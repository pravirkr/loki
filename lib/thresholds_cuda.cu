#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <random>

#include <cuda/std/atomic>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <curanddx.hpp>
#include <highfive/highfive.hpp>
#include <math_constants.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/progress.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/utils.hpp"

namespace loki::detection {

// Define the cuRANDDx Generator Descriptor.
using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                     curanddx::PhiloxRounds<10>() +
                     curanddx::SM<CURANDDX_SM>() + curanddx::Thread());

namespace {

/**
 * Device-side handle for CUDA memory - lightweight POD type
 */
struct FoldVectorHandleDevice {
    float* data = nullptr; // Pointer to the fold data in device memory
    SizeType ntrials{};
    SizeType capacity_ntrials{};
    SizeType nbins{};
    float variance{};
    SizeType pool_id{};  // 0 for pool A, 1 for pool B
    SizeType slot_idx{}; // Slot index within the pool

    __device__ __host__ FoldVectorHandleDevice() = default;
    __device__ __host__ SizeType size() const { return ntrials * nbins; }
    __device__ __host__ bool is_valid() const { return data != nullptr; }
};

/**
 * Device-side allocator interface - can be called from kernels
 */
struct DevicePoolAllocator {
    float* pool_a_data;
    float* pool_b_data;
    int* free_slots_a; // Pointer to free_slots array
    int* free_slots_b;
    int* free_top_a;
    int* free_top_b;
    SizeType slot_size;
    SizeType max_ntrials;
    SizeType nbins;
    SizeType slots_per_pool;
    int current_out_pool; // 0 for A, 1 for B

    /**
     * Allocate from the current "out" pool - callable from device
     */
    __device__ FoldVectorHandleDevice allocate(SizeType ntrials,
                                               float variance) const {
        FoldVectorHandleDevice handle; // handle.data is nullptr by default

        int* free_top   = (current_out_pool == 0) ? free_top_a : free_top_b;
        int* free_slots = (current_out_pool == 0) ? free_slots_a : free_slots_b;
        float* pool_data = (current_out_pool == 0) ? pool_a_data : pool_b_data;

        // DEBUG: Check initial state
        int current_top = atomicAdd(free_top, 0); // Read without modifying
        if (threadIdx.x == 0 && blockIdx.x < 5) { // Only first few blocks
            printf(
                "Block %d: Before alloc, current_top=%d, slots_per_pool=%d\n",
                blockIdx.x, current_top, static_cast<int>(slots_per_pool));
        }

        // Atomically decrement and get the NEW top value
        int new_top = atomicAdd(free_top, -1) - 1;

        if (threadIdx.x == 0 && blockIdx.x < 5) {
            printf("Block %d: After atomicAdd, new_top=%d\n", blockIdx.x,
                   new_top);
        }

        if (new_top >= 0 && new_top < static_cast<int>(slots_per_pool)) {
            int slot_idx = free_slots[new_top];

            if (threadIdx.x == 0 && blockIdx.x < 5) {
                printf("Block %d: slot_idx=%d, slot_size=%d\n", blockIdx.x,
                       slot_idx, static_cast<int>(slot_size));
            }

            if (slot_idx >= 0 && slot_idx < static_cast<int>(slots_per_pool)) {
                handle.data             = pool_data + (slot_idx * slot_size);
                handle.ntrials          = ntrials;
                handle.capacity_ntrials = max_ntrials;
                handle.nbins            = nbins;
                handle.variance         = variance;
                handle.pool_id          = current_out_pool;
                handle.slot_idx         = static_cast<SizeType>(slot_idx);

                if (threadIdx.x == 0 && blockIdx.x < 5) {
                    printf("Block %d: SUCCESS - allocated slot %d\n",
                           blockIdx.x, slot_idx);
                }
            } else {
                // CRITICAL: Restore the counter if slot_idx is invalid
                atomicAdd(free_top, 1);
                if (threadIdx.x == 0 && blockIdx.x < 3) {
                    printf("Block %d: INVALID slot_idx=%d, restored counter\n",
                           blockIdx.x, slot_idx);
                }
            }
        } else {
            // CRITICAL: Restore the counter if new_top is out of range
            atomicAdd(free_top, 1);
            if (threadIdx.x == 0 && blockIdx.x < 3) {
                printf(
                    "Block %d: Pool exhausted, new_top=%d, restored counter\n",
                    blockIdx.x, new_top);
            }
        }
        return handle;
    }

    /**
     * Deallocate - callable from device
     */
    __device__ void deallocate(const FoldVectorHandleDevice& handle) const {
        if (!handle.is_valid()) {
            return;
        }

        int* free_top   = (handle.pool_id == 0) ? free_top_a : free_top_b;
        int* free_slots = (handle.pool_id == 0) ? free_slots_a : free_slots_b;

        // Atomically increment the top pointer. The return value is the old top
        // index.
        int old_top  = atomicAdd(free_top, 1);
        int push_idx = old_top + 1;

        if (push_idx < static_cast<int>(slots_per_pool)) {
            // Write the freed slot index to the new top of the stack.
            free_slots[push_idx] = static_cast<int>(handle.slot_idx);
        } else {
            // This indicates a severe error, like a double-free, causing an
            // overflow. Revert the counter to prevent further corruption.
            atomicAdd(free_top, -1);
        }
    }
};

/**
 * CUDA Dual-Pool Memory Manager using thrust::device_vector for safety
 */
class DualPoolFoldManagerDevice {
public:
    DualPoolFoldManagerDevice(SizeType nbins,
                              SizeType ntrials_min,
                              SizeType slots_per_pool)
        : m_nbins(nbins),
          m_max_ntrials(2 * ntrials_min),
          m_slot_size(m_max_ntrials * nbins),
          m_slots_per_pool(slots_per_pool) {

        // Allocate device memory pools
        m_pool_a.resize(m_slots_per_pool * m_slot_size);
        m_pool_b.resize(m_slots_per_pool * m_slot_size);

        // Allocate free slot stacks (indices 0 to slots-1)
        m_free_slots_a.resize(m_slots_per_pool);
        m_free_slots_b.resize(m_slots_per_pool);
        thrust::sequence(thrust::device, m_free_slots_a.begin(),
                         m_free_slots_a.end(), 0);
        thrust::sequence(thrust::device, m_free_slots_b.begin(),
                         m_free_slots_b.end(), 0);

        // Allocate tops (init to slots_per_pool - 1)
        cudaMalloc(&m_free_top_a, sizeof(int));
        cudaMalloc(&m_free_top_b, sizeof(int));
        int initial_top = static_cast<int>(slots_per_pool) - 1;
        cudaMemcpy(m_free_top_a, &initial_top, sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_free_top_b, &initial_top, sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    ~DualPoolFoldManagerDevice() {
        cudaFree(m_free_top_a);
        cudaFree(m_free_top_b);
    }

    // Delete copy/move operations
    DualPoolFoldManagerDevice(const DualPoolFoldManagerDevice&) = delete;
    DualPoolFoldManagerDevice&
    operator=(const DualPoolFoldManagerDevice&)                       = delete;
    DualPoolFoldManagerDevice(DualPoolFoldManagerDevice&&)            = delete;
    DualPoolFoldManagerDevice& operator=(DualPoolFoldManagerDevice&&) = delete;

    SizeType get_max_ntrials() const { return m_max_ntrials; }
    /**
     * Get device allocator for use in kernels
     */
    DevicePoolAllocator get_device_allocator() {
        return {.pool_a_data  = thrust::raw_pointer_cast(m_pool_a.data()),
                .pool_b_data  = thrust::raw_pointer_cast(m_pool_b.data()),
                .free_slots_a = thrust::raw_pointer_cast(m_free_slots_a.data()),
                .free_slots_b = thrust::raw_pointer_cast(m_free_slots_b.data()),
                .free_top_a   = m_free_top_a,
                .free_top_b   = m_free_top_b,
                .slot_size    = m_slot_size,
                .max_ntrials  = m_max_ntrials,
                .nbins        = m_nbins,
                .slots_per_pool   = m_slots_per_pool,
                .current_out_pool = m_current_out_pool};
    }

    /**
     * Swap pools - must be called from host
     */
    void swap_pools() { m_current_out_pool = 1 - m_current_out_pool; }

    void debug_pool_status() {
        int top_a, top_b;
        cudaMemcpy(&top_a, m_free_top_a, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&top_b, m_free_top_b, sizeof(int), cudaMemcpyDeviceToHost);

        spdlog::info("Pool status: A_top={}, B_top={}, slots_per_pool={}, "
                     "current_out={}",
                     top_a, top_b, m_slots_per_pool, m_current_out_pool);

        if (top_a < -100 || top_b < -100 ||
            top_a >= static_cast<int>(m_slots_per_pool) ||
            top_b >= static_cast<int>(m_slots_per_pool)) {
            spdlog::error("POOL CORRUPTION DETECTED!");
        }
    }

private:
    thrust::device_vector<float> m_pool_a;
    thrust::device_vector<float> m_pool_b;
    thrust::device_vector<int> m_free_slots_a;
    thrust::device_vector<int> m_free_slots_b;
    int* m_free_top_a = nullptr;
    int* m_free_top_b = nullptr;

    SizeType m_nbins;
    SizeType m_max_ntrials;
    SizeType m_slot_size;
    SizeType m_slots_per_pool;
    int m_current_out_pool{};
};

struct FoldsTypeDevice {
    FoldVectorHandleDevice folds_h0;
    FoldVectorHandleDevice folds_h1;

    __device__ __host__ FoldsTypeDevice() = default;

    __device__ __host__ FoldsTypeDevice(FoldVectorHandleDevice h0,
                                        FoldVectorHandleDevice h1)
        : folds_h0(h0),
          folds_h1(h1) {}

    __device__ __host__ bool is_empty() const {
        return !folds_h0.is_valid() || !folds_h1.is_valid() ||
               folds_h0.size() == 0 || folds_h1.size() == 0;
    }
};

// Kernel to properly deallocate folds before clearing
struct DeallocateFunctor {
    DevicePoolAllocator allocator;

    __device__ void operator()(cuda::std::optional<FoldsTypeDevice>& fold) {
        if (fold.has_value()) {
            const auto& f = fold.value();
            if (f.folds_h0.is_valid()) {
                allocator.deallocate(f.folds_h0);
            }
            if (f.folds_h1.is_valid()) {
                allocator.deallocate(f.folds_h1);
            }
            fold = cuda::std::nullopt;
        }
    }
};

// Batch transition data for parallel processing
struct TransitionWorkItem { // NOLINT
    int threshold_idx;
    int prob_idx;
    int input_fold_idx;
    StateD input_state;
    float threshold;
    float nbranches;
    FoldsTypeDevice folds_sim;
    FoldsTypeDevice folds_in_initial; // Only valid for initial stage
    bool is_initial{false};
};

struct TransitionBatch {
    thrust::device_vector<TransitionWorkItem> work_items_d;

    void reserve(SizeType max_items) { work_items_d.reserve(max_items); }
};

__device__ int find_bin_index_device(const float* __restrict__ probs,
                                     int nprobs,
                                     float value) {
    // value below first bin
    if (value < probs[0]) {
        return -1;
    }
    // scan for the first bin > value
    for (int i = 1; i < nprobs; ++i) {
        if (value < probs[i]) {
            return i - 1;
        }
    }
    // value >= last edge
    return nprobs - 1;
}

__device__ void simulate_transition_phase(
    const TransitionWorkItem& work_item,
    const cuda::std::optional<FoldsTypeDevice>* __restrict__ folds_in_ptr,
    const float* __restrict__ profile,
    int nbins,
    float bias_snr,
    float var_add,
    uint64_t seed,
    uint64_t offset) {
    extern __shared__ float shared_profile_scaled[]; // NOLINT
    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    const int block_id   = static_cast<int>(blockIdx.x);

    // Pre-scale profile for H1 (shared across all threads in block)
    for (int i = tid; i < nbins; i += block_size) {
        shared_profile_scaled[i] = profile[i] * bias_snr;
    }
    __syncthreads();

    // Calculate total elements to process for both H0 and H1
    const int total_elements_h0 =
        static_cast<int>(work_item.folds_sim.folds_h0.ntrials) * nbins;
    const int total_elements_h1 =
        static_cast<int>(work_item.folds_sim.folds_h1.ntrials) * nbins;
    const float noise_stddev = sqrtf(var_add);

    // Lambda for processing 4 elements to match cuRANDDx generate4()
    auto process_batch = [&](int base_i, int total_elements,
                             const float* in_data, float* out_data,
                             int input_ntrials, int seq_id_base,
                             bool add_signal) {
        const int seq_id =
            seq_id_base + ((block_id * total_elements + base_i) / 4);
        // Generate noise using cuRANDDx (Use unique sequence ID for each
        // element)
        RNG rng(seed, seq_id, offset);
        curanddx::normal<float, curanddx::box_muller> dist(0.0F, noise_stddev);
        const float4 noise = dist.generate4(rng);

        int trial_idx   = base_i / nbins;
        int bin_idx     = base_i % nbins;
        int orig_trial  = trial_idx % input_ntrials;
        int orig_offset = (orig_trial * nbins) + bin_idx;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = base_i + j;
            if (idx >= total_elements) {
                break;
            }
            float noise_val = j == 0   ? noise.x
                              : j == 1 ? noise.y  // NOLINT
                              : j == 2 ? noise.z  // NOLINT
                                       : noise.w; // NOLINT
            const float profile_val =
                add_signal ? shared_profile_scaled[bin_idx] : 0.0F;
            out_data[idx] = in_data[orig_offset] + noise_val + profile_val;

            bin_idx++;
            if (bin_idx >= nbins) {
                bin_idx = 0;
                trial_idx++;
                orig_trial = trial_idx % input_ntrials;
            }
            orig_offset = orig_trial * nbins + bin_idx;
        }
    };

    // Get the correct input fold data based on whether this is the initial
    // stage
    const FoldsTypeDevice& fold_in =
        work_item.is_initial ? work_item.folds_in_initial
                             : folds_in_ptr[work_item.input_fold_idx].value();
    // H0: no signal bias
    for (int base = tid * 4; base < total_elements_h0; base += block_size * 4) {
        process_batch(base, total_elements_h0, fold_in.folds_h0.data,
                      work_item.folds_sim.folds_h0.data,
                      static_cast<int>(fold_in.folds_h0.ntrials), 0, false);
    }
    // H1: with signal bias
    const int h1_seq_offset = (total_elements_h0 + 3) / 4;
    for (int base = tid * 4; base < total_elements_h1; base += block_size * 4) {
        process_batch(base, total_elements_h1, fold_in.folds_h1.data,
                      work_item.folds_sim.folds_h1.data,
                      static_cast<int>(fold_in.folds_h1.ntrials), h1_seq_offset,
                      true);
    }
}

__device__ float
compute_trial_snr_on_demand(const float* __restrict__ trial_data,
                            int nbins,
                            const SizeType* __restrict__ widths,
                            int nwidths,
                            float stdnoise = 1.0F) {
    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    // Step 1: Compute total sum collaboratively
    float total_sum = 0.0F;
    for (int i = tid; i < nbins; i += block_size) {
        total_sum += trial_data[i];
    }

    // Block-wide reduction for total sum
    for (unsigned int offset = block_size / 2; offset > 0; offset >>= 1U) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }
    // Broadcast to all threads
    total_sum = __shfl_sync(0xFFFFFFFF, total_sum, 0);

    float max_snr = -CUDART_INF_F;

    // Step 2: Process each width
    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = -CUDART_INF_F;

        // Each thread processes multiple starting positions
        for (int start = tid; start < nbins; start += block_size) {
            // Compute windowed sum on-the-fly
            float window_sum = 0.0F;

            for (int i = 0; i < w; ++i) {
                int idx = (start + i) % nbins; // Handle circular wrapping
                window_sum += trial_data[idx];
            }

            thread_max_diff = fmaxf(thread_max_diff, window_sum);
        }

        // Block-wide reduction to find maximum difference for this width
        for (unsigned int offset = block_size / 2; offset > 0; offset >>= 1U) {
            float temp = __shfl_down_sync(0xFFFFFFFF, thread_max_diff, offset);
            thread_max_diff = fmaxf(thread_max_diff, temp);
        }

        if (tid == 0) {
            float snr =
                (((h + b) * thread_max_diff) - (b * total_sum)) / stdnoise;
            max_snr = fmaxf(max_snr, snr);
        }
        __syncthreads();
    }

    // Broadcast final result to all threads
    float result_snr = __shfl_sync(0xFFFFFFFF, max_snr, 0);
    return result_snr;
}

//------------------------------------------------------------------------
// A small device‐side bitonic sort on `data[0..count)`, all in shared memory.
// Rounds `count` up to the next power‐of‐2 internally, but only swaps
// indices < count.
//------------------------------------------------------------------------
__inline__ __device__ void bitonic_sort_shared(int* data, int count) {
    // round up to next power‐of‐2
    unsigned int m = 1;
    while (m < static_cast<unsigned int>(count)) {
        m <<= 1U;
    }

    const unsigned int idx = threadIdx.x;

    for (unsigned int k = 2; k <= m; k <<= 1U) {
        for (unsigned int j = k >> 1U; j > 0; j >>= 1U) {
            if (idx < static_cast<unsigned int>(count)) {
                unsigned int ixj = idx ^ j;
                if (ixj < static_cast<unsigned int>(count)) {
                    bool up = ((idx & k) == 0);
                    int a = data[idx], b = data[ixj];
                    // swap to enforce ascending order if up==true,
                    // or descending if up==false (we only use ascending)
                    if ((a > b) == up) {
                        data[idx] = b;
                        data[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

//------------------------------------------------------------------------
// In‐place "score and prune" kernel.
// - Reads + scores trials from folds_sim.[h0|h1]
// - Builds two shared‐memory lists of surviving indices
// - Sorts each list ascending
// - Copies each surviving trial "to the left" in-place
// - Writes back n_success for h0 / h1
//------------------------------------------------------------------------

__device__ void
score_and_prune_fused(FoldsTypeDevice& folds_sim,
                      const SizeType* __restrict__ box_score_widths,
                      int nwidths,
                      int nbins,
                      float threshold) {
    extern __shared__ int shm[]; // NOLINT

    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);

    // max trials in each sim‐buffer
    const int max_trials_h0 = static_cast<int>(folds_sim.folds_h0.ntrials);
    const int max_trials_h1 = static_cast<int>(folds_sim.folds_h1.ntrials);

    // shared counters + index arrays
    __shared__ int shared_count_h0;
    __shared__ int shared_count_h1;
    int* shared_indices_h0 = &shm[0];
    int* shared_indices_h1 = &shm[max_trials_h0];

    if (tid == 0) {
        shared_count_h0 = 0;
        shared_count_h1 = 0;
    }
    __syncthreads();

    // Collect surviving H0 and H1 trials into shared lists
    auto collect = [&](const float* trial_data, int sim_trials, float variance,
                       int& shared_count, int* shared_idx) {
        // Local buffer for good indices
        int local_good[32]; // NOLINT
        int local_count      = 0;
        const float stdnoise = sqrtf(variance);

        // strided over trials
        for (int i = tid; i < sim_trials; i += block_size) {
            float score = compute_trial_snr_on_demand(
                trial_data + static_cast<IndexType>(i * nbins), nbins,
                box_score_widths, nwidths, stdnoise);
            if (score > threshold) {
                local_good[local_count++] = i;
                if (local_count == 32) {
                    // Flush local buffer to shared memory
                    int pos = atomicAdd(&shared_count, 32);
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        if (pos + j < sim_trials) {
                            shared_idx[pos + j] = local_good[j];
                        }
                    }
                    local_count = 0;
                }
            }
        }
        // flush remainder
        if (local_count > 0) {
            int pos = atomicAdd(&shared_count, local_count);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                if (j < local_count && pos + j < sim_trials) {
                    shared_idx[pos + j] = local_good[j];
                }
            }
        }
    };

    // pointers into the single in‐place buffers
    float* folds_sim_h0 = folds_sim.folds_h0.data;
    float* folds_sim_h1 = folds_sim.folds_h1.data;

    collect(folds_sim_h0, static_cast<int>(folds_sim.folds_h0.ntrials),
            folds_sim.folds_h0.variance, shared_count_h0, shared_indices_h0);
    collect(folds_sim_h1, static_cast<int>(folds_sim.folds_h1.ntrials),
            folds_sim.folds_h1.variance, shared_count_h1, shared_indices_h1);
    __syncthreads();

    if (tid == 0) {
        printf("Block %d: H0 %d→%d, H1 %d→%d, thr=%.3f\n", blockIdx.x,
               static_cast<int>(folds_sim.folds_h0.ntrials), shared_count_h0,
               static_cast<int>(folds_sim.folds_h1.ntrials), shared_count_h1,
               threshold);
    }

    // *** Sort the surviving‐trial indices ascending ***
    bitonic_sort_shared(shared_indices_h0, shared_count_h0);
    bitonic_sort_shared(shared_indices_h1, shared_count_h1);
    __syncthreads();

    // *** In‐place copy “to the left” ***
    for (int i = tid; i < shared_count_h0; i += block_size) {
        const int orig_trial = shared_indices_h0[i];
        if (orig_trial != i) {
            const int in_off  = orig_trial * nbins;
            const int out_off = i * nbins;
            for (int j = 0; j < nbins; ++j) {
                folds_sim_h0[out_off + j] = folds_sim_h0[in_off + j];
            }
        }
    }
    for (int i = tid; i < shared_count_h1; i += block_size) {
        const int orig_trial = shared_indices_h1[i];
        if (orig_trial != i) {
            const int in_off  = orig_trial * nbins;
            const int out_off = i * nbins;
            for (int j = 0; j < nbins; ++j) {
                folds_sim_h1[out_off + j] = folds_sim_h1[in_off + j];
            }
        }
    }
    __syncthreads();
    // Update ntrials in the handle itself
    if (tid == 0) {
        folds_sim.folds_h0.ntrials = shared_count_h0;
        folds_sim.folds_h1.ntrials = shared_count_h1;
    }
}

__global__ void merged_transition_kernel(
    TransitionWorkItem* __restrict__ work_items,
    int num_items,
    const cuda::std::optional<FoldsTypeDevice>* __restrict__ folds_in_ptr,
    const float* __restrict__ profile,
    int nbins,
    const SizeType* __restrict__ box_score_widths,
    int nwidths,
    float bias_snr,
    float var_add,
    const float* __restrict__ probs,
    int nprobs,
    int stage_offset_cur,
    StateD* __restrict__ states_out_ptr,
    cuda::std::optional<FoldsTypeDevice>* __restrict__ folds_out_ptr,
    int* __restrict__ locks_ptr,
    DevicePoolAllocator allocator, // Pass by value since it's lightweight
    uint64_t seed,
    uint64_t offset) {

    const auto item_idx = static_cast<int>(blockIdx.x);
    if (item_idx >= num_items) {
        return;
    }

    auto& work_item = work_items[item_idx];
    const auto tid  = static_cast<int>(threadIdx.x);

    // Shared memory for pre-pruning trial counts
    __shared__ int ntrials_sim_h0_before_prune;
    __shared__ int ntrials_sim_h1_before_prune;

    // Phase 1: Simulation (threads collaborate)
    simulate_transition_phase(work_item, folds_in_ptr, profile, nbins, bias_snr,
                              var_add, seed, offset);
    __syncthreads();

    if (tid == 0) {
        ntrials_sim_h0_before_prune =
            static_cast<int>(work_item.folds_sim.folds_h0.ntrials);
        ntrials_sim_h1_before_prune =
            static_cast<int>(work_item.folds_sim.folds_h1.ntrials);
    }

    // Phase 2: Fused Score and Prune (threads collaborate)
    score_and_prune_fused(work_item.folds_sim, box_score_widths, nwidths, nbins,
                          work_item.threshold);
    __syncthreads();

    // Phase 3: Compute and update (thread 0 only)
    if (tid == 0) {
        // Store the number of trials after score and prune
        const auto ntrials_h0_out = work_item.folds_sim.folds_h0.ntrials;
        const auto ntrials_h1_out = work_item.folds_sim.folds_h1.ntrials;

        // Calculate success rates (handle division by zero)
        const auto success_h0 =
            ntrials_sim_h0_before_prune > 0
                ? static_cast<float>(ntrials_h0_out) /
                      static_cast<float>(ntrials_sim_h0_before_prune)
                : 0.0F;
        const auto success_h1 =
            ntrials_sim_h1_before_prune > 0
                ? static_cast<float>(ntrials_h1_out) /
                      static_cast<float>(ntrials_sim_h1_before_prune)
                : 0.0F;

        // Generate next state
        const auto state_next = work_item.input_state.gen_next(
            work_item.threshold, success_h0, success_h1, work_item.nbranches);

        const auto iprob =
            find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);

        bool stored_new_folds = false;
        if (iprob >= 0 && iprob < nprobs) {
            const int fold_idx  = (work_item.threshold_idx * nprobs) + iprob;
            const int state_idx = stage_offset_cur + fold_idx;

            // Acquire lock
            int* lock = &locks_ptr[state_idx];
            while (atomicCAS(lock, 0, 1) != 0) {
            }

            auto& existing_state = states_out_ptr[state_idx];
            auto& existing_folds = folds_out_ptr[fold_idx];

            if (existing_state.is_empty ||
                state_next.complexity_cumul < existing_state.complexity_cumul) {
                // Deallocate old folds before overwriting to prevent leaks
                if (existing_folds.has_value()) {
                    allocator.deallocate(existing_folds->folds_h0);
                    allocator.deallocate(existing_folds->folds_h1);
                }
                // Update to better state
                existing_state = state_next;
                // Move pruned folds to persistent storage
                existing_folds   = work_item.folds_sim;
                stored_new_folds = true;
            }
            // Release lock
            atomicExch(lock, 0);
        }

        // Deallocate temporary simulation folds if they were not stored
        if (!stored_new_folds) {
            allocator.deallocate(work_item.folds_sim.folds_h0);
            allocator.deallocate(work_item.folds_sim.folds_h1);
        }

        // Deallocate temporary initial folds if this is the initial stage
        if (work_item.is_initial) {
            allocator.deallocate(work_item.folds_in_initial.folds_h0);
            allocator.deallocate(work_item.folds_in_initial.folds_h1);
        }
    }
}

struct CountValidTransitions {
    const StateD* states_ptr;
    const cuda::std::optional<FoldsTypeDevice>* folds_ptr;
    SizeType stage_offset_prev;
    SizeType nprobs;

    __device__ SizeType
    operator()(const std::pair<SizeType, SizeType>& pair) const {
        SizeType count         = 0;
        const SizeType jthresh = pair.second;
        for (SizeType kprob = 0; kprob < nprobs; ++kprob) {
            const auto prev_fold_idx = (jthresh * nprobs) + kprob;
            const auto& prev_state =
                states_ptr[stage_offset_prev + prev_fold_idx];

            if (prev_state.is_empty) {
                continue;
            }

            const auto& prev_fold_state = folds_ptr[prev_fold_idx];

            if (blockIdx.x < 1) { // Print for first few thread blocks only
                bool is_invalid =
                    !prev_fold_state.has_value() || prev_fold_state->is_empty();
                printf("Count: j=%d,k=%d,idx=%d | has_val=%d,is_empty=%d -> "
                       "is_invalid=%d\n",
                       (int)jthresh, (int)kprob, (int)prev_fold_idx,
                       (int)prev_fold_state.has_value(),
                       (int)prev_fold_state->is_empty(), (int)is_invalid);

                if (prev_fold_state.has_value()) {
                    printf("     h0_ptr=%p, h0_ntrials=%d | h1_ptr=%p, "
                           "h1_ntrials=%d\n",
                           prev_fold_state->folds_h0.data,
                           (int)prev_fold_state->folds_h0.ntrials,
                           prev_fold_state->folds_h1.data,
                           (int)prev_fold_state->folds_h1.ntrials);
                }
            }
            if (!prev_fold_state.has_value() || prev_fold_state->is_empty()) {
                continue;
            }

            count++;
        }
        return count;
    }
};

struct InitialWorkItemsFunctor {
    const float* thresholds_ptr;
    const float* branching_pattern_ptr;
    SizeType ntrials;
    float var_init;
    TransitionWorkItem* work_items_ptr;
    DevicePoolAllocator* allocator;

    __device__ void operator()(SizeType ithres) const {
        // Allocate initial folds (zero-filled)
        auto folds_h0_init = allocator->allocate(ntrials, 0.0F);
        auto folds_h1_init = allocator->allocate(ntrials, 0.0F);
        auto folds_h0_sim  = allocator->allocate(ntrials, var_init);
        auto folds_h1_sim  = allocator->allocate(ntrials, var_init);

        // Create work item
        TransitionWorkItem item;
        item.threshold_idx                = static_cast<int>(ithres);
        item.prob_idx                     = -1; // Special marker for initial
        item.input_fold_idx               = -1;
        item.input_state                  = StateD(); // Default initial state
        item.input_state.is_empty         = false;
        item.input_state.complexity_cumul = 1.0F;
        item.input_state.success_h1_cumul = 1.0F;
        item.threshold                    = thresholds_ptr[ithres];
        item.nbranches        = branching_pattern_ptr[0]; // First stage
        item.folds_in_initial = FoldsTypeDevice(folds_h0_init, folds_h1_init);
        item.folds_sim        = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);
        item.is_initial       = true;

        work_items_ptr[ithres] = item;
    }
};

__global__ void
zero_fill_initial_kernel(const TransitionWorkItem* __restrict__ work_items,
                         int num_items,
                         int nbins) {

    const auto item_idx = static_cast<int>(blockIdx.x);
    if (item_idx >= num_items)
        return;

    const auto& work_item = work_items[item_idx];
    const auto tid        = static_cast<int>(threadIdx.x);
    const auto block_size = static_cast<int>(blockDim.x);

    // Zero-fill input folds collaboratively
    const int total_elements =
        static_cast<int>(work_item.folds_in_initial.folds_h0.ntrials) * nbins;

    for (int idx = tid; idx < total_elements; idx += block_size) {
        work_item.folds_in_initial.folds_h0.data[idx] = 0.0F;
        work_item.folds_in_initial.folds_h1.data[idx] = 0.0F;
    }
}

struct TransitionFunctor {
    const StateD* states_ptr;
    const cuda::std::optional<FoldsTypeDevice>* folds_ptr;
    const float* thresholds_ptr;
    const float* branching_pattern_ptr;
    SizeType nprobs;
    SizeType ntrials;
    SizeType stage_offset_prev;
    SizeType istage;
    float var_add;
    TransitionWorkItem* work_items_ptr;
    const SizeType* offset_ptr;
    DevicePoolAllocator* allocator;
    SizeType batch_start;
    SizeType batch_end;

    TransitionFunctor(
        const thrust::device_vector<StateD>& states_d,
        const thrust::device_vector<cuda::std::optional<FoldsTypeDevice>>&
            folds_current_d,
        const thrust::device_vector<float>& thresholds,
        const thrust::device_vector<float>& branching_pattern,
        SizeType nprobs,
        SizeType ntrials,
        SizeType offset_prev,
        SizeType stage,
        float var,
        thrust::device_vector<TransitionWorkItem>& items_d,
        const thrust::device_vector<SizeType>& offsets,
        DevicePoolAllocator* allocator,
        SizeType batch_start,
        SizeType batch_end)
        : states_ptr(thrust::raw_pointer_cast(states_d.data())),
          folds_ptr(thrust::raw_pointer_cast(folds_current_d.data())),
          thresholds_ptr(thrust::raw_pointer_cast(thresholds.data())),
          branching_pattern_ptr(
              thrust::raw_pointer_cast(branching_pattern.data())),
          nprobs(nprobs),
          ntrials(ntrials),
          stage_offset_prev(offset_prev),
          istage(stage),
          var_add(var),
          work_items_ptr(thrust::raw_pointer_cast(items_d.data())),
          offset_ptr(thrust::raw_pointer_cast(offsets.data())),
          allocator(allocator),
          batch_start(batch_start),
          batch_end(batch_end) {}

    __device__ void operator()(
        const thrust::tuple<std::pair<SizeType, SizeType>, SizeType>& input)
        const {
        const auto& pair        = thrust::get<0>(input);
        const SizeType pair_idx = thrust::get<1>(input);

        SizeType base_offset = offset_ptr[pair_idx];
        if (base_offset >= batch_end) {
            return;
        }

        SizeType slot = 0;
        for (SizeType kprob = 0; kprob < nprobs; ++kprob) {
            SizeType global_slot = base_offset + slot;
            if (global_slot < batch_start) {
                slot++;
                continue;
            }
            if (global_slot >= batch_end) {
                break;
            }
            const auto prev_fold_idx = (pair.second * nprobs) + kprob;
            const auto& prev_state =
                states_ptr[stage_offset_prev + prev_fold_idx];

            if (prev_state.is_empty) {
                continue;
            }

            const auto& prev_fold_state = folds_ptr[prev_fold_idx];
            if (!prev_fold_state.has_value() || prev_fold_state->is_empty()) {
                continue;
            }

            // Pre-allocate output buffers
            const auto ntrials_in_h0 = prev_fold_state->folds_h0.ntrials;
            const auto ntrials_in_h1 = prev_fold_state->folds_h1.ntrials;
            if (ntrials_in_h0 == 0 || ntrials_in_h1 == 0) {
                continue;
            }
            const auto repeat_h0 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h0)));
            const auto ntrials_out_h0 = repeat_h0 * ntrials_in_h0;
            const auto repeat_h1 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h1)));
            const auto ntrials_out_h1 = repeat_h1 * ntrials_in_h1;

            auto folds_h0_sim = allocator->allocate(
                ntrials_out_h0, prev_fold_state->folds_h0.variance + var_add);
            auto folds_h1_sim = allocator->allocate(
                ntrials_out_h1, prev_fold_state->folds_h1.variance + var_add);

            // Create work item
            TransitionWorkItem item;
            item.threshold_idx  = static_cast<int>(pair.first);
            item.prob_idx       = static_cast<int>(kprob);
            item.input_fold_idx = static_cast<int>(prev_fold_idx);
            item.input_state    = prev_state;
            item.threshold      = thresholds_ptr[pair.first];
            item.nbranches      = branching_pattern_ptr[istage];
            item.folds_sim      = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);

            work_items_ptr[global_slot - batch_start] = item;
            slot++;
        }
    }
};

// Create a compound type for State
HighFive::CompoundType create_compound_state() {
    return {{"success_h0", HighFive::create_datatype<float>()},
            {"success_h1", HighFive::create_datatype<float>()},
            {"complexity", HighFive::create_datatype<float>()},
            {"complexity_cumul", HighFive::create_datatype<float>()},
            {"success_h1_cumul", HighFive::create_datatype<float>()},
            {"nbranches", HighFive::create_datatype<float>()},
            {"threshold", HighFive::create_datatype<float>()},
            {"cost", HighFive::create_datatype<float>()},
            {"threshold_prev", HighFive::create_datatype<float>()},
            {"success_h1_cumul_prev", HighFive::create_datatype<float>()},
            {"is_empty", HighFive::create_datatype<bool>()}};
}

} // namespace

__host__ __device__ StateD::StateD() noexcept
    : success_h0(1.0F),
      success_h1(1.0F),
      complexity(1.0F),
      complexity_cumul(std::numeric_limits<float>::infinity()),
      success_h1_cumul(1.0F),
      nbranches(1.0F),
      threshold(-1.0F),
      cost(std::numeric_limits<float>::infinity()),
      threshold_prev(-1.0F),
      success_h1_cumul_prev(1.0F),
      is_empty(true) {}

__host__ __device__ StateD StateD::gen_next(float threshold,
                                            float success_h0,
                                            float success_h1,
                                            float nbranches) const noexcept {
    const auto nleaves_next          = this->complexity * nbranches;
    const auto nleaves_surv          = nleaves_next * success_h0;
    const auto complexity_cumul_next = this->complexity_cumul + nleaves_next;
    const auto success_h1_cumul_next = this->success_h1_cumul * success_h1;
    const auto cost_next = complexity_cumul_next / success_h1_cumul_next;

    // Create a new state struct
    StateD state_next;
    state_next.success_h0       = success_h0;
    state_next.success_h1       = success_h1;
    state_next.complexity       = nleaves_surv;
    state_next.complexity_cumul = complexity_cumul_next;
    state_next.success_h1_cumul = success_h1_cumul_next;
    state_next.nbranches        = nbranches;
    state_next.threshold        = threshold;
    state_next.cost             = cost_next;
    state_next.is_empty         = false;
    // For backtracking
    state_next.threshold_prev        = this->threshold;
    state_next.success_h1_cumul_prev = this->success_h1_cumul;
    return state_next;
}

__host__ __device__ State StateD::to_state() const {
    State result;
    result.success_h0            = this->success_h0;
    result.success_h1            = this->success_h1;
    result.complexity            = this->complexity;
    result.complexity_cumul      = this->complexity_cumul;
    result.success_h1_cumul      = this->success_h1_cumul;
    result.nbranches             = this->nbranches;
    result.threshold             = this->threshold;
    result.cost                  = this->cost;
    result.threshold_prev        = this->threshold_prev;
    result.success_h1_cumul_prev = this->success_h1_cumul_prev;
    result.is_empty              = this->is_empty;
    return result;
}

struct StateConversionFunctor {
    __device__ __host__ State operator()(const StateD& state_d) const {
        return state_d.to_state();
    }
};

// CUDA-specific implementation
class DynamicThresholdSchemeCUDA::Impl {
public:
    Impl(std::span<const float> branching_pattern,
         float ref_ducy,
         SizeType nbins,
         SizeType ntrials,
         SizeType nprobs,
         float prob_min,
         float snr_final,
         SizeType nthresholds,
         float ducy_max,
         float wtsp,
         float beam_width,
         SizeType trials_start,
         int device_id)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_trials_start(trials_start),
          m_device_id(device_id),
          m_batch_size(256) {

        cuda_utils::set_device(m_device_id);
        if (m_branching_pattern.empty()) {
            throw std::invalid_argument("Branching pattern is empty");
        }
        // Host-side computations
        m_profile = simulation::generate_folded_profile(nbins, ref_ducy);
        m_thresholds =
            detection::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs       = detection::compute_probs(nprobs, prob_min);
        m_nprobs      = m_probs.size();
        m_nbins       = m_profile.size();
        m_nstages     = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        m_box_score_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_guess_path = detection::guess_scheme(
            m_nstages, snr_final, m_branching_pattern, m_trials_start);

        // Copy data to device
        m_branching_pattern_d = m_branching_pattern;
        m_profile_d           = m_profile;
        m_thresholds_d        = m_thresholds;
        m_probs_d             = m_probs;
        m_box_score_widths_d  = m_box_score_widths;

        // Initialize memory management
        const auto slots_per_pool = compute_max_allocations_needed();
        m_device_manager          = std::make_unique<DualPoolFoldManagerDevice>(
            m_nbins, m_ntrials, slots_per_pool);
        spdlog::info("Pre-allocated 2 CUDA pools of {} slots each",
                     slots_per_pool);

        // Initialize state management
        m_folds_current_d.resize(m_nthresholds * m_nprobs);
        m_folds_next_d.resize(m_nthresholds * m_nprobs);
        const auto grid_size = m_nstages * m_nthresholds * m_nprobs;
        m_states_d.resize(grid_size, StateD());
        m_states_locks_d.resize(grid_size);
        thrust::fill(m_states_locks_d.begin(), m_states_locks_d.end(), 0);
        m_states.resize(grid_size, State{});
        auto allocator = m_device_manager->get_device_allocator();
        init_states(allocator);
        m_device_manager->swap_pools();
    }
    ~Impl()                          = default;
    Impl(const Impl&)                = delete;
    Impl& operator=(const Impl&)     = delete;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;

    // Methods
    void run(SizeType thres_neigh = 10) {
        spdlog::info("Running dynamic threshold scheme on CUDA");
        progress::ProgressGuard progress_guard(true);
        auto bar = progress::make_standard_bar("Computing scheme...");

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            // Get device allocator for this stage
            auto allocator = m_device_manager->get_device_allocator();
            run_segment(istage, thres_neigh, allocator);
            m_device_manager->swap_pools();
            std::swap(m_folds_current_d, m_folds_next_d);
            spdlog::info(
                "After swap for stage {}, checking folds_current_d validity",
                istage);
            // Deallocate using thrust with a fresh allocator
            auto deallocator = m_device_manager->get_device_allocator();
            thrust::for_each(thrust::device, m_folds_next_d.begin(),
                             m_folds_next_d.end(),
                             DeallocateFunctor{deallocator});
            cudaDeviceSynchronize();

            const auto progress = static_cast<float>(istage) /
                                  static_cast<float>(m_nstages - 1) * 100.0F;
            bar.set_progress(static_cast<SizeType>(progress));
        }
        // Copy final states back to host
        thrust::transform(thrust::device, m_states_d.begin(), m_states_d.end(),
                          m_states.begin(), StateConversionFunctor{});
    }

    std::string save(const std::string& outdir = "./") const {
        const std::filesystem::path filebase = std::format(
            "dynscheme_nstages_{:03d}_nthresh_{:03d}_nprobs_{:03d}_"
            "ntrials_{:04d}_snr_{:04.1f}_ducy_{:04.2f}_beam_{:03.1f}.h5",
            m_nstages, m_nthresholds, m_nprobs, m_ntrials, m_thresholds.back(),
            m_ref_ducy, m_beam_width);
        const std::filesystem::path filepath =
            std::filesystem::path(outdir) / filebase;
        HighFive::File file(filepath, HighFive::File::Overwrite);
        // Save simple attributes
        file.createAttribute("ntrials", m_ntrials);
        file.createAttribute("snr_final", m_thresholds.back());
        file.createAttribute("ref_ducy", m_ref_ducy);
        file.createAttribute("ducy_max", m_ducy_max);
        file.createAttribute("wtsp", m_wtsp);
        file.createAttribute("beam_width", m_beam_width);

        // Create dataset creation property list and enable compression
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1024}));
        props.add(HighFive::Deflate(9));

        // Save arrays
        file.createDataSet("branching_pattern", m_branching_pattern);
        file.createDataSet("profile", m_profile);
        file.createDataSet("thresholds", m_thresholds);
        file.createDataSet("probs", m_probs);
        file.createDataSet("guess_path", m_guess_path);
        // Define the 3D dataspace for states
        std::vector<SizeType> dims = {m_nstages, m_nthresholds, m_nprobs};
        HighFive::DataSetCreateProps props_states;
        std::vector<hsize_t> chunk_dims(dims.begin(), dims.end());
        props_states.add(HighFive::Chunking(chunk_dims));
        auto dataset =
            file.createDataSet("states", HighFive::DataSpace(dims),
                               create_compound_state(), props_states);
        dataset.write_raw(m_states.data());
        spdlog::info("Saved dynamic threshold scheme to {}", filepath.string());
        return filepath.string();
    }

private:
    // Host-side parameters and metadata
    std::vector<float> m_branching_pattern;
    float m_ref_ducy;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    SizeType m_trials_start;
    int m_device_id;
    SizeType m_batch_size{};

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    std::vector<SizeType> m_box_score_widths;
    float m_bias_snr;
    std::vector<float> m_guess_path;
    std::vector<State> m_states;

    // Memory and RNG management
    std::unique_ptr<DualPoolFoldManagerDevice> m_device_manager;

    // Device-side data
    thrust::device_vector<float> m_branching_pattern_d;
    thrust::device_vector<float> m_profile_d;
    thrust::device_vector<float> m_thresholds_d;
    thrust::device_vector<float> m_probs_d;
    thrust::device_vector<SizeType> m_box_score_widths_d;
    thrust::device_vector<StateD> m_states_d;
    thrust::device_vector<int> m_states_locks_d;
    thrust::device_vector<cuda::std::optional<FoldsTypeDevice>>
        m_folds_current_d;
    thrust::device_vector<cuda::std::optional<FoldsTypeDevice>> m_folds_next_d;

    SizeType compute_max_allocations_needed() {
        SizeType max_active_per_stage = 0;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            auto active_thresholds = get_current_thresholds_idx(istage);
            max_active_per_stage =
                std::max(max_active_per_stage, active_thresholds.size());
        }
        // h0 + h1 per cell
        const auto max_persistent = max_active_per_stage * m_nprobs * 2;
        // 2 simulated folds per transition
        const auto max_temporary =
            std::max((m_batch_size * 2), (max_active_per_stage * 2));
        const auto slots_per_pool = max_persistent + max_temporary;
        spdlog::info(
            "CUDA allocation analysis: {} active thresholds max, {} prob "
            "bins",
            max_active_per_stage, m_nprobs);
        spdlog::info("Need {} persistent + {} temporary = {} slots per pool",
                     max_persistent, max_temporary, slots_per_pool);
        return slots_per_pool;
    }

    void init_states(DevicePoolAllocator& allocator) {
        const float var_init      = 1.0F;
        const auto thresholds_idx = get_current_thresholds_idx(0);

        spdlog::info("Initial stage: {} threshold indices",
                     thresholds_idx.size());

        if (thresholds_idx.empty()) {
            spdlog::warn("No initial thresholds!");
            return;
        }

        // Create work items for initial stage
        thrust::device_vector<TransitionWorkItem> initial_work_items_d(
            thresholds_idx.size());

        // Create indices array on device
        thrust::device_vector<SizeType> indices_d = thresholds_idx;

        // Populate initial work items
        thrust::for_each(
            thrust::device, indices_d.begin(), indices_d.end(),
            InitialWorkItemsFunctor{
                .thresholds_ptr =
                    thrust::raw_pointer_cast(m_thresholds_d.data()),
                .branching_pattern_ptr =
                    thrust::raw_pointer_cast(m_branching_pattern_d.data()),
                .ntrials  = m_ntrials,
                .var_init = var_init,
                .work_items_ptr =
                    thrust::raw_pointer_cast(initial_work_items_d.data()),
                .allocator = &allocator});

        // Zero-fill the input folds
        dim3 block_dim(256);
        dim3 grid_dim(static_cast<int>(thresholds_idx.size()));

        zero_fill_initial_kernel<<<grid_dim, block_dim>>>(
            thrust::raw_pointer_cast(initial_work_items_d.data()),
            static_cast<int>(thresholds_idx.size()), static_cast<int>(m_nbins));

        cuda_utils::check_last_cuda_error("zero_fill_initial_kernel");
        cudaDeviceSynchronize();

        // Process initial work items through the main kernel
        process_transition_batch(
            initial_work_items_d, var_init, 0, nullptr,
            thrust::raw_pointer_cast(m_folds_current_d.data()), allocator);

        // Check results
        thrust::host_vector<StateD> states_h(m_nthresholds * m_nprobs);
        thrust::copy(m_states_d.begin(),
                     m_states_d.begin() + (m_nthresholds * m_nprobs),
                     states_h.begin());

        int non_empty_count = 0;
        for (const auto& state : states_h) {
            if (!state.is_empty) {
                non_empty_count++;
            }
        }
        spdlog::info("Initial stage created {} non-empty states",
                     non_empty_count);
    }

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const {
        const auto guess       = m_guess_path[istage];
        const auto half_extent = m_beam_width;
        const auto lower_bound = std::max(0.0F, guess - half_extent);
        const auto upper_bound =
            std::min(m_thresholds.back(), guess + half_extent);

        std::vector<SizeType> result;
        for (SizeType i = 0; i < m_thresholds.size(); ++i) {
            if (m_thresholds[i] >= lower_bound &&
                m_thresholds[i] <= upper_bound) {
                result.push_back(i);
            }
        }
        return result;
    }

    void run_segment(SizeType istage,
                     SizeType thres_neigh,
                     DevicePoolAllocator& allocator) {
        const float var_add          = 1.0F;
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;
        spdlog::info("Running segment {} of {} ({} thresholds)", istage,
                     m_nstages, beam_idx_cur.size());

        // Step 1: Generate all possible (ithres, jthresh) pairs
        std::vector<std::pair<SizeType, SizeType>> threshold_pairs;
        for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
            const auto ithres = beam_idx_cur[i];
            const auto neighbour_beam_indices =
                utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                 thres_neigh);
            for (SizeType jthresh : neighbour_beam_indices) {
                threshold_pairs.emplace_back(ithres, jthresh);
            }
        }

        if (threshold_pairs.empty()) {
            spdlog::info("All keys size: 0");
            return;
        }

        // Step 2: Count valid transitions (fast pass)
        thrust::device_vector<std::pair<SizeType, SizeType>> threshold_pairs_d =
            threshold_pairs;
        thrust::device_vector<SizeType> transition_counts_d(
            threshold_pairs.size());

        // Count transitions per pair
        thrust::transform(
            thrust::device, threshold_pairs_d.begin(), threshold_pairs_d.end(),
            transition_counts_d.begin(),
            CountValidTransitions{
                .states_ptr = thrust::raw_pointer_cast(m_states_d.data()),
                .folds_ptr = thrust::raw_pointer_cast(m_folds_current_d.data()),
                .stage_offset_prev = stage_offset_prev,
                .nprobs            = m_nprobs});

        // Compute prefix sum for offsets
        thrust::device_vector<SizeType> offsets_d(threshold_pairs.size() + 1);
        thrust::exclusive_scan(thrust::device, transition_counts_d.begin(),
                               transition_counts_d.end(), offsets_d.begin(), 0);

        SizeType total_transitions = offsets_d.back();
        spdlog::info("All transitions count: {}", total_transitions);

        if (total_transitions == 0) {
            return;
        }

        // Step 3: Process in batches
        const SizeType num_batches =
            (total_transitions + m_batch_size - 1) / m_batch_size;

        for (SizeType b = 0; b < num_batches; ++b) {
            const SizeType start = b * m_batch_size;
            const SizeType end =
                std::min(start + m_batch_size, total_transitions);
            const SizeType current_batch_size = end - start;

            // Create work items for this batch
            thrust::device_vector<TransitionWorkItem> work_items_d(
                current_batch_size);

            // Populate work items using TransitionFunctor
            auto zip_begin = thrust::make_zip_iterator(
                thrust::make_tuple(threshold_pairs_d.begin(),
                                   thrust::counting_iterator<SizeType>(0)));
            auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
                threshold_pairs_d.end(),
                thrust::counting_iterator<SizeType>(threshold_pairs.size())));

            thrust::for_each(
                thrust::device, zip_begin, zip_end,
                TransitionFunctor(m_states_d, m_folds_current_d, m_thresholds_d,
                                  m_branching_pattern_d, m_nprobs, m_ntrials,
                                  stage_offset_prev, istage, var_add,
                                  work_items_d, offsets_d, &allocator, start,
                                  end));

            // Process this batch
            process_transition_batch(
                work_items_d, var_add, stage_offset_cur,
                thrust::raw_pointer_cast(m_folds_current_d.data()),
                thrust::raw_pointer_cast(m_folds_next_d.data()), allocator);
        }
    }

    void process_transition_batch(
        thrust::device_vector<TransitionWorkItem>& work_items_d,
        float var_add,
        SizeType stage_offset_cur,
        const cuda::std::optional<FoldsTypeDevice>* __restrict__ folds_in_ptr,
        cuda::std::optional<FoldsTypeDevice>* __restrict__ folds_out_ptr,
        DevicePoolAllocator& allocator) {
        const auto num_items = static_cast<int>(work_items_d.size());
        if (num_items == 0) {
            return;
        }

        // Generate random seed and offset
        const uint64_t seed   = std::random_device{}();
        const uint64_t offset = 0;
        // Launch unified kernel: one block per transition
        const dim3 block_dim(256);
        const dim3 grid_dim(num_items);
        // simulate phase and score_and_prune phase
        const SizeType profile_mem = m_nbins * sizeof(float);
        const SizeType pruning_mem =
            (2 * sizeof(int)) +
            (2 * m_device_manager->get_max_ntrials() * sizeof(int));
        const SizeType shared_mem_size = std::max(profile_mem, pruning_mem);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                               shared_mem_size);

        merged_transition_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            thrust::raw_pointer_cast(work_items_d.data()), num_items,
            folds_in_ptr, thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins),
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            static_cast<int>(m_box_score_widths_d.size()), m_bias_snr, var_add,
            thrust::raw_pointer_cast(m_probs_d.data()),
            static_cast<int>(m_nprobs), static_cast<int>(stage_offset_cur),
            thrust::raw_pointer_cast(m_states_d.data()), folds_out_ptr,
            thrust::raw_pointer_cast(m_states_locks_d.data()), allocator, seed,
            offset);
        cuda_utils::check_last_cuda_error("merged_transition_kernel");
        cudaDeviceSynchronize();
    }
};

DynamicThresholdSchemeCUDA::DynamicThresholdSchemeCUDA(
    std::span<const float> branching_pattern,
    float ref_ducy,
    SizeType nbins,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType trials_start,
    int device_id)
    : m_impl(std::make_unique<Impl>(branching_pattern,
                                    ref_ducy,
                                    nbins,
                                    ntrials,
                                    nprobs,
                                    prob_min,
                                    snr_final,
                                    nthresholds,
                                    ducy_max,
                                    wtsp,
                                    beam_width,
                                    trials_start,
                                    device_id)) {}
DynamicThresholdSchemeCUDA::~DynamicThresholdSchemeCUDA() = default;
DynamicThresholdSchemeCUDA::DynamicThresholdSchemeCUDA(
    DynamicThresholdSchemeCUDA&&) noexcept = default;
DynamicThresholdSchemeCUDA& DynamicThresholdSchemeCUDA::operator=(
    DynamicThresholdSchemeCUDA&&) noexcept = default;

void DynamicThresholdSchemeCUDA::run(SizeType thres_neigh) {
    m_impl->run(thres_neigh);
}
std::string DynamicThresholdSchemeCUDA::save(const std::string& outdir) const {
    return m_impl->save(outdir);
}

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)