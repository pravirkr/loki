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
#include <cuda_runtime.h>
#include <curanddx.hpp>
#include <highfive/highfive.hpp>
#include <math_constants.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/progress.hpp"

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
    int* next_free_a;
    int* next_free_b;
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

        // Select the appropriate pool and counter
        int* next_free_counter =
            (current_out_pool == 0) ? next_free_a : next_free_b;
        float* pool_data = (current_out_pool == 0) ? pool_a_data : pool_b_data;

        // Atomically get a free slot index (bump allocation)
        int slot_idx = atomicAdd(next_free_counter, 1);

        if (slot_idx < static_cast<int>(slots_per_pool)) {
            handle.data             = pool_data + (slot_idx * slot_size);
            handle.ntrials          = ntrials;
            handle.capacity_ntrials = max_ntrials;
            handle.nbins            = nbins;
            handle.variance         = variance;
            handle.pool_id          = current_out_pool;
            handle.slot_idx         = slot_idx;
        }

        return handle;
    }

    /**
     * Deallocate - callable from device (no-op for bump allocator)
     */
    __device__ void deallocate(const FoldVectorHandleDevice& /*handle*/) const {
        // This is a no-op because the entire pool is reset by swap_pools.
        // The slot becomes reusable after swap_pools resets the counter.
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

        // Allocate counters for bump allocation
        cudaMalloc(&m_next_free_a, sizeof(int));
        cudaMalloc(&m_next_free_b, sizeof(int));
        cudaMemset(m_next_free_a, 0, sizeof(int));
        cudaMemset(m_next_free_b, 0, sizeof(int));
    }

    ~DualPoolFoldManagerDevice() {
        cudaFree(m_next_free_a);
        cudaFree(m_next_free_b);
    }

    // Delete copy/move operations
    DualPoolFoldManagerDevice(const DualPoolFoldManagerDevice&) = delete;
    DualPoolFoldManagerDevice&
    operator=(const DualPoolFoldManagerDevice&)                       = delete;
    DualPoolFoldManagerDevice(DualPoolFoldManagerDevice&&)            = delete;
    DualPoolFoldManagerDevice& operator=(DualPoolFoldManagerDevice&&) = delete;

    /**
     * Get device allocator for use in kernels
     */
    DevicePoolAllocator get_device_allocator() {
        return {.pool_a_data      = thrust::raw_pointer_cast(m_pool_a.data()),
                .pool_b_data      = thrust::raw_pointer_cast(m_pool_b.data()),
                .next_free_a      = m_next_free_a,
                .next_free_b      = m_next_free_b,
                .slot_size        = m_slot_size,
                .max_ntrials      = m_max_ntrials,
                .nbins            = m_nbins,
                .slots_per_pool   = m_slots_per_pool,
                .current_out_pool = m_current_out_pool};
    }

    /**
     * Swap pools - must be called from host
     */
    void swap_pools() {
        m_current_out_pool = 1 - m_current_out_pool;

        // Reset the "in" pool's counter for next use
        if (m_current_out_pool == 0) {
            // B is now "in", reset its counter
            cudaMemset(m_next_free_b, 0, sizeof(int));
        } else {
            // A is now "in", reset its counter
            cudaMemset(m_next_free_a, 0, sizeof(int));
        }
    }

private:
    thrust::device_vector<float> m_pool_a;
    thrust::device_vector<float> m_pool_b;
    int* m_next_free_a = nullptr;
    int* m_next_free_b = nullptr;

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
    FoldsTypeDevice folds_in;
    FoldsTypeDevice folds_sim;
    FoldsTypeDevice folds_pruned;
};

struct TransitionResult { // NOLINT
    int threshold_idx;
    int prob_idx;
    StateD computed_state;
    FoldsTypeDevice folds_out;
    bool invalid;
};

struct TransitionBatch {
    thrust::device_vector<TransitionWorkItem> work_items_d;
    thrust::device_vector<TransitionResult> results_d;

    void reserve(SizeType max_items) {
        work_items_d.reserve(max_items);
        results_d.resize(max_items);
    }
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

__device__ void simulate_transition_phase(const TransitionWorkItem& work_item,
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

    // H0: no signal bias
    for (int base = tid * 4; base < total_elements_h0; base += block_size * 4) {
        process_batch(base, total_elements_h0, work_item.folds_in.folds_h0.data,
                      work_item.folds_sim.folds_h0.data,
                      static_cast<int>(work_item.folds_in.folds_h0.ntrials), 0,
                      false);
    }
    // H1: with signal bias
    const int h1_seq_offset = (total_elements_h0 + 3) / 4;
    for (int base = tid * 4; base < total_elements_h1; base += block_size * 4) {
        process_batch(base, total_elements_h1, work_item.folds_in.folds_h1.data,
                      work_item.folds_sim.folds_h1.data,
                      static_cast<int>(work_item.folds_in.folds_h1.ntrials),
                      h1_seq_offset, true);
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

__device__ void
score_and_prune_fused(const TransitionWorkItem& work_item,
                      const SizeType* __restrict__ box_score_widths,
                      int nwidths,
                      int nbins,
                      float threshold,
                      int* output_ntrials_h0,
                      int* output_ntrials_h1) {
    extern __shared__ int shm[]; // NOLINT
    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);

    // Shared memory for reduction
    int& shared_count_h0   = shm[0];
    int& shared_count_h1   = shm[1];
    int* shared_indices_h0 = &shm[2];
    int* shared_indices_h1 = &shm[2 + block_size];

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
        int local_count = 0;

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
                        if (pos + j < block_size) {
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
                if (j < local_count && pos + j < block_size) {
                    shared_idx[pos + j] = local_good[j];
                }
            }
        }
    };

    collect(work_item.folds_sim.folds_h0.data,
            static_cast<int>(work_item.folds_sim.folds_h0.ntrials),
            work_item.folds_sim.folds_h0.variance, shared_count_h0,
            shared_indices_h0);
    collect(work_item.folds_sim.folds_h1.data,
            static_cast<int>(work_item.folds_sim.folds_h1.ntrials),
            work_item.folds_sim.folds_h1.variance, shared_count_h1,
            shared_indices_h1);
    __syncthreads();

    // Copy H0 surviving trials
    float* __restrict__ folds_h0_sim    = work_item.folds_sim.folds_h0.data;
    float* __restrict__ folds_h1_sim    = work_item.folds_sim.folds_h1.data;
    float* __restrict__ folds_h0_pruned = work_item.folds_pruned.folds_h0.data;
    float* __restrict__ folds_h1_pruned = work_item.folds_pruned.folds_h1.data;
    for (int i = tid; i < shared_count_h0; i += block_size) {
        const int orig_trial    = shared_indices_h0[i];
        const int input_offset  = orig_trial * nbins;
        const int output_offset = i * nbins;
        for (int j = 0; j < nbins; ++j) {
            folds_h0_pruned[output_offset + j] = folds_h0_sim[input_offset + j];
        }
    }

    // Copy H1 surviving trials
    for (int i = tid; i < shared_count_h1; i += block_size) {
        const int orig_trial    = shared_indices_h1[i];
        const int input_offset  = orig_trial * nbins;
        const int output_offset = i * nbins;
        for (int j = 0; j < nbins; ++j) {
            folds_h1_pruned[output_offset + j] = folds_h1_sim[input_offset + j];
        }
    }

    // Store final counts (single thread)
    if (tid == 0) {
        *output_ntrials_h0 = shared_count_h0;
        *output_ntrials_h1 = shared_count_h1;
    }
}

__global__ void process_transitions_unified_kernel(
    const TransitionWorkItem* __restrict__ work_items,
    int num_items,
    const float* __restrict__ profile,
    int nbins,
    const SizeType* __restrict__ box_score_widths,
    int nwidths,
    float bias_snr,
    float var_add,
    const float* __restrict__ probs,
    int nprobs,
    TransitionResult* __restrict__ results,
    uint64_t seed,
    uint64_t offset) {

    const auto item_idx = static_cast<int>(blockIdx.x);
    if (item_idx >= num_items) {
        return;
    }

    const auto& work_item = work_items[item_idx];
    const auto tid        = static_cast<int>(threadIdx.x);

    // Shared memory for output counts
    __shared__ int shared_ntrials_h0_out;
    __shared__ int shared_ntrials_h1_out;

    // Phase 1: Simulation (threads collaborate)
    simulate_transition_phase(work_item, profile, nbins, bias_snr, var_add,
                              seed, offset);
    __syncthreads();

    // Phase 2: Fused Score and Prune (threads collaborate)
    score_and_prune_fused(work_item, box_score_widths, nwidths, nbins,
                          work_item.threshold, &shared_ntrials_h0_out,
                          &shared_ntrials_h1_out);
    __syncthreads();

    // Phase 3: Compute final state and result (single thread per block)
    if (tid == 0) {
        const auto ntrials_h0_out = shared_ntrials_h0_out;
        const auto ntrials_h1_out = shared_ntrials_h1_out;
        // Calculate success rates
        const auto success_h0 =
            static_cast<float>(ntrials_h0_out) /
            static_cast<float>(work_item.folds_sim.folds_h0.ntrials);
        const auto success_h1 =
            static_cast<float>(ntrials_h1_out) /
            static_cast<float>(work_item.folds_sim.folds_h1.ntrials);

        // Generate next state
        const auto state_next = work_item.input_state.gen_next(
            work_item.threshold, success_h0, success_h1, work_item.nbranches);

        // Find probability bin
        const auto iprob =
            find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);

        // Store result
        TransitionResult& result = results[item_idx];
        result.threshold_idx     = work_item.threshold_idx;
        result.prob_idx          = iprob;
        result.computed_state    = state_next;
        result.folds_out         = work_item.folds_pruned;
        // Critical part. After pruning, need to update the correct ntrials
        result.folds_out.folds_h0.ntrials = ntrials_h0_out;
        result.folds_out.folds_h1.ntrials = ntrials_h1_out;
        result.invalid                    = (iprob < 0 || iprob >= nprobs);
    }
}

__global__ void
update_states_kernel(const TransitionResult* results,
                     int num_results,
                     int nprobs,
                     int stage_offset_cur,
                     StateD* states_out_ptr,
                     int* locks_ptr,
                     cuda::std::optional<FoldsTypeDevice>* folds_out_ptr) {

    const auto idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (idx >= num_results) {
        return;
    }
    const auto& result = results[idx];
    if (result.invalid) {
        return;
    }
    const int fold_idx  = (result.threshold_idx * nprobs) + result.prob_idx;
    const int state_idx = stage_offset_cur + fold_idx;

    // Acquire lock for this grid cell using a spinlock
    int* lock = &locks_ptr[state_idx];
    while (atomicCAS(lock, 0, 1) != 0) {
        // Spin while waiting for the lock to be released
    }
    const float current_complexity = states_out_ptr[state_idx].complexity_cumul;
    const float new_complexity     = result.computed_state.complexity_cumul;

    if (states_out_ptr[state_idx].is_empty || new_complexity < current_complexity) {
        // Our new state is better, so update everything.
        states_out_ptr[state_idx] = result.computed_state;
        folds_out_ptr[fold_idx]   = result.folds_out;
    }
    // Release the lock
    atomicExch(lock, 0);
}

struct IndexPair {
    SizeType ithres;
    SizeType jthresh;
};

struct PairWithCount {
    IndexPair pair;
    SizeType count; // Number of valid kprob iterations
};

struct CountValidWorkItems {
    const StateD* states_ptr;
    const cuda::std::optional<FoldsTypeDevice>* folds_ptr;
    SizeType stage_offset_prev;
    SizeType nprobs;

    CountValidWorkItems(
        const thrust::device_vector<StateD>& states_d,
        const thrust::device_vector<cuda::std::optional<FoldsTypeDevice>>&
            folds_current_d,
        SizeType stage_offset_prev,
        SizeType nprobs)
        : states_ptr(thrust::raw_pointer_cast(states_d.data())),
          folds_ptr(thrust::raw_pointer_cast(folds_current_d.data())),
          stage_offset_prev(stage_offset_prev),
          nprobs(nprobs) {}

    __device__ PairWithCount operator()(const IndexPair& pair) const {
        SizeType count = 0;
        for (SizeType kprob = 0; kprob < nprobs; ++kprob) {
            const auto prev_fold_idx = (pair.jthresh * nprobs) + kprob;
            const auto& prev_state =
                states_ptr[stage_offset_prev + prev_fold_idx];
            if (prev_state.is_empty) {
                continue;
            }
            const auto& prev_fold_state = folds_ptr[prev_fold_idx];
            if (!prev_fold_state.has_value() || prev_fold_state->is_empty()) {
                continue;
            }
            count++;
        }
        if (count == 0) {
            printf("Count is 0 for pair (%zu, %zu)\n", pair.ithres, pair.jthresh);
        }
        return {.pair = pair, .count = count};
    }
};

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
        DevicePoolAllocator* allocator)
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
          allocator(allocator) {}

    __device__ void
    operator()(const thrust::tuple<PairWithCount, SizeType>& input) const {
        const auto& pair_with_count = thrust::get<0>(input);
        const SizeType index        = thrust::get<1>(input);
        const auto& pair            = pair_with_count.pair;
        SizeType ithres             = pair.ithres;
        SizeType jthresh            = pair.jthresh;
        // Current offset
        SizeType base_offset = offset_ptr[index];
        SizeType slot        = base_offset;
        for (SizeType kprob = 0; kprob < nprobs; ++kprob) {
            const auto prev_fold_idx = (jthresh * nprobs) + kprob;
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
            const auto repeat_factor_h0 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h0)));
            const auto ntrials_out_h0 = repeat_factor_h0 * ntrials_in_h0;
            const auto repeat_factor_h1 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h1)));
            const auto ntrials_out_h1 = repeat_factor_h1 * ntrials_in_h1;

            auto folds_h0_sim = allocator->allocate(
                ntrials_out_h0, prev_fold_state->folds_h0.variance + var_add);
            auto folds_h1_sim = allocator->allocate(
                ntrials_out_h1, prev_fold_state->folds_h1.variance + var_add);
            auto folds_h0_prn =
                allocator->allocate(ntrials_out_h0, folds_h0_sim.variance);
            auto folds_h1_prn =
                allocator->allocate(ntrials_out_h1, folds_h1_sim.variance);

            // Populate the TransitionWorkItem
            TransitionWorkItem item;
            item.threshold_idx   = static_cast<int>(ithres);
            item.prob_idx        = static_cast<int>(kprob);
            item.input_fold_idx  = static_cast<int>(prev_fold_idx);
            item.input_state     = prev_state;
            item.threshold       = thresholds_ptr[ithres];
            item.nbranches       = branching_pattern_ptr[istage];
            item.folds_in        = FoldsTypeDevice(prev_fold_state->folds_h0,
                                                   prev_fold_state->folds_h1);
            item.folds_sim       = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);
            item.folds_pruned    = FoldsTypeDevice(folds_h0_prn, folds_h1_prn);
            work_items_ptr[slot] = item;
            slot++;
        }
    }
};

struct ExtractCount {
    __device__ SizeType operator()(const PairWithCount& pwc) const {
        return pwc.count;
    }
};

__device__ void simulate_folds_init_device(FoldVectorHandleDevice folds,
                                           const float* profile,
                                           int nbins,
                                           float bias_snr,
                                           float var_add,
                                           uint64_t seed,
                                           uint64_t base_offset) {
    const int ntrials    = static_cast<int>(folds.ntrials);
    const int total_size = ntrials * nbins;

    if (!folds.is_valid() || total_size == 0) {
        return;
    }

    const float noise_stddev = sqrtf(var_add);

    // Each thread handles multiple elements
    const int tid = static_cast<int>(threadIdx.x + (blockIdx.x * blockDim.x));
    const int grid_size = static_cast<int>(gridDim.x * blockDim.x);

    // Process 4 elements at a time
    for (int base_idx = tid * 4; base_idx < total_size;
         base_idx += grid_size * 4) {
        // Generate noise using cuRANDDx - 4 values at once
        const uint64_t seq_id = base_offset + (base_idx / 4);
        RNG rng(seed, seq_id, 0);
        curanddx::normal<float, curanddx::box_muller> dist(0.0F, noise_stddev);
        const float4 noise_vals = dist.generate4(rng);

        // Convert float4 to array for easier indexing
        const float noise_array[4] = {noise_vals.x, noise_vals.y, noise_vals.z,
                                      noise_vals.w};

        // Fill up to 4 elements, handling edge case
        const int elements_to_process = min(4, total_size - base_idx);

#pragma unroll
        for (int i = 0; i < elements_to_process; ++i) {
            const int idx           = base_idx + i;
            const int bin_idx       = idx % nbins;
            const float profile_val = profile[bin_idx];

            folds.data[idx] = profile_val * bias_snr + noise_array[i];
        }
    }
}

struct InitialStateFunctor {
    const SizeType* threshold_indices_ptr;
    const float* thresholds_ptr;
    const float* branching_pattern_ptr;
    const float* profile_ptr;
    SizeType nbins;
    SizeType ntrials;
    float bias_snr;
    float var_init;
    TransitionWorkItem* work_items_ptr;
    DevicePoolAllocator* allocator;
    uint64_t seed;

    InitialStateFunctor(
        const thrust::device_vector<SizeType>& threshold_indices,
        const thrust::device_vector<float>& thresholds,
        const thrust::device_vector<float>& branching_pattern,
        const thrust::device_vector<float>& profile,
        SizeType nbins,
        SizeType ntrials,
        float bias_snr,
        float var_init,
        thrust::device_vector<TransitionWorkItem>& work_items,
        DevicePoolAllocator* allocator,
        uint64_t seed)
        : threshold_indices_ptr(
              thrust::raw_pointer_cast(threshold_indices.data())),
          thresholds_ptr(thrust::raw_pointer_cast(thresholds.data())),
          branching_pattern_ptr(
              thrust::raw_pointer_cast(branching_pattern.data())),
          profile_ptr(thrust::raw_pointer_cast(profile.data())),
          nbins(nbins),
          ntrials(ntrials),
          bias_snr(bias_snr),
          var_init(var_init),
          work_items_ptr(thrust::raw_pointer_cast(work_items.data())),
          allocator(allocator),
          seed(seed) {}

    __device__ void operator()(SizeType idx) const {
        const SizeType ithres = threshold_indices_ptr[idx];

        // Allocate initial fold vectors
        auto folds_h0_init = allocator->allocate(ntrials, var_init);
        auto folds_h1_init = allocator->allocate(ntrials, var_init);

        // Simulate the initial folds
        simulate_folds_init_device(folds_h0_init, profile_ptr,
                                   static_cast<int>(nbins), 0.0F, var_init,
                                   seed, idx * ntrials);
        simulate_folds_init_device(folds_h1_init, profile_ptr,
                                   static_cast<int>(nbins), bias_snr, var_init,
                                   seed, (idx * ntrials) + ntrials);

        // Allocate simulation and pruned output buffers
        auto folds_h0_sim = allocator->allocate(ntrials, var_init);
        auto folds_h1_sim = allocator->allocate(ntrials, var_init);
        auto folds_h0_prn = allocator->allocate(ntrials, var_init);
        auto folds_h1_prn = allocator->allocate(ntrials, var_init);

        // Create work item
        TransitionWorkItem item;
        item.threshold_idx  = static_cast<int>(ithres);
        item.prob_idx       = -1; // Will be determined after processing
        item.input_fold_idx = -1; // Not applicable for initial state
        // Create proper initial state (not empty)
        StateD initial_state;
        initial_state.is_empty         = false;
        initial_state.complexity_cumul = 1.0F;
        item.input_state               = initial_state;
        item.threshold                 = thresholds_ptr[ithres];
        item.nbranches                 = branching_pattern_ptr[0];
        item.folds_in     = FoldsTypeDevice(folds_h0_init, folds_h1_init);
        item.folds_sim    = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);
        item.folds_pruned = FoldsTypeDevice(folds_h0_prn, folds_h1_prn);

        work_items_ptr[idx] = item;
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
          m_device_id(device_id) {

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
        init_states();
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
            spdlog::info("After swap for stage {}, checking folds_current_d validity", istage);
            // Deallocate using thrust
            thrust::for_each(thrust::device, m_folds_next_d.begin(),
                             m_folds_next_d.end(),
                             DeallocateFunctor{allocator});
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
        // 2 simulated and 2 pruned folds per transition
        const auto max_temporary  = 8; // Conservative for CUDA
        const auto slots_per_pool = max_persistent + max_temporary;
        spdlog::info(
            "CUDA allocation analysis: {} active thresholds max, {} prob "
            "bins",
            max_active_per_stage, m_nprobs);
        spdlog::info("Need {} persistent + {} temporary = {} slots per pool",
                     max_persistent, max_temporary, slots_per_pool);
        return slots_per_pool;
    }

    void init_states() {
        const float var_init = 1.0F;
        auto allocator       = m_device_manager->get_device_allocator();

        // Create initial batch for all thresholds in the initial beam
        const auto thresholds_idx = get_current_thresholds_idx(0);
        thrust::device_vector<SizeType> threshold_indices_d(thresholds_idx);

        // Create initial batch
        TransitionBatch initial_batch;
        initial_batch.reserve(thresholds_idx.size());
        initial_batch.work_items_d.resize(thresholds_idx.size());

        // Generate random seed
        const uint64_t seed = std::random_device{}();
        // Launch functor to create initial work items
        thrust::for_each(
            thrust::counting_iterator<SizeType>(0),
            thrust::counting_iterator<SizeType>(thresholds_idx.size()),
            InitialStateFunctor(threshold_indices_d, m_thresholds_d,
                                m_branching_pattern_d, m_profile_d, m_nbins,
                                m_ntrials, m_bias_snr, var_init,
                                initial_batch.work_items_d, &allocator, seed));

        // Process initial batch
        process_initial_batch(initial_batch, var_init);
        // Wait for the GPU to finish computing the initial states before
        // proceeding.
        cudaDeviceSynchronize();
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
                     DevicePoolAllocator allocator) {
        const float var_add          = 1.0F;
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;
        spdlog::info("Running segment {} of {} ({} thresholds)", istage,
                     m_nstages, beam_idx_cur.size());
        // Step 1: Collect all transitions into batch
        TransitionBatch batch;
        const SizeType max_transitions =
            beam_idx_cur.size() * thres_neigh * m_nprobs;
        batch.reserve(max_transitions);
        spdlog::info(
            "Collecting transitions for segment {} of {} ({} thresholds)",
            istage, m_nstages, beam_idx_cur.size());
        collect_transitions(batch, beam_idx_cur, beam_idx_prev,
                            stage_offset_prev, istage, thres_neigh, var_add,
                            allocator);

        if (batch.work_items_d.empty()) {
            return;
        }
        spdlog::info(
            "Processing transitions for segment {} of {} ({} thresholds)",
            istage, m_nstages, beam_idx_cur.size());
        process_transition_batch(batch, var_add, stage_offset_cur);
        cudaDeviceSynchronize();
    }

    void process_initial_batch(TransitionBatch& batch, float var_init) {
        const auto num_items = static_cast<int>(batch.work_items_d.size());
        if (num_items == 0) {
            return;
        }

        const uint64_t seed   = std::random_device{}();
        const uint64_t offset = 0;

        // Launch unified kernel for initial processing
        const dim3 block_dim(256);
        const dim3 grid_dim(num_items);
        // simulate phase and score_and_prune phase
        const SizeType shared_mem_size = std::max(
            {static_cast<SizeType>(m_nbins * sizeof(float)),
             static_cast<SizeType>(2 * (1 + block_dim.x)) * sizeof(int)});
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                               shared_mem_size);

        process_transitions_unified_kernel<<<grid_dim, block_dim,
                                             shared_mem_size>>>(
            thrust::raw_pointer_cast(batch.work_items_d.data()), num_items,
            thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins),
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            static_cast<int>(m_box_score_widths_d.size()), m_bias_snr, var_init,
            thrust::raw_pointer_cast(m_probs_d.data()),
            static_cast<int>(m_nprobs),
            thrust::raw_pointer_cast(batch.results_d.data()), seed, offset);

        cuda_utils::check_last_cuda_error("process_initial_batch");

        // Process results and populate initial states using the device kernel
        const dim3 update_block_dim(256);
        const dim3 update_grid_dim((num_items + update_block_dim.x - 1) /
                                   update_block_dim.x);
        cuda_utils::check_kernel_launch_params(update_grid_dim,
                                               update_block_dim);
        update_states_kernel<<<update_grid_dim, update_block_dim>>>(
            thrust::raw_pointer_cast(batch.results_d.data()), num_items,
            static_cast<int>(m_nprobs), 0, /* stage_offset_cur is 0 */
            thrust::raw_pointer_cast(m_states_d.data()),
            thrust::raw_pointer_cast(m_states_locks_d.data()),
            thrust::raw_pointer_cast(m_folds_current_d.data()));
        cuda_utils::check_last_cuda_error("update_states_kernel_initial");
    }

    void collect_transitions(TransitionBatch& batch,
                             const std::vector<SizeType>& beam_idx_cur,
                             const std::vector<SizeType>& beam_idx_prev,
                             SizeType stage_offset_prev,
                             SizeType istage,
                             SizeType thres_neigh,
                             float var_add,
                             DevicePoolAllocator& allocator) {
        std::vector<IndexPair> h_pairs;
        h_pairs.reserve(beam_idx_cur.size() * thres_neigh);
        for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
            const auto ithres = beam_idx_cur[i];
            // Find nearest neighbors in the previous beam
            const auto neighbour_beam_indices =
                utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                 thres_neigh);

            for (SizeType jthresh : neighbour_beam_indices) {
                h_pairs.push_back({ithres, jthresh});
            }
        }
        spdlog::info("Collected {} pairs", h_pairs.size());
        thrust::device_vector<IndexPair> d_pairs = h_pairs;
        thrust::device_vector<PairWithCount> d_pairs_with_count(d_pairs.size());
        thrust::transform(thrust::device, d_pairs.begin(), d_pairs.end(),
                          d_pairs_with_count.begin(),
                          CountValidWorkItems(m_states_d, m_folds_current_d,
                                              stage_offset_prev, m_nprobs));
        spdlog::info("Counted {} valid work items", d_pairs_with_count.size());

        // Extract counts into a separate vector
        thrust::device_vector<SizeType> d_counts(d_pairs.size());
        thrust::transform(thrust::device, d_pairs_with_count.begin(),
                          d_pairs_with_count.end(), d_counts.begin(),
                          ExtractCount{});

        // Create offsets by exclusive scan of counts
        thrust::device_vector<SizeType> d_offsets(d_pairs.size());
        thrust::exclusive_scan(thrust::device, d_counts.begin(), d_counts.end(),
                               d_offsets.begin());

        spdlog::info("Scanned {} offsets", d_offsets.size());
        // Calculate total items
        SizeType total_items = 0;
        if (!d_offsets.empty()) {
            SizeType last_offset{};
            SizeType last_count{};
            thrust::copy(d_offsets.end() - 1, d_offsets.end(), &last_offset);
            thrust::copy(d_counts.end() - 1, d_counts.end(), &last_count);
            total_items = last_offset + last_count;
        }
        batch.work_items_d.resize(total_items);
        spdlog::info("Resized batch to {} work items", total_items);
        thrust::for_each(
            thrust::device,
            thrust::make_zip_iterator(
                thrust::make_tuple(d_pairs_with_count.begin(),
                                   thrust::counting_iterator<SizeType>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_pairs_with_count.end(), thrust::counting_iterator<SizeType>(
                                              d_pairs_with_count.size()))),
            TransitionFunctor(m_states_d, m_folds_current_d, m_thresholds_d,
                              m_branching_pattern_d, m_nprobs, m_ntrials,
                              stage_offset_prev, istage, var_add,
                              batch.work_items_d, d_offsets, &allocator));

        cudaDeviceSynchronize();
    }

    void process_transition_batch(TransitionBatch& batch,
                                  float var_add,
                                  SizeType stage_offset_cur) {
        const auto num_items = static_cast<int>(batch.work_items_d.size());
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
        const SizeType shared_mem_size = std::max(
            {static_cast<SizeType>(m_nbins * sizeof(float)),
             static_cast<SizeType>(2 * (1 + block_dim.x)) * sizeof(int)});
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                               shared_mem_size);

        process_transitions_unified_kernel<<<grid_dim, block_dim,
                                             shared_mem_size>>>(
            thrust::raw_pointer_cast(batch.work_items_d.data()), num_items,
            thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins),
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            static_cast<int>(m_box_score_widths_d.size()), m_bias_snr, var_add,
            thrust::raw_pointer_cast(m_probs_d.data()),
            static_cast<int>(m_nprobs),
            thrust::raw_pointer_cast(batch.results_d.data()), seed, offset);
        cuda_utils::check_last_cuda_error("process_transitions_unified_kernel");

        // Phase 2: Launch state update kernel
        const dim3 update_block_dim(256);
        const dim3 update_grid_dim((num_items + update_block_dim.x - 1) /
                                   update_block_dim.x);
        cuda_utils::check_kernel_launch_params(update_grid_dim,
                                               update_block_dim);

        update_states_kernel<<<update_grid_dim, update_block_dim>>>(
            thrust::raw_pointer_cast(batch.results_d.data()), num_items,
            static_cast<int>(m_nprobs), static_cast<int>(stage_offset_cur),
            thrust::raw_pointer_cast(m_states_d.data()),
            thrust::raw_pointer_cast(m_states_locks_d.data()),
            thrust::raw_pointer_cast(m_folds_next_d.data()));
        cuda_utils::check_last_cuda_error("update_states_kernel");
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