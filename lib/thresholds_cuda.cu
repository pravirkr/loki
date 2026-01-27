#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <random>

#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <cuda/std/climits>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/math_cuda.cuh"
#include "loki/progress.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/utils.hpp"
#include "loki/cub_helpers.cuh"

namespace loki::detection {

using RNG = loki::math::DefaultDeviceRNG;

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

/* Allocate `ntrials × nbins` floats on the device and wrap the pointer */
inline FoldVectorHandleDevice
make_fold(SizeType ntrials, SizeType nbins, float variance = 0.0F) {
    FoldVectorHandleDevice h;
    h.ntrials          = ntrials;
    h.capacity_ntrials = ntrials; // full capacity = size
    h.nbins            = nbins;
    h.variance         = variance;
    h.pool_id          = std::numeric_limits<SizeType>::max();
    h.slot_idx         = std::numeric_limits<SizeType>::max();

    const auto bytes = ntrials * nbins * sizeof(float);
    cuda_utils::check_cuda_call(cudaMalloc(&h.data, bytes));
    return h;
}

/* Free a buffer created with make_fold */
inline void free_fold(FoldVectorHandleDevice& h) {
    if (h.data != nullptr) {
        cudaFree(h.data);
    }
    h.data = nullptr;
}

/**
 * Device-side allocator interface - can be called from kernels
 */
struct DevicePoolAllocator {
    float* pool_a_data;
    float* pool_b_data;
    int* free_slots_a; // Stack of available slot indices for pool A
    int* free_slots_b; // Stack of available slot indices for pool B
    int* free_top_a;   // Pointer to the count of free slots in A
    int* free_top_b;   // Pointer to the count of free slots in B
    SizeType slot_size;
    SizeType max_ntrials;
    SizeType nbins;
    SizeType slots_per_pool;
    int current_out_pool; // 0 for A, 1 for B

    /**
     * Allocate from the current "out" pool - callable from device.
     * This implements a lock-free pop from a concurrent stack.
     */
    __device__ FoldVectorHandleDevice allocate(SizeType ntrials,
                                               float variance) const {
        FoldVectorHandleDevice handle; // handle.data is nullptr by default

        int* free_top   = (current_out_pool == 0) ? free_top_a : free_top_b;
        int* free_slots = (current_out_pool == 0) ? free_slots_a : free_slots_b;
        float* pool_data = (current_out_pool == 0) ? pool_a_data : pool_b_data;

        // Atomically decrement the free slot counter. The return value is the
        // *new* count. The index we want is this new count.
        int stack_idx = atomicSub(free_top, 1) - 1;

        if (stack_idx >= 0) {
            // Get the actual slot index from the free list stack
            int slot_idx = free_slots[stack_idx];

            handle.data             = pool_data + (slot_idx * slot_size);
            handle.ntrials          = ntrials;
            handle.capacity_ntrials = max_ntrials;
            handle.nbins            = nbins;
            handle.variance         = variance;
            handle.pool_id          = current_out_pool;
            handle.slot_idx         = static_cast<SizeType>(slot_idx);
        }
        // If stack_idx < 0, the pool is exhausted. The handle remains invalid
        // (handle.data == nullptr), which is the correct failure signal.
        return handle;
    }

    /**
     * Deallocate - callable from device.
     * This implements a lock-free push to a concurrent stack.
     */
    __device__ void deallocate(FoldVectorHandleDevice& handle) const {
        if (!handle.is_valid()) {
            return;
        }

        int* free_top   = (handle.pool_id == 0) ? free_top_a : free_top_b;
        int* free_slots = (handle.pool_id == 0) ? free_slots_a : free_slots_b;

        // Atomically increment the free slot counter. The return value is the
        // *old* count, which is the correct index to push our slot_idx into.
        int stack_idx = atomicAdd(free_top, 1);

        if (stack_idx < static_cast<int>(slots_per_pool)) {
            // Return the slot index to the free list stack
            free_slots[stack_idx] = static_cast<int>(handle.slot_idx);
        }
        // If stack_idx is out of bounds, it implies a double-free or memory
        // corruption. We do not revert the counter, as this would hide the
        // error and lead to more corruption.
    }
};

/**
 * CUDA Dual-Pool Memory Manager using thrust::device_vector for safety
 */
class DualPoolFoldManagerDevice {
public:
    DualPoolFoldManagerDevice(SizeType nbins,
                              SizeType max_ntrials_per_slot,
                              SizeType slots_per_pool)
        : m_nbins(nbins),
          m_max_ntrials(max_ntrials_per_slot),
          m_slot_size(m_max_ntrials * nbins),
          m_slots_per_pool(slots_per_pool) {

        // Allocate device memory pools
        m_pool_a.resize(m_slots_per_pool * m_slot_size);
        m_pool_b.resize(m_slots_per_pool * m_slot_size);

        // Allocate and initialize free slot stacks (indices 0 to slots-1)
        m_free_slots_a.resize(m_slots_per_pool);
        m_free_slots_b.resize(m_slots_per_pool);
        thrust::sequence(thrust::device, m_free_slots_a.begin(),
                         m_free_slots_a.end(), 0);
        thrust::sequence(thrust::device, m_free_slots_b.begin(),
                         m_free_slots_b.end(), 0);

        // Allocate and initialize the top-of-stack counters.
        // The value represents the *count* of free slots.
        cudaMalloc(&m_free_top_a, sizeof(int));
        cudaMalloc(&m_free_top_b, sizeof(int));
        int initial_count = static_cast<int>(slots_per_pool);
        cudaMemcpy(m_free_top_a, &initial_count, sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_free_top_b, &initial_count, sizeof(int),
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
        int top_a = -1, top_b = -1;
        cudaMemcpy(&top_a, m_free_top_a, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&top_b, m_free_top_b, sizeof(int), cudaMemcpyDeviceToHost);

        const int used_a = static_cast<int>(m_slots_per_pool) - top_a;
        const int used_b = static_cast<int>(m_slots_per_pool) - top_b;

        const float usage_pct_a = 100.0F * static_cast<float>(used_a) /
                                  static_cast<float>(m_slots_per_pool);
        const float usage_pct_b = 100.0F * static_cast<float>(used_b) /
                                  static_cast<float>(m_slots_per_pool);

        const std::string cur_pool = (m_current_out_pool != 0) ? "B" : "A";
        spdlog::info("Cur Pool: {}; Pool A: used = {}/{} ({:.1f}%)  |  Pool B: "
                     "used = {}/{} ({:.1f}%)",
                     cur_pool, used_a, m_slots_per_pool, usage_pct_a, used_b,
                     m_slots_per_pool, usage_pct_b);

        // Corruption checks
        if (top_a < 0 || top_b < 0 ||
            top_a > static_cast<int>(m_slots_per_pool) ||
            top_b > static_cast<int>(m_slots_per_pool)) {
            spdlog::error("POOL CORRUPTION DETECTED! (top_a={}, top_b={})",
                          top_a, top_b);
        }

        // Optional warning when near exhaustion
        if (top_a == 0 || top_b == 0) {
            spdlog::warn("Pool A or B is completely exhausted!");
        } else if (usage_pct_a > 95.0F || usage_pct_b > 95.0F) {
            spdlog::warn(
                "Pool usage above 95%. Consider increasing slots per pool.");
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

    __device__ void operator()(FoldsTypeDevice& fold) const {
        allocator.deallocate(fold.folds_h0);
        allocator.deallocate(fold.folds_h1);
        // Reset to default (nullptrs)
        fold = FoldsTypeDevice();
    }
};

struct BoxcarWidthsCacheDeviceView {
    const int* box_score_widths;
    const float* h_vals;
    const float* b_vals;
    float* psum_workspace;

    int nsum; // Size of the psum_workspace for each thread
    int nwidths;
    int nbins;
    int wmax;
};

struct BoxcarWidthsCacheDevice {
    thrust::device_vector<int> box_score_widths_d;
    thrust::device_vector<float> h_vals_d;
    thrust::device_vector<float> b_vals_d;
    thrust::device_vector<float> psum_workspace_d;
    int nsum;
    int nwidths;
    int nbins;
    int wmax;

    BoxcarWidthsCacheDevice(std::span<const SizeType> box_score_widths,
                            int nbins,
                            int max_concurrent_threads)
        : nbins(nbins) {

        nwidths = static_cast<int>(box_score_widths.size());
        wmax    = static_cast<int>(*std::ranges::max_element(box_score_widths));
        std::vector<float> h_vals(nwidths);
        std::vector<float> b_vals(nwidths);

        box_score_widths_d.resize(nwidths);
        for (int i = 0; i < nwidths; ++i) {
            const int w           = static_cast<int>(box_score_widths[i]);
            const auto nbins_f    = static_cast<float>(nbins);
            const auto w_f        = static_cast<float>(w);
            box_score_widths_d[i] = w;
            h_vals[i] = std::sqrt((nbins_f - w_f) / (nbins_f * w_f));
            b_vals[i] = w_f * h_vals[i] / (nbins_f - w_f);
        }
        h_vals_d = h_vals;
        b_vals_d = b_vals;
        nsum     = nbins + wmax;

        // Allocate the per-thread psum workspace
        // Each thread needs a buffer of size (nbins + wmax).
        const auto total_workspace_size = nsum * max_concurrent_threads;
        psum_workspace_d.resize(total_workspace_size);
    }

    BoxcarWidthsCacheDeviceView get_device_view() {
        return BoxcarWidthsCacheDeviceView{
            .box_score_widths =
                thrust::raw_pointer_cast(box_score_widths_d.data()),
            .h_vals         = thrust::raw_pointer_cast(h_vals_d.data()),
            .b_vals         = thrust::raw_pointer_cast(b_vals_d.data()),
            .psum_workspace = thrust::raw_pointer_cast(psum_workspace_d.data()),
            .nsum           = nsum,
            .nwidths        = nwidths,
            .nbins          = nbins,
            .wmax           = wmax};
    }
};

// Batch transition data for parallel processing
struct TransitionWorkItem { // NOLINT
    int thres_idx;
    StateD input_state;
    float threshold;
    float nbranches;
    FoldsTypeDevice folds_in;
    FoldsTypeDevice folds_sim;
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

__device__ __forceinline__ void
simulate_folds(const FoldsTypeDevice* __restrict__ folds_in_ptr,
               FoldsTypeDevice* __restrict__ folds_sim_ptr,
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
        static_cast<int>(folds_sim_ptr->folds_h0.ntrials) * nbins;
    const int total_elements_h1 =
        static_cast<int>(folds_sim_ptr->folds_h1.ntrials) * nbins;
    const float noise_stddev = sqrtf(var_add);

    // Lambda for processing 4 elements to match cuRANDDx generate4()
    auto process_batch = [&](int base_i, int total_elements,
                             const float* in_data, float* out_data,
                             int input_ntrials, int seq_id_base,
                             bool add_signal) {
        const int seq_id =
            seq_id_base + ((block_id * total_elements + base_i) / 4);
        typename RNG::Generator rng(seed, seq_id, offset);
        typename RNG::NormalFloat dist(0.0F, noise_stddev);
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
        process_batch(base, total_elements_h0, folds_in_ptr->folds_h0.data,
                      folds_sim_ptr->folds_h0.data,
                      static_cast<int>(folds_in_ptr->folds_h0.ntrials), 0,
                      false);
    }
    // H1: with signal bias
    const int h1_seq_offset = (total_elements_h0 + 3) / 4;
    for (int base = tid * 4; base < total_elements_h1; base += block_size * 4) {
        process_batch(base, total_elements_h1, folds_in_ptr->folds_h1.data,
                      folds_sim_ptr->folds_h1.data,
                      static_cast<int>(folds_in_ptr->folds_h1.ntrials),
                      h1_seq_offset, true);
    }
}

// Wrapper kernel for simulate_folds
__global__ void
simulate_folds_kernel(const FoldsTypeDevice* __restrict__ folds_in_ptr,
                      FoldsTypeDevice* __restrict__ folds_sim_ptr,
                      const float* __restrict__ profile,
                      int nbins,
                      float bias_snr,
                      float var_add,
                      uint64_t seed,
                      uint64_t offset) {

    simulate_folds(folds_in_ptr, folds_sim_ptr, profile, nbins, bias_snr,
                   var_add, seed, offset);
}

constexpr int kLocalFilterMax = 32;

__device__ float
compute_trial_snr_on_demand(const float* __restrict__ trial_data,
                            int nbins,
                            const BoxcarWidthsCacheDeviceView& box_cache,
                            float stdnoise) {
    const int nwidths                = box_cache.nwidths;
    const int nsum                   = box_cache.nsum;
    const int* __restrict__ widths   = box_cache.box_score_widths;
    const float* __restrict__ h_vals = box_cache.h_vals;
    const float* __restrict__ b_vals = box_cache.b_vals;

    // Calculate the pointer to this thread's private workspace
    const auto tid = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    float* psum =
        box_cache.psum_workspace + (static_cast<IndexType>(tid * nsum));

    // Compute circular prefix sum (inclusive scan with wrapping)
    psum[0] = trial_data[0];
    for (int i = 1; i < nbins; ++i) {
        psum[i] = psum[i - 1] + trial_data[i];
    }
    const float total_sum = psum[nbins - 1];
    for (int i = nbins; i < nsum; ++i) {
        const int wrap_count   = i / nbins;
        const int pos_in_cycle = i % nbins;
        psum[i] =
            psum[pos_in_cycle] + (static_cast<float>(wrap_count) * total_sum);
    }

    float max_snr = -cuda::std::numeric_limits<float>::infinity();
    for (int iw = 0; iw < nwidths; ++iw) {
        float max_diff = -cuda::std::numeric_limits<float>::infinity();
        for (int j = 0; j < nbins; ++j) {
            const float diff = psum[widths[iw] + j] - psum[j];
            max_diff         = max(diff, max_diff);
        }
        const float snr = (((h_vals[iw] + b_vals[iw]) * max_diff) -
                           (b_vals[iw] * total_sum)) /
                          stdnoise;
        max_snr = max(snr, max_snr);
    }
    return max_snr;
}

/**
 * @brief Computes the Signal-to-Noise Ratio (SNR) for a single trial profile.
 *
 * This function is the heart of the optimization. An entire 32-thread warp
 * collaborates to score one trial, leveraging shared memory and CUB primitives
 * for maximum efficiency.
 *
 * @param trial_data Pointer to the start of the trial's data in global memory.
 * @param nbins The number of bins in the trial profile.
 * @param box_cache A view containing pre-calculated boxcar filter parameters.
 * @param stdnoise The standard deviation of the noise for this trial.
 * @param shm_warp_psum A pointer to the warp's dedicated section of shared
 * memory for the prefix sum.
 * @param temp_scan Temporary storage for the CUB WarpScan primitive.
 * @param temp_reduce_1 Temporary storage for the first CUB WarpReduce
 * primitive.
 * @param temp_reduce_2 Temporary storage for the second CUB WarpReduce
 * primitive.
 * @return The maximum SNR found for the trial across all boxcar widths.
 */
__device__ float compute_trial_snr_warp_level(
    const float* __restrict__ trial_data,
    int nbins,
    const BoxcarWidthsCacheDeviceView& box_cache,
    float stdnoise,
    float* __restrict__ shm_warp_psum,
    typename cub::WarpScan<float>::TempStorage& temp_scan,
    typename cub::WarpReduce<float>::TempStorage& temp_reduce_1,
    typename cub::WarpReduce<float>::TempStorage& temp_reduce_2) {

    constexpr int kWarpSize = 32;
    using WarpScan          = cub::WarpScan<float, kWarpSize>;
    using WarpReduce        = cub::WarpReduce<float, kWarpSize>;

    const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;

    // CRITICAL: Zero out the shared memory workspace before use. This prevents
    // data corruption from previous trials processed by the same warp.
    for (int i = lane_id; i < box_cache.nsum; i += kWarpSize) {
        shm_warp_psum[i] = 0.0f;
    }
    __syncthreads();

    // Coalesced load from global to shared memory.
    for (int i = lane_id; i < nbins; i += kWarpSize) {
        shm_warp_psum[i] = trial_data[i];
    }
    __syncthreads();

    // Perform a warp-level inclusive prefix sum (scan) in shared memory.
    float running_sum    = 0.0f;
    const int num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int idx   = (chunk * kWarpSize) + lane_id;
        float val = (idx < nbins) ? shm_warp_psum[idx] : 0.0f;
        WarpScan(temp_scan).InclusiveSum(val, val);
        if (idx < nbins) {
            shm_warp_psum[idx] = val + running_sum;
        }
        float chunk_sum = __shfl_sync(0xFFFFFFFF, val, kWarpSize - 1);
        if (lane_id == 0) {
            running_sum += chunk_sum;
        }
        running_sum = __shfl_sync(0xFFFFFFFF, running_sum, 0);
    }
    __syncthreads();

    // Find the maximum SNR across all boxcar widths
    const float total_sum = shm_warp_psum[nbins - 1];
    float warp_max_snr    = -cuda::std::numeric_limits<float>::infinity();
    const int nwidths     = box_cache.nwidths;
    const int* __restrict__ widths   = box_cache.box_score_widths;
    const float* __restrict__ h_vals = box_cache.h_vals;
    const float* __restrict__ b_vals = box_cache.b_vals;

    for (int iw = 0; iw < nwidths; ++iw) {
        float thread_max_diff = -cuda::std::numeric_limits<float>::infinity();
        for (int j = lane_id; j < nbins; j += kWarpSize) {
            float sum_before_start = (j > 0) ? shm_warp_psum[j - 1] : 0.0F;
            float current_sum;
            const int end_idx = j + widths[iw] - 1;
            if (end_idx < nbins) {
                // Normal case: sum from j to j+w-1
                current_sum = shm_warp_psum[end_idx] - sum_before_start;
            } else {
                // Circular case: sum wraps around
                current_sum = (total_sum - sum_before_start) +
                              shm_warp_psum[end_idx % nbins];
            }
            thread_max_diff = fmaxf(thread_max_diff, current_sum);
        }
        float max_diff = WarpReduce(temp_reduce_1)
                             .Reduce(thread_max_diff, CubMaxOp<float>());

        if (lane_id == 0) {
            const float snr = (((h_vals[iw] + b_vals[iw]) * max_diff) -
                               (b_vals[iw] * total_sum)) /
                              stdnoise;
            warp_max_snr = fmaxf(warp_max_snr, snr);
        }
    }
    // Final reduction to get max SNR across all widths for this warp
    // A separate temporary storage object is required for correctness.
    float final_max_snr =
        WarpReduce(temp_reduce_2).Reduce(warp_max_snr, CubMaxOp<float>());
    return final_max_snr;
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

    for (unsigned int k = 2U; k <= m; k <<= 1U) {
        for (unsigned int j = k >> 1U; j > 0; j >>= 1U) {
            __syncthreads();
            if (idx < static_cast<unsigned int>(count)) {
                unsigned int ixj = idx ^ j;
                if (ixj > idx && ixj < static_cast<unsigned int>(count)) {
                    bool up = ((idx & k) == 0);
                    int a = data[idx], b = data[ixj];
                    // Swap to enforce ascending order if up==true
                    if ((a > b) == up) {
                        data[idx] = b;
                        data[ixj] = a;
                    }
                }
            }
        }
    }
    __syncthreads(); // Ensure all threads have completed sorting
}

__device__ void compact_in_place(int count,
                                 const int* __restrict__ indices,
                                 float* __restrict__ data,
                                 int nbins,
                                 int ntrials_max) {
    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);

    // Moves one surviving trial to its new, compacted position.
    for (int i = 0; i < count; ++i) {
        const int original_idx = indices[i];
        if (original_idx != i) {
            // Threads in the block cooperate to copy the elements of a single
            // trial. This ensures memory accesses are coalesced.
            for (int j = tid; j < nbins; j += block_size) {
                const int in_offset  = (original_idx * nbins) + j;
                const int out_offset = (i * nbins) + j;
                if (in_offset < ntrials_max * nbins) {
                    data[out_offset] = data[in_offset];
                }
            }
        }
        __syncthreads();
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
score_and_prune_fused_old(FoldsTypeDevice& folds_sim,
                          const BoxcarWidthsCacheDeviceView& box_cache,
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
    auto collect = [&](const float* __restrict__ trial_data, int sim_trials,
                       float variance, int& shared_count,
                       int* __restrict__ shared_idx) {
        // Local buffer for good indices
        int local_good[kLocalFilterMax]; // NOLINT
        int local_count      = 0;
        const float stdnoise = sqrtf(variance);

        // strided over trials
        for (int i = tid; i < sim_trials; i += block_size) {
            float score = compute_trial_snr_on_demand(
                trial_data + static_cast<IndexType>(i * nbins), nbins,
                box_cache, stdnoise);
            if (score > threshold) {
                local_good[local_count++] = i;
                if (local_count == kLocalFilterMax) {
                    // Flush local buffer to shared memory
                    int pos = atomicAdd(&shared_count, local_count);
                    for (int j = 0; j < local_count; ++j) {
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
            for (int j = 0; j < local_count; ++j) {
                if (pos + j < sim_trials) {
                    shared_idx[pos + j] = local_good[j];
                }
            }
        }
    };

    collect(folds_sim.folds_h0.data, max_trials_h0, folds_sim.folds_h0.variance,
            shared_count_h0, shared_indices_h0);
    __syncthreads();
    collect(folds_sim.folds_h1.data, max_trials_h1, folds_sim.folds_h1.variance,
            shared_count_h1, shared_indices_h1);
    __syncthreads();

    // *** Sort the surviving‐trial indices ascending ***
    bitonic_sort_shared(shared_indices_h0, shared_count_h0);
    bitonic_sort_shared(shared_indices_h1, shared_count_h1);

    // In-place copy using the coalesced helper function
    compact_in_place(shared_count_h0, shared_indices_h0,
                     folds_sim.folds_h0.data, nbins, max_trials_h0);
    compact_in_place(shared_count_h1, shared_indices_h1,
                     folds_sim.folds_h1.data, nbins, max_trials_h1);

    // Update ntrials in the handle itself
    if (tid == 0) {
        folds_sim.folds_h0.ntrials = static_cast<SizeType>(shared_count_h0);
        folds_sim.folds_h1.ntrials = static_cast<SizeType>(shared_count_h1);
    }
}

/**
 * @brief In-place scores, filters, and compacts trial profiles for a single
 * work item.
 *
 * This kernel orchestrates the score-and-prune process. Warps within the block
 * are assigned trials to score in parallel using the
 * `compute_trial_snr_warp_level` helper. Surviving trial indices are collected,
 * sorted, and then used to compact the data in-place.
 */
__device__ void
score_and_prune_fused(FoldsTypeDevice& folds_sim,
                      const BoxcarWidthsCacheDeviceView& box_cache,
                      int nbins,
                      float threshold) {
    extern __shared__ int shm_int[];

    // --- Thread & Warp Indexing ---
    constexpr int kWarpSize = 32;
    using WarpScan          = cub::WarpScan<float, kWarpSize>;
    using WarpReduce        = cub::WarpReduce<float, kWarpSize>;

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp_id   = tid / kWarpSize;
    const int lane_id   = tid % kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    const int max_trials_h0 = static_cast<int>(folds_sim.folds_h0.ntrials);
    const int max_trials_h1 = static_cast<int>(folds_sim.folds_h1.ntrials);

    // --- Shared Memory Layout ---
    __shared__ int shared_count_h0;
    __shared__ int shared_count_h1;

    int* shared_indices_h0 = &shm_int[0];
    int* shared_indices_h1 = &shm_int[max_trials_h0];
    char* shmem_base_ptr =
        reinterpret_cast<char*>(&shm_int[max_trials_h0 + max_trials_h1]);

    auto* temp_scan_base =
        reinterpret_cast<typename WarpScan::TempStorage*>(shmem_base_ptr);
    shmem_base_ptr += num_warps * sizeof(typename WarpScan::TempStorage);

    // Two separate temporary storage areas are required for the two reductions
    // in the helper function to prevent race conditions.
    auto* temp_reduce_1_base =
        reinterpret_cast<typename WarpReduce::TempStorage*>(shmem_base_ptr);
    shmem_base_ptr += num_warps * sizeof(typename WarpReduce::TempStorage);
    auto* temp_reduce_2_base =
        reinterpret_cast<typename WarpReduce::TempStorage*>(shmem_base_ptr);
    shmem_base_ptr += num_warps * sizeof(typename WarpReduce::TempStorage);

    // Workspace for the parallel prefix sum (per warp)
    float* psum_base = reinterpret_cast<float*>(shmem_base_ptr);
    // --- End Shared Memory Layout ---

    if (tid == 0) {
        shared_count_h0 = 0;
        shared_count_h1 = 0;
    }
    __syncthreads();

    // --- Collection Phase ---
    auto collect_warp_level =
        [&](const float* __restrict__ all_trials_data, int sim_trials,
            float variance, int& shared_count, int* __restrict__ shared_idx) {
            int local_good[kLocalFilterMax]; // NOLINT
            int local_count      = 0;
            const float stdnoise = sqrtf(variance);

            // Per-warp shared memory psum workspace
            float* shm_warp_psum = psum_base + warp_id * box_cache.nsum;
            auto& temp_scan      = temp_scan_base[warp_id];
            auto& temp_reduce_1  = temp_reduce_1_base[warp_id];
            auto& temp_reduce_2  = temp_reduce_2_base[warp_id];

            // This strided loop correctly and efficiently distributes trials
            // among the warps.
            for (int i = warp_id; i < sim_trials; i += num_warps) {
                float score = compute_trial_snr_warp_level(
                    all_trials_data + static_cast<IndexType>(i * nbins), nbins,
                    box_cache, stdnoise, shm_warp_psum, temp_scan,
                    temp_reduce_1, temp_reduce_2);

                if (lane_id == 0) {
                    if (score > threshold) {
                        local_good[local_count++] = i;
                        if (local_count == kLocalFilterMax) {
                            int pos = atomicAdd(&shared_count, local_count);
                            for (int j = 0; j < local_count; ++j) {
                                // Add bounds check as in old kernel
                                if (pos + j < sim_trials) {
                                    shared_idx[pos + j] = local_good[j];
                                }
                            }
                            local_count = 0; // Reset local buffer
                        }
                    }
                }
                __syncthreads();
            }

            if (lane_id == 0 && local_count > 0) {
                int pos = atomicAdd(&shared_count, local_count);
                for (int j = 0; j < local_count; ++j) {
                    if (pos + j < sim_trials) {
                        shared_idx[pos + j] = local_good[j];
                    }
                }
            }
            __syncthreads();
        };

    collect_warp_level(folds_sim.folds_h0.data, max_trials_h0,
                       folds_sim.folds_h0.variance, shared_count_h0,
                       shared_indices_h0);
    collect_warp_level(folds_sim.folds_h1.data, max_trials_h1,
                       folds_sim.folds_h1.variance, shared_count_h1,
                       shared_indices_h1);

    // --- Sort & Compact Phase ---
    bitonic_sort_shared(shared_indices_h0, shared_count_h0);
    bitonic_sort_shared(shared_indices_h1, shared_count_h1);
    compact_in_place(shared_count_h0, shared_indices_h0,
                     folds_sim.folds_h0.data, nbins, max_trials_h0);
    compact_in_place(shared_count_h1, shared_indices_h1,
                     folds_sim.folds_h1.data, nbins, max_trials_h1);

    if (tid == 0) {
        folds_sim.folds_h0.ntrials = static_cast<SizeType>(shared_count_h0);
        folds_sim.folds_h1.ntrials = static_cast<SizeType>(shared_count_h1);
    }
}

__global__ void transition_kernel(
    TransitionWorkItem* __restrict__ work_items,
    int num_items,
    const float* __restrict__ profile,
    int nbins,
    const BoxcarWidthsCacheDeviceView& box_cache,
    float bias_snr,
    float var_add,
    const float* __restrict__ probs,
    int nprobs,
    int stage_offset_cur,
    StateD* __restrict__ states_out_ptr,
    FoldsTypeDevice* __restrict__ folds_out_ptr,
    int* __restrict__ locks_ptr,
    DevicePoolAllocator* allocator, // Pass by value since it's lightweight
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
    simulate_folds(&work_item.folds_in, &work_item.folds_sim, profile, nbins,
                   bias_snr, var_add, seed, offset);
    __syncthreads();

    if (tid == 0) {
        ntrials_sim_h0_before_prune =
            static_cast<int>(work_item.folds_sim.folds_h0.ntrials);
        ntrials_sim_h1_before_prune =
            static_cast<int>(work_item.folds_sim.folds_h1.ntrials);
    }

    // Phase 2: Fused Score and Prune (threads collaborate)
    score_and_prune_fused(work_item.folds_sim, box_cache, nbins,
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
        // Find the bin index for the next state
        const auto iprob =
            find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);

        bool stored_new_folds = false;
        if (iprob >= 0 && iprob < nprobs) {
            const int fold_idx  = (work_item.thres_idx * nprobs) + iprob;
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
                allocator->deallocate(existing_folds.folds_h0);
                allocator->deallocate(existing_folds.folds_h1);

                // Update to better state
                existing_state = state_next;
                // Move pruned folds to persistent storage
                existing_folds = work_item.folds_sim;

                stored_new_folds = true;
            }
            // Release lock
            atomicExch(lock, 0);
        }

        // Deallocate temporary simulation folds if they were not stored
        if (!stored_new_folds) {
            allocator->deallocate(work_item.folds_sim.folds_h0);
            allocator->deallocate(work_item.folds_sim.folds_h1);
        }
    }
}

struct CountValidTransitions {
    const StateD* __restrict__ states_ptr;
    const FoldsTypeDevice* __restrict__ folds_ptr;
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
            if (prev_fold_state.is_empty()) {
                continue;
            }

            // If we get here, the state is valid for transition.
            count++;
        }
        return count;
    }
};

struct InitialWorkItemsFunctor {
    FoldsTypeDevice* __restrict__ folds_in_sim_ptr;
    const float* __restrict__ thresholds_ptr;
    float nbranches;
    SizeType ntrials;
    float var_init;
    TransitionWorkItem* __restrict__ work_items_ptr;
    DevicePoolAllocator* __restrict__ allocator;

    __device__ void operator()(int ithres) const {
        auto folds_h0_sim = allocator->allocate(ntrials, var_init);
        auto folds_h1_sim = allocator->allocate(ntrials, var_init);

        // Create work item
        TransitionWorkItem item;
        item.thres_idx                    = ithres;
        item.input_state                  = StateD();
        item.input_state.is_empty         = false;
        item.input_state.complexity_cumul = 1.0F;
        item.threshold                    = thresholds_ptr[ithres];
        item.nbranches                    = nbranches;
        item.folds_in                     = *folds_in_sim_ptr;
        item.folds_sim = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);

        work_items_ptr[ithres] = item;
    }
};

struct TransitionFunctor {
    const StateD* __restrict__ states_ptr;
    const FoldsTypeDevice* __restrict__ folds_ptr;
    const float* __restrict__ thresholds_ptr;
    const float* __restrict__ branching_pattern_ptr;
    SizeType nprobs;
    SizeType ntrials;
    SizeType stage_offset_prev;
    SizeType istage;
    float var_add;
    TransitionWorkItem* __restrict__ work_items_ptr;
    const SizeType* __restrict__ offset_ptr;
    DevicePoolAllocator* __restrict__ allocator;
    SizeType batch_start;
    SizeType batch_end;

    __device__ void operator()(
        const thrust::tuple<std::pair<SizeType, SizeType>, SizeType>& input)
        const {
        const auto& pair       = thrust::get<0>(input);
        const SizeType ithres  = pair.first;
        const SizeType jthresh = pair.second;
        const SizeType ipair   = thrust::get<1>(input);

        SizeType base_offset = offset_ptr[ipair];
        if (base_offset >= batch_end) {
            return;
        }

        SizeType slot = 0;
        for (SizeType kprob = 0; kprob < nprobs; ++kprob) {
            SizeType global_slot = base_offset + slot;
            if (global_slot < batch_start) {
                // Only increment slot if this would have been a valid
                // transition! So, replicate the checks here:
                const auto prev_fold_idx = (jthresh * nprobs) + kprob;
                const auto& prev_state =
                    states_ptr[stage_offset_prev + prev_fold_idx];
                if (prev_state.is_empty) {
                    continue;
                }
                const auto& prev_fold_state = folds_ptr[prev_fold_idx];
                if (prev_fold_state.is_empty()) {
                    continue;
                }
                const auto ntrials_in_h0 = prev_fold_state.folds_h0.ntrials;
                const auto ntrials_in_h1 = prev_fold_state.folds_h1.ntrials;
                if (ntrials_in_h0 == 0 || ntrials_in_h1 == 0) {
                    continue;
                }
                slot++;
                continue;
            }
            if (global_slot >= batch_end) {
                break;
            }
            const auto prev_fold_idx = (jthresh * nprobs) + kprob;
            const auto& prev_state =
                states_ptr[stage_offset_prev + prev_fold_idx];

            if (prev_state.is_empty) {
                continue;
            }

            const auto& prev_fold_state = folds_ptr[prev_fold_idx];
            if (prev_fold_state.is_empty()) {
                continue;
            }

            // Pre-allocate output buffers
            const auto ntrials_in_h0 = prev_fold_state.folds_h0.ntrials;
            const auto ntrials_in_h1 = prev_fold_state.folds_h1.ntrials;
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
                ntrials_out_h0, prev_fold_state.folds_h0.variance + var_add);
            auto folds_h1_sim = allocator->allocate(
                ntrials_out_h1, prev_fold_state.folds_h1.variance + var_add);

            // Create work item
            TransitionWorkItem item;
            item.thres_idx   = static_cast<int>(ithres);
            item.input_state = prev_state;
            item.threshold   = thresholds_ptr[ithres];
            item.nbranches   = branching_pattern_ptr[istage];
            item.folds_in    = prev_fold_state;
            item.folds_sim   = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);

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
          m_device_id(device_id) {

        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
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
        const auto slots_per_pool      = compute_max_allocations_needed();
        const auto max_trials_per_slot = m_ntrials * 2;
        m_device_manager = std::make_unique<DualPoolFoldManagerDevice>(
            m_nbins, max_trials_per_slot, slots_per_pool);
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
        m_boxcar_cache_d = std::make_unique<BoxcarWidthsCacheDevice>(
            m_box_score_widths, m_nbins, m_batch_size * m_threads_per_block);

        // Initialize states
        init_states();
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
        auto bar =
            progress::make_standard_bar("Computing scheme", m_nstages - 1);

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            auto allocator = m_device_manager->get_device_allocator();
            run_segment(istage, thres_neigh, allocator);

            // Deallocate before swapping pools
            thrust::for_each(thrust::device, m_folds_current_d.begin(),
                             m_folds_current_d.end(),
                             DeallocateFunctor{allocator});
            m_device_manager->swap_pools();
            std::swap(m_folds_current_d, m_folds_next_d);
            cudaDeviceSynchronize();

            bar.set_progress(istage);
        }
        bar.mark_as_completed();
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
    SizeType m_batch_size{256};
    SizeType m_threads_per_block{256};

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
    thrust::device_vector<FoldsTypeDevice> m_folds_current_d;
    thrust::device_vector<FoldsTypeDevice> m_folds_next_d;

    // Boxcar cache for scoring
    std::unique_ptr<BoxcarWidthsCacheDevice> m_boxcar_cache_d;

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

    void init_states() {
        const float var_init  = 1.0F;
        const float var_add   = 1.0F;
        const uint64_t seed   = std::random_device{}();
        const uint64_t offset = 0;

        // Create initial fold vectors
        auto folds_h0_init = make_fold(m_ntrials, m_nbins, 0.0F);
        auto folds_h1_init = make_fold(m_ntrials, m_nbins, 0.0F);
        auto folds_h0_sim  = make_fold(m_ntrials, m_nbins, var_init);
        auto folds_h1_sim  = make_fold(m_ntrials, m_nbins, var_init);
        thrust::fill(thrust::device, folds_h0_init.data,
                     folds_h0_init.data + folds_h0_init.size(), 0.0F);
        thrust::fill(thrust::device, folds_h1_init.data,
                     folds_h1_init.data + folds_h1_init.size(), 0.0F);

        // Simulate the initial folds
        const auto folds_in = FoldsTypeDevice(folds_h0_init, folds_h1_init);
        auto folds_in_sim   = FoldsTypeDevice(folds_h0_sim, folds_h1_sim);
        const dim3 block_dim_init(256);
        const dim3 grid_dim_init(1);
        const auto shmem_size_init = m_nbins * sizeof(float);
        cuda_utils::check_kernel_launch_params(grid_dim_init, block_dim_init,
                                               shmem_size_init);

        simulate_folds_kernel<<<grid_dim_init, block_dim_init,
                                shmem_size_init>>>(
            &folds_in, &folds_in_sim,
            thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins), m_bias_snr, var_init, seed, offset);
        cuda_utils::check_last_cuda_error("simulate_folds_kernel");
        cudaDeviceSynchronize();

        const auto thresholds_idx = get_current_thresholds_idx(0);
        const auto num_items      = thresholds_idx.size();
        thrust::device_vector<SizeType> thresholds_idx_d = thresholds_idx;

        // Create work items for initial stage
        thrust::device_vector<TransitionWorkItem> work_items_d(num_items);

        // Populate initial work items
        auto allocator = m_device_manager->get_device_allocator();
        InitialWorkItemsFunctor functor{
            .folds_in_sim_ptr = &folds_in_sim,
            .thresholds_ptr   = thrust::raw_pointer_cast(m_thresholds_d.data()),
            .nbranches        = m_branching_pattern_d[0],
            .ntrials          = m_ntrials,
            .var_init         = var_init,
            .work_items_ptr   = thrust::raw_pointer_cast(work_items_d.data()),
            .allocator        = &allocator};
        thrust::for_each(thrust::device, thresholds_idx_d.begin(),
                         thresholds_idx_d.end(), functor);

        // Process initial work items through the main kernel
        // Launch unified kernel: one block per transition
        const dim3 block_dim(256);
        const dim3 grid_dim(num_items);
        // simulate phase and score_and_prune phase
        const SizeType profile_mem = m_nbins * sizeof(float);
        const SizeType pruning_mem =
            (2 * sizeof(int)) +
            (2 * m_device_manager->get_max_ntrials() * sizeof(int));

        const SizeType index_shmem_size = std::max(profile_mem, pruning_mem);
        const int num_warps             = block_dim.x / 32;
        const size_t cub_storage_size =
            num_warps *
            (sizeof(typename cub::WarpScan<float>::TempStorage) +
             2 * sizeof(typename cub::WarpReduce<float>::TempStorage));
        const size_t psum_workspace_size =
            num_warps * m_boxcar_cache_d->nsum * sizeof(float);

        const size_t shmem_size =
            index_shmem_size + cub_storage_size + psum_workspace_size;

        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

        transition_kernel<<<grid_dim, block_dim, shmem_size>>>(
            thrust::raw_pointer_cast(work_items_d.data()),
            static_cast<int>(num_items),
            thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins), m_boxcar_cache_d->get_device_view(),
            m_bias_snr, var_add, thrust::raw_pointer_cast(m_probs_d.data()),
            static_cast<int>(m_nprobs), 0,
            thrust::raw_pointer_cast(m_states_d.data()),
            thrust::raw_pointer_cast(m_folds_current_d.data()),
            thrust::raw_pointer_cast(m_states_locks_d.data()), &allocator, seed,
            offset);
        cuda_utils::check_last_cuda_error("transition_kernel");
        cudaDeviceSynchronize();

        // Deallocate initial folds
        free_fold(folds_h0_init);
        free_fold(folds_h1_init);
        free_fold(folds_h0_sim);
        free_fold(folds_h1_sim);
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

        // Step 2: Count valid transitions (fast pass)
        thrust::device_vector<std::pair<SizeType, SizeType>> threshold_pairs_d =
            threshold_pairs;
        thrust::device_vector<SizeType> transition_counts_d(
            threshold_pairs.size());

        // Count transitions per pair
        CountValidTransitions functor_count{
            .states_ptr = thrust::raw_pointer_cast(m_states_d.data()),
            .folds_ptr  = thrust::raw_pointer_cast(m_folds_current_d.data()),
            .stage_offset_prev = stage_offset_prev,
            .nprobs            = m_nprobs};
        thrust::transform(thrust::device, threshold_pairs_d.begin(),
                          threshold_pairs_d.end(), transition_counts_d.begin(),
                          functor_count);

        // Compute prefix sum for offsets
        thrust::device_vector<SizeType> offsets_d(threshold_pairs.size());
        thrust::exclusive_scan(thrust::device, transition_counts_d.begin(),
                               transition_counts_d.end(), offsets_d.begin(), 0);

        SizeType total_transitions;
        thrust::copy_n(offsets_d.end() - 1, 1, &total_transitions);

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

            TransitionFunctor functor{
                .states_ptr = thrust::raw_pointer_cast(m_states_d.data()),
                .folds_ptr = thrust::raw_pointer_cast(m_folds_current_d.data()),
                .thresholds_ptr =
                    thrust::raw_pointer_cast(m_thresholds_d.data()),
                .branching_pattern_ptr =
                    thrust::raw_pointer_cast(m_branching_pattern_d.data()),
                .nprobs            = m_nprobs,
                .ntrials           = m_ntrials,
                .stage_offset_prev = stage_offset_prev,
                .istage            = istage,
                .var_add           = var_add,
                .work_items_ptr = thrust::raw_pointer_cast(work_items_d.data()),
                .offset_ptr     = thrust::raw_pointer_cast(offsets_d.data()),
                .allocator      = &allocator,
                .batch_start    = start,
                .batch_end      = end,
            };
            thrust::for_each(thrust::device, zip_begin, zip_end, functor);

            // Process this batch
            // Generate random seed and offset
            const uint64_t seed   = std::random_device{}();
            const uint64_t offset = 0;
            // Launch unified kernel: one block per transition
            const dim3 block_dim(256);
            const dim3 grid_dim(current_batch_size);
            // simulate phase and score_and_prune phase
            const SizeType profile_mem = m_nbins * sizeof(float);
            const SizeType pruning_mem =
                (2 * sizeof(int)) +
                (2 * m_device_manager->get_max_ntrials() * sizeof(int));
            const SizeType index_shmem_size =
                std::max(profile_mem, pruning_mem);
            const int num_warps = block_dim.x / 32;
            const size_t cub_storage_size =
                num_warps *
                (sizeof(typename cub::WarpScan<float>::TempStorage) +
                 2 * sizeof(typename cub::WarpReduce<float>::TempStorage));
            const size_t psum_workspace_size =
                num_warps * m_boxcar_cache_d->nsum * sizeof(float);

            const size_t shmem_size =
                index_shmem_size + cub_storage_size + psum_workspace_size;

            cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                                   shmem_size);

            transition_kernel<<<grid_dim, block_dim, shmem_size>>>(
                thrust::raw_pointer_cast(work_items_d.data()),
                static_cast<int>(current_batch_size),
                thrust::raw_pointer_cast(m_profile_d.data()),
                static_cast<int>(m_nbins), m_boxcar_cache_d->get_device_view(),
                m_bias_snr, var_add, thrust::raw_pointer_cast(m_probs_d.data()),
                static_cast<int>(m_nprobs), static_cast<int>(stage_offset_cur),
                thrust::raw_pointer_cast(m_states_d.data()),
                thrust::raw_pointer_cast(m_folds_next_d.data()),
                thrust::raw_pointer_cast(m_states_locks_d.data()), &allocator,
                seed, offset);
            cuda_utils::check_last_cuda_error("transition_kernel");
            cudaDeviceSynchronize();
        }
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