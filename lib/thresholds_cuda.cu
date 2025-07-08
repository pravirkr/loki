#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <queue>
#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curanddx.hpp>
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda/std/optional>
#include <cuda/std/span>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/math_cuda.cuh"
#include "loki/simulation/simulation.hpp"
#include "loki/utils.hpp"

namespace loki::detection {

// Define the cuRANDDx Generator Descriptor.
using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                     curanddx::PhiloxRounds<10>() +
                     curanddx::SM<CURANDDX_SM>() + curanddx::Thread());

namespace {
class DualPoolFoldManager;

/**
 * CUDA FoldVector Handle - RAII wrapper for GPU memory with automatic
 * deallocation
 */
class FoldVectorHandle {
public:
    FoldVectorHandle(float* data,
                     SizeType actual_ntrials,
                     SizeType capacity_ntrials,
                     SizeType nbins,
                     float variance,
                     DualPoolFoldManager* manager)
        : m_data(data),
          m_actual_ntrials(actual_ntrials),
          m_capacity_ntrials(capacity_ntrials),
          m_nbins(nbins),
          m_variance(variance),
          m_manager(manager) {}

    // Move-only semantics
    FoldVectorHandle(const FoldVectorHandle&)            = delete;
    FoldVectorHandle& operator=(const FoldVectorHandle&) = delete;

    FoldVectorHandle(FoldVectorHandle&& other) noexcept
        : m_data(other.m_data),
          m_actual_ntrials(other.m_actual_ntrials),
          m_capacity_ntrials(other.m_capacity_ntrials),
          m_nbins(other.m_nbins),
          m_variance(other.m_variance),
          m_manager(other.m_manager) {
        other.m_manager = nullptr; // Prevent double deallocation
    }

    FoldVectorHandle& operator=(FoldVectorHandle&& other) noexcept {
        if (this != &other) {
            release();
            m_data             = other.m_data;
            m_actual_ntrials   = other.m_actual_ntrials;
            m_capacity_ntrials = other.m_capacity_ntrials;
            m_nbins            = other.m_nbins;
            m_variance         = other.m_variance;
            m_manager          = other.m_manager;
            other.m_manager    = nullptr;
        }
        return *this;
    }

    ~FoldVectorHandle() { release(); }

    // Interface
    cuda::std::span<float> data() { return {m_data, size()}; }
    cuda::std::span<const float> data() const { return {m_data, size()}; }
    float* raw_data() { return m_data; }
    const float* raw_data() const { return m_data; }
    SizeType size() const { return m_actual_ntrials * m_nbins; }
    SizeType ntrials() const { return m_actual_ntrials; }
    SizeType nbins() const { return m_nbins; }
    float variance() const { return m_variance; }
    void set_ntrials(SizeType ntrials) {
        assert(ntrials <= m_capacity_ntrials);
        m_actual_ntrials = ntrials;
    }
    void set_variance(float variance) { m_variance = variance; }

private:
    void release() noexcept; // Implemented after DualPoolFoldManager

    float* m_data;
    SizeType m_actual_ntrials;
    SizeType m_capacity_ntrials;
    SizeType m_nbins;
    float m_variance;
    DualPoolFoldManager* m_manager;
};

/**
 * CUDA Dual-Pool Memory Manager using thrust::device_vector for safety
 */
class DualPoolFoldManager {
public:
    DualPoolFoldManager(SizeType nbins,
                        SizeType ntrials_min,
                        SizeType slots_per_pool)
        : m_nbins(nbins),
          m_max_ntrials(2 * ntrials_min),
          m_slot_size(m_max_ntrials * nbins),
          m_slots_per_pool(slots_per_pool) {

        // Pre-allocate all memory for both pools using thrust
        m_data_a.resize(m_slots_per_pool * m_slot_size);
        m_slot_occupied_a.resize(m_slots_per_pool, false);
        m_data_b.resize(m_slots_per_pool * m_slot_size);
        m_slot_occupied_b.resize(m_slots_per_pool, false);

        // Initialize free slots for both pools
        for (SizeType i = 0; i < m_slots_per_pool; ++i) {
            m_free_slots_a.push(i);
            m_free_slots_b.push(i);
        }

        // Start with A as the "out" pool and B as the "in" pool
        set_pools_a_out_b_in();
    }

    DualPoolFoldManager(const DualPoolFoldManager&)            = delete;
    DualPoolFoldManager& operator=(const DualPoolFoldManager&) = delete;
    DualPoolFoldManager(DualPoolFoldManager&&)                 = delete;
    DualPoolFoldManager& operator=(DualPoolFoldManager&&)      = delete;

    ~DualPoolFoldManager() = default;

    /**
     * Allocates a new FoldVector from the current "out" pool.
     */
    [[nodiscard]] std::unique_ptr<FoldVectorHandle>
    allocate(SizeType initial_ntrials = 0, float variance = 0.0F) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_free_slots_out->empty()) {
            throw std::runtime_error(
                "DualPoolFoldManager 'out' pool exhausted!");
        }

        const auto slot_idx = m_free_slots_out->front();
        m_free_slots_out->pop();
        (*m_slot_occupied_out)[slot_idx] = true;
        float* slot_data = thrust::raw_pointer_cast(m_data_out->data()) +
                           (slot_idx * m_slot_size);
        return std::make_unique<FoldVectorHandle>(
            slot_data, initial_ntrials, m_max_ntrials, m_nbins, variance, this);
    }

    /**
     * Deallocates a handle's memory, returning it to the correct pool's free
     * list.
     */
    void deallocate(const float* data_ptr) noexcept {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Determine which pool the pointer belongs to and release it.
        auto* pool_a_start = thrust::raw_pointer_cast(m_data_a.data());
        auto* pool_b_start = thrust::raw_pointer_cast(m_data_b.data());

        if (data_ptr >= pool_a_start &&
            data_ptr < pool_a_start + m_data_a.size()) {
            deallocate_from_pool(data_ptr, pool_a_start, m_slot_occupied_a,
                                 m_free_slots_a);
        } else if (data_ptr >= pool_b_start &&
                   data_ptr < pool_b_start + m_data_b.size()) {
            deallocate_from_pool(data_ptr, pool_b_start, m_slot_occupied_b,
                                 m_free_slots_b);
        } else {
            // This should not happen if used correctly
            assert(false &&
                   "Attempted to deallocate memory not owned by this manager.");
            std::terminate();
        }
    }

    /**
     * Swaps the roles of the "in" and "out" pools.
     */
    void swap_pools() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_data_out == &m_data_a) {
            set_pools_b_out_a_in();
        } else {
            set_pools_a_out_b_in();
        }
    }

private:
    void deallocate_from_pool(const float* data_ptr,
                              const float* pool_start,
                              std::vector<bool>& occupied_pool,
                              std::queue<SizeType>& free_pool) const {
        // Calculate slot index from pointer offset
        const auto byte_offset = (data_ptr - pool_start);
        const auto slot_idx    = byte_offset / m_slot_size;

        assert(slot_idx < m_slots_per_pool);
        assert(occupied_pool[slot_idx]);

        occupied_pool[slot_idx] = false;
        free_pool.push(slot_idx);
    }

    void set_pools_a_out_b_in() {
        m_data_out          = &m_data_a;
        m_slot_occupied_out = &m_slot_occupied_a;
        m_free_slots_out    = &m_free_slots_a;
        m_data_in           = &m_data_b;
        m_slot_occupied_in  = &m_slot_occupied_b;
        m_free_slots_in     = &m_free_slots_b;
    }

    void set_pools_b_out_a_in() {
        m_data_out          = &m_data_b;
        m_slot_occupied_out = &m_slot_occupied_b;
        m_free_slots_out    = &m_free_slots_b;
        m_data_in           = &m_data_a;
        m_slot_occupied_in  = &m_slot_occupied_a;
        m_free_slots_in     = &m_free_slots_a;
    }

    // Pool A - using thrust::device_vector for safety
    thrust::device_vector<float> m_data_a;
    std::vector<bool> m_slot_occupied_a;
    std::queue<SizeType> m_free_slots_a;

    // Pool B
    thrust::device_vector<float> m_data_b;
    std::vector<bool> m_slot_occupied_b;
    std::queue<SizeType> m_free_slots_b;

    // Pointers to current in/out pools
    thrust::device_vector<float>* m_data_out = nullptr;
    std::vector<bool>* m_slot_occupied_out   = nullptr;
    std::queue<SizeType>* m_free_slots_out   = nullptr;
    thrust::device_vector<float>* m_data_in  = nullptr;
    std::vector<bool>* m_slot_occupied_in    = nullptr;
    std::queue<SizeType>* m_free_slots_in    = nullptr;

    // Config
    SizeType m_nbins;
    SizeType m_max_ntrials;
    SizeType m_slot_size;
    SizeType m_slots_per_pool;
    mutable std::mutex m_mutex;
};

inline void FoldVectorHandle::release() noexcept {
    if (m_manager != nullptr) {
        try {
            m_manager->deallocate(m_data);
        } catch (...) {
            assert(false && "Exception in FoldVectorHandle::release");
            std::terminate();
        }
        m_manager = nullptr;
    }
}

struct FoldsType {
    std::unique_ptr<FoldVectorHandle> folds_h0;
    std::unique_ptr<FoldVectorHandle> folds_h1;

    FoldsType() = default;
    FoldsType(std::unique_ptr<FoldVectorHandle> h0,
              std::unique_ptr<FoldVectorHandle> h1)
        : folds_h0(std::move(h0)),
          folds_h1(std::move(h1)) {}

    bool is_empty() const {
        return !folds_h0 || !folds_h1 || folds_h0->data().empty() ||
               folds_h1->data().empty();
    }

    FoldsType(const FoldsType&)            = delete;
    FoldsType& operator=(const FoldsType&) = delete;
    FoldsType(FoldsType&&)                 = default;
    FoldsType& operator=(FoldsType&&)      = default;

    ~FoldsType() = default;
};

struct FoldsDeviceData {
    const float* folds_h0_data;
    const float* folds_h1_data;
    int folds_h0_ntrials;
    int folds_h1_ntrials;
    float folds_h0_variance;
    float folds_h1_variance;
    bool is_empty;

    __device__ __host__ FoldsDeviceData()
        : folds_h0_data(nullptr),
          folds_h1_data(nullptr),
          folds_h0_ntrials(0),
          folds_h1_ntrials(0),
          folds_h0_variance(0.0f),
          folds_h1_variance(0.0f),
          is_empty(true) {}

    __device__ __host__ FoldsDeviceData(const FoldsType& folds)
        : folds_h0_data(folds.folds_h0 ? folds.folds_h0->raw_data() : nullptr),
          folds_h1_data(folds.folds_h1 ? folds.folds_h1->raw_data() : nullptr),
          folds_h0_ntrials(folds.folds_h0 ? folds.folds_h0->ntrials() : 0),
          folds_h1_ntrials(folds.folds_h1 ? folds.folds_h1->ntrials() : 0),
          folds_h0_variance(folds.folds_h0 ? folds.folds_h0->variance() : 0.0f),
          folds_h1_variance(folds.folds_h1 ? folds.folds_h1->variance() : 0.0f),
          is_empty(folds.is_empty()) {}
};

// Batch transition data for parallel processing
struct TransitionWorkItem { // NOLINT
    int threshold_idx;
    int prob_idx;
    int input_fold_idx;
    StateD input_state;
    float threshold;
    float nbranches;
    // Input fold info
    const float* input_h0_data;
    const float* input_h1_data;
    int input_h0_ntrials;
    int input_h1_ntrials;
    float input_variance;
    // Output buffer pointers (pre-allocated)
    float* output_h0_sim_data;
    float* output_h1_sim_data;
    int output_h0_sim_ntrials;
    int output_h1_sim_ntrials;
    float* output_h0_pruned_data;
    float* output_h1_pruned_data;
};

struct TransitionResult { // NOLINT
    int threshold_idx;
    int prob_idx;
    StateD computed_state;
    float* output_h0_data;
    float* output_h1_data;
    float output_variance;
    int output_ntrials_h0;
    int output_ntrials_h1;
    bool invalid;
};

struct TransitionBatch {
    thrust::device_vector<TransitionWorkItem> work_items_d;
    thrust::device_vector<float> scores_pool_d;
    thrust::device_vector<int> indices_pool_d;
    thrust::device_vector<TransitionResult> results_d;
    thrust::device_vector<float> probs_d;

    void reserve(SizeType max_items, SizeType max_trials) {
        work_items_d.reserve(max_items);
        scores_pool_d.resize(max_items * max_trials * 2);
        indices_pool_d.resize(max_items * max_trials * 2);
        results_d.resize(max_items);
        probs_d.resize(max_items);
    }
};

IndexType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::ranges::upper_bound(bins, value);
    return std::distance(bins.begin(), it) - 1;
}

__global__ void simulate_folds_kernel(float* __restrict__ folds_out,
                                      const float* __restrict__ folds_in,
                                      const float* __restrict__ profile,
                                      const float* __restrict__ noise,
                                      int ntrials_out,
                                      int ntrials_in,
                                      int nbins,
                                      float bias_snr) {
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= ntrials_out * nbins)
        return;

    const int trial_idx = i / nbins;
    const int bin_idx   = i % nbins;

    if (trial_idx >= ntrials_out)
        return;

    const int orig_trial  = (ntrials_in == 0) ? 0 : (trial_idx % ntrials_in);
    const int orig_offset = orig_trial * nbins + bin_idx;

    float fold_in_val = (ntrials_in == 0) ? 0.0f : folds_in[orig_offset];
    folds_out[i]      = fold_in_val + noise[i] + profile[bin_idx] * bias_snr;
}

__global__ void prune_trials_kernel(const float* __restrict__ folds_in,
                                    float* __restrict__ folds_out,
                                    const int* __restrict__ good_indices,
                                    int ntrials_out,
                                    int nbins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ntrials_out * nbins)
        return;

    int trial_idx = idx / nbins;
    int bin_idx   = idx % nbins;

    if (trial_idx < ntrials_out) {
        int orig_trial = good_indices[trial_idx];
        int orig_idx   = orig_trial * nbins + bin_idx;
        folds_out[idx] = folds_in[orig_idx];
    }
}

// Unified kernel for simulating both H0 and H1 folds in one launch
__global__ void
simulate_folds_dual_kernel(float* __restrict__ folds_h0_out,
                           float* __restrict__ folds_h1_out,
                           const float* __restrict__ folds_h0_in,
                           const float* __restrict__ folds_h1_in,
                           const float* __restrict__ profile,
                           const float* __restrict__ profile_scaled,
                           const float* __restrict__ noise_h0,
                           const float* __restrict__ noise_h1,
                           int ntrials_out,
                           int ntrials_in,
                           int nbins) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntrials_out * nbins)
        return;

    const int trial_idx = i / nbins;
    const int bin_idx   = i % nbins;

    if (trial_idx >= ntrials_out)
        return;

    // FIX 4: Match CPU logic exactly - no zero handling since we throw
    // exception above
    const int orig_trial  = trial_idx % ntrials_in;
    const int orig_offset = orig_trial * nbins + bin_idx;

    // H0 processing (no signal bias) - matches CPU line 349-350
    folds_h0_out[i] = folds_h0_in[orig_offset] + noise_h0[i];

    // H1 processing (with pre-scaled profile) - matches CPU line 349-350
    folds_h1_out[i] =
        folds_h1_in[orig_offset] + noise_h1[i] + profile_scaled[bin_idx];
}

// Unified kernel for pruning both H0 and H1 folds in one launch
__global__ void prune_folds_dual_kernel(const float* __restrict__ folds_h0_in,
                                        const float* __restrict__ folds_h1_in,
                                        float* __restrict__ folds_h0_out,
                                        float* __restrict__ folds_h1_out,
                                        const int* __restrict__ good_indices_h0,
                                        const int* __restrict__ good_indices_h1,
                                        int ntrials_h0_out,
                                        int ntrials_h1_out,
                                        int nbins) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // FIX 5: Process H0 and H1 separately with proper bounds checking
    // Process H0 trials
    if (idx < ntrials_h0_out * nbins) {
        const int trial_idx     = idx / nbins;
        const int bin_idx       = idx % nbins;
        const int orig_trial_h0 = good_indices_h0[trial_idx];
        const int orig_idx_h0   = orig_trial_h0 * nbins + bin_idx;
        folds_h0_out[idx]       = folds_h0_in[orig_idx_h0];
    }

    // Process H1 trials (offset by H0 total size)
    const int h1_idx = idx - (ntrials_h0_out * nbins);
    if (h1_idx >= 0 && h1_idx < ntrials_h1_out * nbins) {
        const int trial_idx     = h1_idx / nbins;
        const int bin_idx       = h1_idx % nbins;
        const int orig_trial_h1 = good_indices_h1[trial_idx];
        const int orig_idx_h1   = orig_trial_h1 * nbins + bin_idx;
        folds_h1_out[h1_idx]    = folds_h1_in[orig_idx_h1];
    }
}

__device__ float compute_snr_boxcar_score(const float* folds,
                                          SizeType ntrials,
                                          SizeType nbins,
                                          const SizeType* box_widths,
                                          SizeType nwidths,
                                          float noise_std) {
    // Inline the SNR boxcar scoring logic here
    // This replaces the external kernel call
    float max_score = 0.0f;
    // ... implement the scoring logic as device function
    return max_score;
}

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
    const int total_elements_h0 = work_item.output_h0_sim_ntrials * nbins;
    const int total_elements_h1 = work_item.output_h1_sim_ntrials * nbins;
    const float noise_stddev    = sqrtf(var_add);

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
        process_batch(base, total_elements_h0, work_item.input_h0_data,
                      work_item.output_h0_sim_data, work_item.input_h0_ntrials,
                      0, false);
    }
    // H1: with signal bias
    const int h1_seq_offset = (total_elements_h0 + 3) / 4;
    for (int base = tid * 4; base < total_elements_h1; base += block_size * 4) {
        process_batch(base, total_elements_h1, work_item.input_h1_data,
                      work_item.output_h1_sim_data, work_item.input_h1_ntrials,
                      h1_seq_offset, true);
    }
}

__device__ void prune_transition_phase(const TransitionWorkItem& work_item,
                                       const float* __restrict__ scores_h0,
                                       const float* __restrict__ scores_h1,
                                       float threshold,
                                       int nbins,
                                       int* output_ntrials_h0,
                                       int* output_ntrials_h1) {
    extern __shared__ int shm[]; // NOLINT
    const int tid        = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    // Shared memory for reduction
    int& shared_count_h0   = *(&shm[0]);
    int& shared_count_h1   = *(&shm[1]);
    int* shared_indices_h0 = &shm[2];
    int* shared_indices_h1 = &shm[2 + block_size];

    if (tid == 0) {
        shared_count_h0 = 0;
        shared_count_h1 = 0;
    }
    __syncthreads();

    // Collect surviving H0 and H1 trials into shared lists
    auto collect = [&](const float* scores, int max_trials, int& shared_count,
                       int* shared_idx) {
        int local_good[32]; // Local buffer for good indices
        int local_count = 0;

        // strided over trials
        for (int i = tid; i < max_trials; i += block_size) {
            if (scores[i] > threshold) {
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

    collect(scores_h0, work_item.output_h0_sim_ntrials, shared_count_h0,
            shared_indices_h0);
    collect(scores_h1, work_item.output_h1_sim_ntrials, shared_count_h1,
            shared_indices_h1);
    __syncthreads();

    // Copy H0 surviving trials
    for (int i = tid; i < shared_count_h0; i += block_size) {
        const int orig_trial    = shared_indices_h0[i];
        const int input_offset  = orig_trial * nbins;
        const int output_offset = i * nbins;

        // Copy entire trial (all bins)
        for (int j = 0; j < nbins; ++j) {
            work_item.output_h0_pruned_data[output_offset + j] =
                work_item.output_h0_sim_data[input_offset + j];
        }
    }
    // Copy H1 surviving trials
    for (int i = tid; i < shared_count_h1; i += block_size) {
        const int orig_trial    = shared_indices_h1[i];
        const int input_offset  = orig_trial * nbins;
        const int output_offset = i * nbins;

        for (int j = 0; j < nbins; ++j) {
            work_item.output_h1_pruned_data[output_offset + j] =
                work_item.output_h1_sim_data[input_offset + j];
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
    int ntrials_target,
    const float* __restrict__ probs,
    int nprobs,
    float* __restrict__ scores_pool,
    TransitionResult* __restrict__ results,
    uint64_t seed,
    uint64_t offset) {

    const auto item_idx = static_cast<int>(blockIdx.x);
    if (item_idx >= num_items) {
        return;
    }

    const auto& work_item = work_items[item_idx];
    const auto tid        = static_cast<int>(threadIdx.x);
    const auto block_size = static_cast<int>(blockDim.x);

    // Calculate memory offsets for this work item
    const auto score_offset_h0   = item_idx * ntrials_target;
    const auto score_offset_h1   = score_offset_h0 + ntrials_target;
    const auto indices_offset_h0 = item_idx * ntrials_target;
    const auto indices_offset_h1 = indices_offset_h0 + ntrials_target;

    // Shared memory for output counts
    __shared__ int shared_ntrials_h0_out;
    __shared__ int shared_ntrials_h1_out;

    // Phase 1: Simulation (threads collaborate)
    simulate_transition_phase(work_item, profile, nbins, bias_snr, var_add,
                              seed, offset);
    __syncthreads();

    // Phase 2: Scoring (threads collaborate)
    score_transition_phase(work_item, box_score_widths, nwidths, nbins,
                           scores_pool + score_offset_h0,
                           scores_pool + score_offset_h1, tid, block_size);
    __syncthreads();

    // Phase 3: Pruning (threads collaborate)
    prune_transition_phase(work_item, scores_pool + score_offset_h0,
                           scores_pool + score_offset_h1, work_item.threshold,
                           nbins, tid, block_size, &shared_ntrials_h0_out,
                           &shared_ntrials_h1_out);
    __syncthreads();

    // Phase 4: Compute final state and result (single thread per block)
    if (tid == 0) {
        const auto ntrials_h0_out = shared_ntrials_h0_out;
        const auto ntrials_h1_out = shared_ntrials_h1_out;
        // Calculate success rates
        const auto success_h0 =
            static_cast<float>(ntrials_h0_out) /
            static_cast<float>(work_item.output_h0_sim_ntrials);
        const auto success_h1 =
            static_cast<float>(ntrials_h1_out) /
            static_cast<float>(work_item.output_h1_sim_ntrials);

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
        result.output_h0_data    = work_item.output_h0_pruned_data;
        result.output_h1_data    = work_item.output_h1_pruned_data;
        result.output_variance   = work_item.input_variance + var_add;
        result.output_ntrials_h0 = ntrials_h0_out;
        result.output_ntrials_h1 = ntrials_h1_out;
        result.invalid = (iprob < 0 || iprob >= static_cast<IndexType>(nprobs));
    }
}

__global__ void update_states_kernel(const TransitionResult* results,
                                     int num_results,
                                     int nprobs,
                                     int stage_offset_cur,
                                     StateD* states,
                                     float** fold_h0_ptrs,
                                     float** fold_h1_ptrs,
                                     int* fold_ntrials_h0,
                                     int* fold_ntrials_h1,
                                     float* fold_variances_h0,
                                     float* fold_variances_h1) {

    const auto idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (idx >= num_results) {
        return;
    }
    const TransitionResult& result = results[idx];
    if (result.invalid) {
        return;
    }
    const int fold_idx  = (result.threshold_idx * nprobs) + result.prob_idx;
    const int state_idx = stage_offset_cur + fold_idx;
    // Use atomic compare-and-swap to handle race conditions
    // Convert float to int for atomic operations (assumes IEEE 754)
    const int new_complexity_bits =
        __float_as_int(result.computed_state.complexity_cumul);
    int* state_complexity_ptr =
        reinterpret_cast<int*>(&states[state_idx].complexity_cumul);

    // Atomic compare-and-swap loop
    int old_complexity_bits = atomicOr(state_complexity_ptr, 0);

    bool should_update = false;
    if (states[state_idx].is_empty) {
        // Try to claim this empty slot
        int expected_empty = __float_as_int(0.0F);
        if (atomicCAS(state_complexity_ptr, expected_empty,
                      new_complexity_bits) == expected_empty) {
            should_update = true;
        }
    } else {
        // Compare complexities and try to update if we're better
        float old_complexity = __int_as_float(old_complexity_bits);
        if (result.computed_state.complexity_cumul < old_complexity) {
            if (atomicCAS(state_complexity_ptr, old_complexity_bits,
                          new_complexity_bits) == old_complexity_bits) {
                should_update = true;
            }
        }
    }

    if (should_update) {
        states[state_idx]           = result.computed_state;
        fold_h0_ptrs[fold_idx]      = result.output_h0_data;
        fold_h1_ptrs[fold_idx]      = result.output_h1_data;
        fold_ntrials_h0[fold_idx]   = result.output_ntrials_h0;
        fold_ntrials_h1[fold_idx]   = result.output_ntrials_h1;
        fold_variances_h0[fold_idx] = result.output_variance;
        fold_variances_h1[fold_idx] = result.output_variance;
    }
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
    const cuda::std::optional<FoldsType>* folds_ptr;
    SizeType stage_offset_prev;
    SizeType nprobs;

    CountValidWorkItems(
        const thrust::device_vector<StateD>& states_d,
        const thrust::device_vector<cuda::std::optional<FoldsType>>&
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
        return {.pair = pair, .count = count};
    }
};

struct TransitionFunctor {
    const StateD* states_ptr;
    const cuda::std::optional<FoldsDeviceData>* folds_ptr;
    const float* thresholds_ptr;
    const float* branching_pattern_ptr;
    SizeType nprobs;
    SizeType ntrials;
    SizeType stage_offset_prev;
    SizeType istage;
    float var_add;
    TransitionWorkItem* work_items_ptr;
    const SizeType* offset_ptr;
    DualPoolFoldManager* manager;

    TransitionFunctor(
        const thrust::device_vector<StateD>& states_d,
        const thrust::device_vector<cuda::std::optional<FoldsDeviceData>>&
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
        DualPoolFoldManager* manager)
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
          manager(manager) {}

    __device__ void operator()(const PairWithCount& pair_with_count) {
        const auto& pair = pair_with_count.pair;
        SizeType ithres  = pair.ithres;
        SizeType jthresh = pair.jthresh;
        // Current offset
        SizeType base_offset =
            offset_ptr[&pair_with_count - &pair_with_count[0]];
        SizeType slot = base_offset;
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
            const auto ntrials_in_h0 = prev_fold_state->folds_h0_ntrials;
            const auto ntrials_in_h1 = prev_fold_state->folds_h1_ntrials;
            const auto repeat_factor_h0 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h0)));
            const auto ntrials_out_h0 = repeat_factor_h0 * ntrials_in_h0;
            const auto repeat_factor_h1 =
                static_cast<SizeType>(ceilf(static_cast<float>(ntrials) /
                                            static_cast<float>(ntrials_in_h1)));
            const auto ntrials_out_h1 = repeat_factor_h1 * ntrials_in_h1;

            auto output_h0_sim = manager->allocate(
                ntrials_out_h0, prev_fold_state->folds_h0_variance + var_add);
            auto output_h1_sim = manager->allocate(
                ntrials_out_h1, prev_fold_state->folds_h1_variance + var_add);
            auto output_h0_pruned =
                manager->allocate(ntrials_out_h0, output_h0_sim->variance());
            auto output_h1_pruned =
                manager->allocate(ntrials_out_h1, output_h1_sim->variance());

            // Populate the TransitionWorkItem
            TransitionWorkItem item;
            item.threshold_idx         = static_cast<int>(ithres);
            item.prob_idx              = static_cast<int>(kprob);
            item.input_fold_idx        = static_cast<int>(prev_fold_idx);
            item.input_state           = prev_state;
            item.threshold             = thresholds_ptr[ithres];
            item.nbranches             = branching_pattern_ptr[istage];
            item.input_h0_data         = prev_fold_state->folds_h0->raw_data();
            item.input_h1_data         = prev_fold_state->folds_h1->raw_data();
            item.input_h0_ntrials      = prev_fold_state->folds_h0->ntrials();
            item.input_h1_ntrials      = prev_fold_state->folds_h1->ntrials();
            item.output_h0_sim_data    = output_h0_sim->raw_data();
            item.output_h1_sim_data    = output_h1_sim->raw_data();
            item.output_h0_sim_ntrials = ntrials_out_h0;
            item.output_h1_sim_ntrials = ntrials_out_h1;
            item.output_h0_pruned_data = output_h0_pruned->raw_data();
            item.output_h1_pruned_data = output_h1_pruned->raw_data();
            work_items_ptr[slot]       = item;
            slot++;
        }
    }
};

std::unique_ptr<FoldVectorHandle>
simulate_folds(const FoldVectorHandle& folds_in,
               cuda::std::span<const float> profile,
               math::CuRandRNG& rng,
               DualPoolFoldManager& manager,
               float bias_snr       = 0.0F,
               float var_add        = 1.0F,
               SizeType ntrials_min = 1024) {
    const auto ntrials_in = folds_in.ntrials();
    const auto nbins      = folds_in.nbins();

    if (ntrials_in == 0 && ntrials_min == 0) {
        return manager.allocate(0, folds_in.variance() + var_add);
    }

    // Calculate output size
    const auto repeat_factor =
        (ntrials_in == 0)
            ? 1
            : static_cast<SizeType>(std::ceil(static_cast<float>(ntrials_min) /
                                              static_cast<float>(ntrials_in)));
    const auto ntrials_out = std::max(ntrials_min, repeat_factor * ntrials_in);

    // Allocate output
    auto folds_out =
        manager.allocate(ntrials_out, folds_in.variance() + var_add);

    // Generate noise
    thrust::device_vector<float> noise(ntrials_out * nbins);
    rng.generate_range(
        cuda::std::span<float>(thrust::raw_pointer_cast(noise.data()),
                               noise.size()),
        0.0f, std::sqrt(var_add));

    // Launch simulation kernel
    const int threads = 256;
    const int blocks  = (ntrials_out * nbins + threads - 1) / threads;
    simulate_folds_kernel<<<blocks, threads>>>(
        folds_out->raw_data(), folds_in.raw_data(),
        thrust::raw_pointer_cast(profile.data()),
        thrust::raw_pointer_cast(noise.data()), ntrials_out, ntrials_in, nbins,
        bias_snr);
    cuda_utils::check_last_cuda_error("simulate_folds_kernel");

    return folds_out;
}

std::unique_ptr<FoldVectorHandle>
prune_folds(const FoldVectorHandle& folds_in,
            cuda::std::span<const float> scores,
            float threshold,
            DualPoolFoldManager& manager) {
    const auto ntrials = folds_in.ntrials();
    const auto nbins   = folds_in.nbins();

    if (ntrials == 0) {
        return manager.allocate(0, folds_in.variance());
    }

    // Find indices of trials that survive the threshold
    thrust::device_vector<int> trial_indices(ntrials);
    thrust::sequence(trial_indices.begin(), trial_indices.end());

    auto new_end = thrust::copy_if(
        trial_indices.begin(), trial_indices.end(),
        thrust::device_pointer_cast(scores.data()), trial_indices.begin(),
        [threshold] __device__(float score) { return score > threshold; });

    const int ntrials_out = thrust::distance(trial_indices.begin(), new_end);

    if (ntrials_out == 0) {
        return manager.allocate(0, folds_in.variance());
    }

    // Allocate output
    auto folds_out = manager.allocate(ntrials_out, folds_in.variance());

    // Launch pruning kernel
    const int threads = 256;
    const int blocks  = (ntrials_out * nbins + threads - 1) / threads;
    prune_trials_kernel<<<blocks, threads>>>(
        folds_in.raw_data(), folds_out->raw_data(),
        thrust::raw_pointer_cast(trial_indices.data()), ntrials_out, nbins);
    cuda_utils::check_last_cuda_error("prune_trials_kernel");

    return folds_out;
}

std::tuple<State, FoldsType>
gen_next_using_thresh(const State& state_cur,
                      const FoldsType& folds_cur,
                      float threshold,
                      float nbranches,
                      float bias_snr,
                      cuda::std::span<const float> profile,
                      cuda::std::span<const SizeType> box_score_widths,
                      math::CuRandRNG& rng,
                      DualPoolFoldManager& manager,
                      float var_add    = 1.0F,
                      SizeType ntrials = 1024) {

    const auto ntrials_in = folds_cur.folds_h0->ntrials();
    const auto nbins      = folds_cur.folds_h0->nbins();

    // FIX 1: Match CPU zero input handling exactly
    if (ntrials_in == 0) {
        throw std::invalid_argument("No trials in the input folds");
    }

    // FIX 2: Match CPU memory allocation logic exactly
    const auto repeat_factor = static_cast<SizeType>(std::ceil(
        static_cast<float>(ntrials) / static_cast<float>(ntrials_in)));
    const auto ntrials_out   = repeat_factor * ntrials_in;

    // Allocate output folds
    auto folds_h0_sim =
        manager.allocate(ntrials_out, folds_cur.folds_h0->variance() + var_add);
    auto folds_h1_sim =
        manager.allocate(ntrials_out, folds_cur.folds_h1->variance() + var_add);

    // FIX 3: Pre-scale profile like CPU version for H1
    thrust::device_vector<float> profile_scaled(nbins);
    thrust::transform(profile.begin(), profile.end(), profile_scaled.begin(),
                      [bias_snr] __device__(float x) { return x * bias_snr; });

    // Generate noise for both H0 and H1
    thrust::device_vector<float> noise_h0(ntrials_out * nbins);
    thrust::device_vector<float> noise_h1(ntrials_out * nbins);
    rng.generate_range(
        cuda::std::span<float>(thrust::raw_pointer_cast(noise_h0.data()),
                               noise_h0.size()),
        0.0f, std::sqrt(var_add));
    rng.generate_range(
        cuda::std::span<float>(thrust::raw_pointer_cast(noise_h1.data()),
                               noise_h1.size()),
        0.0f, std::sqrt(var_add));

    // Kernel 1: Unified simulation for both H0 and H1
    const int threads = 256;
    const int blocks  = (ntrials_out * nbins + threads - 1) / threads;
    simulate_folds_dual_kernel<<<blocks, threads>>>(
        folds_h0_sim->raw_data(), folds_h1_sim->raw_data(),
        folds_cur.folds_h0->raw_data(), folds_cur.folds_h1->raw_data(),
        profile.data(), thrust::raw_pointer_cast(profile_scaled.data()),
        thrust::raw_pointer_cast(noise_h0.data()),
        thrust::raw_pointer_cast(noise_h1.data()), ntrials_out, ntrials_in,
        nbins);
    cuda_utils::check_last_cuda_error("simulate_folds_dual_kernel");

    // Allocate device memory for scores
    thrust::device_vector<float> scores_h0(ntrials_out);
    thrust::device_vector<float> scores_h1(ntrials_out);

    // Kernel 2: Score H0 folds
    {
        cuda::std::span<const float> folds_h0_span(folds_h0_sim->data());
        cuda::std::span<float> scores_h0_span(
            thrust::raw_pointer_cast(scores_h0.data()), scores_h0.size());
        detection::snr_boxcar_2d_max_cuda_d(
            folds_h0_span, ntrials_out, box_score_widths, scores_h0_span,
            std::sqrt(folds_h0_sim->variance()));
    }

    // Kernel 3: Score H1 folds
    {
        cuda::std::span<const float> folds_h1_span(folds_h1_sim->data());
        cuda::std::span<float> scores_h1_span(
            thrust::raw_pointer_cast(scores_h1.data()), scores_h1.size());
        detection::snr_boxcar_2d_max_cuda_d(
            folds_h1_span, ntrials_out, box_score_widths, scores_h1_span,
            std::sqrt(folds_h1_sim->variance()));
    }

    // Find surviving trials for both H0 and H1
    thrust::device_vector<int> trial_indices_h0(ntrials_out);
    thrust::device_vector<int> trial_indices_h1(ntrials_out);
    thrust::sequence(trial_indices_h0.begin(), trial_indices_h0.end());
    thrust::sequence(trial_indices_h1.begin(), trial_indices_h1.end());

    auto new_end_h0 = thrust::copy_if(
        trial_indices_h0.begin(), trial_indices_h0.end(), scores_h0.begin(),
        trial_indices_h0.begin(),
        [threshold] __device__(float score) { return score > threshold; });
    auto new_end_h1 = thrust::copy_if(
        trial_indices_h1.begin(), trial_indices_h1.end(), scores_h1.begin(),
        trial_indices_h1.begin(),
        [threshold] __device__(float score) { return score > threshold; });

    const int ntrials_h0_out =
        thrust::distance(trial_indices_h0.begin(), new_end_h0);
    const int ntrials_h1_out =
        thrust::distance(trial_indices_h1.begin(), new_end_h1);

    // Allocate output pruned folds
    auto folds_h0_pruned =
        manager.allocate(ntrials_h0_out, folds_h0_sim->variance());
    auto folds_h1_pruned =
        manager.allocate(ntrials_h1_out, folds_h1_sim->variance());

    // Kernel 4: Unified pruning for both H0 and H1
    if (ntrials_h0_out > 0 || ntrials_h1_out > 0) {
        const int total_elements = (ntrials_h0_out + ntrials_h1_out) * nbins;
        const int prune_blocks   = (total_elements + threads - 1) / threads;
        prune_folds_dual_kernel<<<prune_blocks, threads>>>(
            folds_h0_sim->raw_data(), folds_h1_sim->raw_data(),
            folds_h0_pruned->raw_data(), folds_h1_pruned->raw_data(),
            thrust::raw_pointer_cast(trial_indices_h0.data()),
            thrust::raw_pointer_cast(trial_indices_h1.data()), ntrials_h0_out,
            ntrials_h1_out, nbins);
        cuda_utils::check_last_cuda_error("prune_folds_dual_kernel");
    }

    // Calculate success rates
    const auto success_h0 =
        static_cast<float>(ntrials_h0_out) / static_cast<float>(ntrials_out);
    const auto success_h1 =
        static_cast<float>(ntrials_h1_out) / static_cast<float>(ntrials_out);

    const auto state_next =
        gen_next_state(state_cur, threshold, success_h0, success_h1, nbranches);
    return {state_next,
            FoldsType{std::move(folds_h0_pruned), std::move(folds_h1_pruned)}};
}

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
        m_profile_h = simulation::generate_folded_profile(nbins, ref_ducy);
        m_thresholds =
            detection::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs       = detection::compute_probs(nprobs, prob_min);
        m_nprobs      = m_probs.size();
        m_nbins       = m_profile_h.size();
        m_nstages     = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        auto box_score_widths_h =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_guess_path = detection::guess_scheme(
            m_nstages, snr_final, m_branching_pattern, m_trials_start);

        // Copy data to device
        m_profile_d          = m_profile_h;
        m_box_score_widths_d = box_score_widths_h;

        m_rng = std::make_unique<math::CuRandRNG>();
        // Initialize memory management
        const auto slots_per_pool = compute_max_allocations_needed();
        m_manager = std::make_unique<DualPoolFoldManager>(m_nbins, m_ntrials,
                                                          slots_per_pool);
        spdlog::info("Pre-allocated 2 CUDA pools of {} slots each",
                     slots_per_pool);

        // Initialize state management
        m_folds_current_d.resize(m_nthresholds * m_nprobs);
        m_folds_next_d.resize(m_nthresholds * m_nprobs);
        m_states_d.resize(m_nstages * m_nthresholds * m_nprobs, State{});
        m_states_h.resize(m_nstages * m_nthresholds * m_nprobs, State{});
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
        utils::ProgressGuard progress_guard(true);
        auto bar = utils::make_standard_bar("Computing scheme...");

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            run_segment(istage, thres_neigh);
            m_manager->swap_pools();
            std::swap(m_folds_current_d, m_folds_next_d);
            thrust::fill(thrust::device, m_folds_next_d.begin(),
                         m_folds_next_d.end(), cuda::std::nullopt);
            const auto progress = static_cast<float>(istage) /
                                  static_cast<float>(m_nstages - 1) * 100.0F;
            bar.set_progress(static_cast<SizeType>(progress));
        }
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
        file.createDataSet("profile", m_profile_h);
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
    std::vector<float> m_profile_h;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    std::vector<float> m_guess_path;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    float m_bias_snr;
    int m_device_id;

    // Device-side data
    thrust::device_vector<float> m_profile_d;
    thrust::device_vector<SizeType> m_box_score_widths_d;

    // Memory and RNG management
    std::unique_ptr<DualPoolFoldManager> m_manager;
    std::unique_ptr<math::CuRandRNG> m_rng;

    // State management
    std::vector<State> m_states;
    thrust::device_vector<StateD> m_states_d;
    thrust::device_vector<cuda::std::optional<FoldsType>> m_folds_current_d;
    thrust::device_vector<cuda::std::optional<FoldsType>> m_folds_next_d;

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
        // Create initial fold vectors
        auto folds_h0_init = m_manager->allocate(m_ntrials, 0.0F);
        auto folds_h1_init = m_manager->allocate(m_ntrials, 0.0F);

        // Zero-initialize on GPU
        cudaMemset(folds_h0_init->raw_data(), 0,
                   folds_h0_init->size() * sizeof(float));
        cudaMemset(folds_h1_init->raw_data(), 0,
                   folds_h1_init->size() * sizeof(float));
        cuda_utils::check_last_cuda_error("init_states memset");
        const auto m_profile_d_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(m_profile_d.data()), m_profile_d.size());
        const auto m_box_score_widths_d_span = cuda::std::span<const SizeType>(
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            m_box_score_widths_d.size());

        // Simulate the initial folds
        auto folds_h0_sim =
            simulate_folds(*folds_h0_init, m_profile_d_span, *m_rng, *m_manager,
                           0.0F, var_init, m_ntrials);
        auto folds_h1_sim =
            simulate_folds(*folds_h1_init, m_profile_d_span, *m_rng, *m_manager,
                           m_bias_snr, var_init, m_ntrials);

        State initial_state;
        FoldsType fold_state{std::move(folds_h0_sim), std::move(folds_h1_sim)};
        const auto thresholds_idx = get_current_thresholds_idx(0);
        for (SizeType ithres : thresholds_idx) {
            auto [cur_state, cur_fold_state] = gen_next_using_thresh(
                initial_state, fold_state, m_thresholds[ithres],
                m_branching_pattern[0], m_bias_snr, m_profile_d_span,
                m_box_score_widths_d_span, *m_rng, *m_manager, 1.0F, m_ntrials);

            const auto iprob =
                find_bin_index(m_probs, cur_state.success_h1_cumul);
            if (iprob < 0 || iprob >= static_cast<IndexType>(m_nprobs)) {
                continue;
            }
            const auto fold_idx         = (ithres * m_nprobs) + iprob;
            m_states_d[fold_idx]        = cur_state;
            m_folds_current_d[fold_idx] = std::move(cur_fold_state);
        }
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

    void collect_transitions_new(TransitionBatch& batch,
                                 const std::vector<SizeType>& beam_idx_cur,
                                 const std::vector<SizeType>& beam_idx_prev,
                                 SizeType stage_offset_prev,
                                 SizeType istage,
                                 SizeType thres_neigh,
                                 float var_add) {
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
        thrust::device_vector<IndexPair> d_pairs = h_pairs;
        thrust::device_vector<PairWithCount> d_pairs_with_count(d_pairs.size());
        thrust::transform(thrust::device, d_pairs.begin(), d_pairs.end(),
                          d_pairs_with_count.begin(),
                          CountValidWorkItems(m_states_d, m_folds_current_d,
                                              stage_offset_prev, m_nprobs));

        thrust::device_vector<SizeType> d_offsets(d_pairs.size());
        thrust::transform(
            thrust::device, d_pairs_with_count.begin(),
            d_pairs_with_count.end(), d_offsets.begin(),
            [] __device__(const PairWithCount& pwc) { return pwc.count; });
        thrust::exclusive_scan(thrust::device, d_offsets.begin(),
                               d_offsets.end(), d_offsets.begin());

        SizeType total_items = 0;
        if (!d_offsets.empty()) {
            const auto count =
                thrust::raw_pointer_cast(
                    d_pairs_with_count.data())[d_pairs_with_count.size() - 1]
                    .count;
            total_items = d_offsets.back() + count;
        }
        batch.work_items_d.resize(total_items);

        thrust::for_each(
            thrust::device, d_pairs_with_count.begin(),
            d_pairs_with_count.end(),
            TransitionFunctor(m_states_d, m_folds_current_d, m_thresholds,
                              m_branching_pattern, m_nprobs, m_ntrials,
                              stage_offset_prev, istage, var_add,
                              batch.work_items_d, d_offsets, m_manager.get()));

        cudaDeviceSynchronize();
    }

    void collect_transitions(TransitionBatch& batch,
                             const std::vector<SizeType>& beam_idx_cur,
                             const std::vector<SizeType>& beam_idx_prev,
                             SizeType stage_offset_prev,
                             SizeType istage,
                             SizeType thres_neigh,
                             float var_add) {
        for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
            const auto ithres = beam_idx_cur[i];
            // Find nearest neighbors in the previous beam
            const auto neighbour_beam_indices =
                utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                 thres_neigh);

            for (SizeType jthresh : neighbour_beam_indices) {
                for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                    const auto prev_fold_idx = (jthresh * m_nprobs) + kprob;
                    const auto prev_state =
                        m_states_d[stage_offset_prev + prev_fold_idx];

                    if (prev_state.is_empty) {
                        continue;
                    }

                    const auto& prev_fold_state =
                        m_folds_current_d[prev_fold_idx];
                    if (!prev_fold_state.has_value() ||
                        prev_fold_state->is_empty())
                        continue;

                    // Pre-allocate output buffers
                    const auto ntrials_in_h0 =
                        *prev_fold_state->folds_h0.ntrials();
                    const auto ntrials_in_h1 =
                        *prev_fold_state->folds_h1.ntrials();
                    if (ntrials_in_h0 == 0 || ntrials_in_h1 == 0) {
                        throw std::invalid_argument(
                            "No trials in the input folds");
                    }
                    const auto repeat_factor_h0 = static_cast<SizeType>(
                        std::ceil(static_cast<float>(m_ntrials) /
                                  static_cast<float>(ntrials_in_h0)));
                    const auto ntrials_out_h0 =
                        repeat_factor_h0 * ntrials_in_h0;
                    const auto repeat_factor_h1 = static_cast<SizeType>(
                        std::ceil(static_cast<float>(m_ntrials) /
                                  static_cast<float>(ntrials_in_h1)));
                    const auto ntrials_out_h1 =
                        repeat_factor_h1 * ntrials_in_h1;
                    auto output_h0_sim = m_manager->allocate(
                        ntrials_out_h0,
                        *prev_fold_state->folds_h0.variance() + var_add);
                    auto output_h1_sim = m_manager->allocate(
                        ntrials_out_h1,
                        *prev_fold_state->folds_h1.variance() + var_add);
                    auto output_h0_pruned = m_manager->allocate(
                        ntrials_out_h0, output_h0_sim->variance());
                    auto output_h1_pruned = m_manager->allocate(
                        ntrials_out_h1, output_h1_sim->variance());

                    TransitionWorkItem item;
                    item.threshold_idx  = ithres;
                    item.prob_idx       = kprob;
                    item.input_fold_idx = prev_fold_idx;
                    item.input_state    = prev_state;
                    item.threshold      = m_thresholds[ithres];
                    item.nbranches      = m_branching_pattern[istage];
                    item.input_h0_data  = prev_fold_state->folds_h0->raw_data();
                    item.input_h1_data  = prev_fold_state->folds_h1->raw_data();
                    item.input_h0_ntrials =
                        prev_fold_state->folds_h0->ntrials();
                    item.input_h1_ntrials =
                        prev_fold_state->folds_h1->ntrials();
                    item.output_h0_sim_data    = output_h0_sim->raw_data();
                    item.output_h1_sim_data    = output_h1_sim->raw_data();
                    item.output_h0_sim_ntrials = ntrials_out_h0;
                    item.output_h1_sim_ntrials = ntrials_out_h1;
                    item.output_h0_pruned_data = output_h0_pruned->raw_data();
                    item.output_h1_pruned_data = output_h1_pruned->raw_data();

                    batch.work_items_d.push_back(item);
                }
            }
        }
    }

    void run_segment(SizeType istage, SizeType thres_neigh) {
        const float var_add          = 1.0F;
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;

        // Step 1: Collect all transitions into batch
        TransitionBatch batch;
        const SizeType max_transitions =
            beam_idx_cur.size() * thres_neigh * m_nprobs;
        batch.reserve(max_transitions, m_ntrials * 2);

        collect_transitions(batch, beam_idx_cur, beam_idx_prev,
                            stage_offset_prev, istage, thres_neigh, var_add);

        if (batch.work_items_d.empty())
            return;

        // Step 2: Process entire batch on GPU
        process_transition_batch(batch, var_add);

        // Step 3: Update states and folds from results
        update_states_from_batch(batch, stage_offset_cur);
    }

    void update_states_from_batch(TransitionBatch& batch,
                                  SizeType stage_offset_cur) {
        // Copy results back from device (states and fold metadata)
        thrust::copy(m_states_d.begin() + stage_offset_cur,
                     m_states_d.begin() + stage_offset_cur +
                         (m_nthresholds * m_nprobs),
                     m_states.begin() + stage_offset_cur);

        // Update m_folds_next with the new pointers and metadata
        // (This part depends on how you want to structure the fold management)

        // Clean up temporary handles
        m_temp_handles.clear();
    }

    void process_transition_batch(TransitionBatch& batch, float var_add) {
        const auto num_items = static_cast<int>(batch.work_items_d.size());

        // Generate random seed and offset
        const unsigned long long seed   = std::random_device{}();
        const unsigned long long offset = 0;
        // Launch unified kernel: one block per transition
        const dim3 block_dim(256);
        const dim3 grid_dim(num_items);
        const size_t shared_mem_size = m_nbins * sizeof(float);
        cuda::utils::check_kernel_launch_params(grid_dim, block_dim,
                                                shared_mem_size);

        process_transitions_unified_kernel<<<grid_dim, block_dim,
                                             shared_mem_size>>>(
            thrust::raw_pointer_cast(batch.work_items_d.data()), num_items,
            thrust::raw_pointer_cast(m_profile_d.data()),
            static_cast<int>(m_nbins),
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            static_cast<int>(m_box_score_widths_d.size()), m_bias_snr, var_add,
            m_ntrials, thrust::raw_pointer_cast(batch.probs_d.data()),
            static_cast<int>(m_nprobs),
            thrust::raw_pointer_cast(batch.scores_pool_d.data()),
            thrust::raw_pointer_cast(batch.indices_pool_d.data()),
            thrust::raw_pointer_cast(batch.results_d.data()), seed, offset);
        cuda_utils::check_last_cuda_error("process_transitions_unified_kernel");

        // Phase 2: Launch state update kernel
        const dim3 update_block_dim(256);
        const dim3 update_grid_dim((num_items + update_block_dim.x - 1) /
                                   update_block_dim.x);

        update_states_kernel<<<update_grid_dim, update_block_dim>>>(
            thrust::raw_pointer_cast(batch.results_d.data()), num_items,
            static_cast<int>(m_nprobs), stage_offset_cur,
            thrust::raw_pointer_cast(m_states_d.data()),
            thrust::raw_pointer_cast(m_fold_h0_ptrs_d.data()),
            thrust::raw_pointer_cast(m_fold_h1_ptrs_d.data()),
            thrust::raw_pointer_cast(m_fold_ntrials_h0_d.data()),
            thrust::raw_pointer_cast(m_fold_ntrials_h1_d.data()),
            thrust::raw_pointer_cast(m_fold_variances_h0_d.data()),
            thrust::raw_pointer_cast(m_fold_variances_h1_d.data()));
        cuda_utils::check_last_cuda_error("update_states_kernel");
    }

    void run_segment(SizeType istage, SizeType thres_neigh) {
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;
        const auto m_profile_d_span  = cuda::std::span<const float>(
            thrust::raw_pointer_cast(m_profile_d.data()), m_profile_d.size());
        const auto m_box_score_widths_d_span = cuda::std::span<const SizeType>(
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            m_box_score_widths_d.size());

        for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
            const auto ithres = beam_idx_cur[i];
            // Find nearest neighbors in the previous beam
            const auto neighbour_beam_indices =
                utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                 thres_neigh);
            for (SizeType jthresh : neighbour_beam_indices) {
                for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                    const auto prev_fold_idx = (jthresh * m_nprobs) + kprob;
                    const auto prev_state =
                        m_states[stage_offset_prev + prev_fold_idx];

                    if (prev_state.is_empty) {
                        continue;
                    }
                    const auto& prev_fold_state =
                        m_folds_current[prev_fold_idx];
                    if (!prev_fold_state.has_value() ||
                        prev_fold_state->is_empty()) {
                        continue;
                    }
                    auto [cur_state, cur_fold_state] = gen_next_using_thresh(
                        prev_state, *prev_fold_state, m_thresholds[ithres],
                        m_branching_pattern[istage], m_bias_snr,
                        m_profile_d_span, m_box_score_widths_d_span, *m_rng,
                        *m_manager, 1.0F, m_ntrials);

                    const auto iprob =
                        find_bin_index(m_probs, cur_state.success_h1_cumul);
                    if (iprob < 0 ||
                        iprob >= static_cast<IndexType>(m_nprobs)) {
                        continue;
                    }

                    const auto cur_idx       = (ithres * m_nprobs) + iprob;
                    const auto cur_state_idx = stage_offset_cur + cur_idx;

                    auto& existing_state = m_states[cur_state_idx];
                    auto& existing_folds = m_folds_next[cur_idx];

                    if (existing_state.is_empty ||
                        cur_state.complexity_cumul <
                            existing_state.complexity_cumul) {
                        existing_state = cur_state;
                        existing_folds = std::move(cur_fold_state);
                    }
                }
            }
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