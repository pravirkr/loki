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
#include "loki/cub_helpers.cuh"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/math_cuda.cuh"
#include "loki/progress.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"

namespace loki::detection {

using RNG = loki::math::DefaultDeviceRNG;

namespace {

struct ThresholdPairItem {
    uint32_t ithres_abs;
    uint32_t islot_cur;
    uint32_t jthres_abs;
    uint32_t jslot_prev;
};

// Batch transition data for parallel processing
struct TransitionWorkItem {
    uint32_t ithres_abs; // state lookup/write key
    uint32_t islot_cur;  // output pool slot
    uint32_t jthres_abs; // input state key
    uint32_t jslot_prev; // input pool slot
    uint32_t kprob;
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

__global__ void simulate_folds_init_kernel(float* __restrict__ folds_sim,
                                           const float* __restrict__ profile,
                                           uint32_t nbins,
                                           float bias_snr,
                                           float var_add,
                                           uint64_t seed,
                                           uint64_t offset,
                                           uint32_t ntrials) {
    const uint32_t tid          = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_trials = 2 * ntrials;
    if (tid >= total_trials) {
        return;
    }

    const uint32_t branch     = tid / ntrials; // 0=H0, 1=H1
    const uint32_t trial_id   = tid % ntrials;
    const uint32_t out_offset = branch * ntrials * nbins + trial_id * nbins;
    const float branch_scale  = (branch == 1) ? bias_snr : 0.0F;

    // Generate fold locally
    const float noise_stddev = sqrtf(var_add);
    typename RNG::Generator rng_noise(seed, tid, offset);
    typename RNG::NormalFloat dist_noise(0.0F, noise_stddev);

    const uint32_t main_loop = nbins / 4;
    for (uint32_t j = 0; j < main_loop * 4; j += 4) {
        float4 noise                  = dist_noise.generate4(rng_noise);
        folds_sim[out_offset + j + 0] = noise.x + profile[j + 0] * branch_scale;
        folds_sim[out_offset + j + 1] = noise.y + profile[j + 1] * branch_scale;
        folds_sim[out_offset + j + 2] = noise.z + profile[j + 2] * branch_scale;
        folds_sim[out_offset + j + 3] = noise.w + profile[j + 3] * branch_scale;
    }

    // Handle remaining bins (if nbins is not multiple of 4)
    for (uint32_t j = main_loop * 4; j < nbins; ++j) {
        float4 noise              = dist_noise.generate4(rng_noise);
        folds_sim[out_offset + j] = noise.x + profile[j] * branch_scale;
    }
}

__device__ __forceinline__ bool
snr_threshold_trial_streaming(const float* __restrict__ trial_arr,
                              uint32_t nbins,
                              const uint32_t* __restrict__ widths,
                              uint32_t nwidths,
                              float threshold,
                              float stdnoise) {
    const float inv_stdnoise = 1.0f / stdnoise;

    // compute total_sum once (identical to prefix[nbins-1])
    float total_sum = 0.0f;
    for (uint32_t i = 0; i < nbins; ++i)
        total_sum += trial_arr[i];

    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const uint32_t w = widths[iw];
        const float h    = sqrtf(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));

        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        // --- initial window sum (j=0)
        float window_sum = 0.0f;
        for (uint32_t i = 0; i < w; ++i)
            window_sum += trial_arr[i];

        float max_sum = window_sum;

        // --- slide through all circular positions
        for (uint32_t j = 1; j < nbins; ++j) {

            uint32_t add_idx = j + w - 1;
            if (add_idx >= nbins)
                add_idx -= nbins;

            uint32_t sub_idx = j - 1;

            window_sum += trial_arr[add_idx];
            window_sum -= trial_arr[sub_idx];

            max_sum = fmaxf(max_sum, window_sum);
        }

        const float snr =
            (((h + b) * max_sum) - (b * total_sum)) * inv_stdnoise;

        if (snr > threshold)
            return true;
    }

    return false;
}

template <uint32_t MAX_BINS>
__device__ __forceinline__ bool
snr_threshold_trial(const float* __restrict__ trial_arr,
                    uint32_t nbins,
                    const uint32_t* __restrict__ widths,
                    uint32_t nwidths,
                    float threshold,
                    float stdnoise) {
    float prefix[MAX_BINS];
    prefix[0] = trial_arr[0];
    for (uint32_t i = 1; i < nbins; ++i) {
        prefix[i] = prefix[i - 1] + trial_arr[i];
    }
    const float total_sum    = prefix[nbins - 1];
    const float inv_stdnoise = 1.0f / stdnoise;

    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const uint32_t w = static_cast<uint32_t>(widths[iw]);
        const float h    = sqrtf(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float max_diff = cuda::std::numeric_limits<float>::lowest();

        // Split the sliding window loop to eliminate branch in hot path
        // A. Window starts at 0 (Special case: no subtraction)
        // Sum = P[w-1]
        max_diff = fmaxf(max_diff, prefix[w - 1]);

        // B. Non-wrapping window: j from 1 to nbins-w
        // Sum = P[j+w-1] - P[j-1]
        const uint32_t loop_limit = nbins - w;

        for (uint32_t j = 1; j <= loop_limit; ++j) {
            float diff = prefix[j + w - 1] - prefix[j - 1];
            max_diff   = fmaxf(max_diff, diff);
        }

        // C. Wrapping window: j from nbins-w+1 to nbins-1
        // Sum = (Total - P[j-1]) + P[wrap_idx]
        // wrap_idx = (j + w - 1) - nbins
        for (uint32_t j = loop_limit + 1; j < nbins; ++j) {
            float before_window = prefix[j - 1];
            float after_wrap    = prefix[j + w - 1 - nbins];
            float diff          = (total_sum - before_window) + after_wrap;
            max_diff            = fmaxf(max_diff, diff);
        }

        const float snr =
            (((h + b) * max_diff) - (b * total_sum)) * inv_stdnoise;
        if (snr > threshold) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ uint32_t compute_pool_grid_offset(uint32_t slot_idx,
                                                             uint32_t prob_idx,
                                                             uint32_t branch,
                                                             uint32_t nprobs,
                                                             uint32_t ntrials,
                                                             uint32_t nbins) {
    // cell = slot_idx * nprobs + prob_idx
    return ((slot_idx * nprobs + prob_idx) * 2 + branch) * ntrials * nbins;
}

__device__ __forceinline__ uint32_t compute_pool_ntrials_offset(
    uint32_t slot_idx, uint32_t prob_idx, uint32_t branch, uint32_t nprobs) {
    return ((slot_idx * nprobs + prob_idx) * 2) + branch;
}

// Helper to check 16-byte alignment
__device__ __forceinline__ bool is_aligned_float4(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

__global__ __launch_bounds__(256, 4) // Hint: Max 256 threads, min 4 blocks/SM
    void simulate_folds_kernel(
        const TransitionWorkItem* __restrict__ work_items,
        uint32_t batch_size,
        const float* __restrict__ folds_current,
        const uint32_t* __restrict__ ntrials_current,
        float* __restrict__ scratch_folds,
        const float* __restrict__ profile,
        uint32_t nbins,
        float bias_snr,
        float var_in,
        float var_add,
        uint64_t seed,
        uint64_t offset,
        uint32_t ntrials,
        uint32_t nprobs) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Total number of trials across the entire batch
    // logical structure: [Batch] -> [H0/H1] -> [Trials]
    const uint32_t trials_per_item = 2 * ntrials;
    const uint32_t total_trials    = batch_size * trials_per_item;
    if (tid >= total_trials) {
        return;
    }

    const uint32_t work_idx  = tid / trials_per_item;
    const uint32_t local_idx = tid % trials_per_item;
    const uint32_t branch    = local_idx / ntrials; // 0=H0, 1=H1
    const uint32_t trial_id  = local_idx % ntrials;

    const TransitionWorkItem& item = work_items[work_idx];

    const uint32_t grid_offset_in = compute_pool_grid_offset(
        item.jslot_prev, item.kprob, branch, nprobs, ntrials, nbins);
    const uint32_t ntrials_count_idx = compute_pool_ntrials_offset(
        item.jslot_prev, item.kprob, branch, nprobs);
    const uint32_t ntrials_in = ntrials_current[ntrials_count_idx];

    uint32_t src_trial_idx = trial_id;
    if (trial_id >= ntrials_in) {
        // We are in the "fill" zone. Pick a random source trial.
        typename RNG::Generator rng_select(seed, trial_id, 0xCAFEBABE);
        typename RNG::UniformFloat dist_select(0.0F, 1.0F);
        float4 u_vec = dist_select.generate4(rng_select);
        // Map to integer index [0, ntrials_in - 1]
        src_trial_idx = static_cast<uint32_t>(u_vec.x * ntrials_in);
        if (src_trial_idx >= ntrials_in) {
            src_trial_idx = ntrials_in - 1;
        }
    }

    const uint32_t in_offset         = grid_offset_in + src_trial_idx * nbins;
    const uint32_t out_offset        = tid * nbins;
    const float branch_scale         = (branch == 1) ? bias_snr : 0.0F;
    const float* __restrict__ in_ptr = folds_current + in_offset;
    float* __restrict__ out_ptr      = scratch_folds + out_offset;

    const float noise_stddev = sqrtf(var_add);
    typename RNG::Generator rng_noise(seed, tid, offset);
    typename RNG::NormalFloat dist_noise(0.0F, noise_stddev);

    // Check if all pointers are 16-byte aligned for safe float4 access
    const bool can_use_float4 = is_aligned_float4(in_ptr) &&
                                is_aligned_float4(out_ptr) &&
                                is_aligned_float4(profile);

    if (can_use_float4 && (nbins % 4 == 0)) {
        // FAST PATH: Vectorized float4 loads/stores
        const float4* __restrict__ in_ptr4 =
            reinterpret_cast<const float4*>(in_ptr);
        float4* __restrict__ out_ptr4 = reinterpret_cast<float4*>(out_ptr);
        const float4* __restrict__ prof_ptr4 =
            reinterpret_cast<const float4*>(profile);

        const uint32_t vec_count = nbins / 4;

#pragma unroll 4
        for (uint32_t j = 0; j < vec_count; ++j) {
            // Single 128-bit load instead of 4x 32-bit loads
            const float4 data  = in_ptr4[j];
            const float4 prof  = prof_ptr4[j];
            const float4 noise = dist_noise.generate4(rng_noise);

            // Compute in registers
            out_ptr4[j] =
                make_float4(fmaf(prof.x, branch_scale, noise.x + data.x),
                            fmaf(prof.y, branch_scale, noise.y + data.y),
                            fmaf(prof.z, branch_scale, noise.z + data.z),
                            fmaf(prof.w, branch_scale, noise.w + data.w));
        }
    } else {
        // SAFE PATH: Scalar loads but still vectorized RNG
        const uint32_t vec_iters = nbins / 4;

// Process 4 elements at a time with vectorized RNG
#pragma unroll 4
        for (uint32_t j = 0; j < vec_iters; ++j) {
            const float4 noise = dist_noise.generate4(rng_noise);
            const uint32_t idx = j * 4;

            out_ptr[idx + 0] =
                in_ptr[idx + 0] + noise.x + profile[idx + 0] * branch_scale;
            out_ptr[idx + 1] =
                in_ptr[idx + 1] + noise.y + profile[idx + 1] * branch_scale;
            out_ptr[idx + 2] =
                in_ptr[idx + 2] + noise.z + profile[idx + 2] * branch_scale;
            out_ptr[idx + 3] =
                in_ptr[idx + 3] + noise.w + profile[idx + 3] * branch_scale;
        }

        // Handle remainder (if nbins % 4 != 0)
        const uint32_t remainder_start = vec_iters * 4;
        if (remainder_start < nbins) {
            const float4 noise = dist_noise.generate4(rng_noise);
            for (uint32_t j = remainder_start; j < nbins; ++j) {
                const uint32_t noise_idx = j - remainder_start;
                const float n            = (noise_idx == 0)   ? noise.x
                                           : (noise_idx == 1) ? noise.y
                                           : (noise_idx == 2) ? noise.z
                                                              : noise.w;
                out_ptr[j] = in_ptr[j] + n + profile[j] * branch_scale;
            }
        }
    }
}

template <uint32_t MAX_BINS>
__global__ __launch_bounds__(256, 4) // Hint: Max 256 threads, min 4 blocks/SM
    void score_filter_kernel(const TransitionWorkItem* __restrict__ work_items,
                             uint32_t batch_size,
                             const float* __restrict__ scratch_folds,
                             uint32_t* __restrict__ survive_flags,
                             uint32_t nbins,
                             const uint32_t* __restrict__ widths,
                             uint32_t nwidths,
                             const float* __restrict__ thresholds,
                             float var_in,
                             float var_add,
                             uint32_t ntrials) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Total number of trials across the entire batch
    // logical structure: [Batch] -> [H0/H1] -> [Trials]
    const uint32_t trials_per_item = 2 * ntrials;
    const uint32_t total_trials    = batch_size * trials_per_item;
    if (tid >= total_trials) {
        return;
    }

    const uint32_t work_idx = tid / trials_per_item;

    const TransitionWorkItem& item      = work_items[work_idx];
    const float threshold               = thresholds[item.ithres_abs];
    const float* __restrict__ trial_arr = scratch_folds + tid * nbins;

    // Compute SNR with early exit
    float prefix[MAX_BINS];
    prefix[0] = trial_arr[0];
    for (uint32_t i = 1; i < nbins; ++i) {
        prefix[i] = prefix[i - 1] + trial_arr[i];
    }
    const float total_sum    = prefix[nbins - 1];
    const float inv_stdnoise = 1.0f / sqrtf(var_in + var_add);

    bool survive = false;
    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const uint32_t w = static_cast<uint32_t>(widths[iw]);
        const float h    = sqrtf(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float max_diff = cuda::std::numeric_limits<float>::lowest();

        // Split the sliding window loop to eliminate branch in hot path
        // A. Window starts at 0 (Special case: no subtraction)
        // Sum = P[w-1]
        max_diff = fmaxf(max_diff, prefix[w - 1]);

        // B. Non-wrapping window: j from 1 to nbins-w
        // Sum = P[j+w-1] - P[j-1]
        const uint32_t loop_limit = nbins - w;

        for (uint32_t j = 1; j <= loop_limit; ++j) {
            float diff = prefix[j + w - 1] - prefix[j - 1];
            max_diff   = fmaxf(max_diff, diff);
        }

        // C. Wrapping window: j from nbins-w+1 to nbins-1
        // Sum = (Total - P[j-1]) + P[wrap_idx]
        // wrap_idx = (j + w - 1) - nbins
        for (uint32_t j = loop_limit + 1; j < nbins; ++j) {
            float before_window = prefix[j - 1];
            float after_wrap    = prefix[j + w - 1 - nbins];
            float diff          = (total_sum - before_window) + after_wrap;
            max_diff            = fmaxf(max_diff, diff);
        }

        const float snr =
            (((h + b) * max_diff) - (b * total_sum)) * inv_stdnoise;
        if (snr > threshold) {
            survive = true;
            break;
        }
    }
    survive_flags[tid] = survive ? 1u : 0u;
}

__global__ void
transition_decision_kernel(const TransitionWorkItem* __restrict__ work_items,
                           uint32_t batch_size,
                           const float* __restrict__ scratch_folds,
                           const uint32_t* __restrict__ scan_indices,
                           const uint32_t* __restrict__ survive_flags,
                           float* __restrict__ folds_next,
                           uint32_t* __restrict__ ntrials_next,
                           State* __restrict__ states,
                           int* __restrict__ locks,
                           const float* __restrict__ thresholds,
                           const float* __restrict__ probs,
                           uint32_t ntrials,
                           uint32_t nbins,
                           uint32_t nprobs,
                           float nbranches,
                           uint32_t stage_offset_prev,
                           uint32_t stage_offset_cur) {
    const uint32_t work_idx     = blockIdx.x + (blockIdx.y * gridDim.x);
    const uint32_t tid_in_block = threadIdx.x;
    if (work_idx >= batch_size) {
        return;
    }
    __shared__ bool should_update;
    __shared__ uint32_t count_h0, count_h1;
    __shared__ uint32_t base_h0, base_h1; // Base indices in scratch
    __shared__ uint32_t grid_offset_h0, grid_offset_h1;

    // Thread 0: Decision logic
    if (tid_in_block == 0) {
        const TransitionWorkItem& item = work_items[work_idx];
        const float threshold          = thresholds[item.ithres_abs];
        const uint32_t trials_per_item = 2 * ntrials;

        // Compute survivor counts using scan indices
        base_h0                = work_idx * trials_per_item;
        base_h1                = base_h0 + ntrials;
        const uint32_t last_h0 = base_h0 + ntrials - 1;
        const uint32_t last_h1 = base_h1 + ntrials - 1;

        count_h0 = scan_indices[last_h0] - scan_indices[base_h0] +
                   survive_flags[last_h0];
        count_h1 = scan_indices[last_h1] - scan_indices[base_h1] +
                   survive_flags[last_h1];
        const float succ_h0 = static_cast<float>(count_h0) / ntrials;
        const float succ_h1 = static_cast<float>(count_h1) / ntrials;

        // Generate next state
        const uint32_t state_idx_in =
            stage_offset_prev + (item.jthres_abs * nprobs) + item.kprob;
        const auto state_next = states[state_idx_in].gen_next(
            threshold, succ_h0, succ_h1, nbranches);
        const int iprob =
            find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);

        should_update = false;
        if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
            const int state_idx_out =
                stage_offset_cur + (item.ithres_abs * nprobs) + iprob;
            // Lock grid cell
            while (atomicCAS(&locks[state_idx_out], 0, 1) != 0)
                ;
            State& existing_state = states[state_idx_out];
            if (existing_state.is_empty || (state_next.complexity_cumul <
                                            existing_state.complexity_cumul)) {
                existing_state = state_next;
                should_update  = true;
                grid_offset_h0 = compute_pool_grid_offset(
                    item.islot_cur, iprob, 0, nprobs, ntrials, nbins);
                grid_offset_h1 = compute_pool_grid_offset(
                    item.islot_cur, iprob, 1, nprobs, ntrials, nbins);
                const uint32_t ntrials_offset_h0 = compute_pool_ntrials_offset(
                    item.islot_cur, iprob, 0, nprobs);
                const uint32_t ntrials_offset_h1 = compute_pool_ntrials_offset(
                    item.islot_cur, iprob, 1, nprobs);
                ntrials_next[ntrials_offset_h0] = count_h0;
                ntrials_next[ntrials_offset_h1] = count_h1;
            }
            // Unlock
            atomicExch(&locks[state_idx_out], 0);
        }
    }
    __syncthreads();

    // All threads: Cooperative sparse-to-dense copy

    if (should_update) {
        // Copy H0 survivors using scan-based mapping
        for (uint32_t trial = tid_in_block; trial < ntrials;
             trial += blockDim.x) {
            const uint32_t trial_idx = base_h0 + trial;

            if (survive_flags[trial_idx]) {
                // Dense position = scan value relative to base
                const uint32_t dense_idx =
                    scan_indices[trial_idx] - scan_indices[base_h0];

                // Copy this trial's data
                const uint32_t src_offset = trial_idx * nbins;
                const uint32_t dst_offset = grid_offset_h0 + dense_idx * nbins;

                for (uint32_t j = 0; j < nbins; ++j) {
                    folds_next[dst_offset + j] = scratch_folds[src_offset + j];
                }
            }
        }

        // Copy H1 survivors
        for (uint32_t trial = tid_in_block; trial < ntrials;
             trial += blockDim.x) {
            const uint32_t trial_idx = base_h1 + trial;

            if (survive_flags[trial_idx]) {
                const uint32_t dense_idx =
                    scan_indices[trial_idx] - scan_indices[base_h1];

                const uint32_t src_offset = trial_idx * nbins;
                const uint32_t dst_offset = grid_offset_h1 + dense_idx * nbins;

                for (uint32_t j = 0; j < nbins; ++j) {
                    folds_next[dst_offset + j] = scratch_folds[src_offset + j];
                }
            }
        }
    }
}

void simulate_score_kernel_launcher_cuda(
    const TransitionWorkItem* __restrict__ work_items,
    uint32_t batch_size,
    const float* __restrict__ folds_current,
    const uint32_t* __restrict__ ntrials_current,
    float* __restrict__ scratch_folds,
    uint32_t* __restrict__ survive_flags,
    const float* __restrict__ profile,
    uint32_t nbins,
    const uint32_t* __restrict__ widths,
    uint32_t nwidths,
    const float* __restrict__ thresholds,
    float bias_snr,
    float var_in,
    float var_add,
    uint64_t seed,
    uint64_t offset,
    uint32_t ntrials,
    uint32_t nprobs,
    cudaStream_t stream) {

    constexpr SizeType kThreadsPerBlock = 256;
    const SizeType total_work           = batch_size * 2 * ntrials;
    const SizeType blocks_per_grid =
        (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    simulate_folds_kernel<<<grid_dim, block_dim, 0, stream>>>(
        work_items, batch_size, folds_current, ntrials_current, scratch_folds,
        profile, nbins, bias_snr, var_in, var_add, seed, offset, ntrials,
        nprobs);
    cuda_utils::check_last_cuda_error("simulate_folds_kernel");

    auto dispatch_kernel = [&](auto... args) {
        if (nbins <= 32) {
            score_filter_kernel<32>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else if (nbins <= 64) {
            score_filter_kernel<64>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else if (nbins <= 128) {
            score_filter_kernel<128>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else if (nbins <= 256) {
            score_filter_kernel<256>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else if (nbins <= 512) {
            score_filter_kernel<512>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else if (nbins <= 1024) {
            score_filter_kernel<1024>
                <<<grid_dim, block_dim, 0, stream>>>(args...);
        } else {
            throw std::runtime_error(
                "score_filter_kernel_launcher_cuda: nbins exceeds compiled "
                "limit of 1024");
        }
    };
    dispatch_kernel(work_items, batch_size, scratch_folds, survive_flags, nbins,
                    widths, nwidths, thresholds, var_in, var_add, ntrials);
    cuda_utils::check_last_cuda_error("score_filter_kernel_launcher_cuda");
}

struct CountValidTransitionsFunctor {
    const State* __restrict__ states_ptr;
    const uint32_t* __restrict__ ntrials_current;
    uint32_t stage_offset_prev;
    uint32_t nprobs;

    __device__ uint32_t operator()(const ThresholdPairItem& pair) const {
        uint32_t count = 0;
        for (uint32_t kprob = 0; kprob < nprobs; ++kprob) {
            const uint32_t idx      = (pair.jthres_abs * nprobs) + kprob;
            const State& prev_state = states_ptr[stage_offset_prev + idx];
            if (prev_state.is_empty) {
                continue;
            }
            const uint32_t ntrials_offset_h0 =
                compute_pool_ntrials_offset(pair.jslot_prev, kprob, 0, nprobs);
            const uint32_t ntrials_offset_h1 =
                compute_pool_ntrials_offset(pair.jslot_prev, kprob, 1, nprobs);
            if (ntrials_current[ntrials_offset_h0] == 0 ||
                ntrials_current[ntrials_offset_h1] == 0) {
                continue;
            }
            // If we get here, the state is valid for transition.
            ++count;
        }
        return count;
    }
};

struct WriteValidTransitionsFunctor {
    const ThresholdPairItem* __restrict__ pairs_ptr;
    const uint32_t* __restrict__ offsets_ptr;
    TransitionWorkItem* __restrict__ out_ptr;

    const State* __restrict__ states_ptr;
    const uint32_t* __restrict__ ntrials_current;
    uint32_t stage_offset_prev;
    uint32_t nprobs;

    __device__ void operator()(uint32_t pair_id) const {
        const ThresholdPairItem& pair = pairs_ptr[pair_id];

        uint32_t write_pos = offsets_ptr[pair_id];
        for (uint32_t kprob = 0; kprob < nprobs; ++kprob) {
            const uint32_t idx      = (pair.jthres_abs * nprobs) + kprob;
            const State& prev_state = states_ptr[stage_offset_prev + idx];
            if (prev_state.is_empty) {
                continue;
            }
            const uint32_t ntrials_offset_h0 =
                compute_pool_ntrials_offset(pair.jslot_prev, kprob, 0, nprobs);
            const uint32_t ntrials_offset_h1 =
                compute_pool_ntrials_offset(pair.jslot_prev, kprob, 1, nprobs);
            if (ntrials_current[ntrials_offset_h0] == 0 ||
                ntrials_current[ntrials_offset_h1] == 0) {
                continue;
            }
            out_ptr[write_pos++] =
                TransitionWorkItem{pair.ithres_abs, pair.islot_cur,
                                   pair.jthres_abs, pair.jslot_prev, kprob};
        }
    }
};

struct WriteValidInitialTransitionsFunctor {
    const uint32_t* __restrict__ th_indices_ptr;

    __device__ TransitionWorkItem operator()(uint32_t i) const {
        return TransitionWorkItem{.ithres_abs = th_indices_ptr[i],
                                  .islot_cur  = i,
                                  .jthres_abs = 0u,
                                  .jslot_prev = 0u,
                                  .kprob      = 0u};
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
         SizeType batch_size,
         int device_id)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_trials_start(trials_start),
          m_batch_size(batch_size),
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

        if (m_nstages < 2) {
            throw std::invalid_argument(
                "DynamicThresholdSchemeCUDA requires at least 2 stages");
        }

        // Copy data to device
        m_thresholds_d       = m_thresholds;
        m_profile_d          = m_profile;
        m_probs_d            = m_probs;
        m_box_score_widths_d = m_box_score_widths;

        // Initialize memory management
        const auto slots_per_pool = compute_max_slots_needed();
        m_folds_current_d.resize(slots_per_pool * m_ntrials * m_nbins);
        m_folds_next_d.resize(slots_per_pool * m_ntrials * m_nbins);
        m_ntrials_current_d.resize(slots_per_pool);
        m_ntrials_next_d.resize(slots_per_pool);
        spdlog::info("Pre-allocated 2 CUDA pools of {} slots each",
                     slots_per_pool);

        // Initialize state management
        const auto grid_size = m_nstages * m_nthresholds * m_nprobs;
        m_states.resize(grid_size, State{});
        m_states_locks_d.resize(grid_size, 0);
        m_states_d = m_states;

        const SizeType n_batch_trials = m_batch_size * 2 * m_ntrials;
        m_survive_flags_d.resize(n_batch_trials);
        m_write_indices_d.resize(n_batch_trials);
        m_scratch_folds_d.resize(n_batch_trials * m_nbins);

        // Initialize CUB Temp Storage
        cuda_utils::check_cuda_call(
            cub::DeviceScan::ExclusiveSum(
                nullptr, m_cub_temp_bytes, static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), m_batch_size * 2 * m_ntrials),
            "cub::DeviceScan::ExclusiveSum failed");
        cuda_utils::check_cuda_call(
            cudaMalloc(&m_cub_temp_storage, m_cub_temp_bytes),
            "cudaMalloc failed");

        // Log memory usage
        const auto bytes_needed_persistent =
            (2 * slots_per_pool * m_ntrials * m_nbins * sizeof(float)) +
            (2 * slots_per_pool * sizeof(uint32_t)) +
            (2 * grid_size * sizeof(State)) + (grid_size * sizeof(int));
        const auto bytes_needed_workspace =
            (n_batch_trials * m_nbins * sizeof(float)) +
            (2 * n_batch_trials * sizeof(uint32_t));
        spdlog::info(
            "CUDA Memory usage: Allocated {:.2f} GiB (persistent) + {:.2f} "
            "GiB (workspace)",
            utils::to_gib(bytes_needed_persistent),
            utils::to_gib(bytes_needed_workspace));
    }
    ~Impl() {
        if (m_cub_temp_storage != nullptr) {
            cudaFree(m_cub_temp_storage);
        }
    }
    Impl(const Impl&)                = delete;
    Impl& operator=(const Impl&)     = delete;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;

    // Methods
    void run(SizeType thres_neigh = 10) {
        timing::ScopeTimer timer("DynamicThresholdSchemeCUDA::run");
        spdlog::info("Running dynamic threshold scheme on CUDA");
        const float var_init = 1.0F;
        const float var_add  = 1.0F;

        cudaStream_t stream = nullptr;
        cuda_utils::check_cuda_call(cudaStreamCreate(&stream),
                                    "cudaStreamCreate failed");

        // Initialize states
        float var_in = var_init;
        init_states(var_init, var_add, stream);
        var_in += var_add;
        // Swap fold grids and ntrials grids
        std::swap(m_folds_current_d, m_folds_next_d);
        std::swap(m_ntrials_current_d, m_ntrials_next_d);

        const bool show_progress = false;
        progress::ProgressGuard progress_guard(show_progress);
        auto bar =
            progress::make_standard_bar("Computing scheme", m_nstages - 1);

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            thrust::fill(thrust::cuda::par.on(stream), m_ntrials_next_d.begin(),
                         m_ntrials_next_d.end(), 0u);
            run_segment(istage, thres_neigh, var_in, var_add, stream);
            var_in += var_add;
            // Swap fold grids and ntrials grids
            std::swap(m_folds_current_d, m_folds_next_d);
            std::swap(m_ntrials_current_d, m_ntrials_next_d);
            if (show_progress) {
                bar.set_progress(istage);
            }
        }
        bar.mark_as_completed();
        // Copy final states back to host
        thrust::copy(thrust::cuda::par.on(stream), m_states_d.begin(),
                     m_states_d.end(), m_states.begin());
        cuda_utils::check_cuda_call(cudaStreamDestroy(stream),
                                    "cudaStreamDestroy failed");
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
    SizeType m_batch_size;
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

    // Device-side data
    thrust::device_vector<float> m_thresholds_d;
    thrust::device_vector<float> m_profile_d;
    thrust::device_vector<float> m_probs_d;
    thrust::device_vector<uint32_t> m_box_score_widths_d;
    thrust::device_vector<State> m_states_d;
    thrust::device_vector<int> m_states_locks_d;

    // Persistent Grid Storage (ping-pong between stages)
    // Shape: [nthresholds × nprobs × 2 × ntrials × nbins]
    //        Grid cell 0: [H0: ntrials×nbins | H1: ntrials×nbins]
    thrust::device_vector<float> m_folds_current_d;
    thrust::device_vector<float> m_folds_next_d;
    // Shape: [nthresholds × nprobs × 2]
    //        Each grid cell has 2 counts: [count_h0, count_h1]
    thrust::device_vector<uint32_t> m_ntrials_current_d;
    thrust::device_vector<uint32_t> m_ntrials_next_d;

    // Per-Batch Scratch Buffers (reallocated each batch)
    // Shape: [batch_size × 2 × ntrials × nbins]
    thrust::device_vector<float> m_scratch_folds_d;

    // Shape: [batch_size × 2 × ntrials]
    thrust::device_vector<uint32_t> m_survive_flags_d;
    thrust::device_vector<uint32_t> m_write_indices_d;

    // CUB Temp Storage
    void* m_cub_temp_storage  = nullptr;
    SizeType m_cub_temp_bytes = 0;

    SizeType compute_max_slots_needed() const noexcept {
        SizeType max_active_per_stage = 0;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto active_thresholds = get_current_thresholds_idx(istage);
            max_active_per_stage =
                std::max(max_active_per_stage, active_thresholds.size());
        }
        // h0 + h1 per cell
        const auto slots_per_pool = max_active_per_stage * m_nprobs * 2;
        spdlog::info(
            "CUDA allocation analysis: {} active thresholds max, {} prob "
            "bins",
            max_active_per_stage, m_nprobs);
        return slots_per_pool;
    }

    void init_states(float var_init, float var_add, cudaStream_t stream) {
        const uint64_t seed   = std::random_device{}();
        const uint64_t offset = 0;
        const auto nbranches  = m_branching_pattern[0];

        // Simulate the initial folds
        constexpr SizeType kThreadsPerBlock = 256;
        const SizeType total_work_init      = 2 * m_ntrials;
        const SizeType blocks_per_grid_init =
            (total_work_init + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim_init(kThreadsPerBlock);
        const dim3 grid_dim_init(blocks_per_grid_init);
        cuda_utils::check_kernel_launch_params(grid_dim_init, block_dim_init);
        simulate_folds_init_kernel<<<grid_dim_init, block_dim_init, 0,
                                     stream>>>(
            thrust::raw_pointer_cast(m_folds_current_d.data()),
            thrust::raw_pointer_cast(m_profile_d.data()), m_nbins, m_bias_snr,
            var_init, seed, offset, m_ntrials);
        cuda_utils::check_last_cuda_error("simulate_folds_init_kernel");

        // Simulate the intial state (reuse m_states_d as scratch space)
        // This will be eventually rewritten in the next stage
        const auto dummy_stage_offset_prev  = 1 * m_nthresholds * m_nprobs;
        m_states_d[dummy_stage_offset_prev] = State::initial();
        m_ntrials_current_d[0]              = m_ntrials;
        m_ntrials_current_d[1]              = m_ntrials;

        // Create work items for initial stage
        const auto thresholds_idx    = get_current_thresholds_idx(0);
        const auto total_transitions = thresholds_idx.size();
        thrust::device_vector<uint32_t> thresholds_idx_d = thresholds_idx;
        thrust::device_vector<TransitionWorkItem> work_items_d(
            total_transitions);
        WriteValidInitialTransitionsFunctor functor_write{
            .th_indices_ptr =
                thrust::raw_pointer_cast(thresholds_idx_d.data())};
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<uint32_t>(0),
                          thrust::counting_iterator<uint32_t>(
                              static_cast<uint32_t>(total_transitions)),
                          work_items_d.begin(), functor_write);
        spdlog::info("Initial work items created: {}", total_transitions);

        simulate_score_kernel_launcher_cuda(
            thrust::raw_pointer_cast(work_items_d.data()), total_transitions,
            thrust::raw_pointer_cast(m_folds_current_d.data()),
            thrust::raw_pointer_cast(m_ntrials_current_d.data()),
            thrust::raw_pointer_cast(m_scratch_folds_d.data()),
            thrust::raw_pointer_cast(m_survive_flags_d.data()),
            thrust::raw_pointer_cast(m_profile_d.data()), m_nbins,
            thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            m_box_score_widths.size(),
            thrust::raw_pointer_cast(m_thresholds_d.data()), m_bias_snr,
            var_init, var_add, seed, offset, m_ntrials, m_nprobs, stream);

        // CUB scan
        const SizeType total_work = total_transitions * 2 * m_ntrials;
        cuda_utils::check_cuda_call(
            cub::DeviceScan::ExclusiveSum(
                m_cub_temp_storage, m_cub_temp_bytes,
                thrust::raw_pointer_cast(m_survive_flags_d.data()),
                thrust::raw_pointer_cast(m_write_indices_d.data()), total_work,
                stream),
            "cub::DeviceScan::ExclusiveSum failed");

        // Transition decision (1 block per transition)
        dim3 block_dim(kThreadsPerBlock);
        dim3 grid_dim(total_transitions);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        transition_decision_kernel<<<grid_dim, block_dim, 0, stream>>>(
            thrust::raw_pointer_cast(work_items_d.data()), total_transitions,
            thrust::raw_pointer_cast(m_scratch_folds_d.data()),
            thrust::raw_pointer_cast(m_write_indices_d.data()),
            thrust::raw_pointer_cast(m_survive_flags_d.data()),
            thrust::raw_pointer_cast(m_folds_next_d.data()),
            thrust::raw_pointer_cast(m_ntrials_next_d.data()),
            thrust::raw_pointer_cast(m_states_d.data()),
            thrust::raw_pointer_cast(m_states_locks_d.data()),
            thrust::raw_pointer_cast(m_thresholds_d.data()),
            thrust::raw_pointer_cast(m_probs_d.data()), m_ntrials, m_nbins,
            m_nprobs, nbranches, dummy_stage_offset_prev, 0);
        cuda_utils::check_last_cuda_error("transition_decision_kernel");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
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
                     float var_in,
                     float var_add,
                     cudaStream_t stream) {
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;
        const auto nbranches         = m_branching_pattern[istage];

        std::vector<int32_t> prev_slot_of_thresh(m_nthresholds, -1);
        for (uint32_t s = 0; s < beam_idx_prev.size(); ++s) {
            prev_slot_of_thresh[beam_idx_prev[s]] = static_cast<int32_t>(s);
        }

        // Step 1: Generate all possible (ithres, jthres) pairs
        std::vector<ThresholdPairItem> threshold_pairs;
        threshold_pairs.reserve(beam_idx_cur.size() * thres_neigh * m_nprobs);
        for (SizeType islot = 0; islot < beam_idx_cur.size(); ++islot) {
            const auto ithres = beam_idx_cur[islot];
            const auto neighbour_beam_indices =
                utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                 thres_neigh);
            for (SizeType jthres : neighbour_beam_indices) {
                const int32_t jslot = prev_slot_of_thresh[jthres];
                if (jslot < 0) {
                    continue;
                }
                threshold_pairs.emplace_back(static_cast<uint32_t>(ithres),
                                             static_cast<uint32_t>(islot),
                                             static_cast<uint32_t>(jthres),
                                             static_cast<uint32_t>(jslot));
            }
        }
        const SizeType n_pairs = threshold_pairs.size();

        // Step 2: Count valid transitions (fast pass)
        thrust::device_vector<ThresholdPairItem> threshold_pairs_d =
            threshold_pairs;
        thrust::device_vector<SizeType> transition_counts_d(n_pairs);

        // Count transitions per pair
        CountValidTransitionsFunctor functor_count{
            .states_ptr = thrust::raw_pointer_cast(m_states_d.data()),
            .ntrials_current =
                thrust::raw_pointer_cast(m_ntrials_current_d.data()),
            .stage_offset_prev = static_cast<uint32_t>(stage_offset_prev),
            .nprobs            = static_cast<uint32_t>(m_nprobs)};
        thrust::transform(thrust::cuda::par.on(stream),
                          threshold_pairs_d.begin(), threshold_pairs_d.end(),
                          transition_counts_d.begin(), functor_count);

        // Compute prefix sum for offsets
        thrust::device_vector<uint32_t> offsets_d(n_pairs);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream), transition_counts_d.begin(),
            transition_counts_d.end(), offsets_d.begin(), uint32_t{0});
        const SizeType total_transitions = thrust::reduce(
            thrust::cuda::par.on(stream), transition_counts_d.begin(),
            transition_counts_d.end(), SizeType{0});

        if (total_transitions == 0) {
            return;
        }

        thrust::device_vector<TransitionWorkItem> work_items_d(
            total_transitions);
        WriteValidTransitionsFunctor functor_write{
            .pairs_ptr   = thrust::raw_pointer_cast(threshold_pairs_d.data()),
            .offsets_ptr = thrust::raw_pointer_cast(offsets_d.data()),
            .out_ptr     = thrust::raw_pointer_cast(work_items_d.data()),
            .states_ptr  = thrust::raw_pointer_cast(m_states_d.data()),
            .ntrials_current =
                thrust::raw_pointer_cast(m_ntrials_current_d.data()),
            .stage_offset_prev = static_cast<uint32_t>(stage_offset_prev),
            .nprobs            = static_cast<uint32_t>(m_nprobs)};

        thrust::for_each(thrust::cuda::par.on(stream),
                         thrust::counting_iterator<SizeType>(0),
                         thrust::counting_iterator<SizeType>(n_pairs),
                         functor_write);

        // Step 3: Process in batches
        const SizeType num_batches =
            (total_transitions + m_batch_size - 1) / m_batch_size;
        spdlog::info(
            "Stage {}, total transitions: {}, num batches: {}, batch size: {}",
            istage, total_transitions, num_batches, m_batch_size);

        for (SizeType b = 0; b < num_batches; ++b) {
            const SizeType start = b * m_batch_size;
            const SizeType end =
                std::min(start + m_batch_size, total_transitions);
            const SizeType current_batch_size = end - start;
            const auto work_items_span =
                cuda_utils::as_span(work_items_d)
                    .subspan(start, current_batch_size);

            // Process this batch
            // Generate random seed and offset
            const uint64_t seed   = std::random_device{}();
            const uint64_t offset = 0;

            simulate_score_kernel_launcher_cuda(
                work_items_span.data(), current_batch_size,
                thrust::raw_pointer_cast(m_folds_current_d.data()),
                thrust::raw_pointer_cast(m_ntrials_current_d.data()),
                thrust::raw_pointer_cast(m_scratch_folds_d.data()),
                thrust::raw_pointer_cast(m_survive_flags_d.data()),
                thrust::raw_pointer_cast(m_profile_d.data()), m_nbins,
                thrust::raw_pointer_cast(m_box_score_widths_d.data()),
                m_box_score_widths.size(),
                thrust::raw_pointer_cast(m_thresholds_d.data()), m_bias_snr,
                var_in, var_add, seed, offset, m_ntrials, m_nprobs, stream);

            // CUB scan
            const SizeType total_work = current_batch_size * 2 * m_ntrials;
            cuda_utils::check_cuda_call(
                cub::DeviceScan::ExclusiveSum(
                    m_cub_temp_storage, m_cub_temp_bytes,
                    thrust::raw_pointer_cast(m_survive_flags_d.data()),
                    thrust::raw_pointer_cast(m_write_indices_d.data()),
                    total_work, stream),
                "cub::DeviceScan::ExclusiveSum failed");

            // Transition decision (1 block per transition)
            constexpr SizeType kThreadsPerBlock = 256;
            dim3 block_dim(kThreadsPerBlock);
            dim3 grid_dim(current_batch_size);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
            transition_decision_kernel<<<grid_dim, block_dim, 0, stream>>>(
                work_items_span.data(), current_batch_size,
                thrust::raw_pointer_cast(m_scratch_folds_d.data()),
                thrust::raw_pointer_cast(m_write_indices_d.data()),
                thrust::raw_pointer_cast(m_survive_flags_d.data()),
                thrust::raw_pointer_cast(m_folds_next_d.data()),
                thrust::raw_pointer_cast(m_ntrials_next_d.data()),
                thrust::raw_pointer_cast(m_states_d.data()),
                thrust::raw_pointer_cast(m_states_locks_d.data()),
                thrust::raw_pointer_cast(m_thresholds_d.data()),
                thrust::raw_pointer_cast(m_probs_d.data()), m_ntrials, m_nbins,
                m_nprobs, nbranches, stage_offset_prev, stage_offset_cur);
            cuda_utils::check_last_cuda_error("transition_decision_kernel");
        }
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
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
    SizeType batch_size,
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
                                    batch_size,
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