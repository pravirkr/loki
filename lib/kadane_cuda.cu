#include "loki/detection/kadane.hpp"

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/climits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/cub_helpers.cuh"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::detection {

namespace {

template <int BlockThreads, int MaxBins, int ProfilesPerWarp>
__global__ void
kernel_kadane_warp_multiprofile(const float* __restrict__ folds,
                                int nprofiles,
                                int nbins,
                                const float* __restrict__ biases,
                                float* __restrict__ scores,
                                const uint8_t* __restrict__ validation_mask,
                                uint8_t* __restrict__ filtered_mask,
                                float threshold) {
    constexpr int kWarpSize      = 32;
    constexpr int kWarpsPerBlock = BlockThreads / kWarpSize;

    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp_global_profile_idx =
        (static_cast<int>(blockIdx.x) * kWarpsPerBlock) + warp_id;
    const int profile_start_idx = warp_global_profile_idx * ProfilesPerWarp;

    if (profile_start_idx >= nprofiles) {
        return;
    }

    // Early validation: one lane per profile (b_idx == 0) clears filtered_mask
    if ((lane_id % 3) == 0) {
        const int p_sub = lane_id / 3;
        if (p_sub < ProfilesPerWarp) {
            const int g_profile = profile_start_idx + p_sub;
            if (g_profile < nprofiles && validation_mask[g_profile] == 0) {
                filtered_mask[g_profile] = 0;
            }
        }
    }
    __syncwarp();

    constexpr int nbins_padded = MaxBins + 1;

    extern __shared__ float s_fold[]; // NOLINT
    float* s_fold_warp = s_fold + (warp_id * ProfilesPerWarp * nbins_padded);

    // 1. Cooperative, conflict-free loading and on-the-fly normalization of
    // profiles
    for (int p_sub = 0; p_sub < ProfilesPerWarp; ++p_sub) {
        const int g_profile = profile_start_idx + p_sub;
        if (g_profile < nprofiles && validation_mask[g_profile] != 0) {
            const float* e_ptr    = folds + g_profile * 2 * nbins;
            const float* v_ptr    = e_ptr + nbins;
            float* s_fold_profile = s_fold_warp + p_sub * nbins_padded;
            for (int bin_idx = lane_id; bin_idx < nbins; bin_idx += kWarpSize) {
                s_fold_profile[bin_idx] =
                    __ldg(&e_ptr[bin_idx]) * rsqrtf(__ldg(&v_ptr[bin_idx]));
            }
        }
    }

    __syncwarp();

    // 2. Cooperative mean calculation: threads of the warp sum sections of each
    // profile
    float partial_sum            = 0.0F;
    constexpr int active_threads = ProfilesPerWarp * 3;
    const int p_sub              = lane_id / 3;
    const int b_idx              = lane_id % 3;
    const int base_lane          = p_sub * 3;

    if (lane_id < active_threads) {
        const int g_profile = profile_start_idx + p_sub;
        if (g_profile < nprofiles && validation_mask[g_profile] != 0) {
            const float* s_fold_profile = s_fold_warp + p_sub * nbins_padded;
            const int chunk             = (nbins + 2) / 3; // split into parts
            const int start             = b_idx * chunk;
            const int end               = min(nbins, start + chunk);
            for (int j = start; j < end; ++j) {
                partial_sum += s_fold_profile[j];
            }
        }
    }

    // Shuffle within each 3-lane group to get total sum per profile
    float total_sum                 = partial_sum;
    constexpr uint32_t kShuffleMask = 0xFFFFFFFF; // all 32 lanes participate

    int next1 = b_idx + 1;
    if (next1 >= 3)
        next1 = 0;
    int next2 = b_idx + 2;
    if (next2 >= 3)
        next2 -= 3;

    total_sum += __shfl_sync(kShuffleMask, partial_sum, base_lane + next1);
    total_sum += __shfl_sync(kShuffleMask, partial_sum, base_lane + next2);

    const float mean_val = total_sum / static_cast<float>(nbins);

    // 3. Parallel Bias Kadane scans
    if (lane_id < active_threads) {
        const int g_profile = profile_start_idx + p_sub;
        if (g_profile < nprofiles && validation_mask[g_profile] != 0) {
            const float bias            = biases[b_idx];
            const float* s_fold_profile = s_fold_warp + p_sub * nbins_padded;

            float max_sum = cuda::std::numeric_limits<float>::lowest();
            float cur_max = 0.0F;
            float w_cur   = 0.0F;
            float best_w  = 1.0F;

            float min_sum = cuda::std::numeric_limits<float>::max();
            float cur_min = 0.0F;
            float wc_min  = 0.0F;
            float best_wm = 1.0F;

            const float shift = mean_val + bias;
            for (int j = 0; j < nbins; ++j) {
                const float val = s_fold_profile[j] - shift;

                // Track Max Contiguous Subarray
                cur_max += val;
                w_cur += 1.0F;

                const bool update_max = (cur_max > max_sum);
                max_sum               = update_max ? cur_max : max_sum;
                best_w                = update_max ? w_cur : best_w;

                const bool reset_max = (cur_max < 0.0F);
                cur_max              = reset_max ? 0.0F : cur_max;
                w_cur                = reset_max ? 0.0F : w_cur;

                // Track Min Contiguous Subarray (for wrapped candidate)
                cur_min += val;
                wc_min += 1.0F;

                const bool update_min = (cur_min < min_sum);
                min_sum               = update_min ? cur_min : min_sum;
                best_wm               = update_min ? wc_min : best_wm;

                const bool reset_min = (cur_min > 0.0F);
                cur_min              = reset_min ? 0.0F : cur_min;
                wc_min               = reset_min ? 0.0F : wc_min;
            }

            // Candidate 1: Standard Non-wrapped
            float best_biased_sum = max_sum;
            int best_width        = static_cast<int>(best_w);

            // Candidate 2: Wrapped maximum
            const int excluded_width = static_cast<int>(best_wm);
            const int wrapped_width  = nbins - excluded_width;

            const float wrapped_sum =
                (-static_cast<float>(nbins) * bias) - min_sum;
            const bool use_wrapped =
                (wrapped_width > 0) && (wrapped_sum > best_biased_sum);
            best_biased_sum = use_wrapped ? wrapped_sum : best_biased_sum;
            best_width      = use_wrapped ? wrapped_width : best_width;

            float snr = cuda::std::numeric_limits<float>::lowest();
            if (best_width > 0 && best_width < nbins) {
                const float unbiased_sum =
                    best_biased_sum + (static_cast<float>(best_width) * bias);

                const float scale = sqrtf(
                    static_cast<float>(nbins) /
                    static_cast<float>(best_width * (nbins - best_width)));

                snr = unbiased_sum * scale;
            }

            // Shuffle-based max reduction across the 3 biases for this profile
            float max_snr = snr;
            const float other_snr1 =
                __shfl_sync(kShuffleMask, snr, base_lane + next1);
            const float other_snr2 =
                __shfl_sync(kShuffleMask, snr, base_lane + next2);
            max_snr = fmaxf(max_snr, fmaxf(other_snr1, other_snr2));

            if (b_idx == 0) {
                scores[g_profile]        = max_snr;
                filtered_mask[g_profile] = (max_snr >= threshold) ? 1 : 0;
            }
        }
    }
}

template <int BlockThreads>
__global__ void
kernel_kadane_segment_parallel(const float* __restrict__ folds,
                               int nprofiles,
                               int nbins,
                               const float* __restrict__ biases,
                               float* __restrict__ scores,
                               const uint8_t* __restrict__ validation_mask,
                               uint8_t* __restrict__ filtered_mask,
                               float threshold) {
    constexpr int kWarpSize      = 32;
    constexpr int kWarpsPerBlock = BlockThreads / kWarpSize;
    constexpr int kSegments      = 10;
    constexpr int kBiases        = 3;
    constexpr int kActiveThreads = kSegments * kBiases; // 30

    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
    const int profile_idx =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;

    if (profile_idx >= nprofiles) {
        return;
    }
    if (validation_mask[profile_idx] == 0) {
        filtered_mask[profile_idx] = 0;
        return;
    }

    // Shared memory: 1 profile per warp (same footprint as V5 at PPW=1)
    extern __shared__ float s_fold[]; // NOLINT
    float* s_profile = s_fold + warp_id * nbins;

    // ─── Phase 1: Cooperative coalesced loading (all 32 lanes) ──────────
    const float* e_ptr = folds + profile_idx * 2 * nbins;
    const float* v_ptr = e_ptr + nbins;
    for (int j = lane_id; j < nbins; j += kWarpSize) {
        s_profile[j] = __ldg(&e_ptr[j]) * rsqrtf(__ldg(&v_ptr[j]));
    }
    __syncwarp();

    // ─── Phase 2: Cooperative mean computation (all 32 lanes) ───────────
    float lane_sum = 0.0F;
    for (int j = lane_id; j < nbins; j += kWarpSize) {
        lane_sum += s_profile[j];
    }
    // Full warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, offset);
    }
    const float total_profile_sum = __shfl_sync(0xFFFFFFFF, lane_sum, 0);
    const float mean_val = total_profile_sum / static_cast<float>(nbins);

    // ─── Phase 3: Segment-parallel Kadane (30 active lanes) ─────────────
    // Lane mapping: seg_idx = lane_id / 3 (0-9), b_idx = lane_id % 3 (0-2)
    const int seg_idx   = lane_id / kBiases;
    const int b_idx     = lane_id % kBiases;
    const int seg_chunk = (nbins + kSegments - 1) / kSegments;
    const int seg_start = seg_idx * seg_chunk;
    const int seg_end   = min(nbins, seg_start + seg_chunk);
    const int seg_len   = (lane_id < kActiveThreads && seg_start < nbins)
                              ? (seg_end - seg_start)
                              : 0;

    const float bias =
        (lane_id < kActiveThreads) ? __ldg(&biases[b_idx]) : 0.0F;
    const float shift = mean_val + bias;

    // Segment metadata — initialized as merge identity (empty segment)
    float s_total     = 0.0F;
    float s_total_w   = 0.0F;
    float s_max_pfx   = cuda::std::numeric_limits<float>::lowest();
    float s_max_pfx_w = 0.0F;
    float s_min_pfx   = cuda::std::numeric_limits<float>::max();
    float s_min_pfx_w = 0.0F;
    float s_max_sfx   = cuda::std::numeric_limits<float>::lowest();
    float s_max_sfx_w = 0.0F;
    float s_min_sfx   = cuda::std::numeric_limits<float>::max();
    float s_min_sfx_w = 0.0F;
    float s_max_sub   = cuda::std::numeric_limits<float>::lowest();
    float s_max_sub_w = 0.0F;
    float s_min_sub   = cuda::std::numeric_limits<float>::max();
    float s_min_sub_w = 0.0F;

    if (seg_len > 0) {
        // Running state for sequential Kadane on this segment
        float running_sum = 0.0F;

        // For max_suffix = total - min(0, S_0, ..., S_{n-1})
        float min_pfx_inc0     = 0.0F; // includes the "virtual" S_{-1}=0
        float min_pfx_inc0_pos = 0.0F; // suffix_w = seg_len - pos

        // For min_suffix = total - max(0, S_0, ..., S_{n-1})
        float max_pfx_inc0     = 0.0F;
        float max_pfx_inc0_pos = 0.0F;

        // Kadane max-subarray state
        float cur_max      = 0.0F;
        float w_cur        = 0.0F;
        float kadane_max   = cuda::std::numeric_limits<float>::lowest();
        float kadane_max_w = 0.0F;

        // Kadane min-subarray state
        float cur_min      = 0.0F;
        float wc_min       = 0.0F;
        float kadane_min   = cuda::std::numeric_limits<float>::max();
        float kadane_min_w = 0.0F;

        for (int i = 0; i < seg_len; ++i) {
            const float val = s_profile[seg_start + i] - shift;
            running_sum += val;

            // Max prefix (max of S_0..S_i, not including virtual S_{-1}=0)
            if (running_sum > s_max_pfx) {
                s_max_pfx   = running_sum;
                s_max_pfx_w = static_cast<float>(i + 1);
            }
            // Min prefix
            if (running_sum < s_min_pfx) {
                s_min_pfx   = running_sum;
                s_min_pfx_w = static_cast<float>(i + 1);
            }
            // Track min(0, S_0, ...) for max_suffix computation
            if (running_sum < min_pfx_inc0) {
                min_pfx_inc0     = running_sum;
                min_pfx_inc0_pos = static_cast<float>(i + 1);
            }
            // Track max(0, S_0, ...) for min_suffix computation
            if (running_sum > max_pfx_inc0) {
                max_pfx_inc0     = running_sum;
                max_pfx_inc0_pos = static_cast<float>(i + 1);
            }

            // Kadane max
            cur_max += val;
            w_cur += 1.0F;
            const bool upd_max = (cur_max > kadane_max);
            kadane_max         = upd_max ? cur_max : kadane_max;
            kadane_max_w       = upd_max ? w_cur : kadane_max_w;
            const bool rst_max = (cur_max < 0.0F);
            cur_max            = rst_max ? 0.0F : cur_max;
            w_cur              = rst_max ? 0.0F : w_cur;

            // Kadane min
            cur_min += val;
            wc_min += 1.0F;
            const bool upd_min = (cur_min < kadane_min);
            kadane_min         = upd_min ? cur_min : kadane_min;
            kadane_min_w       = upd_min ? wc_min : kadane_min_w;
            const bool rst_min = (cur_min > 0.0F);
            cur_min            = rst_min ? 0.0F : cur_min;
            wc_min             = rst_min ? 0.0F : wc_min;
        }

        const float seg_len_f = static_cast<float>(seg_len);
        s_total               = running_sum;
        s_total_w             = seg_len_f;
        s_max_sub             = kadane_max;
        s_max_sub_w           = kadane_max_w;
        s_min_sub             = kadane_min;
        s_min_sub_w           = kadane_min_w;
        // max_suffix = total - min(0, S_0..S_{n-1})
        s_max_sfx   = s_total - min_pfx_inc0;
        s_max_sfx_w = seg_len_f - min_pfx_inc0_pos;
        // min_suffix = total - max(0, S_0..S_{n-1})
        s_min_sfx   = s_total - max_pfx_inc0;
        s_min_sfx_w = seg_len_f - max_pfx_inc0_pos;
    }

    // ─── Phase 4: Merge tree — 4 rounds to combine 10 segments ──────────
    // Round r: left segments at seg_idx % (2<<r) == 0 merge with partner
    //          at seg_idx + (1<<r), using warp lane stride 3*(1<<r).

    // Not using kActiveMask as it will be UB as per PTX ISA documentation
    constexpr uint32_t kShuffleMask = 0xFFFFFFFF; // all 32 lanes participate
#pragma unroll
    for (int round = 0; round < 4; ++round) {
        const int seg_stride   = (1 << round);   // 1, 2, 4, 8
        const int lane_stride  = 3 * seg_stride; // 3, 6, 12, 24
        const int partner_lane = lane_id + lane_stride;

        // All lanes shuffle partner's 14 segment values
        const float R_total = __shfl_sync(kShuffleMask, s_total, partner_lane);
        const float R_total_w =
            __shfl_sync(kShuffleMask, s_total_w, partner_lane);
        const float R_max_pfx =
            __shfl_sync(kShuffleMask, s_max_pfx, partner_lane);
        const float R_max_pfx_w =
            __shfl_sync(kShuffleMask, s_max_pfx_w, partner_lane);
        const float R_min_pfx =
            __shfl_sync(kShuffleMask, s_min_pfx, partner_lane);
        const float R_min_pfx_w =
            __shfl_sync(kShuffleMask, s_min_pfx_w, partner_lane);
        const float R_max_sfx =
            __shfl_sync(kShuffleMask, s_max_sfx, partner_lane);
        const float R_max_sfx_w =
            __shfl_sync(kShuffleMask, s_max_sfx_w, partner_lane);
        const float R_min_sfx =
            __shfl_sync(kShuffleMask, s_min_sfx, partner_lane);
        const float R_min_sfx_w =
            __shfl_sync(kShuffleMask, s_min_sfx_w, partner_lane);
        const float R_max_sub =
            __shfl_sync(kShuffleMask, s_max_sub, partner_lane);
        const float R_max_sub_w =
            __shfl_sync(kShuffleMask, s_max_sub_w, partner_lane);
        const float R_min_sub =
            __shfl_sync(kShuffleMask, s_min_sub, partner_lane);
        const float R_min_sub_w =
            __shfl_sync(kShuffleMask, s_min_sub_w, partner_lane);

        // Only "left" lanes that have a valid right partner perform the merge
        const bool is_left = (lane_id < kActiveThreads) &&
                             ((seg_idx % (2 * seg_stride)) == 0) &&
                             ((seg_idx + seg_stride) < kSegments);

        if (is_left) {
            // Cross terms MUST be computed before suffix updates
            const float cross_max   = s_max_sfx + R_max_pfx;
            const float cross_max_w = s_max_sfx_w + R_max_pfx_w;
            const float cross_min   = s_min_sfx + R_min_pfx;
            const float cross_min_w = s_min_sfx_w + R_min_pfx_w;

            // Max prefix: extends through all of L into R?
            const float new_mpfx = s_total + R_max_pfx;
            if (new_mpfx > s_max_pfx) {
                s_max_pfx   = new_mpfx;
                s_max_pfx_w = s_total_w + R_max_pfx_w;
            }

            // Min prefix
            const float new_minpfx = s_total + R_min_pfx;
            if (new_minpfx < s_min_pfx) {
                s_min_pfx   = new_minpfx;
                s_min_pfx_w = s_total_w + R_min_pfx_w;
            }

            // Max suffix: R's suffix, or R extends back through all of R into
            // L?
            const float new_msfx = R_total + s_max_sfx;
            if (new_msfx > R_max_sfx) {
                s_max_sfx   = new_msfx;
                s_max_sfx_w = R_total_w + s_max_sfx_w;
            } else {
                s_max_sfx   = R_max_sfx;
                s_max_sfx_w = R_max_sfx_w;
            }

            // Min suffix
            const float new_minsfx = R_total + s_min_sfx;
            if (new_minsfx < R_min_sfx) {
                s_min_sfx   = new_minsfx;
                s_min_sfx_w = R_total_w + s_min_sfx_w;
            } else {
                s_min_sfx   = R_min_sfx;
                s_min_sfx_w = R_min_sfx_w;
            }

            // Total
            s_total += R_total;
            s_total_w += R_total_w;

            // Max subarray: in L, in R, or crossing the boundary
            if (R_max_sub > s_max_sub) {
                s_max_sub   = R_max_sub;
                s_max_sub_w = R_max_sub_w;
            }
            if (cross_max > s_max_sub) {
                s_max_sub   = cross_max;
                s_max_sub_w = cross_max_w;
            }

            // Min subarray
            if (R_min_sub < s_min_sub) {
                s_min_sub   = R_min_sub;
                s_min_sub_w = R_min_sub_w;
            }
            if (cross_min < s_min_sub) {
                s_min_sub   = cross_min;
                s_min_sub_w = cross_min_w;
            }
        }
    }

    // ─── Phase 5: Compute SNR from merged result ────────────────────────
    // After 4 merge rounds, seg_idx==0 lanes (0, 1, 2) hold the final
    // merged result for biases 0, 1, 2 respectively.
    if (lane_id < kBiases) {
        // s_max_sub = max non-wrapping subarray (Kadane result)
        float best_biased_sum = s_max_sub;
        int best_width        = static_cast<int>(s_max_sub_w);

        // Wrapped candidate: complement of the min subarray
        const int excluded_width = static_cast<int>(s_min_sub_w);
        const int wrapped_width  = nbins - excluded_width;

        const float wrapped_sum =
            (-static_cast<float>(nbins) * bias) - s_min_sub;
        const bool use_wrapped =
            (wrapped_width > 0) && (wrapped_sum > best_biased_sum);
        best_biased_sum = use_wrapped ? wrapped_sum : best_biased_sum;
        best_width      = use_wrapped ? wrapped_width : best_width;

        float snr = cuda::std::numeric_limits<float>::lowest();
        if (best_width > 0 && best_width < nbins) {
            const float unbiased_sum =
                best_biased_sum + (static_cast<float>(best_width) * bias);

            const float scale =
                sqrtf(static_cast<float>(nbins) /
                      static_cast<float>(best_width * (nbins - best_width)));

            snr = unbiased_sum * scale;
        }

        // Max across 3 biases
        constexpr uint32_t kBiasMask = 0x7; // lanes 0-2
        const float snr1             = __shfl_sync(kBiasMask, snr, 1);
        const float snr2             = __shfl_sync(kBiasMask, snr, 2);
        const float max_snr          = fmaxf(snr, fmaxf(snr1, snr2));

        if (lane_id == 0) {
            scores[profile_idx]        = max_snr;
            filtered_mask[profile_idx] = (max_snr >= threshold) ? 1 : 0;
        }
    }
}

template <SizeType MaxBins, SizeType ProfilesPerWarp>
void launch_kernel_kadane_warp_multiprofile(
    cuda::std::span<const float> folds,
    cuda::std::span<const float> biases,
    cuda::std::span<float> scores,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint8_t> filtered_mask,
    float threshold,
    SizeType nprofiles,
    SizeType nbins,
    cudaStream_t stream) {
    constexpr SizeType kWarpSize        = 32;
    constexpr SizeType kThreadsPerBlock = 256;
    constexpr SizeType kWarpsPerBlock   = kThreadsPerBlock / kWarpSize;
    constexpr SizeType nbins_padded     = MaxBins + 1;
    const SizeType shmem_size =
        kWarpsPerBlock * ProfilesPerWarp * nbins_padded * sizeof(float);
    const dim3 block_dim(kThreadsPerBlock);
    const SizeType total_warps =
        (nprofiles + ProfilesPerWarp - 1) / ProfilesPerWarp;
    const SizeType blocks_per_grid =
        (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

    kernel_kadane_warp_multiprofile<kThreadsPerBlock, MaxBins, ProfilesPerWarp>
        <<<grid_dim, block_dim, shmem_size, stream>>>(
            folds.data(), static_cast<int>(nprofiles), static_cast<int>(nbins),
            biases.data(), scores.data(), validation_mask.data(),
            filtered_mask.data(), threshold);
    cuda_utils::check_last_cuda_error(
        "kernel_kadane_warp_multiprofile launch failed");
}

} // namespace

SizeType score_and_filter_max_cuda_kadane_d(
    cuda::std::span<const float> folds,
    cuda::std::span<const float> biases,
    cuda::std::span<float> scores,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint8_t> filtered_mask,
    float threshold,
    SizeType nprofiles,
    SizeType nbins,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) {
    // Dispatch mechanism: Use warp-based when nbins<=256
    constexpr SizeType kWarpRegimeMaxBins = 256;
    const bool use_warp_regime            = (nbins <= kWarpRegimeMaxBins);

    if (biases.size() != 3) {
        throw std::runtime_error(
            "score_and_filter_max_cuda_kadane_d: biases must be of size 3");
    }

    if (use_warp_regime) {
        if (nbins <= 32) {
            launch_kernel_kadane_warp_multiprofile<32, 10>(
                folds, biases, scores, validation_mask, filtered_mask,
                threshold, nprofiles, nbins, stream);
        } else if (nbins <= 64) {
            launch_kernel_kadane_warp_multiprofile<64, 10>(
                folds, biases, scores, validation_mask, filtered_mask,
                threshold, nprofiles, nbins, stream);
        } else if (nbins <= 128) {
            launch_kernel_kadane_warp_multiprofile<128, 10>(
                folds, biases, scores, validation_mask, filtered_mask,
                threshold, nprofiles, nbins, stream);
        } else if (nbins <= 256) {
            launch_kernel_kadane_warp_multiprofile<256, 5>(
                folds, biases, scores, validation_mask, filtered_mask,
                threshold, nprofiles, nbins, stream);
        } else {
            throw std::runtime_error(
                "kernel_kadane_warp_multiprofile launch failed: nbins "
                "exceeds compiled limit of 256");
        }
    } else {
        constexpr SizeType kWarpSize        = 32;
        constexpr SizeType kThreadsPerBlock = 256;
        constexpr SizeType kWarpsPerBlock   = kThreadsPerBlock / kWarpSize;
        // 1 profile per warp, nbins floats in shmem
        const SizeType shmem_size = kWarpsPerBlock * nbins * sizeof(float);
        const dim3 block_dim(kThreadsPerBlock);
        const SizeType blocks_per_grid =
            (nprofiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

        kernel_kadane_segment_parallel<kThreadsPerBlock>
            <<<grid_dim, block_dim, shmem_size, stream>>>(
                folds.data(), static_cast<int>(nprofiles),
                static_cast<int>(nbins), biases.data(), scores.data(),
                validation_mask.data(), filtered_mask.data(), threshold);
        cuda_utils::check_last_cuda_error(
            "kernel_kadane_segment_parallel launch failed");
    }

    // Count number of passing profiles

    auto transform_it =
        thrust::make_transform_iterator(filtered_mask.data(), Uint8ToUint32{});
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Sum(scratch_ws.cub_temp_storage,
                               scratch_ws.cub_temp_bytes, transform_it,
                               scratch_ws.d_reduce_out, nprofiles, stream),
        "cub::DeviceReduce::Sum failed");

    // Copy result back
    uint32_t nprofiles_passing = 0;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&nprofiles_passing, scratch_ws.d_reduce_out,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    // You already sync elsewhere usually, but if needed:
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "stream sync failed");

    return nprofiles_passing;
}
} // namespace loki::detection