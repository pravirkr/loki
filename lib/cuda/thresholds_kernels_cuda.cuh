#pragma once

/**
 * @file thresholds_kernels_cuda.cuh
 * @brief Scorers, device helpers and scorer-templated kernels of the CUDA
 * DynamicThresholdScheme. Internal.
 *
 * Each scorer type instantiates heavy, fully unrolled kernels. To keep
 * compile time down they are compiled in their own translation units:
 * thresholds_reg_cuda.cu (RegScorer) and thresholds_generic_cuda.cu
 * (GenericScorer). thresholds_cuda.cu reaches them only through
 * ScorerLaunch<Scorer>, whose instantiations are declared extern below.
 * The kernel code itself is unchanged by the split.
 */

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/cuda/device_rng.cuh"
#include "lib/cuda/kernel_utils.cuh"

namespace loki::detection::detail {

using RNG = loki::device_rng::DefaultDeviceRNG;

struct ThresholdPairItem {
    uint32_t ithres_abs;
    uint32_t islot_cur;
    uint32_t jthres_abs;
    uint32_t jslot_prev;
};

// Philox stream for one logical stream: subsequence = base, offset 0. Each
// (global thread, purpose) gets a distinct base via trial_rng_bases or init.
__device__ __forceinline__ typename RNG::Generator make_rng(uint64_t seed,
                                                            uint64_t base) {
    return typename RNG::Generator(seed, base, 0);
}

// evaluate() uses eval_subseq bases; search uses trial_rng_bases / make_rng.
template <bool kFixedStreams>
__device__ __forceinline__ typename RNG::Generator open_rng(uint64_t seed,
                                                            uint64_t base) {
    if constexpr (kFixedStreams) {
        return typename RNG::Generator(seed, base, 0);
    }
    return make_rng(seed, base);
}

// Unique subsequence for evaluate(). purpose, stage, parent, branch and
// trial each occupy a disjoint bit field, so streams do not overlap.
__device__ __forceinline__ uint64_t eval_subseq(uint32_t purpose,
                                                uint32_t stage,
                                                uint32_t parent,
                                                uint32_t branch,
                                                uint32_t trial) {
    return (static_cast<uint64_t>(purpose & 7U) << 61U) |
           (static_cast<uint64_t>(parent & 0xFFFFFFU) << 37U) |
           (static_cast<uint64_t>(stage & 0xFFFFU) << 21U) |
           (static_cast<uint64_t>(branch & 1U) << 20U) |
           static_cast<uint64_t>(trial & 0xFFFFFU);
}

// Boxcar template with its normalisation precomputed on the host:
// S/N = ((h + b) * window_sum - b * total_sum) / sigma.
struct BoxWidth {
    uint32_t w;
    float hb; // h + b
    float b;
};

// 1 / sqrt(variance), correctly rounded (same bits in every kernel).
__device__ __forceinline__ float inv_stdnoise_of(float var_in, float var_add) {
    return __frsqrt_rn(__fadd_rn(var_in, var_add));
}

// Max boxcar S/N over all widths and circular start bins, given the inclusive
// prefix sum of one profile (prefix[i] = x[0] + ... + x[i], summed in order).
// Only exactly-rounded ops (adds/subs/max, explicit _rn products), so the
// result does not depend on how the caller is compiled or inlined.
__device__ __forceinline__ float
max_boxcar_snr_from_prefix(const float* __restrict__ prefix,
                           uint32_t nbins,
                           const BoxWidth* __restrict__ widths,
                           uint32_t nwidths,
                           float inv_stdnoise) {
    const float total_sum = prefix[nbins - 1];

    float max_snr = cuda::std::numeric_limits<float>::lowest();
    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const BoxWidth box = widths[iw];
        const uint32_t w   = box.w;
        float max_diff     = cuda::std::numeric_limits<float>::lowest();
        max_diff           = fmaxf(max_diff, prefix[w - 1]);

        const uint32_t loop_limit = nbins - w;
        for (uint32_t j = 1; j <= loop_limit; ++j) {
            const float diff = prefix[j + w - 1] - prefix[j - 1];
            max_diff         = fmaxf(max_diff, diff);
        }
        for (uint32_t j = loop_limit + 1; j < nbins; ++j) {
            const float diff =
                (total_sum - prefix[j - 1]) + prefix[j + w - 1 - nbins];
            max_diff = fmaxf(max_diff, diff);
        }
        const float snr = __fmul_rn(
            __fsub_rn(__fmul_rn(box.hb, max_diff), __fmul_rn(box.b, total_sum)),
            inv_stdnoise);
        max_snr = fmaxf(max_snr, snr);
    }
    return max_snr;
}

// Packs a candidate's (complexity_cumul, global work index) into one ordered
// key. complexity_cumul >= 0, so its IEEE bits order like the float. The
// smallest key per output cell is the winner: lowest complexity_cumul, ties
// broken by the earliest work item (the order a sequential walk would keep).
__device__ __forceinline__ unsigned long long
transition_key(float complexity_cumul, uint32_t global_work_idx) {
    return (static_cast<unsigned long long>(__float_as_uint(complexity_cumul))
            << 32U) |
           global_work_idx;
}

// Orders (score, trial) so that the largest key is the highest score, ties
// broken by the lowest trial index (as std::max_element). 0 is below every
// key, so it marks "no trial".
__device__ __forceinline__ unsigned long long score_trial_key(float score,
                                                              uint32_t trial) {
    const uint32_t bits = __float_as_uint(score);
    const uint32_t ord =
        (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
    return (static_cast<unsigned long long>(ord) << 32U) |
           (0xFFFFFFFFU - trial);
}

__device__ __forceinline__ uint32_t trial_of_key(unsigned long long key) {
    return 0xFFFFFFFFU - static_cast<uint32_t>(key & 0xFFFFFFFFULL);
}

struct ParentCandidate {
    uint32_t jslot_prev;
    uint32_t jthres_abs;
    uint32_t kprob;
};

// ---------------------------------------------------------------------------
// Stage pipelines (no host syncs, no per-stage allocations).
//
// The transitions of stage s are enumerated as (pair, kprob) items,
// item = pair * nprobs + kprob, in the order of the original batched walk,
// so "lowest item index" is the original tie-break order.
//
// Modes improved and shared (one simulation per parent cell, shared by its
// children; mode shared counts H1 on bitmasks of shared trials):
//   compact_parents_kernel -> simulate_score_parents_kernel ->
//   decide_parents_kernel -> commit_parents_kernel.
//   Parent folds are gathered through the (src cell, survivor index) map
//   written by the previous commit; simulated folds ping-pong between two
//   buffers, so survivors are never copied.
// ---------------------------------------------------------------------------

// Device-side RNG bookkeeping: [0] = cumulative offset, [1] = this stage's
// base offset.
inline constexpr uint32_t kRngCumulative = 0;
inline constexpr uint32_t kRngStageBase  = 1;
inline constexpr uint32_t kInvalidIndex  = 0xFFFFFFFFU;
inline constexpr unsigned long long kEmptyKey =
    cuda::std::numeric_limits<unsigned long long>::max();

// Parameters shared by every simulated trial of a stage.
struct TrialSimParams {
    const float* __restrict__ profile;
    const BoxWidth* __restrict__ widths;     // nwidths entries
    const BoxWidth* __restrict__ box_by_len; // nbins + 1 entries, w=0: unused
    uint32_t nwidths;
    uint32_t nbins;
    uint32_t nbins_padded;
    float bias_snr;
    float var_in;
    float var_add;
    uint64_t seed;
};

// Philox stream bases for one simulated trial: distinct subsequence per
// global thread index for threshold selection vs noise.
__device__ __forceinline__ void trial_rng_bases(uint64_t global_tid,
                                                uint32_t /*nbins_padded*/,
                                                uint64_t& select_base,
                                                uint64_t& noise_base) {
    select_base = 2 * global_tid;
    noise_base  = select_base + 1;
}

// Source trial for output trial `trial_id`: itself while survivors last,
// then a uniformly drawn survivor (bootstrap fill).
__device__ __forceinline__ uint32_t select_source_trial(uint32_t trial_id,
                                                        uint32_t ntrials_in,
                                                        uint64_t seed,
                                                        uint64_t select_base) {
    if (trial_id < ntrials_in) {
        return trial_id;
    }
    auto rng_select = make_rng(seed, select_base);
    typename RNG::UniformFloat dist_select(0.0F, 1.0F);
    const float u = dist_select.generate4(rng_select).x;
    return min(static_cast<uint32_t>(u * ntrials_in), ntrials_in - 1U);
}

// Source survivor for a failed slot (mode shared): the failed slots of a
// parent, taken in order, are refilled stratified over its @p nsurv
// survivors: slot number failed_rank of nfailed draws its quantile
// (failed_rank + u) / nfailed, u uniform. Stratification keeps the expected
// offspring of a survivor at ntrials / nsurv with less variance than
// independent draws.
__device__ __forceinline__ uint32_t
select_failed_source_trial(uint32_t failed_rank,
                           uint32_t nfailed,
                           uint32_t nsurv,
                           uint64_t seed,
                           uint64_t select_base) {
    auto rng_select = make_rng(seed, select_base);
    typename RNG::UniformFloat dist_select(0.0F, 1.0F);
    const float u = dist_select.generate4(rng_select).x;
    const float quantile =
        __fdiv_rn(__fadd_rn(static_cast<float>(failed_rank), u),
                  static_cast<float>(nfailed));
    return min(
        static_cast<uint32_t>(__fmul_rn(quantile, static_cast<float>(nsurv))),
        nsurv - 1U);
}

// One simulated trial: out = in + N(0, var_add) + branch_scale * profile,
// optionally stored, and its max boxcar S/N. Two implementations with
// identical arithmetic:
//  - RegScorer<NB, WMAX>: nbins == padded == NB, widths <= WMAX; fully
//    unrolled, the prefix sum lives in registers.
//  - GenericScorer<MAX_BINS>: any nbins <= MAX_BINS (local-memory prefix).
template <uint32_t NB, uint32_t WMAX> struct RegScorer {
    static_assert(NB % 4 == 0 && WMAX <= NB);
    static constexpr uint32_t kMaxVec  = NB / 4;
    static constexpr uint32_t kMaxBins = NB;
    static constexpr bool kHasWarpRun  = false;
    __device__ __forceinline__ static uint32_t
    vec_count(const TrialSimParams& /*p*/) {
        return NB / 4;
    }

    template <bool kStore, bool kFixedStreams = false>
    __device__ __forceinline__ static float
    run(const float* __restrict__ in_row,
        float* __restrict__ out_row,
        float branch_scale,
        uint64_t noise_base,
        const TrialSimParams& p) {
        auto rng_noise = open_rng<kFixedStreams>(p.seed, noise_base);
        typename RNG::NormalFloat dist_noise(0.0F, __fsqrt_rn(p.var_add));
        const auto* in4   = reinterpret_cast<const float4*>(in_row);
        auto* out4        = reinterpret_cast<float4*>(out_row);
        const auto* prof4 = reinterpret_cast<const float4*>(p.profile);

        float prefix[NB];
#pragma unroll
        for (uint32_t j = 0; j < NB / 4; ++j) {
            const float4 data  = in4[j];
            const float4 prof  = prof4[j];
            const float4 noise = dist_noise.generate4(rng_noise);
            const float4 out =
                make_float4(fmaf(prof.x, branch_scale, noise.x + data.x),
                            fmaf(prof.y, branch_scale, noise.y + data.y),
                            fmaf(prof.z, branch_scale, noise.z + data.z),
                            fmaf(prof.w, branch_scale, noise.w + data.w));
            if constexpr (kStore) {
                out4[j] = out;
            }
            prefix[(4 * j) + 0] =
                (j == 0) ? out.x : prefix[(4 * j) - 1] + out.x;
            prefix[(4 * j) + 1] = prefix[(4 * j) + 0] + out.y;
            prefix[(4 * j) + 2] = prefix[(4 * j) + 1] + out.z;
            prefix[(4 * j) + 3] = prefix[(4 * j) + 2] + out.w;
        }

        const float total_sum    = prefix[NB - 1];
        const float inv_stdnoise = inv_stdnoise_of(p.var_in, p.var_add);
        float max_snr            = cuda::std::numeric_limits<float>::lowest();
#pragma unroll
        for (uint32_t w = 1; w <= WMAX; ++w) {
            const BoxWidth box = p.box_by_len[w];
            if (box.w == 0) {
                continue; // not a box width (uniform across the grid)
            }
            float max_diff = cuda::std::numeric_limits<float>::lowest();
            max_diff       = fmaxf(max_diff, prefix[w - 1]);
#pragma unroll
            for (uint32_t j = 1; j <= NB - w; ++j) {
                max_diff = fmaxf(max_diff, prefix[j + w - 1] - prefix[j - 1]);
            }
#pragma unroll
            for (uint32_t j = NB - w + 1; j < NB; ++j) {
                max_diff = fmaxf(max_diff, (total_sum - prefix[j - 1]) +
                                               prefix[j + w - 1 - NB]);
            }
            const float snr = __fmul_rn(__fsub_rn(__fmul_rn(box.hb, max_diff),
                                                  __fmul_rn(box.b, total_sum)),
                                        inv_stdnoise);
            max_snr         = fmaxf(max_snr, snr);
        }
        return max_snr;
    }
};

template <uint32_t MAX_BINS> struct GenericScorer {
    static constexpr uint32_t kMaxVec  = MAX_BINS / 4;
    static constexpr uint32_t kMaxBins = MAX_BINS;
    static constexpr bool kHasWarpRun  = MAX_BINS >= 128;
    __device__ __forceinline__ static uint32_t
    vec_count(const TrialSimParams& p) {
        return p.nbins_padded / 4;
    }
    template <bool kStore, bool kFixedStreams = false>
    __device__ __forceinline__ static float
    run(const float* __restrict__ in_row,
        float* __restrict__ out_row,
        float branch_scale,
        uint64_t noise_base,
        const TrialSimParams& p) {
        auto rng_noise = open_rng<kFixedStreams>(p.seed, noise_base);
        typename RNG::NormalFloat dist_noise(0.0F, __fsqrt_rn(p.var_add));
        const auto* in4   = reinterpret_cast<const float4*>(in_row);
        auto* out4        = reinterpret_cast<float4*>(out_row);
        const auto* prof4 = reinterpret_cast<const float4*>(p.profile);

        // Prefix sum built in bin order while the fold is generated.
        float prefix[MAX_BINS];
        float acc                = 0.0F;
        const uint32_t vec_count = p.nbins_padded / 4;
#pragma unroll 4
        for (uint32_t j = 0; j < vec_count; ++j) {
            const float4 data  = in4[j];
            const float4 prof  = prof4[j];
            const float4 noise = dist_noise.generate4(rng_noise);
            const float4 out =
                make_float4(fmaf(prof.x, branch_scale, noise.x + data.x),
                            fmaf(prof.y, branch_scale, noise.y + data.y),
                            fmaf(prof.z, branch_scale, noise.z + data.z),
                            fmaf(prof.w, branch_scale, noise.w + data.w));
            if constexpr (kStore) {
                out4[j] = out;
            }
            acc                 = (j == 0) ? out.x : acc + out.x;
            prefix[(4 * j) + 0] = acc;
            acc                 = acc + out.y;
            prefix[(4 * j) + 1] = acc;
            acc                 = acc + out.z;
            prefix[(4 * j) + 2] = acc;
            acc                 = acc + out.w;
            prefix[(4 * j) + 3] = acc;
        }
        return max_boxcar_snr_from_prefix(prefix, p.nbins, p.widths, p.nwidths,
                                          inv_stdnoise_of(p.var_in, p.var_add));
    }

    /// Warp-collaborative counterpart of run(), bit-identical to it: all 32
    /// lanes generate the fold coalesced in shared memory @p s_prefix
    /// (MAX_BINS floats for this warp), lane 0 forms the sequential IEEE
    /// prefix sum in bin order, and all 32 lanes scan the boxcar windows and
    /// reduce max_diff via exact warp-shuffle fmaxf. Eliminates per-thread
    /// local-memory spilling at nbins >= 128.
    template <bool kStore>
    __device__ __forceinline__ static float
    run_warp(const float* __restrict__ in_row,
             float* __restrict__ out_row,
             float* __restrict__ s_prefix,
             float branch_scale,
             uint64_t noise_base,
             const TrialSimParams& p) {
        const uint32_t lane = threadIdx.x & 31U;
        typename RNG::NormalFloat dist_noise(0.0F, __fsqrt_rn(p.var_add));
        const auto* in4          = reinterpret_cast<const float4*>(in_row);
        auto* out4               = reinterpret_cast<float4*>(out_row);
        const auto* prof4        = reinterpret_cast<const float4*>(p.profile);
        auto* sp4                = reinterpret_cast<float4*>(s_prefix);
        const uint32_t vec_count = p.nbins_padded / 4;
        for (uint32_t j = lane; j < vec_count; j += 32U) {
            auto rng_noise     = RNG::make_vec4(p.seed, noise_base, j);
            const float4 data  = in4[j];
            const float4 prof  = prof4[j];
            const float4 noise = dist_noise.generate4(rng_noise);
            const float4 out =
                make_float4(fmaf(prof.x, branch_scale, noise.x + data.x),
                            fmaf(prof.y, branch_scale, noise.y + data.y),
                            fmaf(prof.z, branch_scale, noise.z + data.z),
                            fmaf(prof.w, branch_scale, noise.w + data.w));
            if constexpr (kStore) {
                out4[j] = out;
            }
            sp4[j] = out;
        }
        __syncwarp();
        if (lane == 0U) {
            float acc = 0.0F;
            for (uint32_t j = 0; j < vec_count; ++j) {
                const float4 out = sp4[j];
                float4 pref;
                acc    = (j == 0U) ? out.x : __fadd_rn(acc, out.x);
                pref.x = acc;
                acc    = __fadd_rn(acc, out.y);
                pref.y = acc;
                acc    = __fadd_rn(acc, out.z);
                pref.z = acc;
                acc    = __fadd_rn(acc, out.w);
                pref.w = acc;
                sp4[j] = pref;
            }
        }
        __syncwarp();

        const float total_sum    = s_prefix[p.nbins - 1];
        const float inv_stdnoise = inv_stdnoise_of(p.var_in, p.var_add);
        float max_snr            = cuda::std::numeric_limits<float>::lowest();
        for (uint32_t iw = 0; iw < p.nwidths; ++iw) {
            const BoxWidth box = p.widths[iw];
            const uint32_t w   = box.w;
            float max_diff     = cuda::std::numeric_limits<float>::lowest();
            if (lane == 0U) {
                max_diff = s_prefix[w - 1];
            }
            const uint32_t loop_limit = p.nbins - w;
            for (uint32_t j = 1U + lane; j <= loop_limit; j += 32U) {
                const float diff =
                    __fsub_rn(s_prefix[j + w - 1], s_prefix[j - 1]);
                max_diff = fmaxf(max_diff, diff);
            }
            for (uint32_t j = loop_limit + 1U + lane; j < p.nbins; j += 32U) {
                const float diff =
                    __fadd_rn(__fsub_rn(total_sum, s_prefix[j - 1]),
                              s_prefix[j + w - 1 - p.nbins]);
                max_diff = fmaxf(max_diff, diff);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_diff = fmaxf(
                    max_diff, __shfl_xor_sync(0xFFFFFFFFU, max_diff, offset));
            }
            const float snr = __fmul_rn(__fsub_rn(__fmul_rn(box.hb, max_diff),
                                                  __fmul_rn(box.b, total_sum)),
                                        inv_stdnoise);
            max_snr         = fmaxf(max_snr, snr);
        }
        return max_snr;
    }
};

// Calls f.template operator()<Scorer>() with the fastest scorer that covers
// (nbins, nbins_padded, max box width).
template <typename F>
void dispatch_scorer(SizeType nbins,
                     SizeType nbins_padded,
                     SizeType wmax,
                     bool allow_reg,
                     F&& f) {
    // Register path instantiations: NB in {32, 64, 128} and the smallest
    // WMAX in {16, 32, 64} (capped at NB) covering the widest box. Wider
    // boxes use the generic scorer, keeping the unrolled code bounded.
    auto by_wmax = [&]<uint32_t NB>() {
        if (wmax <= 16) {
            return f.template operator()<RegScorer<NB, 16>>();
        }
        if (wmax <= 32) {
            return f.template operator()<RegScorer<NB, 32>>();
        }
        if constexpr (NB >= 64) {
            return f.template operator()<RegScorer<NB, 64>>();
        }
    };
    if (allow_reg && nbins == nbins_padded &&
        wmax <= std::min<SizeType>(nbins, 64)) {
        switch (nbins) {
        case 32:
            return by_wmax.template operator()<32>();
        case 64:
            return by_wmax.template operator()<64>();
        case 128:
            return by_wmax.template operator()<128>();
        default:
            break;
        }
    }
    if (nbins_padded <= 32) {
        return f.template operator()<GenericScorer<32>>();
    }
    if (nbins_padded <= 64) {
        return f.template operator()<GenericScorer<64>>();
    }
    if (nbins_padded <= 128) {
        return f.template operator()<GenericScorer<128>>();
    }
    if (nbins_padded <= 256) {
        return f.template operator()<GenericScorer<256>>();
    }
    if (nbins_padded <= 512) {
        return f.template operator()<GenericScorer<512>>();
    }
    if (nbins_padded <= 1024) {
        return f.template operator()<GenericScorer<1024>>();
    }
    throw std::runtime_error("ThresholdsCuda: nbins exceeds the "
                             "compiled limit of 1024");
}

// Advances the cumulative RNG offset like the original batched launchers:
// every started batch of `round_to` items reserves round_to * 2 * ntrials
// thread indices.
__device__ __forceinline__ void advance_rng(unsigned long long* rng_state,
                                            uint32_t n,
                                            uint32_t round_to,
                                            uint32_t ntrials) {
    const unsigned long long cum      = rng_state[kRngCumulative];
    const unsigned long long nbatches = (n + round_to - 1) / round_to;
    rng_state[kRngStageBase]          = cum;
    rng_state[kRngCumulative] = cum + (nbatches * round_to * 2ULL * ntrials);
}

__device__ __forceinline__ bool parent_is_valid(const State* prev_states,
                                                const uint32_t* ntrials_par,
                                                uint32_t jthres,
                                                uint32_t par_cell,
                                                uint32_t nprobs,
                                                uint32_t kprob) {
    return !prev_states[(jthres * nprobs) + kprob].is_empty &&
           ntrials_par[2 * par_cell] != 0 &&
           ntrials_par[(2 * par_cell) + 1] != 0;
}

// Memory bound. The explicit minBlocks=1 measured ~6% faster than the bare
// __launch_bounds__(256) (different register allocation); capping registers
// lower (minBlocks=3) spills and is ~30% slower.
template <typename Scorer>
__global__ __launch_bounds__(256, 1) void simulate_score_parents_kernel(
    const ParentCandidate* __restrict__ parents,
    const uint32_t* __restrict__ n_parents,
    const unsigned long long* __restrict__ rng_state,
    const float* __restrict__ sim_prev,
    const uint32_t* __restrict__ ntrials_par,
    const uint32_t* __restrict__ src_par,
    const uint32_t* __restrict__ idx_par,
    float* __restrict__ sim_cur,
    float* __restrict__ scores_cur,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs,
    uint32_t nbranches) {
    // nbranches = 1 simulates H0 only (shared mode: H1 is not per cell)
    const uint64_t tid =
        (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const uint64_t trials_per_item = static_cast<uint64_t>(nbranches) * ntrials;
    const uint64_t iparent         = tid / trials_per_item;
    if (iparent >= *n_parents) {
        return;
    }
    const auto local_idx    = static_cast<uint32_t>(tid % trials_per_item);
    const uint32_t branch   = local_idx / ntrials; // 0=H0, 1=H1
    const uint32_t trial_id = local_idx % ntrials;

    uint64_t select_base = 0;
    uint64_t noise_base  = 0;
    // nbranches = 1 (shared mode): every parent draws the same noise for
    // trial t at this stage (common random numbers), so the costs that cells
    // compare share their noise as far as their populations allow.
    const uint64_t rng_idx = nbranches == 1 ? local_idx : tid;
    trial_rng_bases(rng_state[kRngStageBase] + rng_idx, p.nbins_padded,
                    select_base, noise_base);

    const ParentCandidate par = parents[iparent];
    const uint32_t cell       = (par.jslot_prev * nprobs) + par.kprob;
    uint32_t src_trial        = 0;
    if (nbranches == 1) {
        // Shared mode: a trial that survived keeps its slot; the failed slots
        // are refilled stratified over the survivors, so a candidate's noise
        // population differs from its parent's only where it must.
        const uint32_t count = ntrials_par[2 * cell];
        const uint32_t* survivors =
            idx_par + (static_cast<uint64_t>(cell) * ntrials);
        uint32_t lo = 0;
        uint32_t hi = count;
        while (lo < hi) {
            const uint32_t mid = lo + ((hi - lo) / 2);
            if (survivors[mid] < trial_id) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        src_trial =
            (lo < count && survivors[lo] == trial_id)
                ? lo
                : select_failed_source_trial(trial_id - lo, ntrials - count,
                                             count, p.seed, select_base);
    } else {
        src_trial = select_source_trial(
            trial_id, ntrials_par[(2 * cell) + branch], p.seed, select_base);
    }
    // Pool rows per cell: nbranches (H0 only in shared mode)
    const uint64_t cell_branch =
        (static_cast<uint64_t>(nbranches) * cell) + branch;
    const uint64_t src_row =
        (((static_cast<uint64_t>(nbranches) * src_par[cell]) + branch) *
         ntrials) +
        idx_par[(cell_branch * ntrials) + src_trial];
    const uint64_t out_row = (cell_branch * ntrials) + trial_id;

    scores_cur[out_row] = Scorer::template run<true>(
        sim_prev + (src_row * p.nbins_padded),
        sim_cur + (out_row * p.nbins_padded), (branch == 1) ? p.bias_snr : 0.0F,
        noise_base, p);
}

template <typename Scorer, uint32_t kBlock = 256>
__global__ __launch_bounds__(kBlock, 1) void simulate_score_parents_warp_kernel(
    const ParentCandidate* __restrict__ parents,
    const uint32_t* __restrict__ n_parents,
    const unsigned long long* __restrict__ rng_state,
    const float* __restrict__ sim_prev,
    const uint32_t* __restrict__ ntrials_par,
    const uint32_t* __restrict__ src_par,
    const uint32_t* __restrict__ idx_par,
    float* __restrict__ sim_cur,
    float* __restrict__ scores_cur,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs,
    uint32_t nbranches) {
    static constexpr uint32_t kWarps = kBlock / 32U;
    __shared__ float s_prefix[kWarps * Scorer::kMaxBins];
    const uint32_t warp_in_block = threadIdx.x / 32U;
    const uint32_t lane          = threadIdx.x & 31U;
    const uint64_t tid =
        (static_cast<uint64_t>(blockIdx.x) * kWarps) + warp_in_block;
    const uint64_t trials_per_item = static_cast<uint64_t>(nbranches) * ntrials;
    const uint64_t iparent         = tid / trials_per_item;
    if (iparent >= *n_parents) {
        return;
    }
    const auto local_idx  = static_cast<uint32_t>(tid % trials_per_item);
    const uint32_t branch = local_idx / ntrials;
    uint64_t src_row      = 0;
    uint64_t out_row      = 0;
    uint64_t noise_base   = 0;
    if (lane == 0U) {
        const uint32_t trial_id = local_idx % ntrials;
        uint64_t select_base    = 0;
        const uint64_t rng_idx  = nbranches == 1 ? local_idx : tid;
        trial_rng_bases(rng_state[kRngStageBase] + rng_idx, p.nbins_padded,
                        select_base, noise_base);
        const ParentCandidate par = parents[iparent];
        const uint32_t cell       = (par.jslot_prev * nprobs) + par.kprob;
        uint32_t src_trial        = 0;
        if (nbranches == 1) {
            const uint32_t count = ntrials_par[2 * cell];
            const uint32_t* survivors =
                idx_par + (static_cast<uint64_t>(cell) * ntrials);
            uint32_t lo = 0;
            uint32_t hi = count;
            while (lo < hi) {
                const uint32_t mid = lo + ((hi - lo) / 2);
                if (survivors[mid] < trial_id) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            src_trial =
                (lo < count && survivors[lo] == trial_id)
                    ? lo
                    : select_failed_source_trial(trial_id - lo, ntrials - count,
                                                 count, p.seed, select_base);
        } else {
            src_trial =
                select_source_trial(trial_id, ntrials_par[(2 * cell) + branch],
                                    p.seed, select_base);
        }
        const uint64_t cell_branch =
            (static_cast<uint64_t>(nbranches) * cell) + branch;
        src_row =
            (((static_cast<uint64_t>(nbranches) * src_par[cell]) + branch) *
             ntrials) +
            idx_par[(cell_branch * ntrials) + src_trial];
        out_row = (cell_branch * ntrials) + trial_id;
    }
    src_row           = __shfl_sync(0xFFFFFFFFU, src_row, 0);
    out_row           = __shfl_sync(0xFFFFFFFFU, out_row, 0);
    noise_base        = __shfl_sync(0xFFFFFFFFU, noise_base, 0);
    const float score = Scorer::template run_warp<true>(
        sim_prev + (src_row * p.nbins_padded),
        sim_cur + (out_row * p.nbins_padded),
        s_prefix + (warp_in_block * Scorer::kMaxBins),
        (branch == 1) ? p.bias_snr : 0.0F, noise_base, p);
    if (lane == 0U) {
        scores_cur[out_row] = score;
    }
}

// Reads and clears the winning key of output cell (islot, iprob).
__device__ __forceinline__ unsigned long long
take_cell_key(unsigned long long* cell_keys, uint32_t key_idx) {
    __shared__ unsigned long long key_s;
    if (threadIdx.x == 0) {
        key_s              = cell_keys[key_idx];
        cell_keys[key_idx] = kEmptyKey; // ready for the next stage
    }
    __syncthreads();
    return key_s;
}

__device__ __forceinline__ float eval_uniform(uint64_t seed, uint64_t subseq) {
    auto rng = typename RNG::Generator(seed, subseq, 0);
    typename RNG::UniformFloat dist(0.0F, 1.0F);
    const float u = dist.generate4(rng).x;
    return (u >= 1.0F) ? 0.0F : u;
}

// One path, one cell. The first survivors are kept, then a uniform bootstrap
// fills the rest. Noise streams are (stage, branch, trial) with offset 0,
// independent of run().
template <typename Scorer>
__global__ void evaluate_stage_kernel(const float* __restrict__ in_folds,
                                      const uint32_t* __restrict__ n_in,
                                      float* __restrict__ out_folds,
                                      float* __restrict__ scores,
                                      TrialSimParams p,
                                      uint32_t ntrials,
                                      uint32_t stage,
                                      uint32_t tid_offset,
                                      uint32_t trial_offset,
                                      uint32_t tid_limit,
                                      uint32_t row_offset) {
    // tid_offset = ntrials runs the H1 branch only; tid_limit = ntrials the
    // H0 branch only (2 * ntrials: both). trial_offset numbers the trials of
    // a chunk globally: their random streams are those of trials
    // trial_offset + [0, ntrials). row_offset shifts the fold/score rows when
    // only H1 is allocated.
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x + tid_offset;
    if (tid >= tid_limit) {
        return;
    }
    const uint32_t branch   = tid / ntrials;
    const uint32_t trial_id = tid % ntrials;
    const uint32_t nsurv    = n_in[branch];
    uint32_t src            = trial_id;
    if (trial_id >= nsurv) {
        const float u =
            eval_uniform(p.seed, eval_subseq(3U, stage, 0U, branch,
                                             trial_id + trial_offset));
        src = min(static_cast<uint32_t>(u * static_cast<float>(nsurv)),
                  nsurv - 1U);
    }
    const uint64_t noise_subseq =
        eval_subseq(2U, stage, 0U, branch, trial_id + trial_offset);
    const uint64_t src_row =
        (static_cast<uint64_t>(branch) * ntrials) + src - row_offset;
    const uint64_t out_row =
        (static_cast<uint64_t>(branch) * ntrials) + trial_id - row_offset;
    scores[out_row] = Scorer::template run<true, true>(
        in_folds + (src_row * p.nbins_padded),
        out_folds + (out_row * p.nbins_padded),
        (branch == 1) ? p.bias_snr : 0.0F, noise_subseq, p);
}

template <typename Scorer, uint32_t kBlock = 256>
__global__ __launch_bounds__(kBlock, 1) void evaluate_stage_warp_kernel(
    const float* __restrict__ in_folds,
    const uint32_t* __restrict__ n_in,
    float* __restrict__ out_folds,
    float* __restrict__ scores,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t stage,
    uint32_t tid_offset,
    uint32_t trial_offset,
    uint32_t tid_limit,
    uint32_t row_offset) {
    static constexpr uint32_t kWarps = kBlock / 32U;
    __shared__ float s_prefix[kWarps * Scorer::kMaxBins];
    const uint32_t warp_in_block = threadIdx.x / 32U;
    const uint32_t lane          = threadIdx.x & 31U;
    const uint64_t tid64 = (static_cast<uint64_t>(blockIdx.x) * kWarps) +
                           warp_in_block + tid_offset;
    if (tid64 >= tid_limit) {
        return;
    }
    const auto tid          = static_cast<uint32_t>(tid64);
    const uint32_t branch   = tid / ntrials;
    const uint32_t trial_id = tid % ntrials;
    uint32_t src            = trial_id;
    if (lane == 0U) {
        const uint32_t nsurv = n_in[branch];
        if (trial_id >= nsurv) {
            const float u =
                eval_uniform(p.seed, eval_subseq(3U, stage, 0U, branch,
                                                 trial_id + trial_offset));
            src = min(static_cast<uint32_t>(u * static_cast<float>(nsurv)),
                      nsurv - 1U);
        }
    }
    src = __shfl_sync(0xFFFFFFFFU, src, 0);
    const uint64_t noise_subseq =
        eval_subseq(2U, stage, 0U, branch, trial_id + trial_offset);
    const uint64_t src_row =
        (static_cast<uint64_t>(branch) * ntrials) + src - row_offset;
    const uint64_t out_row =
        (static_cast<uint64_t>(branch) * ntrials) + trial_id - row_offset;
    const float score = Scorer::template run_warp<true>(
        in_folds + (src_row * p.nbins_padded),
        out_folds + (out_row * p.nbins_padded),
        s_prefix + (warp_in_block * Scorer::kMaxBins),
        (branch == 1) ? p.bias_snr : 0.0F, noise_subseq, p);
    if (lane == 0U) {
        scores[out_row] = score;
    }
}

template <typename Scorer>
__global__ void
h1_pruned_stage_kernel(const float* __restrict__ in_folds,
                       float* __restrict__ out_folds,
                       const uint32_t* __restrict__ active_slot,
                       const uint32_t* __restrict__ active_trial,
                       float* __restrict__ margins,
                       uint8_t* __restrict__ keep,
                       TrialSimParams p,
                       uint32_t n_active,
                       uint32_t stage,
                       uint32_t trial_offset,
                       float threshold,
                       float min_margin) {
    const uint32_t k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= n_active) {
        return;
    }
    const uint32_t trial_id = active_trial[k];
    const float prev_m      = margins[trial_offset + trial_id];
    if (prev_m <= min_margin) {
        keep[k] = 0U;
        return;
    }
    const uint32_t slot_in = active_slot != nullptr ? active_slot[k] : k;
    const uint64_t noise_subseq =
        eval_subseq(2U, stage, 0U, 1U, trial_id + trial_offset);
    const float score = Scorer::template run<true, true>(
        in_folds + (static_cast<uint64_t>(slot_in) * p.nbins_padded),
        out_folds + (static_cast<uint64_t>(k) * p.nbins_padded), p.bias_snr,
        noise_subseq, p);
    const float m                    = fminf(prev_m, score - threshold);
    margins[trial_offset + trial_id] = m;
    keep[k] = (m > min_margin) ? uint8_t{1} : uint8_t{0};
}

template <typename Scorer, uint32_t kBlock = 256>
__global__ __launch_bounds__(kBlock, 1) void h1_pruned_stage_warp_kernel(
    const float* __restrict__ in_folds,
    float* __restrict__ out_folds,
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ active_trial,
    float* __restrict__ margins,
    uint8_t* __restrict__ keep,
    TrialSimParams p,
    uint32_t n_active,
    uint32_t stage,
    uint32_t trial_offset,
    float threshold,
    float min_margin) {
    static constexpr uint32_t kWarps = kBlock / 32U;
    __shared__ float s_prefix[kWarps * Scorer::kMaxBins];
    const uint32_t warp_in_block = threadIdx.x / 32U;
    const uint32_t lane          = threadIdx.x & 31U;
    const uint64_t k64 =
        (static_cast<uint64_t>(blockIdx.x) * kWarps) + warp_in_block;
    if (k64 >= n_active) {
        return;
    }
    const auto k            = static_cast<uint32_t>(k64);
    const uint32_t trial_id = active_trial[k];
    float prev_m            = 0.0F;
    if (lane == 0U) {
        prev_m = margins[trial_offset + trial_id];
    }
    prev_m = __shfl_sync(0xFFFFFFFFU, prev_m, 0);
    if (prev_m <= min_margin) {
        if (lane == 0U) {
            keep[k] = 0U;
        }
        return;
    }
    const uint32_t slot_in = active_slot != nullptr ? active_slot[k] : k;
    const uint64_t noise_subseq =
        eval_subseq(2U, stage, 0U, 1U, trial_id + trial_offset);
    const float score = Scorer::template run_warp<true>(
        in_folds + (static_cast<uint64_t>(slot_in) * p.nbins_padded),
        out_folds + (static_cast<uint64_t>(k) * p.nbins_padded),
        s_prefix + (warp_in_block * Scorer::kMaxBins), p.bias_snr, noise_subseq,
        p);
    if (lane == 0U) {
        const float m                    = fminf(prev_m, score - threshold);
        margins[trial_offset + trial_id] = m;
        keep[k] = (m > min_margin) ? uint8_t{1} : uint8_t{0};
    }
}

// ---------------------------------------------------------------------------
// Host launchers for the scorer-templated kernels. Each launches exactly the
// kernel configuration the caller passes; the members are defined out of
// class so the extern declarations below keep every other TU from
// instantiating them (and the kernels).
// ---------------------------------------------------------------------------

template <typename Scorer> struct ScorerLaunch {
    static void simulate_score_parents(dim3 grid,
                                       dim3 block,
                                       cudaStream_t stream,
                                       const ParentCandidate* parents,
                                       const uint32_t* n_parents,
                                       const unsigned long long* rng_state,
                                       const float* sim_prev,
                                       const uint32_t* ntrials_par,
                                       const uint32_t* src_par,
                                       const uint32_t* idx_par,
                                       float* sim_cur,
                                       float* scores_cur,
                                       TrialSimParams p,
                                       uint32_t ntrials,
                                       uint32_t nprobs,
                                       uint32_t nbranches);

    static void evaluate_stage(dim3 grid,
                               dim3 block,
                               cudaStream_t stream,
                               const float* in_folds,
                               const uint32_t* n_in,
                               float* out_folds,
                               float* scores,
                               TrialSimParams p,
                               uint32_t ntrials,
                               uint32_t stage,
                               uint32_t tid_offset,
                               uint32_t trial_offset,
                               uint32_t tid_limit,
                               uint32_t row_offset = 0U);

    static void h1_pruned_stage(cudaStream_t stream,
                                const float* in_folds,
                                float* out_folds,
                                const uint32_t* active_slot,
                                const uint32_t* active_trial,
                                float* margins,
                                uint8_t* keep,
                                TrialSimParams p,
                                uint32_t n_active,
                                uint32_t stage,
                                uint32_t trial_offset,
                                float threshold,
                                float min_margin);
};

template <typename Scorer>
void ScorerLaunch<Scorer>::simulate_score_parents(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    const ParentCandidate* parents,
    const uint32_t* n_parents,
    const unsigned long long* rng_state,
    const float* sim_prev,
    const uint32_t* ntrials_par,
    const uint32_t* src_par,
    const uint32_t* idx_par,
    float* sim_cur,
    float* scores_cur,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs,
    uint32_t nbranches) {
    if constexpr (Scorer::kHasWarpRun) {
        constexpr uint32_t kBlock = 256;
        constexpr uint32_t kWarps = kBlock / 32U;
        const uint64_t total_trials =
            static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(block.x);
        const dim3 warp_grid(
            static_cast<uint32_t>((total_trials + kWarps - 1ULL) / kWarps));
        simulate_score_parents_warp_kernel<Scorer, kBlock>
            <<<warp_grid, kBlock, 0, stream>>>(
                parents, n_parents, rng_state, sim_prev, ntrials_par, src_par,
                idx_par, sim_cur, scores_cur, p, ntrials, nprobs, nbranches);
    } else {
        simulate_score_parents_kernel<Scorer><<<grid, block, 0, stream>>>(
            parents, n_parents, rng_state, sim_prev, ntrials_par, src_par,
            idx_par, sim_cur, scores_cur, p, ntrials, nprobs, nbranches);
    }
}

template <typename Scorer>
void ScorerLaunch<Scorer>::evaluate_stage(dim3 grid,
                                          dim3 block,
                                          cudaStream_t stream,
                                          const float* in_folds,
                                          const uint32_t* n_in,
                                          float* out_folds,
                                          float* scores,
                                          TrialSimParams p,
                                          uint32_t ntrials,
                                          uint32_t stage,
                                          uint32_t tid_offset,
                                          uint32_t trial_offset,
                                          uint32_t tid_limit,
                                          uint32_t row_offset) {
    if constexpr (Scorer::kHasWarpRun) {
        (void)grid;
        (void)block;
        constexpr uint32_t kBlock = 256;
        constexpr uint32_t kWarps = kBlock / 32U;
        const uint32_t n_active =
            tid_limit > tid_offset ? (tid_limit - tid_offset) : 0U;
        if (n_active == 0U) {
            return;
        }
        const dim3 warp_grid((n_active + kWarps - 1U) / kWarps);
        evaluate_stage_warp_kernel<Scorer, kBlock>
            <<<warp_grid, kBlock, 0, stream>>>(
                in_folds, n_in, out_folds, scores, p, ntrials, stage,
                tid_offset, trial_offset, tid_limit, row_offset);
    } else {
        evaluate_stage_kernel<Scorer><<<grid, block, 0, stream>>>(
            in_folds, n_in, out_folds, scores, p, ntrials, stage, tid_offset,
            trial_offset, tid_limit, row_offset);
    }
}

template <typename Scorer>
void ScorerLaunch<Scorer>::h1_pruned_stage(cudaStream_t stream,
                                           const float* in_folds,
                                           float* out_folds,
                                           const uint32_t* active_slot,
                                           const uint32_t* active_trial,
                                           float* margins,
                                           uint8_t* keep,
                                           TrialSimParams p,
                                           uint32_t n_active,
                                           uint32_t stage,
                                           uint32_t trial_offset,
                                           float threshold,
                                           float min_margin) {
    if (n_active == 0U) {
        return;
    }
    constexpr uint32_t kBlock = 256;
    if constexpr (Scorer::kHasWarpRun) {
        constexpr uint32_t kWarps = kBlock / 32U;
        const dim3 grid((n_active + kWarps - 1U) / kWarps);
        h1_pruned_stage_warp_kernel<Scorer, kBlock>
            <<<grid, kBlock, 0, stream>>>(
                in_folds, out_folds, active_slot, active_trial, margins, keep,
                p, n_active, stage, trial_offset, threshold, min_margin);
    } else {
        const dim3 grid((n_active + kBlock - 1U) / kBlock);
        h1_pruned_stage_kernel<Scorer><<<grid, kBlock, 0, stream>>>(
            in_folds, out_folds, active_slot, active_trial, margins, keep, p,
            n_active, stage, trial_offset, threshold, min_margin);
    }
}

// Every scorer type dispatch_scorer can select. Instantiated in
// thresholds_reg_cuda.cu and thresholds_generic_cuda.cu.
extern template struct ScorerLaunch<RegScorer<32, 16>>;
extern template struct ScorerLaunch<RegScorer<32, 32>>;
extern template struct ScorerLaunch<RegScorer<64, 16>>;
extern template struct ScorerLaunch<RegScorer<64, 32>>;
extern template struct ScorerLaunch<RegScorer<64, 64>>;
extern template struct ScorerLaunch<RegScorer<128, 16>>;
extern template struct ScorerLaunch<RegScorer<128, 32>>;
extern template struct ScorerLaunch<RegScorer<128, 64>>;
extern template struct ScorerLaunch<GenericScorer<32>>;
extern template struct ScorerLaunch<GenericScorer<64>>;
extern template struct ScorerLaunch<GenericScorer<128>>;
extern template struct ScorerLaunch<GenericScorer<256>>;
extern template struct ScorerLaunch<GenericScorer<512>>;
extern template struct ScorerLaunch<GenericScorer<1024>>;

} // namespace loki::detection::detail
