#pragma once

/**
 * @file thresholds_kernels_cuda.cuh
 * @brief Scorers, device helpers and scorer-templated kernels of the CUDA
 * DynamicThresholdScheme. Internal.
 *
 * Each scorer type instantiates four heavy, fully unrolled kernels. To keep
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
        float max_diff = cuda::std::numeric_limits<float>::lowest();
        max_diff       = fmaxf(max_diff, prefix[w - 1]);

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

struct ParentCandidate {
    uint32_t jslot_prev;
    uint32_t jthres_abs;
    uint32_t kprob;
};

// ---------------------------------------------------------------------------
// Stage pipelines (no host syncs, no per-stage allocations).
//
// Both modes enumerate the transitions of stage s as (pair, kprob) items,
// item = pair * nprobs + kprob, in the order of the original batched walk,
// so "lowest item index" is the original tie-break order.
//
// Improved mode (one simulation per parent cell, shared by its children):
//   compact_parents_kernel -> simulate_score_parents_kernel ->
//   decide_parents_kernel -> commit_parents_kernel.
//   Parent folds are gathered through the (src cell, survivor index) map
//   written by the previous commit; simulated folds ping-pong between two
//   buffers, so survivors are never copied.
//
// Legacy mode (one simulation per transition):
//   compact_items_kernel -> simulate_count_items_kernel ->
//   decide_items_kernel -> commit_items_kernel.
//   Transitions are only scored and counted; the winner of each cell
//   regenerates its trials (same RNG streams) and writes its survivors.
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

// One simulated trial: out = in + N(0, var_add) + branch_scale * profile,
// optionally stored, and its max boxcar S/N. Two implementations with
// identical arithmetic:
//  - RegScorer<NB, WMAX>: nbins == padded == NB, widths <= WMAX; fully
//    unrolled, the prefix sum lives in registers.
//  - GenericScorer<MAX_BINS>: any nbins <= MAX_BINS (local-memory prefix).
template <uint32_t NB, uint32_t WMAX> struct RegScorer {
    static_assert(NB % 4 == 0 && WMAX <= NB);
    static constexpr uint32_t kMaxVec = NB / 4;
    __device__ __forceinline__ static uint32_t
    vec_count(const TrialSimParams& /*p*/) {
        return NB / 4;
    }

    template <bool kStore, bool kFixedStreams = false>
    __device__ __forceinline__ static float run(const float* __restrict__ in_row,
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
            max_snr = fmaxf(max_snr, snr);
        }
        return max_snr;
    }
};

template <uint32_t MAX_BINS> struct GenericScorer {
    static constexpr uint32_t kMaxVec = MAX_BINS / 4;
    __device__ __forceinline__ static uint32_t
    vec_count(const TrialSimParams& p) {
        return p.nbins_padded / 4;
    }
    template <bool kStore, bool kFixedStreams = false>
    __device__ __forceinline__ static float run(const float* __restrict__ in_row,
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
           ntrials_par[2 * par_cell] != 0 && ntrials_par[(2 * par_cell) + 1] != 0;
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
    uint32_t nprobs) {
    const uint64_t tid = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                         threadIdx.x;
    const uint64_t trials_per_item = 2ULL * ntrials;
    const uint64_t iparent         = tid / trials_per_item;
    if (iparent >= *n_parents) {
        return;
    }
    const auto local_idx    = static_cast<uint32_t>(tid % trials_per_item);
    const uint32_t branch   = local_idx / ntrials; // 0=H0, 1=H1
    const uint32_t trial_id = local_idx % ntrials;

    uint64_t select_base = 0;
    uint64_t noise_base  = 0;
    trial_rng_bases(rng_state[kRngStageBase] + tid, p.nbins_padded,
                    select_base, noise_base);

    const ParentCandidate par = parents[iparent];
    const uint32_t cell       = (par.jslot_prev * nprobs) + par.kprob;
    const uint32_t src_trial  = select_source_trial(
        trial_id, ntrials_par[(2 * cell) + branch], p.seed, select_base);
    const uint64_t cell_branch = (2ULL * cell) + branch;
    const uint64_t src_row     = (((2ULL * src_par[cell]) + branch) * ntrials) +
                             idx_par[(cell_branch * ntrials) + src_trial];
    const uint64_t out_row = (cell_branch * ntrials) + trial_id;

    scores_cur[out_row] = Scorer::template run<true>(
        sim_prev + (src_row * p.nbins_padded),
        sim_cur + (out_row * p.nbins_padded),
        (branch == 1) ? p.bias_snr : 0.0F, noise_base, p);
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
                                      uint32_t stage) {
    const uint32_t tid   = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total = 2U * ntrials;
    if (tid >= total) {
        return;
    }
    const uint32_t branch   = tid / ntrials;
    const uint32_t trial_id = tid % ntrials;
    const uint32_t nsurv    = n_in[branch];
    uint32_t src            = trial_id;
    if (trial_id >= nsurv) {
        const float u = eval_uniform(
            p.seed, eval_subseq(3U, stage, 0U, branch, trial_id));
        src = min(static_cast<uint32_t>(u * static_cast<float>(nsurv)),
                  nsurv - 1U);
    }
    const uint64_t noise_subseq = eval_subseq(2U, stage, 0U, branch, trial_id);
    const uint64_t src_row = (static_cast<uint64_t>(branch) * ntrials) + src;
    const uint64_t out_row = (static_cast<uint64_t>(branch) * ntrials) + trial_id;
    scores[out_row]        = Scorer::template run<true, true>(
        in_folds + (src_row * p.nbins_padded),
        out_folds + (out_row * p.nbins_padded),
        (branch == 1) ? p.bias_snr : 0.0F, noise_subseq, p);
}

// Simulates, scores and counts the survivors of every valid item without
// storing folds. Threads walk items in `order` (grouped by parent cell) so
// the items sharing a parent read its folds from L2.
// Compute bound: capping registers at 128 (2 blocks of 256 per SM, the same
// on every 64K-register SM) beats the uncapped ~156 registers by ~25%.
template <typename Scorer>
__global__ __launch_bounds__(256, 2) void simulate_count_items_kernel(
    const uint32_t* __restrict__ order,
    uint32_t n_items,
    const ThresholdPairItem* __restrict__ pairs,
    const uint32_t* __restrict__ item_rng_idx,
    const unsigned long long* __restrict__ rng_state,
    const float* __restrict__ folds_par,
    const uint32_t* __restrict__ ntrials_par,
    const float* __restrict__ thresholds,
    uint32_t* __restrict__ counts,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs) {
    const uint64_t tid = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                         threadIdx.x;
    const uint64_t trials_per_item = 2ULL * ntrials;
    const uint64_t slot            = tid / trials_per_item;
    uint32_t segment               = kInvalidIndex;
    bool survived                  = false;
    if (slot < n_items) {
        const uint32_t item    = order[slot];
        const uint32_t rng_idx = item_rng_idx[item];
        if (rng_idx != kInvalidIndex) {
            const auto local_idx    = static_cast<uint32_t>(tid % trials_per_item);
            const uint32_t branch   = local_idx / ntrials;
            const uint32_t trial_id = local_idx % ntrials;
            const ThresholdPairItem pair = pairs[item / nprobs];
            const uint32_t par_cell = (pair.jslot_prev * nprobs) + (item % nprobs);
            uint64_t select_base    = 0;
            uint64_t noise_base     = 0;
            trial_rng_bases(rng_state[kRngStageBase] +
                                (rng_idx * trials_per_item) + local_idx,
                            p.nbins_padded, select_base, noise_base);
            const uint32_t src_trial = select_source_trial(
                trial_id, ntrials_par[(2 * par_cell) + branch], p.seed,
                select_base);
            const uint64_t src_row =
                (((2ULL * par_cell) + branch) * ntrials) + src_trial;
            const float score = Scorer::template run<false>(
                folds_par + (src_row * p.nbins_padded), nullptr,
                (branch == 1) ? p.bias_snr : 0.0F, noise_base, p);
            survived = score > thresholds[pair.ithres_abs];
            segment  = (2 * item) + branch;
        }
    }
    // Warp-aggregated count per (item, branch) segment.
    const unsigned peers = __match_any_sync(0xFFFFFFFFU, segment);
    const unsigned votes = __ballot_sync(0xFFFFFFFFU, survived) & peers;
    const auto lane      = threadIdx.x % 32;
    if (segment != kInvalidIndex && lane == static_cast<uint32_t>(__ffs(peers) - 1) &&
        votes != 0) {
        atomicAdd(&counts[segment], static_cast<uint32_t>(__popc(votes)));
    }
}

// The winner of each output cell regenerates its trials (same RNG streams
// as simulate_count_items_kernel), compacts the survivors in trial order and
// writes their folds (kept in registers across the scan, so every trial is
// generated once). A survivor count that disagrees with the counting pass
// raises *error_flag.
template <typename Scorer, uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void commit_items_kernel(
    const uint32_t* __restrict__ beam_cur,
    const ThresholdPairItem* __restrict__ pairs,
    const uint32_t* __restrict__ item_rng_idx,
    const unsigned long long* __restrict__ rng_state,
    const uint2* __restrict__ counts,
    const State* __restrict__ cand_states,
    State* __restrict__ states_cur,
    unsigned long long* __restrict__ cell_keys,
    const float* __restrict__ folds_par,
    const uint32_t* __restrict__ ntrials_par,
    const float* __restrict__ thresholds,
    float* __restrict__ folds_next,
    uint32_t* __restrict__ ntrials_next,
    uint32_t* __restrict__ error_flag,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs) {
    const uint32_t out_cell = blockIdx.x; // islot * nprobs + iprob
    const uint32_t ithres   = beam_cur[out_cell / nprobs];
    const uint32_t key_idx  = (ithres * nprobs) + (out_cell % nprobs);
    const unsigned long long key = take_cell_key(cell_keys, key_idx);
    if (key == kEmptyKey) {
        if (threadIdx.x == 0) {
            ntrials_next[2 * out_cell]       = 0;
            ntrials_next[(2 * out_cell) + 1] = 0;
        }
        return;
    }
    const auto item              = static_cast<uint32_t>(key & 0xFFFFFFFFULL);
    const ThresholdPairItem pair = pairs[item / nprobs];
    const uint32_t par_cell = (pair.jslot_prev * nprobs) + (item % nprobs);
    const float threshold   = thresholds[ithres];
    const uint2 cnt         = counts[item];
    if (threadIdx.x == 0) {
        states_cur[key_idx]              = cand_states[item];
        ntrials_next[2 * out_cell]       = cnt.x;
        ntrials_next[(2 * out_cell) + 1] = cnt.y;
    }
    const uint64_t trials_per_item = 2ULL * ntrials;
    const uint64_t rng_item_base =
        rng_state[kRngStageBase] + (item_rng_idx[item] * trials_per_item);
    const uint32_t vec_count = Scorer::vec_count(p);

    using BlockScan = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    for (uint32_t branch = 0; branch < 2; ++branch) {
        const float branch_scale  = (branch == 1) ? p.bias_snr : 0.0F;
        const uint32_t ntrials_in = ntrials_par[(2 * par_cell) + branch];
        uint32_t running          = 0;
        for (uint32_t base = 0; base < ntrials; base += kBlock) {
            const uint32_t t = base + threadIdx.x;
            uint32_t flag    = 0;
            float4 fold[Scorer::kMaxVec];
            if (t < ntrials) {
                uint64_t select_base = 0;
                uint64_t noise_base  = 0;
                trial_rng_bases(rng_item_base + (branch * ntrials) + t,
                                p.nbins_padded, select_base, noise_base);
                const uint32_t src_trial =
                    select_source_trial(t, ntrials_in, p.seed, select_base);
                const float* in_row =
                    folds_par +
                    ((((2ULL * par_cell) + branch) * ntrials) + src_trial) *
                        p.nbins_padded;
                flag = Scorer::template run<true>(
                           in_row, reinterpret_cast<float*>(fold),
                           branch_scale, noise_base, p) > threshold
                           ? 1U
                           : 0U;
            }
            uint32_t pos   = 0;
            uint32_t total = 0;
            BlockScan(temp_storage).ExclusiveSum(flag, pos, total);
            if (flag != 0) {
                auto* dst4 = reinterpret_cast<float4*>(
                    folds_next + ((((2ULL * out_cell) + branch) * ntrials) +
                                  running + pos) *
                                     p.nbins_padded);
#pragma unroll
                for (uint32_t j = 0; j < Scorer::kMaxVec; ++j) {
                    if (j < vec_count) {
                        dst4[j] = fold[j];
                    }
                }
            }
            running += total;
            __syncthreads(); // temp_storage reuse
        }
        if (threadIdx.x == 0 && running != (branch == 0 ? cnt.x : cnt.y)) {
            atomicOr(error_flag, 1U);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers for the scorer-templated kernels. Each launches exactly the
// kernel configuration the caller passes; the members are defined out of
// class so the extern declarations below keep every other TU from
// instantiating them (and the kernels).
// ---------------------------------------------------------------------------

template <typename Scorer> struct ScorerLaunch {
    static constexpr uint32_t kCommitItemsBlock = 256;

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
                                       uint32_t nprobs);

    static void evaluate_stage(dim3 grid,
                               dim3 block,
                               cudaStream_t stream,
                               const float* in_folds,
                               const uint32_t* n_in,
                               float* out_folds,
                               float* scores,
                               TrialSimParams p,
                               uint32_t ntrials,
                               uint32_t stage);

    static void simulate_count_items(dim3 grid,
                                     dim3 block,
                                     cudaStream_t stream,
                                     const uint32_t* order,
                                     uint32_t n_items,
                                     const ThresholdPairItem* pairs,
                                     const uint32_t* item_rng_idx,
                                     const unsigned long long* rng_state,
                                     const float* folds_par,
                                     const uint32_t* ntrials_par,
                                     const float* thresholds,
                                     uint32_t* counts,
                                     TrialSimParams p,
                                     uint32_t ntrials,
                                     uint32_t nprobs);

    static void commit_items(dim3 grid,
                             dim3 block,
                             cudaStream_t stream,
                             const uint32_t* beam_cur,
                             const ThresholdPairItem* pairs,
                             const uint32_t* item_rng_idx,
                             const unsigned long long* rng_state,
                             const uint2* counts,
                             const State* cand_states,
                             State* states_cur,
                             unsigned long long* cell_keys,
                             const float* folds_par,
                             const uint32_t* ntrials_par,
                             const float* thresholds,
                             float* folds_next,
                             uint32_t* ntrials_next,
                             uint32_t* error_flag,
                             TrialSimParams p,
                             uint32_t ntrials,
                             uint32_t nprobs);
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
    uint32_t nprobs) {
    simulate_score_parents_kernel<Scorer><<<grid, block, 0, stream>>>(
        parents, n_parents, rng_state, sim_prev, ntrials_par, src_par, idx_par,
        sim_cur, scores_cur, p, ntrials, nprobs);
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
                                          uint32_t stage) {
    evaluate_stage_kernel<Scorer><<<grid, block, 0, stream>>>(
        in_folds, n_in, out_folds, scores, p, ntrials, stage);
}

template <typename Scorer>
void ScorerLaunch<Scorer>::simulate_count_items(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    const uint32_t* order,
    uint32_t n_items,
    const ThresholdPairItem* pairs,
    const uint32_t* item_rng_idx,
    const unsigned long long* rng_state,
    const float* folds_par,
    const uint32_t* ntrials_par,
    const float* thresholds,
    uint32_t* counts,
    TrialSimParams p,
    uint32_t ntrials,
    uint32_t nprobs) {
    simulate_count_items_kernel<Scorer><<<grid, block, 0, stream>>>(
        order, n_items, pairs, item_rng_idx, rng_state, folds_par, ntrials_par,
        thresholds, counts, p, ntrials, nprobs);
}

template <typename Scorer>
void ScorerLaunch<Scorer>::commit_items(dim3 grid,
                                        dim3 block,
                                        cudaStream_t stream,
                                        const uint32_t* beam_cur,
                                        const ThresholdPairItem* pairs,
                                        const uint32_t* item_rng_idx,
                                        const unsigned long long* rng_state,
                                        const uint2* counts,
                                        const State* cand_states,
                                        State* states_cur,
                                        unsigned long long* cell_keys,
                                        const float* folds_par,
                                        const uint32_t* ntrials_par,
                                        const float* thresholds,
                                        float* folds_next,
                                        uint32_t* ntrials_next,
                                        uint32_t* error_flag,
                                        TrialSimParams p,
                                        uint32_t ntrials,
                                        uint32_t nprobs) {
    commit_items_kernel<Scorer, kCommitItemsBlock>
        <<<grid, block, 0, stream>>>(beam_cur, pairs, item_rng_idx, rng_state,
                                     counts, cand_states, states_cur,
                                     cell_keys, folds_par, ntrials_par,
                                     thresholds, folds_next, ntrials_next,
                                     error_flag, p, ntrials, nprobs);
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
