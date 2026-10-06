#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "cuda/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "cuda/device_rng.cuh"
#include "detection/thresholds_engine.hpp"
#include "detection/scheme.hpp"
#include "loki/simulation/simulation.hpp"
#include "detail/timing.hpp"
#include "detail/utils.hpp"

namespace loki::detection {

using RNG = loki::math::DefaultDeviceRNG;

namespace {

struct ThresholdPairItem {
    uint32_t ithres_abs;
    uint32_t islot_cur;
    uint32_t jthres_abs;
    uint32_t jslot_prev;
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

__global__ void simulate_folds_init_kernel(float* __restrict__ folds_sim,
                                           const float* __restrict__ profile,
                                           uint32_t nbins_padded,
                                           float bias_snr,
                                           float var_add,
                                           uint64_t seed,
                                           uint64_t rng_offset,
                                           uint32_t ntrials) {
    const uint32_t tid          = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_trials = 2 * ntrials;
    if (tid >= total_trials) {
        return;
    }

    const uint32_t branch   = tid / ntrials; // 0=H0, 1=H1
    const uint32_t trial_id = tid % ntrials;
    const uint32_t out_offset =
        branch * ntrials * nbins_padded + trial_id * nbins_padded;
    const float branch_scale = (branch == 1) ? bias_snr : 0.0F;

    const uint64_t global_tid = rng_offset + tid;
    const uint64_t noise_base = (2 * global_tid) + 1;
    const float noise_stddev = __fsqrt_rn(var_add);
    auto rng_noise           = make_rng(seed, noise_base);
    typename RNG::NormalFloat dist_noise(0.0F, noise_stddev);

    // all pointers are 16-byte aligned for safe float4 access
    const uint32_t vec_count = nbins_padded / 4;
    float4* __restrict__ out_ptr4 =
        reinterpret_cast<float4*>(folds_sim + out_offset);
    const float4* __restrict__ prof_ptr4 =
        reinterpret_cast<const float4*>(profile);

#pragma unroll 4
    for (uint32_t j = 0; j < vec_count; ++j) {
        const float4 prof  = prof_ptr4[j];
        const float4 noise = dist_noise.generate4(rng_noise);
        out_ptr4[j]        = make_float4(fmaf(prof.x, branch_scale, noise.x),
                                         fmaf(prof.y, branch_scale, noise.y),
                                         fmaf(prof.z, branch_scale, noise.z),
                                         fmaf(prof.w, branch_scale, noise.w));
    }
}

// Boxcar template with its normalisation precomputed on the host:
// S/N = ((h + b) * window_sum - b * total_sum) / sigma.
struct BoxWidth {
    uint32_t w;
    float hb; // h + b
    float b;
};

std::vector<BoxWidth> make_box_widths(std::span<const SizeType> widths,
                                      SizeType nbins) {
    std::vector<BoxWidth> out;
    out.reserve(widths.size());
    const auto n = static_cast<float>(nbins);
    for (const auto width : widths) {
        const auto w = static_cast<float>(width);
        const float h = std::sqrt((n - w) / (n * w));
        const float b = w * h / (n - w);
        out.push_back(BoxWidth{static_cast<uint32_t>(width), h + b, b});
    }
    return out;
}

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

struct ModuloFunctor {
    uint32_t mod;
    __device__ uint32_t operator()(uint32_t i) const { return i % mod; }
};

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

// ----------------------------- improved mode -------------------------------

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void compact_parents_kernel(
    const uint32_t* __restrict__ beam_prev,
    uint32_t n_beam_prev,
    uint32_t nprobs,
    const State* __restrict__ prev_states,
    const uint32_t* __restrict__ ntrials_par,
    ParentCandidate* __restrict__ parents,
    uint32_t* __restrict__ n_parents,
    unsigned long long* __restrict__ rng_state,
    uint32_t rng_round_to,
    uint32_t ntrials) {
    using BlockScan = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint32_t running           = 0;
    const uint32_t n_candidates = n_beam_prev * nprobs;
    for (uint32_t base = 0; base < n_candidates; base += kBlock) {
        const uint32_t icand = base + threadIdx.x;
        uint32_t valid       = 0;
        ParentCandidate cand{};
        if (icand < n_candidates) {
            const uint32_t jslot = icand / nprobs;
            const uint32_t kprob = icand % nprobs;
            cand  = ParentCandidate{jslot, beam_prev[jslot], kprob};
            valid = parent_is_valid(prev_states, ntrials_par, cand.jthres_abs,
                                    icand, nprobs, kprob)
                        ? 1U
                        : 0U;
        }
        uint32_t pos   = 0;
        uint32_t total = 0;
        BlockScan(temp_storage).ExclusiveSum(valid, pos, total);
        if (valid != 0) {
            parents[running + pos] = cand;
        }
        running += total;
        __syncthreads(); // temp_storage reuse
    }
    if (threadIdx.x == 0) {
        *n_parents = running;
        advance_rng(rng_state, running, rng_round_to, ntrials);
    }
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

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void decide_parents_kernel(
    const ThresholdPairItem* __restrict__ pairs,
    const State* __restrict__ prev_states,
    const uint32_t* __restrict__ ntrials_par,
    const float* __restrict__ scores_cur,
    const float* __restrict__ thresholds,
    const float* __restrict__ probs,
    uint2* __restrict__ counts,
    State* __restrict__ cand_states,
    unsigned long long* __restrict__ cell_keys,
    uint32_t ntrials,
    uint32_t nprobs,
    float nbranches) {
    const uint32_t item          = blockIdx.x;
    const ThresholdPairItem pair = pairs[item / nprobs];
    const uint32_t kprob         = item % nprobs;
    const uint32_t cell          = (pair.jslot_prev * nprobs) + kprob;
    if (!parent_is_valid(prev_states, ntrials_par, pair.jthres_abs, cell,
                         nprobs, kprob)) {
        return; // uniform across the block
    }
    const float threshold               = thresholds[pair.ithres_abs];
    const float* __restrict__ scores_h0 = scores_cur + (2ULL * cell * ntrials);
    const float* __restrict__ scores_h1 = scores_h0 + ntrials;
    // Both counts in one 64-bit sum: low = H0, high = H1.
    unsigned long long packed = 0;
    for (uint32_t t = threadIdx.x; t < ntrials; t += kBlock) {
        packed += (scores_h0[t] > threshold ? 1ULL : 0ULL) +
                  (scores_h1[t] > threshold ? (1ULL << 32U) : 0ULL);
    }
    using BlockReduce = cub::BlockReduce<unsigned long long, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    packed = BlockReduce(temp_storage).Sum(packed);
    if (threadIdx.x != 0) {
        return;
    }
    const auto count_h0 = static_cast<uint32_t>(packed & 0xFFFFFFFFULL);
    const auto count_h1 = static_cast<uint32_t>(packed >> 32U);
    counts[item]        = make_uint2(count_h0, count_h1);
    const float ntrials_f = static_cast<float>(ntrials);
    const auto state_next = prev_states[(pair.jthres_abs * nprobs) + kprob]
                                .gen_next(threshold,
                                          __fdiv_rn(static_cast<float>(count_h0),
                                                    ntrials_f),
                                          __fdiv_rn(static_cast<float>(count_h1),
                                                    ntrials_f),
                                          nbranches);
    cand_states[item] = state_next;
    const int iprob =
        find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);
    if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
        atomicMin(&cell_keys[(pair.ithres_abs * nprobs) + iprob],
                  transition_key(state_next.complexity_cumul, item));
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

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void commit_parents_kernel(
    const uint32_t* __restrict__ beam_cur,
    const ThresholdPairItem* __restrict__ pairs,
    const uint2* __restrict__ counts,
    const State* __restrict__ cand_states,
    State* __restrict__ states_cur,
    unsigned long long* __restrict__ cell_keys,
    const float* __restrict__ scores_cur,
    const float* __restrict__ thresholds,
    uint32_t* __restrict__ ntrials_next,
    uint32_t* __restrict__ src_next,
    uint32_t* __restrict__ idx_next,
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
    if (threadIdx.x == 0) {
        const uint2 cnt                  = counts[item];
        states_cur[key_idx]              = cand_states[item];
        ntrials_next[2 * out_cell]       = cnt.x;
        ntrials_next[(2 * out_cell) + 1] = cnt.y;
        src_next[out_cell]               = par_cell;
    }

    // Ordered compaction of the surviving trial indices.
    using BlockScan = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    for (uint32_t branch = 0; branch < 2; ++branch) {
        const float* __restrict__ scores =
            scores_cur + (((2ULL * par_cell) + branch) * ntrials);
        uint32_t* __restrict__ idx_out =
            idx_next + (((2ULL * out_cell) + branch) * ntrials);
        uint32_t running = 0;
        for (uint32_t base = 0; base < ntrials; base += kBlock) {
            const uint32_t t = base + threadIdx.x;
            const uint32_t flag =
                (t < ntrials && scores[t] > threshold) ? 1U : 0U;
            uint32_t pos   = 0;
            uint32_t total = 0;
            BlockScan(temp_storage).ExclusiveSum(flag, pos, total);
            if (flag != 0) {
                idx_out[running + pos] = t;
            }
            running += total;
            __syncthreads(); // temp_storage reuse
        }
    }
}


__device__ __forceinline__ float eval_uniform(uint64_t seed, uint64_t subseq) {
    auto rng = typename RNG::Generator(seed, subseq, 0);
    typename RNG::UniformFloat dist(0.0F, 1.0F);
    const float u = dist.generate4(rng).x;
    return (u >= 1.0F) ? 0.0F : u;
}

// Initial folds for evaluate(): one non-overlapping subsequence per
// (branch, trial). Same arithmetic as simulate_folds_init_kernel.
__global__ void simulate_folds_init_eval_kernel(float* __restrict__ folds_sim,
                                                  const float* __restrict__ profile,
                                                  uint32_t nbins_padded,
                                                  float bias_snr,
                                                  float var_add,
                                                  uint64_t seed,
                                                  uint32_t ntrials) {
    const uint32_t tid          = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_trials = 2 * ntrials;
    if (tid >= total_trials) {
        return;
    }
    const uint32_t branch   = tid / ntrials;
    const uint32_t trial_id = tid % ntrials;
    const uint32_t out_offset =
        branch * ntrials * nbins_padded + trial_id * nbins_padded;
    const float branch_scale = (branch == 1) ? bias_snr : 0.0F;
    const float noise_stddev = __fsqrt_rn(var_add);
    auto rng_noise           = typename RNG::Generator(
        seed, eval_subseq(0U, 0U, 0U, branch, trial_id), 0);
    typename RNG::NormalFloat dist_noise(0.0F, noise_stddev);
    const uint32_t vec_count = nbins_padded / 4;
    float4* __restrict__ out_ptr4 =
        reinterpret_cast<float4*>(folds_sim + out_offset);
    const float4* __restrict__ prof_ptr4 =
        reinterpret_cast<const float4*>(profile);
#pragma unroll 4
    for (uint32_t j = 0; j < vec_count; ++j) {
        const float4 prof  = prof_ptr4[j];
        const float4 noise = dist_noise.generate4(rng_noise);
        out_ptr4[j]        = make_float4(fmaf(prof.x, branch_scale, noise.x),
                                         fmaf(prof.y, branch_scale, noise.y),
                                         fmaf(prof.z, branch_scale, noise.z),
                                         fmaf(prof.w, branch_scale, noise.w));
    }
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

template <uint32_t kBlock>
__global__ void evaluate_compact_kernel(const float* __restrict__ folds,
                                        const float* __restrict__ scores,
                                        float threshold,
                                        float* __restrict__ compacted,
                                        uint32_t* __restrict__ n_out,
                                        uint32_t ntrials,
                                        uint32_t nbins_padded) {
    const uint32_t branch = blockIdx.x;
    using BlockScan       = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    const float* scores_b = scores + (branch * ntrials);
    const float* folds_b =
        folds + (static_cast<uint64_t>(branch) * ntrials * nbins_padded);
    float* out_b =
        compacted + (static_cast<uint64_t>(branch) * ntrials * nbins_padded);
    uint32_t running = 0;
    for (uint32_t base = 0; base < ntrials; base += kBlock) {
        const uint32_t t    = base + threadIdx.x;
        const uint32_t flag = (t < ntrials && scores_b[t] > threshold) ? 1U : 0U;
        uint32_t pos        = 0;
        uint32_t total      = 0;
        BlockScan(temp_storage).ExclusiveSum(flag, pos, total);
        if (flag != 0) {
            const float* src =
                folds_b + (static_cast<uint64_t>(t) * nbins_padded);
            float* dst =
                out_b + (static_cast<uint64_t>(running + pos) * nbins_padded);
            for (uint32_t j = 0; j < nbins_padded; ++j) {
                dst[j] = src[j];
            }
        }
        running += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        n_out[branch] = running;
    }
}

// ------------------------------ legacy mode --------------------------------

// Compacts the valid items of the stage in item order: item_rng_idx[item] is
// its rank among valid items (its RNG stream), or kInvalidIndex. Also clears
// the per-item survivor counts.
template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void compact_items_kernel(
    const ThresholdPairItem* __restrict__ pairs,
    uint32_t n_items,
    uint32_t nprobs,
    const State* __restrict__ prev_states,
    const uint32_t* __restrict__ ntrials_par,
    uint32_t* __restrict__ item_rng_idx,
    uint2* __restrict__ counts,
    unsigned long long* __restrict__ rng_state,
    uint32_t rng_round_to,
    uint32_t ntrials) {
    using BlockScan = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint32_t running = 0;
    for (uint32_t base = 0; base < n_items; base += kBlock) {
        const uint32_t item = base + threadIdx.x;
        uint32_t valid      = 0;
        if (item < n_items) {
            const ThresholdPairItem pair = pairs[item / nprobs];
            const uint32_t kprob         = item % nprobs;
            valid = parent_is_valid(prev_states, ntrials_par, pair.jthres_abs,
                                    (pair.jslot_prev * nprobs) + kprob, nprobs,
                                    kprob)
                        ? 1U
                        : 0U;
        }
        uint32_t pos   = 0;
        uint32_t total = 0;
        BlockScan(temp_storage).ExclusiveSum(valid, pos, total);
        if (item < n_items) {
            item_rng_idx[item] = (valid != 0) ? running + pos : kInvalidIndex;
            counts[item]       = make_uint2(0U, 0U);
        }
        running += total;
        __syncthreads(); // temp_storage reuse
    }
    if (threadIdx.x == 0) {
        advance_rng(rng_state, running, rng_round_to, ntrials);
    }
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

__global__ void decide_items_kernel(const ThresholdPairItem* __restrict__ pairs,
                                    uint32_t n_items,
                                    const uint32_t* __restrict__ item_rng_idx,
                                    const State* __restrict__ prev_states,
                                    const uint2* __restrict__ counts,
                                    const float* __restrict__ thresholds,
                                    const float* __restrict__ probs,
                                    State* __restrict__ cand_states,
                                    unsigned long long* __restrict__ cell_keys,
                                    uint32_t ntrials,
                                    uint32_t nprobs,
                                    float nbranches) {
    const uint32_t item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (item >= n_items || item_rng_idx[item] == kInvalidIndex) {
        return;
    }
    const ThresholdPairItem pair = pairs[item / nprobs];
    const uint32_t kprob         = item % nprobs;
    const uint2 cnt              = counts[item];
    const float ntrials_f        = static_cast<float>(ntrials);
    const auto state_next = prev_states[(pair.jthres_abs * nprobs) + kprob]
                                .gen_next(thresholds[pair.ithres_abs],
                                          __fdiv_rn(static_cast<float>(cnt.x),
                                                    ntrials_f),
                                          __fdiv_rn(static_cast<float>(cnt.y),
                                                    ntrials_f),
                                          nbranches);
    cand_states[item] = state_next;
    const int iprob =
        find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);
    if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
        atomicMin(&cell_keys[(pair.ithres_abs * nprobs) + iprob],
                  transition_key(state_next.complexity_cumul, item));
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

// Owning wrapper for a CUDA stream.
class CudaStream {
public:
    CudaStream() {
        cuda_utils::check_cuda_call(cudaStreamCreate(&m_stream),
                                    "cudaStreamCreate failed");
    }
    ~CudaStream() { cudaStreamDestroy(m_stream); }
    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&)                 = delete;
    CudaStream& operator=(CudaStream&&)      = delete;
    operator cudaStream_t() const noexcept { return m_stream; } // NOLINT

private:
    cudaStream_t m_stream{nullptr};
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
class ThresholdsCudaCore {
public:
    ThresholdsCudaCore(std::span<const float> branching_pattern,
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
                       std::string_view mode,
                       SizeType batch_size,
                       int device_id,
                       std::optional<uint64_t> seed)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_trials_start(trials_start),
          m_mode(dynamic_threshold_mode_from_string(mode)),
          m_batch_size(batch_size),
          m_device_id(device_id),
          m_seed(seed.value_or(std::random_device{}())),
          m_nbins(nbins) {
        validate_inputs(nprobs, prob_min, snr_final, nthresholds);
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);

        // Host-side setup
        m_nbins_padded = (m_nbins + 3) & ~SizeType{3}; // multiple of 4
        m_profile.assign(m_nbins_padded, 0.0F);
        simulation::generate_folded_profile(m_profile, m_nbins, m_ref_ducy);
        m_thresholds  = detail::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs       = detail::compute_probs(nprobs, prob_min);
        m_nprobs      = m_probs.size();
        m_nstages     = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        m_box_score_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        if (*std::ranges::max_element(m_box_score_widths) >= m_nbins) {
            throw std::invalid_argument(std::format(
                "ducy_max={} gives a box width >= nbins={}", m_ducy_max,
                m_nbins));
        }
        m_bias_snr = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_guess_path = detail::guess_scheme(m_nstages, snr_final,
                                            m_branching_pattern, m_trials_start);
        SizeType max_beam = 0;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto beam_size = get_current_thresholds_idx(istage).size();
            if (beam_size == 0) {
                throw std::invalid_argument(std::format(
                    "Threshold beam of stage {} is empty (guess {:.3f}, "
                    "beam_width {}); increase beam_width or nthresholds",
                    istage, m_guess_path[istage], m_beam_width));
            }
            max_beam = std::max(max_beam, beam_size);
        }

        // Device constants
        m_thresholds_d = m_thresholds;
        m_profile_d    = m_profile;
        m_probs_d      = m_probs;
        const auto boxes = make_box_widths(m_box_score_widths, m_nbins);
        m_box_score_widths_d = boxes;
        std::vector<BoxWidth> by_len(m_nbins + 1, BoxWidth{0, 0.0F, 0.0F});
        for (const auto& box : boxes) {
            by_len[box.w] = box;
        }
        m_box_by_len_d = by_len;

        // Fold pools: one cell per (beam slot, prob bin), H0 + H1 rows each.
        const SizeType cells     = max_beam * m_nprobs;
        const SizeType pool_rows = cells * 2 * m_ntrials;
        if (cells * 2 > std::numeric_limits<uint32_t>::max() ||
            m_ntrials > std::numeric_limits<uint32_t>::max() / 2 ||
            m_nthresholds * m_nprobs > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(
                "ThresholdsCuda: problem size exceeds 32-bit "
                "cell indexing");
        }
        for (auto& v : m_folds_d) {
            v.resize(pool_rows * m_nbins_padded);
        }
        for (auto& v : m_ntrials_d) {
            v.resize(cells * 2);
        }
        const bool parent_map = m_mode != DynamicThresholdMode::kLegacy;
        if (parent_map) {
            for (auto& v : m_src_d) {
                v.resize(cells);
            }
            for (auto& v : m_idx_d) {
                v.resize(pool_rows);
            }
            m_scores_d.resize(pool_rows);
        }
        const auto grid_size = m_nstages * m_nthresholds * m_nprobs;
        m_states.resize(grid_size, State{});
        m_states_d.resize(grid_size);
        m_cell_keys_d.resize(m_nthresholds * m_nprobs);

        const SizeType bytes =
            (2 * pool_rows * m_nbins_padded * sizeof(float)) +
            (parent_map ? pool_rows * (2 * sizeof(uint32_t) + sizeof(float))
                        : 0) +
            (grid_size * sizeof(State));
        spdlog::info("ThresholdsCuda ({} mode): {} max beam "
                     "thresholds x {} prob bins, {:.2f} GiB of device memory",
                     mode_to_string(m_mode), max_beam, m_nprobs,
                     utils::to_gib(bytes));
    }
    ~ThresholdsCudaCore()                                    = default;
    ThresholdsCudaCore(const ThresholdsCudaCore&)            = delete;
    ThresholdsCudaCore& operator=(const ThresholdsCudaCore&) = delete;
    ThresholdsCudaCore(ThresholdsCudaCore&&)                 = delete;
    ThresholdsCudaCore& operator=(ThresholdsCudaCore&&)      = delete;

    void run(SizeType thres_neigh) {
        if (thres_neigh == 0) {
            throw std::invalid_argument("thres_neigh must be positive");
        }
        timing::ScopeTimer timer("ThresholdsCuda::run");
        spdlog::info("Running dynamic threshold scheme on CUDA ({} mode)",
                     mode_to_string(m_mode));
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        constexpr float kVarInit = 1.0F;
        constexpr float kVarAdd  = 1.0F;

        // Fresh grid every run, so run() can be called again.
        thrust::fill(thrust::cuda::par.on(stream), m_states_d.begin(),
                     m_states_d.end(), State{});
        thrust::fill(thrust::cuda::par.on(stream), m_cell_keys_d.begin(),
                     m_cell_keys_d.end(), kEmptyKey);
        for (auto& v : m_ntrials_d) {
            thrust::fill(thrust::cuda::par.on(stream), v.begin(), v.end(), 0U);
        }
        if (m_mode == DynamicThresholdMode::kImproved) {
            run_improved(thres_neigh, kVarInit, kVarAdd, stream);
        } else {
            run_legacy(thres_neigh, kVarInit, kVarAdd, stream);
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(m_states.data(),
                            thrust::raw_pointer_cast(m_states_d.data()),
                            m_states.size() * sizeof(State),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        m_thres_neigh = thres_neigh;
        warn_empty_stages();
    }

    std::vector<State> get_states() const { return m_states; }
    std::vector<float> get_thresholds() const { return m_thresholds; }
    std::vector<float> get_probs() const { return m_probs; }

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

    std::vector<float> get_branching_pattern() const {
        return m_branching_pattern;
    }
    std::vector<float> get_profile() const { return m_profile; }
    SizeType get_nstages() const { return m_nstages; }
    SizeType get_nthresholds() const { return m_nthresholds; }
    SizeType get_nprobs() const { return m_nprobs; }
    std::vector<SizeType> get_box_score_widths() const {
        return m_box_score_widths;
    }

    // Static per-stage layout shared by both pipelines. Stage s reads the
    // beam of stage s-1 (stage 0 reads a virtual one-threshold beam holding
    // the initial state); its transitions are pairs x nprobs items.
    struct StagePlan {
        uint32_t beam_prev_offset;
        uint32_t n_beam_prev;
        uint32_t beam_cur_offset;
        uint32_t n_beam_cur;
        uint32_t pair_offset; // also the offset of the item order / nprobs
        uint32_t n_pairs;
    };

    struct StagePlans {
        std::vector<StagePlan> stages;
        thrust::device_vector<uint32_t> beams_d;
        thrust::device_vector<ThresholdPairItem> pairs_d;
        // Legacy: items of each stage grouped by parent cell.
        thrust::device_vector<uint32_t> item_order_d;
        SizeType max_items{1};
        SizeType max_parents{1};
    };

    StagePlans build_stage_plans(SizeType thres_neigh, bool with_order) const {
        StagePlans plans;
        plans.stages.resize(m_nstages);
        std::vector<uint32_t> beams_flat = {0U}; // virtual beam before stage 0
        std::vector<ThresholdPairItem> pairs_flat;
        std::vector<uint32_t> order_flat;
        std::vector<SizeType> beam_prev_idx = {0};
        uint32_t beam_prev_offset           = 0;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto beam_cur_idx = get_current_thresholds_idx(istage);
            auto& plan              = plans.stages[istage];
            plan.beam_prev_offset   = beam_prev_offset;
            plan.n_beam_prev = static_cast<uint32_t>(beam_prev_idx.size());
            plan.beam_cur_offset = static_cast<uint32_t>(beams_flat.size());
            plan.n_beam_cur  = static_cast<uint32_t>(beam_cur_idx.size());
            plan.pair_offset = static_cast<uint32_t>(pairs_flat.size());
            for (const auto ithres : beam_cur_idx) {
                beams_flat.push_back(static_cast<uint32_t>(ithres));
            }
            if (istage == 0) {
                for (SizeType islot = 0; islot < beam_cur_idx.size();
                     ++islot) {
                    pairs_flat.push_back(ThresholdPairItem{
                        static_cast<uint32_t>(beam_cur_idx[islot]),
                        static_cast<uint32_t>(islot), 0U, 0U});
                }
            } else {
                std::vector<int32_t> prev_slot_of_thresh(m_nthresholds, -1);
                for (uint32_t slot = 0; slot < beam_prev_idx.size(); ++slot) {
                    prev_slot_of_thresh[beam_prev_idx[slot]] =
                        static_cast<int32_t>(slot);
                }
                for (SizeType islot = 0; islot < beam_cur_idx.size();
                     ++islot) {
                    const auto ithres = beam_cur_idx[islot];
                    for (const SizeType jthres :
                         utils::find_neighbouring_indices(beam_prev_idx,
                                                          ithres,
                                                          thres_neigh)) {
                        const int32_t jslot = prev_slot_of_thresh[jthres];
                        if (jslot < 0) {
                            continue;
                        }
                        pairs_flat.push_back(ThresholdPairItem{
                            static_cast<uint32_t>(ithres),
                            static_cast<uint32_t>(islot),
                            static_cast<uint32_t>(jthres),
                            static_cast<uint32_t>(jslot)});
                    }
                }
            }
            plan.n_pairs = static_cast<uint32_t>(pairs_flat.size()) -
                           plan.pair_offset;
            const SizeType n_items =
                static_cast<SizeType>(plan.n_pairs) * m_nprobs;
            plans.max_items = std::max(plans.max_items, n_items);
            plans.max_parents =
                std::max(plans.max_parents,
                         static_cast<SizeType>(plan.n_beam_prev) * m_nprobs);
            if (with_order) {
                std::vector<uint32_t> order(n_items);
                std::iota(order.begin(), order.end(), 0U);
                const auto* pairs = pairs_flat.data() + plan.pair_offset;
                const auto parent_of = [&](uint32_t item) {
                    return (pairs[item / m_nprobs].jslot_prev * m_nprobs) +
                           (item % m_nprobs);
                };
                std::ranges::stable_sort(order, {}, parent_of);
                order_flat.insert(order_flat.end(), order.begin(),
                                  order.end());
            }
            beam_prev_offset = plan.beam_cur_offset;
            beam_prev_idx    = beam_cur_idx;
        }
        if (with_order && pairs_flat.size() * m_nprobs >
                              std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "ThresholdsCuda: too many transitions per run");
        }
        plans.beams_d      = beams_flat;
        plans.pairs_d      = pairs_flat;
        plans.item_order_d = order_flat;
        return plans;
    }

    TrialSimParams trial_sim_params(float var_in, float var_add) const {
        return TrialSimParams{
            .profile    = thrust::raw_pointer_cast(m_profile_d.data()),
            .widths     = thrust::raw_pointer_cast(m_box_score_widths_d.data()),
            .box_by_len = thrust::raw_pointer_cast(m_box_by_len_d.data()),
            .nwidths    = static_cast<uint32_t>(m_box_score_widths.size()),
            .nbins      = static_cast<uint32_t>(m_nbins),
            .nbins_padded = static_cast<uint32_t>(m_nbins_padded),
            .bias_snr     = m_bias_snr,
            .var_in       = var_in,
            .var_add      = var_add,
            .seed         = m_seed};
    }

    template <typename F> void with_scorer(F&& f) const {
        dispatch_scorer(m_nbins, m_nbins_padded,
                        *std::ranges::max_element(m_box_score_widths),
                        /*allow_reg=*/true,
                        std::forward<F>(f));
    }

    // Writes the initial folds as the single parent cell 0 of `folds` and
    // sets up the RNG offsets that follow them.
    void simulate_initial_folds(float* folds,
                                uint32_t* ntrials_desc,
                                unsigned long long* rng_state,
                                float var_init,
                                cudaStream_t stream) const {
        const auto ntrials                  = static_cast<uint32_t>(m_ntrials);
        constexpr SizeType kThreadsPerBlock = 256;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim((2 * m_ntrials + kThreadsPerBlock - 1) /
                            kThreadsPerBlock);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        simulate_folds_init_kernel<<<grid_dim, block_dim, 0, stream>>>(
            folds, thrust::raw_pointer_cast(m_profile_d.data()),
            m_nbins_padded, m_bias_snr, var_init, m_seed, 0, ntrials);
        cuda_utils::check_last_cuda_error("simulate_folds_init_kernel");
        const uint32_t init_desc[2]           = {ntrials, ntrials};
        const unsigned long long rng_init[2] = {2ULL * ntrials, 0ULL};
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(ntrials_desc, init_desc, sizeof(init_desc),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(rng_state, rng_init, sizeof(rng_init),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync failed");
    }

    void run_improved(SizeType thres_neigh,
                      float var_init,
                      float var_add,
                      cudaStream_t stream) {
        constexpr uint32_t kCompactBlock = 1024;
        constexpr uint32_t kSimBlock     = 256;
        constexpr uint32_t kDecideBlock  = 256;
        constexpr uint32_t kCommitBlock  = 256;
        const auto ntrials = static_cast<uint32_t>(m_ntrials);
        const auto nprobs  = static_cast<uint32_t>(m_nprobs);
        const auto grid_stride =
            static_cast<SizeType>(m_nthresholds) * m_nprobs;

        const auto plans = build_stage_plans(thres_neigh, false);
        thrust::device_vector<uint2> counts_d(plans.max_items);
        thrust::device_vector<State> cand_states_d(plans.max_items);
        thrust::device_vector<ParentCandidate> parents_d(plans.max_parents);
        thrust::device_vector<uint32_t> n_parents_d(1, 0U);
        thrust::device_vector<unsigned long long> rng_state_d(2);
        thrust::device_vector<State> init_states_d(m_nprobs, State{});
        init_states_d[0] = State::initial();

        // Ping-pong buffers: index (s & 1) is written at stage s.
        float* sim[2] = {thrust::raw_pointer_cast(m_folds_d[0].data()),
                         thrust::raw_pointer_cast(m_folds_d[1].data())};
        uint32_t* ntrials_desc[2] = {
            thrust::raw_pointer_cast(m_ntrials_d[0].data()),
            thrust::raw_pointer_cast(m_ntrials_d[1].data())};
        uint32_t* src_desc[2] = {thrust::raw_pointer_cast(m_src_d[0].data()),
                                 thrust::raw_pointer_cast(m_src_d[1].data())};
        uint32_t* idx_desc[2] = {thrust::raw_pointer_cast(m_idx_d[0].data()),
                                 thrust::raw_pointer_cast(m_idx_d[1].data())};
        float* scores = thrust::raw_pointer_cast(m_scores_d.data());

        // Initial folds: the single virtual parent cell 0 of the odd
        // buffers, with an identity survivor map.
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(src_desc[1], 0, sizeof(uint32_t), stream),
            "cudaMemsetAsync failed");
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<uint32_t>(0),
                          thrust::counting_iterator<uint32_t>(2 * ntrials),
                          m_idx_d[1].begin(), ModuloFunctor{ntrials});
        simulate_initial_folds(sim[1], ntrials_desc[1],
                               thrust::raw_pointer_cast(rng_state_d.data()),
                               var_init, stream);

        float var_in = var_init;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto& plan     = plans.stages[istage];
            const uint32_t cur   = istage & 1U;
            const uint32_t prev  = cur ^ 1U;
            const auto nbranches = m_branching_pattern[istage];
            const State* prev_states =
                istage == 0
                    ? thrust::raw_pointer_cast(init_states_d.data())
                    : thrust::raw_pointer_cast(m_states_d.data()) +
                          ((istage - 1) * grid_stride);
            State* cur_states = thrust::raw_pointer_cast(m_states_d.data()) +
                                (istage * grid_stride);
            const uint32_t* beams = thrust::raw_pointer_cast(plans.beams_d.data());
            const ThresholdPairItem* pairs =
                thrust::raw_pointer_cast(plans.pairs_d.data()) +
                plan.pair_offset;

            compact_parents_kernel<kCompactBlock>
                <<<1, kCompactBlock, 0, stream>>>(
                    beams + plan.beam_prev_offset, plan.n_beam_prev, nprobs,
                    prev_states, ntrials_desc[prev],
                    thrust::raw_pointer_cast(parents_d.data()),
                    thrust::raw_pointer_cast(n_parents_d.data()),
                    thrust::raw_pointer_cast(rng_state_d.data()),
                    istage == 0 ? 1U : static_cast<uint32_t>(m_batch_size),
                    ntrials);
            cuda_utils::check_last_cuda_error("compact_parents_kernel");

            const SizeType sim_threads =
                static_cast<SizeType>(plan.n_beam_prev) * m_nprobs * 2 *
                m_ntrials;
            if (sim_threads > 0) {
                const dim3 grid_dim((sim_threads + kSimBlock - 1) / kSimBlock);
                const dim3 block_dim(kSimBlock);
                cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
                with_scorer([&]<typename Scorer>() {
                    simulate_score_parents_kernel<Scorer>
                        <<<grid_dim, block_dim, 0, stream>>>(
                            thrust::raw_pointer_cast(parents_d.data()),
                            thrust::raw_pointer_cast(n_parents_d.data()),
                            thrust::raw_pointer_cast(rng_state_d.data()),
                            sim[prev], ntrials_desc[prev], src_desc[prev],
                            idx_desc[prev], sim[cur], scores,
                            trial_sim_params(var_in, var_add), ntrials,
                            nprobs);
                });
                cuda_utils::check_last_cuda_error(
                    "simulate_score_parents_kernel");
            }

            const SizeType n_items =
                static_cast<SizeType>(plan.n_pairs) * m_nprobs;
            if (n_items > 0) {
                decide_parents_kernel<kDecideBlock>
                    <<<n_items, kDecideBlock, 0, stream>>>(
                        pairs, prev_states, ntrials_desc[prev], scores,
                        thrust::raw_pointer_cast(m_thresholds_d.data()),
                        thrust::raw_pointer_cast(m_probs_d.data()),
                        thrust::raw_pointer_cast(counts_d.data()),
                        thrust::raw_pointer_cast(cand_states_d.data()),
                        thrust::raw_pointer_cast(m_cell_keys_d.data()),
                        ntrials, nprobs, nbranches);
                cuda_utils::check_last_cuda_error("decide_parents_kernel");
            }
            const SizeType n_out_cells =
                static_cast<SizeType>(plan.n_beam_cur) * m_nprobs;
            if (n_out_cells > 0) {
                commit_parents_kernel<kCommitBlock>
                    <<<n_out_cells, kCommitBlock, 0, stream>>>(
                        beams + plan.beam_cur_offset, pairs,
                        thrust::raw_pointer_cast(counts_d.data()),
                        thrust::raw_pointer_cast(cand_states_d.data()),
                        cur_states,
                        thrust::raw_pointer_cast(m_cell_keys_d.data()), scores,
                        thrust::raw_pointer_cast(m_thresholds_d.data()),
                        ntrials_desc[cur], src_desc[cur], idx_desc[cur],
                        ntrials, nprobs);
                cuda_utils::check_last_cuda_error("commit_parents_kernel");
            }
            var_in += var_add;
        }
    }

    void run_legacy(SizeType thres_neigh,
                         float var_init,
                         float var_add,
                         cudaStream_t stream) {
        constexpr uint32_t kCompactBlock = 1024;
        constexpr uint32_t kSimBlock     = 256;
        constexpr uint32_t kDecideBlock  = 256;
        constexpr uint32_t kCommitBlock  = 256;
        const auto ntrials = static_cast<uint32_t>(m_ntrials);
        const auto nprobs  = static_cast<uint32_t>(m_nprobs);
        const auto grid_stride =
            static_cast<SizeType>(m_nthresholds) * m_nprobs;

        const auto plans = build_stage_plans(thres_neigh, true);
        thrust::device_vector<uint2> counts_d(plans.max_items);
        thrust::device_vector<State> cand_states_d(plans.max_items);
        thrust::device_vector<uint32_t> item_rng_idx_d(plans.max_items);
        thrust::device_vector<unsigned long long> rng_state_d(2);
        thrust::device_vector<uint32_t> error_flag_d(1, 0U);
        thrust::device_vector<State> init_states_d(m_nprobs, State{});
        init_states_d[0] = State::initial();

        // Ping-pong buffers: index (s & 1) is written at stage s.
        float* folds[2] = {thrust::raw_pointer_cast(m_folds_d[0].data()),
                           thrust::raw_pointer_cast(m_folds_d[1].data())};
        uint32_t* ntrials_desc[2] = {
            thrust::raw_pointer_cast(m_ntrials_d[0].data()),
            thrust::raw_pointer_cast(m_ntrials_d[1].data())};
        simulate_initial_folds(folds[1], ntrials_desc[1],
                               thrust::raw_pointer_cast(rng_state_d.data()),
                               var_init, stream);

        float var_in = var_init;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto& plan     = plans.stages[istage];
            const uint32_t cur   = istage & 1U;
            const uint32_t prev  = cur ^ 1U;
            const auto nbranches = m_branching_pattern[istage];
            const State* prev_states =
                istage == 0
                    ? thrust::raw_pointer_cast(init_states_d.data())
                    : thrust::raw_pointer_cast(m_states_d.data()) +
                          ((istage - 1) * grid_stride);
            State* cur_states = thrust::raw_pointer_cast(m_states_d.data()) +
                                (istage * grid_stride);
            const uint32_t* beams = thrust::raw_pointer_cast(plans.beams_d.data());
            const ThresholdPairItem* pairs =
                thrust::raw_pointer_cast(plans.pairs_d.data()) +
                plan.pair_offset;
            const auto n_items =
                static_cast<uint32_t>(static_cast<SizeType>(plan.n_pairs) *
                                      m_nprobs);
            const auto params = trial_sim_params(var_in, var_add);
            var_in += var_add;
            if (n_items == 0) {
                continue; // no transitions: this and later stages stay empty
            }

            // The original walk simulated the initial transitions in one
            // launch (exact offset) and later stages in batches.
            compact_items_kernel<kCompactBlock>
                <<<1, kCompactBlock, 0, stream>>>(
                    pairs, n_items, nprobs, prev_states, ntrials_desc[prev],
                    thrust::raw_pointer_cast(item_rng_idx_d.data()),
                    thrust::raw_pointer_cast(counts_d.data()),
                    thrust::raw_pointer_cast(rng_state_d.data()),
                    istage == 0 ? 1U : static_cast<uint32_t>(m_batch_size),
                    ntrials);
            cuda_utils::check_last_cuda_error("compact_items_kernel");

            const SizeType sim_threads =
                static_cast<SizeType>(n_items) * 2 * m_ntrials;
            const dim3 sim_grid((sim_threads + kSimBlock - 1) / kSimBlock);
            cuda_utils::check_kernel_launch_params(sim_grid, dim3(kSimBlock));
            with_scorer([&]<typename Scorer>() {
                simulate_count_items_kernel<Scorer>
                    <<<sim_grid, kSimBlock, 0, stream>>>(
                        thrust::raw_pointer_cast(plans.item_order_d.data()) +
                            (static_cast<SizeType>(plan.pair_offset) * m_nprobs),
                        n_items, pairs,
                        thrust::raw_pointer_cast(item_rng_idx_d.data()),
                        thrust::raw_pointer_cast(rng_state_d.data()),
                        folds[prev], ntrials_desc[prev],
                        thrust::raw_pointer_cast(m_thresholds_d.data()),
                        reinterpret_cast<uint32_t*>(
                            thrust::raw_pointer_cast(counts_d.data())),
                        params, ntrials, nprobs);
            });
            cuda_utils::check_last_cuda_error("simulate_count_items_kernel");

            decide_items_kernel<<<(n_items + kDecideBlock - 1) / kDecideBlock,
                                  kDecideBlock, 0, stream>>>(
                pairs, n_items,
                thrust::raw_pointer_cast(item_rng_idx_d.data()), prev_states,
                thrust::raw_pointer_cast(counts_d.data()),
                thrust::raw_pointer_cast(m_thresholds_d.data()),
                thrust::raw_pointer_cast(m_probs_d.data()),
                thrust::raw_pointer_cast(cand_states_d.data()),
                thrust::raw_pointer_cast(m_cell_keys_d.data()), ntrials,
                nprobs, nbranches);
            cuda_utils::check_last_cuda_error("decide_items_kernel");

            const SizeType n_out_cells =
                static_cast<SizeType>(plan.n_beam_cur) * m_nprobs;
            if (n_out_cells > 0) {
                with_scorer([&]<typename Scorer>() {
                    commit_items_kernel<Scorer, kCommitBlock>
                        <<<n_out_cells, kCommitBlock, 0, stream>>>(
                            beams + plan.beam_cur_offset, pairs,
                            thrust::raw_pointer_cast(item_rng_idx_d.data()),
                            thrust::raw_pointer_cast(rng_state_d.data()),
                            thrust::raw_pointer_cast(counts_d.data()),
                            thrust::raw_pointer_cast(cand_states_d.data()),
                            cur_states,
                            thrust::raw_pointer_cast(m_cell_keys_d.data()),
                            folds[prev], ntrials_desc[prev],
                            thrust::raw_pointer_cast(m_thresholds_d.data()),
                            folds[cur], ntrials_desc[cur],
                            thrust::raw_pointer_cast(error_flag_d.data()),
                            params, ntrials, nprobs);
                });
                cuda_utils::check_last_cuda_error("commit_items_kernel");
            }
        }
        const uint32_t error_flag = error_flag_d[0];
        if (error_flag != 0) {
            throw std::runtime_error(std::format(
                "ThresholdsCuda: legacy survivor regeneration "
                "disagreed with the counting pass (flag {})",
                error_flag));
        }
    }

    std::string save(const std::string& outdir = "./") const {
        const std::filesystem::path filebase = std::format(
            "dynscheme_{}_nstages_{:03d}_nthresh_{:03d}_nprobs_{:03d}_"
            "ntrials_{:04d}_snr_{:04.1f}_ducy_{:04.2f}_beam_{:03.1f}.h5",
            mode_to_string(m_mode), m_nstages, m_nthresholds, m_nprobs,
            m_ntrials, m_thresholds.back(), m_ref_ducy, m_beam_width);
        std::filesystem::create_directories(outdir);
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
        file.createAttribute("mode", mode_to_string(m_mode));
        file.createAttribute("seed", m_seed);
        file.createAttribute("batch_size", m_batch_size);
        file.createAttribute("thres_neigh", m_thres_neigh);

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

    std::vector<float> get_best_path_thresholds(float min_pd) const {
        return detail::get_best_path_thresholds(
            std::span<const State>(m_states.data(), m_states.size()),
            m_thresholds, m_probs, m_nstages, m_nthresholds, m_nprobs, min_pd);
    }

    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed) const {
        if (thresholds.size() != m_nstages) {
            throw std::invalid_argument(
                "ThresholdsCuda::evaluate: need one threshold "
                "per stage");
        }
        if (ntrials == 0 || ntrials > (1U << 20U)) {
            throw std::invalid_argument(
                "ThresholdsCuda::evaluate: ntrials must be in "
                "[1, 2^20]");
        }
        for (const float t : thresholds) {
            if (!is_finite(t)) {
                throw std::invalid_argument(
                    "ThresholdsCuda::evaluate: thresholds must "
                    "be finite");
            }
        }
        const uint64_t eval_seed = seed.value_or(std::random_device{}());
        const auto ntrials_u     = static_cast<uint32_t>(ntrials);
        const auto padded        = static_cast<uint32_t>(m_nbins_padded);
        const SizeType rows      = 2 * ntrials * m_nbins_padded;

        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        thrust::device_vector<float> buf_a(rows, 0.0F);
        thrust::device_vector<float> buf_b(rows, 0.0F);
        thrust::device_vector<float> scores(2 * ntrials, 0.0F);
        thrust::device_vector<uint32_t> n_in(2, ntrials_u);
        thrust::device_vector<uint32_t> n_out(2, 0U);

        constexpr uint32_t kBlock = 256;
        {
            const dim3 grid((2 * ntrials + kBlock - 1) / kBlock);
            cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
            simulate_folds_init_eval_kernel<<<grid, kBlock, 0, stream>>>(
                thrust::raw_pointer_cast(buf_a.data()),
                thrust::raw_pointer_cast(m_profile_d.data()), padded,
                m_bias_snr, 1.0F, eval_seed, ntrials_u);
            cuda_utils::check_last_cuda_error(
                "simulate_folds_init_eval_kernel");
        }

        std::vector<State> states(m_nstages);
        State prev         = State::initial();
        float var_in       = 1.0F;
        constexpr float kVarAdd = 1.0F;
        float* cur_in      = thrust::raw_pointer_cast(buf_a.data());
        float* cur_out     = thrust::raw_pointer_cast(buf_b.data());

        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            TrialSimParams params = trial_sim_params(var_in, kVarAdd);
            params.seed           = eval_seed;
            const dim3 grid((2 * ntrials + kBlock - 1) / kBlock);
            cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
            with_scorer([&]<typename Scorer>() {
                evaluate_stage_kernel<Scorer><<<grid, kBlock, 0, stream>>>(
                    cur_in, thrust::raw_pointer_cast(n_in.data()), cur_out,
                    thrust::raw_pointer_cast(scores.data()), params, ntrials_u,
                    static_cast<uint32_t>(istage));
            });
            cuda_utils::check_last_cuda_error("evaluate_stage_kernel");
            evaluate_compact_kernel<kBlock><<<2, kBlock, 0, stream>>>(
                cur_out, thrust::raw_pointer_cast(scores.data()),
                thresholds[istage], cur_in,
                thrust::raw_pointer_cast(n_out.data()), ntrials_u, padded);
            cuda_utils::check_last_cuda_error("evaluate_compact_kernel");
            uint32_t counts[2] = {0, 0};
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(counts, thrust::raw_pointer_cast(n_out.data()),
                                sizeof(counts), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync failed");
            cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                        "cudaStreamSynchronize failed");

            const float ntrials_f = static_cast<float>(ntrials);
            const float success_h0 =
                static_cast<float>(counts[0]) / ntrials_f;
            const float success_h1 =
                static_cast<float>(counts[1]) / ntrials_f;
            State next = prev.gen_next(thresholds[istage], success_h0,
                                       success_h1, m_branching_pattern[istage]);
            states[istage] = next;
            prev           = next;
            var_in += kVarAdd;
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(thrust::raw_pointer_cast(n_in.data()), counts,
                                sizeof(counts), cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync failed");
            if (counts[0] == 0 || counts[1] == 0) {
                break;
            }
            // Compacted survivors were written into cur_in. The next stage
            // reads cur_in and writes cur_out, so the pointers stay as they are.
        }
        return states;
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
    DynamicThresholdMode m_mode;
    SizeType m_batch_size;
    int m_device_id;
    uint64_t m_seed;
    SizeType m_nbins;
    SizeType m_nbins_padded{};
    SizeType m_nprobs{};
    SizeType m_nstages{};
    SizeType m_nthresholds{};
    SizeType m_thres_neigh{0}; // of the last run
    float m_bias_snr{};

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    std::vector<SizeType> m_box_score_widths;
    std::vector<float> m_guess_path;
    std::vector<State> m_states;

    // Device constants
    thrust::device_vector<float> m_thresholds_d;
    thrust::device_vector<float> m_profile_d;
    thrust::device_vector<float> m_probs_d;
    thrust::device_vector<BoxWidth> m_box_score_widths_d;
    // Box widths indexed by length (w == 0: not a width), size nbins + 1.
    thrust::device_vector<BoxWidth> m_box_by_len_d;

    // State grid [nstages x nthresholds x nprobs] and the per-cell winner
    // keys of the stage being processed (kEmptyKey between stages).
    thrust::device_vector<State> m_states_d;
    thrust::device_vector<unsigned long long> m_cell_keys_d;

    // Ping-pong fold pools, index (s & 1) written at stage s. A cell is
    // (beam slot, prob bin) with [H0 | H1] x ntrials rows of nbins_padded.
    //  - legacy: survivor folds of the stage's cells;
    //  - improved: simulated folds of the stage's parent cells.
    std::array<thrust::device_vector<float>, 2> m_folds_d;
    // Per cell survivor counts [H0, H1].
    std::array<thrust::device_vector<uint32_t>, 2> m_ntrials_d;
    // Improved mode: per-cell survivor map into the previous stage's
    // simulated folds (src: sim cell; idx: [cell][branch][trial]) and the
    // scores of the current parent simulation.
    std::array<thrust::device_vector<uint32_t>, 2> m_src_d;
    std::array<thrust::device_vector<uint32_t>, 2> m_idx_d;
    thrust::device_vector<float> m_scores_d;

    // Bit-level check: host code is built with -ffast-math, under which
    // std::isfinite (and NaN comparisons) may be folded away.
    static bool is_finite(float x) noexcept {
        return (std::bit_cast<uint32_t>(x) & 0x7F800000U) != 0x7F800000U;
    }

    void validate_inputs(SizeType nprobs,
                         float prob_min,
                         float snr_final,
                         SizeType nthresholds) const {
        const auto fail = [](const std::string& msg) {
            throw std::invalid_argument("ThresholdsCuda: " + msg);
        };
        for (const float x : {m_ref_ducy, m_ducy_max, m_wtsp, m_beam_width,
                              prob_min, snr_final}) {
            if (!is_finite(x)) {
                fail("float parameters must be finite");
            }
        }
        if (m_branching_pattern.size() < 2) {
            fail("branching pattern needs at least 2 stages");
        }
        for (SizeType i = 0; i < m_branching_pattern.size(); ++i) {
            const float b = m_branching_pattern[i];
            if (!is_finite(b) || b <= 0.0F) {
                fail(std::format("branching_pattern[{}] = {} must be finite "
                                 "and positive",
                                 i, b));
            }
        }
        if (m_nbins < 4 || m_nbins > 1024) {
            fail(std::format("nbins={} must be in [4, 1024]", m_nbins));
        }
        if (m_ntrials == 0 || nprobs == 0 || m_batch_size == 0) {
            fail("ntrials, nprobs and batch_size must be positive");
        }
        if (nthresholds < 2 || !(snr_final > 0.1F)) {
            fail("need nthresholds >= 2 and a finite snr_final > 0.1");
        }
        if (!(prob_min > 0.0F && prob_min < 1.0F)) {
            fail(std::format("prob_min={} must be in (0, 1)", prob_min));
        }
        if (!(m_ref_ducy > 0.0F && m_ref_ducy < 1.0F) ||
            !(m_ducy_max > 0.0F && m_ducy_max < 1.0F)) {
            fail("ref_ducy and ducy_max must be in (0, 1)");
        }
        if (!(m_wtsp > 0.0F) || !(m_beam_width > 0.0F)) {
            fail("wtsp and beam_width must be positive");
        }
    }

    // Logs stages that ended with no state (nothing survived into them).
    void warn_empty_stages() const {
        const auto stride = m_nthresholds * m_nprobs;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto begin = m_states.begin() +
                               static_cast<std::ptrdiff_t>(istage * stride);
            const bool empty = std::all_of(
                begin, begin + static_cast<std::ptrdiff_t>(stride),
                [](const State& st) { return st.is_empty; });
            if (empty) {
                spdlog::warn("ThresholdsCuda: stage {} (and all "
                             "later stages) has no surviving state",
                             istage);
                return;
            }
        }
    }
};

class ThresholdsCudaEngine final : public detail::ThresholdsEngine {
public:
    ThresholdsCudaEngine(std::span<const float> branching_pattern,
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
                         std::string_view mode,
                         SizeType batch_size,
                         int device_id,
                         std::optional<uint64_t> seed)
        : m_impl(branching_pattern,
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
                 mode,
                 batch_size,
                 device_id,
                 seed) {}

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const override {
        return m_impl.get_current_thresholds_idx(istage);
    }
    std::vector<float> get_branching_pattern() const override {
        return m_impl.get_branching_pattern();
    }
    std::vector<float> get_profile() const override {
        return m_impl.get_profile();
    }
    std::vector<float> get_thresholds() const override {
        return m_impl.get_thresholds();
    }
    std::vector<float> get_probs() const override {
        return m_impl.get_probs();
    }
    SizeType get_nstages() const override {
        return m_impl.get_nstages();
    }
    SizeType get_nthresholds() const override {
        return m_impl.get_nthresholds();
    }
    SizeType get_nprobs() const override {
        return m_impl.get_nprobs();
    }
    std::vector<SizeType> get_box_score_widths() const override {
        return m_impl.get_box_score_widths();
    }
    std::vector<State> get_states() const override {
        return m_impl.get_states();
    }
    void run(SizeType thres_neigh) override {
        m_impl.run(thres_neigh);
    }
    std::vector<State>
    evaluate(std::span<const float> thresholds,
             SizeType ntrials,
             std::optional<uint64_t> seed = std::nullopt) const override {
        return m_impl.evaluate(thresholds, ntrials, seed);
    }
    std::string save(const std::string& outdir = "./") const override {
        return m_impl.save(outdir);
    }
    std::vector<float> get_best_path_thresholds(float min_pd = 0.1F) const override {
        return m_impl.get_best_path_thresholds(min_pd);
    }

private:
    ThresholdsCudaCore m_impl;
};

namespace detail {
std::unique_ptr<ThresholdsEngine>
make_thresholds_gpu(std::span<const float> branching_pattern,
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
                     std::string_view mode,
                     SizeType batch_size,
                     int device_id,
                     std::optional<uint64_t> seed) {
    return std::make_unique<ThresholdsCudaEngine>(
        branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
        snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
        mode, batch_size, device_id, seed);
}
} // namespace detail

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)