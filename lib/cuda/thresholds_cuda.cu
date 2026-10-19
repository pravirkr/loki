#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
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
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/simulation/simulation.hpp"

#include "lib/cuda/cub_helpers.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/device_rng.cuh"
#include "lib/cuda/thresholds_kernels_cuda.cuh"
#include "lib/detail/timing.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

namespace loki::detection {

// Scorers, device helpers and the scorer-templated kernels live in
// cuda/thresholds_kernels_cuda.cuh.
using namespace detail; // NOLINT(google-build-using-namespace)

namespace {

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
    const float noise_stddev  = __fsqrt_rn(var_add);
    auto rng_noise            = make_rng(seed, noise_base);
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

std::vector<BoxWidth> make_box_widths(std::span<const SizeType> widths,
                                      SizeType nbins) {
    std::vector<BoxWidth> out;
    out.reserve(widths.size());
    const auto n = static_cast<float>(nbins);
    for (const auto width : widths) {
        const auto w  = static_cast<float>(width);
        const float h = std::sqrt((n - w) / (n * w));
        const float b = w * h / (n - w);
        out.push_back(BoxWidth{static_cast<uint32_t>(width), h + b, b});
    }
    return out;
}

// Seed salt of the pilot simulation of the "pilot" beam guess.
constexpr uint64_t kPilotSalt = 0x9110CA11ULL;
// Seed salt of the shared H1 simulation of the "shared" mode, and its size
// in units of ntrials.
constexpr uint64_t kSharedSalt       = 0x5BA7ED11ULL;
constexpr SizeType kSharedH1PerTrial = 16;
// Fold buffers of one chunk of the H1 simulation (simulate_h1_scores).
constexpr SizeType kH1ChunkBytes = SizeType{64} << 20U;

struct ModuloFunctor {
    uint32_t mod;
    __device__ uint32_t operator()(uint32_t i) const { return i % mod; }
};

// Index of the highest score among scores[0..n) (lowest index on ties, as
// std::max_element), broadcast to the whole block.
template <uint32_t kBlock>
__device__ uint32_t block_argmax(const float* __restrict__ scores, uint32_t n) {
    using BlockReduce = cub::BlockReduce<unsigned long long, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ unsigned long long best_s;
    unsigned long long best = 0;
    for (uint32_t t = threadIdx.x; t < n; t += kBlock) {
        best = max(best, score_trial_key(scores[t], t));
    }
    best = BlockReduce(temp_storage)
               .Reduce(best, cub_helpers::CubMaxOp<unsigned long long>{});
    if (threadIdx.x == 0) {
        best_s = best;
    }
    __syncthreads();
    return trial_of_key(best_s);
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
    uint32_t running            = 0;
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
    float nbranches,
    const uint32_t* __restrict__ h1_mask_par,
    const uint32_t* __restrict__ h1_pass,
    uint32_t h1_words,
    float cost_to_go,
    uint32_t min_support) {
    // Shared mode (h1_mask_par set): H1 survivors are the parent's shared
    // trials that also pass this threshold, counted from bitmasks.
    const uint32_t item          = blockIdx.x;
    const ThresholdPairItem pair = pairs[item / nprobs];
    const uint32_t kprob         = item % nprobs;
    const uint32_t cell          = (pair.jslot_prev * nprobs) + kprob;
    if (!parent_is_valid(prev_states, ntrials_par, pair.jthres_abs, cell,
                         nprobs, kprob)) {
        return; // uniform across the block
    }
    const float threshold = thresholds[pair.ithres_abs];
    // Pool rows per cell: H0 and H1, or H0 only in shared mode
    const uint64_t rows                 = h1_mask_par == nullptr ? 2ULL : 1ULL;
    const float* __restrict__ scores_h0 = scores_cur + (rows * cell * ntrials);
    const float* __restrict__ scores_h1 = scores_h0 + ntrials;
    // Both counts in one 64-bit sum: low = H0, high = H1.
    unsigned long long packed = 0;
    if (h1_mask_par == nullptr) {
        for (uint32_t t = threadIdx.x; t < ntrials; t += kBlock) {
            packed += (scores_h0[t] > threshold ? 1ULL : 0ULL) +
                      (scores_h1[t] > threshold ? (1ULL << 32U) : 0ULL);
        }
    } else {
        for (uint32_t t = threadIdx.x; t < ntrials; t += kBlock) {
            packed += scores_h0[t] > threshold ? 1ULL : 0ULL;
        }
        const uint32_t* mask =
            h1_mask_par + (static_cast<uint64_t>(cell) * h1_words);
        const uint32_t* pass =
            h1_pass + (static_cast<uint64_t>(pair.islot_cur) * h1_words);
        for (uint32_t w = threadIdx.x; w < h1_words; w += kBlock) {
            packed += static_cast<unsigned long long>(__popc(mask[w] & pass[w]))
                      << 32U;
        }
    }
    using BlockReduce = cub::BlockReduce<unsigned long long, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    packed = BlockReduce(temp_storage).Sum(packed);
    if (threadIdx.x != 0) {
        return;
    }
    // No H0 survivor: the best H0 trial stands in, counted as one (the
    // commit keeps it).
    const auto count_h0 =
        max(static_cast<uint32_t>(packed & 0xFFFFFFFFULL), 1U);
    const auto count_h1   = static_cast<uint32_t>(packed >> 32U);
    counts[item]          = make_uint2(count_h0, count_h1);
    const float ntrials_f = static_cast<float>(ntrials);
    // H1 survival is conditional on the parent's survivors: all ntrials
    // resampled trials, or the parent's shared trials in shared mode.
    const float ntrials_h1 =
        h1_mask_par == nullptr
            ? ntrials_f
            : static_cast<float>(ntrials_par[(2 * cell) + 1]);
    // The committed state carries the noise survival with a floor of
    // min_support survivors (a regularised search model: a transition that
    // few of the noise trials survive is mostly luck, and a deep search
    // would exploit it). counts[] and the survivor maps keep the actual
    // observations. min_support is 1 outside mode shared, where it has no
    // effect.
    const auto state_next =
        prev_states[(pair.jthres_abs * nprobs) + kprob].gen_next(
            threshold,
            __fdiv_rn(static_cast<float>(max(count_h0, min_support)),
                      ntrials_f),
            __fdiv_rn(static_cast<float>(count_h1), ntrials_h1), nbranches);
    cand_states[item] = state_next;
    const int iprob =
        find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);
    if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
        // Rank by cost so far, plus the estimated cost to go of the noise
        // survivors in the second pass of the "window" beam. Shared mode
        // ranks the cost per detection: ranking by cost alone favours
        // candidates at the bottom of their Pd bin.
        const float predicted =
            cost_to_go > 0.0F
                ? __fadd_rn(state_next.complexity_cumul,
                            __fmul_rn(cost_to_go, state_next.complexity))
                : state_next.complexity_cumul;
        const float rank =
            h1_mask_par != nullptr
                ? __fdiv_rn(predicted, state_next.success_h1_cumul)
                : predicted;
        atomicMin(&cell_keys[(pair.ithres_abs * nprobs) + iprob],
                  transition_key(rank, item));
    }
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
    uint32_t nprobs,
    const uint32_t* __restrict__ h1_mask_par,
    const uint32_t* __restrict__ h1_pass,
    uint32_t* __restrict__ h1_mask_next,
    uint32_t h1_words) {
    const uint32_t out_cell      = blockIdx.x; // islot * nprobs + iprob
    const uint32_t ithres        = beam_cur[out_cell / nprobs];
    const uint32_t key_idx       = (ithres * nprobs) + (out_cell % nprobs);
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
    const uint32_t par_cell      = (pair.jslot_prev * nprobs) + (item % nprobs);
    const float threshold        = thresholds[ithres];
    if (threadIdx.x == 0) {
        const uint2 cnt                  = counts[item];
        states_cur[key_idx]              = cand_states[item];
        ntrials_next[2 * out_cell]       = cnt.x;
        ntrials_next[(2 * out_cell) + 1] = cnt.y;
        src_next[out_cell]               = par_cell;
    }

    // Shared mode: the surviving shared H1 trials, as a bitmask
    const bool shared = h1_mask_par != nullptr;
    if (shared) {
        const uint32_t* mask =
            h1_mask_par + (static_cast<uint64_t>(par_cell) * h1_words);
        const uint32_t* pass =
            h1_pass + (static_cast<uint64_t>(out_cell / nprobs) * h1_words);
        uint32_t* out =
            h1_mask_next + (static_cast<uint64_t>(out_cell) * h1_words);
        for (uint32_t w = threadIdx.x; w < h1_words; w += kBlock) {
            out[w] = mask[w] & pass[w];
        }
    }

    // Ordered compaction of the surviving trial indices.
    using BlockScan = cub::BlockScan<uint32_t, kBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    for (uint32_t branch = 0; branch < (shared ? 1U : 2U); ++branch) {
        // Pool rows per cell: H0 and H1, or H0 only in shared mode
        const uint64_t rows = shared ? 1ULL : 2ULL;
        const float* __restrict__ scores =
            scores_cur + (((rows * par_cell) + branch) * ntrials);
        uint32_t* __restrict__ idx_out =
            idx_next + (((rows * out_cell) + branch) * ntrials);
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
        if (branch == 0 && running == 0) {
            const uint32_t best = block_argmax<kBlock>(scores, ntrials);
            if (threadIdx.x == 0) {
                idx_out[0] = best;
            }
        }
    }
}

// Shared mode: bit t of word w of row islot is set when shared H1 trial
// 32 * w + t scores above the threshold of beam slot islot at this stage.
// Padding trials (beyond nh1) never pass.
__global__ void h1_pass_kernel(const float* __restrict__ h1_scores_stage,
                               const uint32_t* __restrict__ beam_cur,
                               uint32_t n_beam_cur,
                               const float* __restrict__ thresholds,
                               uint32_t nh1,
                               uint32_t h1_words,
                               uint32_t* __restrict__ h1_pass) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_beam_cur * h1_words) {
        return;
    }
    const uint32_t islot  = tid / h1_words;
    const uint32_t w      = tid % h1_words;
    const float threshold = thresholds[beam_cur[islot]];
    uint32_t bits         = 0;
    for (uint32_t b = 0; b < 32; ++b) {
        const uint32_t t = (w * 32) + b;
        if (t < nh1 && h1_scores_stage[t] > threshold) {
            bits |= 1U << b;
        }
    }
    h1_pass[tid] = bits;
}

// Initial folds for evaluate(): one non-overlapping subsequence per
// (branch, trial). Same arithmetic as simulate_folds_init_kernel.
__global__ void
simulate_folds_init_eval_kernel(float* __restrict__ folds_sim,
                                const float* __restrict__ profile,
                                uint32_t nbins_padded,
                                float bias_snr,
                                float var_add,
                                uint64_t seed,
                                uint32_t ntrials,
                                uint32_t trial_offset,
                                uint32_t nbranches) {
    const uint32_t tid          = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_trials = nbranches * ntrials;
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
        seed, eval_subseq(0U, 0U, 0U, branch, trial_id + trial_offset), 0);
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

// evaluate()'s survivor compaction, device-wide and ordered: survivor indices
// come from cub::DeviceSelect::If, the kernels below gather the folds and
// apply the H0 floor, and the counts stay on the device until the last stage.
// Bit-identical to a one-block compaction with a host sync per stage (which
// took 8.5 s per 2^20-trial path at the production pattern, 0.32 s here).
struct AboveThreshold {
    const float* __restrict__ scores;
    float threshold;
    __device__ bool operator()(uint32_t t) const {
        return scores[t] > threshold;
    }
};

// One thread per (branch, output row): copies survivor idx[branch][i] to
// row i of the branch.
__global__ void evaluate_gather_kernel(const float* __restrict__ folds,
                                       const uint32_t* __restrict__ idx,
                                       const uint32_t* __restrict__ n_sel,
                                       float* __restrict__ compacted,
                                       uint32_t ntrials,
                                       uint32_t nbins_padded,
                                       uint32_t nbranches) {
    const uint32_t tid    = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t branch = tid / ntrials;
    const uint32_t i      = tid % ntrials;
    if (branch >= nbranches || i >= n_sel[branch]) {
        return;
    }
    const uint64_t base = static_cast<uint64_t>(branch) * ntrials;
    const auto* src     = reinterpret_cast<const float4*>(
        folds + ((base + idx[base + i]) * nbins_padded));
    auto* dst =
        reinterpret_cast<float4*>(compacted + ((base + i) * nbins_padded));
    for (uint32_t j = 0; j < nbins_padded / 4; ++j) {
        dst[j] = src[j];
    }
}

// Block = branch; n_sel[branch] survivors were selected. No H0 survivor: the
// best noise trial is kept in row 0, counted as one.
template <uint32_t kBlock>
__global__ void evaluate_finalize_kernel(const float* __restrict__ folds,
                                         const float* __restrict__ scores,
                                         const uint32_t* __restrict__ n_sel,
                                         float* __restrict__ compacted,
                                         uint32_t* __restrict__ n_out,
                                         uint32_t ntrials,
                                         uint32_t nbins_padded) {
    const uint32_t branch = blockIdx.x;
    uint32_t n            = n_sel[branch];
    if (branch == 0 && n == 0) {
        const uint32_t best = block_argmax<kBlock>(scores, ntrials);
        const float* src = folds + (static_cast<uint64_t>(best) * nbins_padded);
        for (uint32_t j = threadIdx.x; j < nbins_padded; j += kBlock) {
            compacted[j] = src[j];
        }
        n = 1;
    }
    if (threadIdx.x == 0) {
        n_out[branch] = n;
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
    unsigned long long* __restrict__ best_h0,
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
            best_h0[item]      = 0ULL;
        }
        running += total;
        __syncthreads(); // temp_storage reuse
    }
    if (threadIdx.x == 0) {
        advance_rng(rng_state, running, rng_round_to, ntrials);
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
    // No H0 survivor: counted as one (commit_items keeps the best trial)
    const uint32_t count_h0 = max(cnt.x, 1U);
    const auto state_next =
        prev_states[(pair.jthres_abs * nprobs) + kprob].gen_next(
            thresholds[pair.ithres_abs],
            __fdiv_rn(static_cast<float>(count_h0), ntrials_f),
            __fdiv_rn(static_cast<float>(cnt.y), ntrials_f), nbranches);
    cand_states[item] = state_next;
    const int iprob =
        find_bin_index_device(probs, nprobs, state_next.success_h1_cumul);
    if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
        atomicMin(&cell_keys[(pair.ithres_abs * nprobs) + iprob],
                  transition_key(state_next.complexity_cumul, item));
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

// Signal bank of solve(): per trial, min over stages of (score - threshold),
// the same float arithmetic as detail::HostSignalBank.
__global__ void bank_margins_kernel(const float* __restrict__ scores,
                                    const float* __restrict__ thresholds,
                                    uint32_t nstages,
                                    uint32_t n,
                                    float* __restrict__ margins) {
    const uint32_t t = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (t >= n) {
        return;
    }
    float m = FLT_MAX;
    for (uint32_t s = 0; s < nstages; ++s) {
        m = fminf(m,
                  scores[(static_cast<SizeType>(s) * n) + t] - thresholds[s]);
    }
    margins[t] = m;
}

struct FloatMin {
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
};

// Order-preserving map of floats to ints (an involution), for atomicMin.
__device__ __forceinline__ int ordered_float_key(int bits) {
    return bits >= 0 ? bits : bits ^ 0x7FFFFFFF;
}

// Per stage (blockIdx.y), the lowest score of the trials with margin > 0.
template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void bank_minima_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ margins,
    uint32_t n,
    int* __restrict__ minima_keys) {
    using BlockReduce = cub::BlockReduce<float, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp;
    const uint32_t s = blockIdx.y;
    float m          = FLT_MAX;
    for (uint32_t t = (blockIdx.x * kBlock) + threadIdx.x; t < n;
         t += gridDim.x * kBlock) {
        if (margins[t] > 0.0F) {
            m = fminf(m, scores[(static_cast<SizeType>(s) * n) + t]);
        }
    }
    const float block_min = BlockReduce(temp).Reduce(m, FloatMin{});
    if (threadIdx.x == 0) {
        atomicMin(&minima_keys[s],
                  ordered_float_key(__float_as_int(block_min)));
    }
}

// SignalBank kept on the device: only margins (n floats) and minima
// (nstages floats) go to the host.
class DeviceSignalBank final : public detail::SignalBank {
public:
    DeviceSignalBank(int device_id, SizeType nstages, SizeType n)
        : m_device_id(device_id),
          m_nstages(nstages),
          m_n(n) {}

    SizeType size() const override { return m_n; }

    std::vector<float>
    margins(std::span<const float> thresholds) const override {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        compute_margins(thresholds, stream);
        std::vector<float> out(m_n);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                out.data(), thrust::raw_pointer_cast(m_margins_d.data()),
                m_n * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        return out;
    }

    std::vector<float>
    survivor_minima(std::span<const float> thresholds) const override {
        constexpr uint32_t kBlock = 256;
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        compute_margins(thresholds, stream);
        thrust::device_vector<int> keys(
            m_nstages, std::bit_cast<int>(std::numeric_limits<float>::max()));
        const auto blocks = static_cast<uint32_t>(
            std::min<SizeType>((m_n + kBlock - 1) / kBlock, 64));
        const dim3 grid(blocks, static_cast<uint32_t>(m_nstages));
        bank_minima_kernel<kBlock><<<grid, kBlock, 0, stream>>>(
            scores(), thrust::raw_pointer_cast(m_margins_d.data()),
            static_cast<uint32_t>(m_n), thrust::raw_pointer_cast(keys.data()));
        cuda_utils::check_last_cuda_error("bank_minima_kernel");
        std::vector<int> host(m_nstages);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(host.data(), thrust::raw_pointer_cast(keys.data()),
                            m_nstages * sizeof(int), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        std::vector<float> out(m_nstages);
        std::ranges::transform(host, out.begin(), [](int key) {
            return std::bit_cast<float>(key >= 0 ? key : key ^ 0x7FFFFFFF);
        });
        return out;
    }

    /// The score matrix of simulate_h1_scores_device (one leading padding
    /// row: stage s at (s + 1) * n).
    thrust::device_vector<float>& matrix() { return m_matrix; }

private:
    const float* scores() const {
        return thrust::raw_pointer_cast(m_matrix.data()) + m_n;
    }

    void compute_margins(std::span<const float> thresholds,
                         cudaStream_t stream) const {
        if (thresholds.size() != m_nstages) {
            throw std::invalid_argument(
                "DeviceSignalBank: one threshold per stage");
        }
        constexpr uint32_t kBlock = 256;
        m_thresholds_d.assign(thresholds.begin(), thresholds.end());
        m_margins_d.resize(m_n);
        const auto blocks = static_cast<uint32_t>((m_n + kBlock - 1) / kBlock);
        bank_margins_kernel<<<blocks, kBlock, 0, stream>>>(
            scores(), thrust::raw_pointer_cast(m_thresholds_d.data()),
            static_cast<uint32_t>(m_nstages), static_cast<uint32_t>(m_n),
            thrust::raw_pointer_cast(m_margins_d.data()));
        cuda_utils::check_last_cuda_error("bank_margins_kernel");
    }

    int m_device_id;
    SizeType m_nstages;
    SizeType m_n;
    thrust::device_vector<float> m_matrix;
    mutable thrust::device_vector<float> m_thresholds_d;
    mutable thrust::device_vector<float> m_margins_d;
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
                       std::optional<uint64_t> seed,
                       std::string_view guess,
                       float min_pd,
                       float w_terminal,
                       float beam_sigma,
                       float reach_sigma)
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
          m_nbins(nbins),
          m_guess(guess) {
        const auto beam_guess = detail::beam_guess_from_string(guess);
        const bool shared     = is_shared_family(m_mode);
        validate_inputs(nprobs, shared ? 0.5F : prob_min, snr_final,
                        nthresholds);
        if (shared) {
            // The grid is anchored at the target: nprobs is the number of
            // bins from min_pd to 1 (prob_min is unused). The terminal
            // weight is a property of this mode only.
            if (!(min_pd > 0.0F && min_pd < 1.0F)) {
                throw std::invalid_argument(std::format(
                    "ThresholdsCuda: min_pd={} must be in (0, 1)", min_pd));
            }
            if (nprobs < 2) {
                throw std::invalid_argument(
                    "ThresholdsCuda: shared mode needs nprobs >= 2");
            }
            if (!(w_terminal >= 0.0F) || !is_finite(w_terminal)) {
                throw std::invalid_argument(
                    "ThresholdsCuda: w_terminal must be finite and >= 0");
            }
            m_min_pd = min_pd;
            m_w      = static_cast<double>(w_terminal);
        }
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);

        // Host-side setup
        m_nbins_padded = (m_nbins + 3) & ~SizeType{3}; // multiple of 4
        m_profile.assign(m_nbins_padded, 0.0F);
        simulation::generate_folded_profile(m_profile, m_nbins, m_ref_ducy);
        m_thresholds = detail::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs      = shared ? detail::shared_probability_grid(min_pd, nprobs)
                              : detail::compute_probs(nprobs, prob_min);
        m_nprobs     = m_probs.size();
        m_nstages    = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        m_box_score_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        if (*std::ranges::max_element(m_box_score_widths) >= m_nbins) {
            throw std::invalid_argument(
                std::format("ducy_max={} gives a box width >= nbins={}",
                            m_ducy_max, m_nbins));
        }
        m_bias_snr = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_nh1      = std::min<SizeType>(kSharedH1PerTrial * m_ntrials,
                                        SizeType{1} << 20U);
        m_h1_words = (m_nh1 + 31) / 32;

        // Device constants
        m_thresholds_d       = m_thresholds;
        m_profile_d          = m_profile;
        m_probs_d            = m_probs;
        const auto boxes     = make_box_widths(m_box_score_widths, m_nbins);
        m_box_score_widths_d = boxes;
        std::vector<BoxWidth> by_len(m_nbins + 1, BoxWidth{0, 0.0F, 0.0F});
        for (const auto& box : boxes) {
            by_len[box.w] = box;
        }
        m_box_by_len_d = by_len;

        if (beam_guess == detail::BeamGuess::kWindow || shared) {
            // Data-driven beam from an unpruned signal simulation: the
            // feasibility window (improved "window": the whole window and
            // every transition), or in the shared modes the guided beam
            // around the pilot guess with reach-limited transitions.
            const auto stats = detail::pilot_stats(
                simulate_h1_scores_sorted(
                    detail::kWindowTrials,
                    detail::derive_seed(m_seed, kPilotSalt)),
                detail::kWindowTrials, m_branching_pattern, m_trials_start,
                m_probs.front());
            m_window          = stats.window;
            m_all_transitions = true;
            if (shared) {
                if (!(beam_sigma > 0.0F) || !(reach_sigma > 0.0F)) {
                    throw std::invalid_argument(
                        "ThresholdsCuda: beam_sigma and reach_sigma must be "
                        "positive");
                }
                m_guide_width = beam_sigma;
                m_guide_reach = reach_sigma;
            }
            if (m_guide_width > 0.0F) {
                m_score_sigma = stats.sigma;
                m_guess_pilot = stats.guess;
                m_guess_path  = m_guess_pilot;
                m_beam = detail::guided_beam(m_window, m_guess_path,
                                             m_score_sigma, m_guide_width);
            } else {
                m_beam = m_window;
                m_guess_path.resize(m_nstages);
                std::ranges::transform(
                    m_beam.lo, m_beam.hi, m_guess_path.begin(),
                    [](float lo, float hi) { return 0.5F * (lo + hi); });
            }
        } else {
            if (beam_guess == detail::BeamGuess::kPilot) {
                m_guess_path = detail::pilot_guess_scheme(
                    simulate_h1_scores(detail::kPilotGuessTrials,
                                       detail::derive_seed(m_seed, kPilotSalt)),
                    detail::kPilotGuessTrials, m_branching_pattern,
                    m_trials_start);
            } else {
                m_guess_path = detail::guess_scheme(
                    m_nstages, snr_final, m_branching_pattern, m_trials_start);
            }
            m_beam = detail::centred_beam(m_guess_path, m_beam_width,
                                          m_thresholds.back());
        }
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
        if (!m_score_sigma.empty()) {
            // Any recentred beam (mode robust's second pass) fits the pools.
            max_beam = std::max(max_beam, detail::guided_beam_capacity(
                                              m_thresholds, m_window,
                                              m_score_sigma, m_guide_width));
        }

        // Fold pools: one cell per (beam slot, prob bin), H0 + H1 rows each.
        // Only run() needs them, so they are allocated there: evaluate() and
        // a scheme that is never run hold no pools.
        m_max_beam           = max_beam;
        const SizeType cells = max_beam * m_nprobs;
        if (cells * 2 > std::numeric_limits<uint32_t>::max() ||
            m_ntrials > std::numeric_limits<uint32_t>::max() / 2 ||
            m_nthresholds * m_nprobs > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(
                "ThresholdsCuda: problem size exceeds 32-bit "
                "cell indexing");
        }
        m_states.resize(m_nstages * m_nthresholds * m_nprobs, State{});
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
        allocate_pools();
        const CudaStream stream;
        constexpr float kVarInit = 1.0F;
        constexpr float kVarAdd  = 1.0F;

        const SizeType neigh = m_all_transitions ? m_nthresholds : thres_neigh;
        // "window" beam: a first pass ranks cell candidates by cost so far;
        // the second (modes improved and robust) ranks them by cost so far
        // plus the first pass's cost to go of their noise survivors
        // (detail::cost_to_go_scheme), and in mode robust searches around
        // the guess recentred on the first pass's path.
        m_cost_to_go.clear();
        m_h1_bank_valid = false;
        m_h1_bank_host.clear();
        if (!m_score_sigma.empty()) {
            m_guess_path = m_guess_pilot;
            m_beam = detail::guided_beam(m_window, m_guess_path, m_score_sigma,
                                         m_guide_width);
        }
        run_pass(neigh, kVarInit, kVarAdd, stream);
        if (m_mode == DynamicThresholdMode::kRobust) {
            m_first_states = m_states;
        }
        // Mode shared stops after one pass.
        if (m_all_transitions && (m_mode == DynamicThresholdMode::kImproved ||
                                  m_mode == DynamicThresholdMode::kRobust)) {
            m_cost_to_go = detail::cost_to_go_scheme(
                m_states, m_thresholds, m_probs, m_nstages, m_nthresholds,
                m_nprobs, 8.0F, 4, m_w);
            if (!m_cost_to_go.empty()) {
                if (!m_score_sigma.empty()) {
                    m_guess_path = detail::recentre_guess(
                        m_guess_pilot, get_best_path_thresholds(m_min_pd));
                    m_beam = detail::guided_beam(m_window, m_guess_path,
                                                 m_score_sigma, m_guide_width);
                }
                run_pass(neigh, kVarInit, kVarAdd, stream);
            }
        }
        if (is_shared_family(m_mode)) {
            detail::store_terminal_cost(m_states, m_nstages, m_nthresholds,
                                        m_nprobs, m_w);
        }
        m_thres_neigh = thres_neigh;
        warn_empty_stages();
    }

    // One forward pass into m_states, from a fresh grid.
    void run_pass(SizeType neigh,
                  float var_init,
                  float var_add,
                  cudaStream_t stream) {
        thrust::fill(thrust::cuda::par.on(stream), m_states_d.begin(),
                     m_states_d.end(), State{});
        thrust::fill(thrust::cuda::par.on(stream), m_cell_keys_d.begin(),
                     m_cell_keys_d.end(), kEmptyKey);
        for (auto& v : m_ntrials_d) {
            thrust::fill(thrust::cuda::par.on(stream), v.begin(), v.end(), 0U);
        }
        if (m_mode != DynamicThresholdMode::kLegacy) {
            run_improved(neigh, var_init, var_add, stream);
        } else {
            run_legacy(neigh, var_init, var_add, stream);
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(m_states.data(),
                            thrust::raw_pointer_cast(m_states_d.data()),
                            m_states.size() * sizeof(State),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
    }

    /// Fold rows of each pool: H0 and H1 per (beam slot, prob bin) cell, or
    /// H0 only in shared mode (its H1 lives in bitmasks). At least the two
    /// rows of the initial virtual parent.
    SizeType pool_rows_count() const {
        const SizeType rows_per_cell = is_shared_family(m_mode) ? 1 : 2;
        return std::max<SizeType>(m_max_beam * m_nprobs * rows_per_cell, 2) *
               m_ntrials;
    }

    /// Device bytes of the pools and state grid that run() allocates.
    SizeType run_memory_bytes() const {
        const SizeType pool_rows = pool_rows_count();
        const bool parent_map    = m_mode != DynamicThresholdMode::kLegacy;
        // Shared mode: the H1 score matrix, its chunked simulation's fold
        // buffers, and the survivor and pass bitmasks.
        const SizeType shared_bytes =
            is_shared_family(m_mode)
                ? ((m_nstages + 1) * m_nh1 * sizeof(float)) +
                      (4 * h1_chunk_trials(m_nh1) * m_nbins_padded *
                       sizeof(float)) +
                      (((2 * m_max_beam * m_nprobs) + m_max_beam) * m_h1_words *
                       sizeof(uint32_t))
                : 0;
        return (2 * pool_rows * m_nbins_padded * sizeof(float)) +
               (parent_map ? pool_rows * (2 * sizeof(uint32_t) + sizeof(float))
                           : 0) +
               (m_states.size() * sizeof(State)) + shared_bytes;
    }

    std::vector<State> get_states() const { return m_states; }
    std::vector<float> get_thresholds() const { return m_thresholds; }
    std::vector<float> get_probs() const { return m_probs; }

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const {
        return detail::beam_indices(m_thresholds, m_beam.lo[istage],
                                    m_beam.hi[istage]);
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

    /// Guided beam: a transition may change the offset from the guess by at
    /// most m_guide_reach sigma_s (0: any transition).
    bool within_reach(SizeType istage, SizeType ithres, SizeType jthres) const {
        if (m_score_sigma.empty() || m_guide_reach <= 0.0F) {
            return true;
        }
        return detail::within_reach(
            m_thresholds[ithres], m_guess_path[istage], m_thresholds[jthres],
            m_guess_path[istage - 1], m_score_sigma[istage], m_guide_reach);
    }

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
            plan.n_beam_prev     = static_cast<uint32_t>(beam_prev_idx.size());
            plan.beam_cur_offset = static_cast<uint32_t>(beams_flat.size());
            plan.n_beam_cur      = static_cast<uint32_t>(beam_cur_idx.size());
            plan.pair_offset     = static_cast<uint32_t>(pairs_flat.size());
            for (const auto ithres : beam_cur_idx) {
                beams_flat.push_back(static_cast<uint32_t>(ithres));
            }
            if (istage == 0) {
                for (SizeType islot = 0; islot < beam_cur_idx.size(); ++islot) {
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
                for (SizeType islot = 0; islot < beam_cur_idx.size(); ++islot) {
                    const auto ithres = beam_cur_idx[islot];
                    for (const SizeType jthres :
                         utils::find_neighbouring_indices(beam_prev_idx, ithres,
                                                          thres_neigh)) {
                        const int32_t jslot = prev_slot_of_thresh[jthres];
                        if (jslot < 0 ||
                            !within_reach(istage, ithres, jthres)) {
                            continue;
                        }
                        pairs_flat.push_back(
                            ThresholdPairItem{static_cast<uint32_t>(ithres),
                                              static_cast<uint32_t>(islot),
                                              static_cast<uint32_t>(jthres),
                                              static_cast<uint32_t>(jslot)});
                    }
                }
            }
            plan.n_pairs =
                static_cast<uint32_t>(pairs_flat.size()) - plan.pair_offset;
            const SizeType n_items =
                static_cast<SizeType>(plan.n_pairs) * m_nprobs;
            plans.max_items = std::max(plans.max_items, n_items);
            plans.max_parents =
                std::max(plans.max_parents,
                         static_cast<SizeType>(plan.n_beam_prev) * m_nprobs);
            if (with_order) {
                std::vector<uint32_t> order(n_items);
                std::iota(order.begin(), order.end(), 0U);
                const auto* pairs    = pairs_flat.data() + plan.pair_offset;
                const auto parent_of = [&](uint32_t item) {
                    return (pairs[item / m_nprobs].jslot_prev * m_nprobs) +
                           (item % m_nprobs);
                };
                std::ranges::stable_sort(order, {}, parent_of);
                order_flat.insert(order_flat.end(), order.begin(), order.end());
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
                        /*allow_reg=*/true, std::forward<F>(f));
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
            folds, thrust::raw_pointer_cast(m_profile_d.data()), m_nbins_padded,
            m_bias_snr, var_init, m_seed, 0, ntrials);
        cuda_utils::check_last_cuda_error("simulate_folds_init_kernel");
        const uint32_t init_desc[2]          = {ntrials, ntrials};
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

    /// Noise survivors a transition must show to count at full weight in the
    /// ranking surrogate: 8 of 1024 trials, the same fraction at any ntrials.
    static uint32_t shared_min_support(uint32_t ntrials) {
        return std::clamp<uint32_t>((ntrials + 64U) / 128U, 1U, ntrials);
    }

    /// Weight of a candidate's noise survivors in its rank at stage @p istage:
    /// the first pass's cost to go; in mode shared the last stage always
    /// weighs w (its survivors go on to the next processing step).
    float rank_weight(SizeType istage) const {
        if (is_shared_family(m_mode) && istage + 1 == m_nstages) {
            return static_cast<float>(m_w);
        }
        return m_cost_to_go.empty() ? 0.0F : m_cost_to_go[istage];
    }

    void run_improved(SizeType thres_neigh,
                      float var_init,
                      float var_add,
                      cudaStream_t stream) {
        constexpr uint32_t kCompactBlock = 1024;
        constexpr uint32_t kSimBlock     = 256;
        constexpr uint32_t kDecideBlock  = 256;
        constexpr uint32_t kCommitBlock  = 256;
        const auto ntrials               = static_cast<uint32_t>(m_ntrials);
        const auto nprobs                = static_cast<uint32_t>(m_nprobs);
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

        // Shared mode: one unpruned H1 simulation scores every stage; cells
        // carry bitmasks of the shared trials they keep.
        const bool shared      = is_shared_family(m_mode);
        const auto nh1         = static_cast<uint32_t>(m_nh1);
        const auto h1_words    = static_cast<uint32_t>(m_h1_words);
        const uint32_t nbranch = shared ? 1U : 2U;
        uint32_t* h1_mask[2]   = {nullptr, nullptr};
        uint32_t* h1_pass      = nullptr;
        if (shared) {
            // One simulation per run(): both passes count on the same trials.
            if (!m_h1_bank_valid) {
                simulate_h1_scores_device(
                    m_nh1, detail::derive_seed(m_seed, kSharedSalt), stream,
                    m_h1_scores_d, m_h1_folds_d);
                m_h1_bank_valid = true;
            }
            h1_mask[0] = thrust::raw_pointer_cast(m_h1_mask_d[0].data());
            h1_mask[1] = thrust::raw_pointer_cast(m_h1_mask_d[1].data());
            h1_pass    = thrust::raw_pointer_cast(m_h1_pass_d.data());
            // The virtual parent (cell 0 of the odd buffer) keeps them all
            std::vector<uint32_t> all(m_h1_words, 0xFFFFFFFFU);
            if (const uint32_t tail = nh1 % 32U; tail != 0) {
                all.back() = (1U << tail) - 1U;
            }
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(h1_mask[1], all.data(),
                                all.size() * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync failed");
        }

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
        float* scores         = thrust::raw_pointer_cast(m_scores_d.data());

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
        if (shared) {
            // H1 count of the virtual parent: every shared trial
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(ntrials_desc[1] + 1, &nh1, sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync failed");
        }

        // Minimum noise support of a transition in the ranking surrogate
        // (shared mode): see decide_parents_kernel. 1 elsewhere (no effect:
        // a transition always has one noise survivor).
        const uint32_t min_support = shared ? shared_min_support(ntrials) : 1U;

        float var_in = var_init;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto& plan     = plans.stages[istage];
            const uint32_t cur   = istage & 1U;
            const uint32_t prev  = cur ^ 1U;
            const auto nbranches = m_branching_pattern[istage];
            const State* prev_states =
                istage == 0 ? thrust::raw_pointer_cast(init_states_d.data())
                            : thrust::raw_pointer_cast(m_states_d.data()) +
                                  ((istage - 1) * grid_stride);
            State* cur_states = thrust::raw_pointer_cast(m_states_d.data()) +
                                (istage * grid_stride);
            const uint32_t* beams =
                thrust::raw_pointer_cast(plans.beams_d.data());
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

            if (shared && plan.n_beam_cur > 0) {
                const SizeType n_words =
                    static_cast<SizeType>(plan.n_beam_cur) * h1_words;
                const dim3 grid_dim((n_words + kSimBlock - 1) / kSimBlock);
                cuda_utils::check_kernel_launch_params(grid_dim,
                                                       dim3(kSimBlock));
                h1_pass_kernel<<<grid_dim, kSimBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(m_h1_scores_d.data()) +
                        ((istage + 1) * m_nh1),
                    beams + plan.beam_cur_offset, plan.n_beam_cur,
                    thrust::raw_pointer_cast(m_thresholds_d.data()), nh1,
                    h1_words, h1_pass);
                cuda_utils::check_last_cuda_error("h1_pass_kernel");
            }
            const SizeType sim_threads =
                static_cast<SizeType>(plan.n_beam_prev) * m_nprobs * nbranch *
                m_ntrials;
            if (sim_threads > 0) {
                const dim3 grid_dim((sim_threads + kSimBlock - 1) / kSimBlock);
                const dim3 block_dim(kSimBlock);
                cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
                with_scorer([&]<typename Scorer>() {
                    ScorerLaunch<Scorer>::simulate_score_parents(
                        grid_dim, block_dim, stream,
                        thrust::raw_pointer_cast(parents_d.data()),
                        thrust::raw_pointer_cast(n_parents_d.data()),
                        thrust::raw_pointer_cast(rng_state_d.data()), sim[prev],
                        ntrials_desc[prev], src_desc[prev], idx_desc[prev],
                        sim[cur], scores, trial_sim_params(var_in, var_add),
                        ntrials, nprobs, nbranch);
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
                        thrust::raw_pointer_cast(m_cell_keys_d.data()), ntrials,
                        nprobs, nbranches, h1_mask[prev], h1_pass, h1_words,
                        rank_weight(istage), min_support);
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
                        ntrials, nprobs, h1_mask[prev], h1_pass, h1_mask[cur],
                        h1_words);
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
        const auto ntrials               = static_cast<uint32_t>(m_ntrials);
        const auto nprobs                = static_cast<uint32_t>(m_nprobs);
        const auto grid_stride =
            static_cast<SizeType>(m_nthresholds) * m_nprobs;

        const auto plans = build_stage_plans(thres_neigh, true);
        thrust::device_vector<uint2> counts_d(plans.max_items);
        thrust::device_vector<State> cand_states_d(plans.max_items);
        thrust::device_vector<uint32_t> item_rng_idx_d(plans.max_items);
        thrust::device_vector<unsigned long long> best_h0_d(plans.max_items);
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
                istage == 0 ? thrust::raw_pointer_cast(init_states_d.data())
                            : thrust::raw_pointer_cast(m_states_d.data()) +
                                  ((istage - 1) * grid_stride);
            State* cur_states = thrust::raw_pointer_cast(m_states_d.data()) +
                                (istage * grid_stride);
            const uint32_t* beams =
                thrust::raw_pointer_cast(plans.beams_d.data());
            const ThresholdPairItem* pairs =
                thrust::raw_pointer_cast(plans.pairs_d.data()) +
                plan.pair_offset;
            const auto n_items = static_cast<uint32_t>(
                static_cast<SizeType>(plan.n_pairs) * m_nprobs);
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
                    thrust::raw_pointer_cast(best_h0_d.data()),
                    thrust::raw_pointer_cast(rng_state_d.data()),
                    istage == 0 ? 1U : static_cast<uint32_t>(m_batch_size),
                    ntrials);
            cuda_utils::check_last_cuda_error("compact_items_kernel");

            const SizeType sim_threads =
                static_cast<SizeType>(n_items) * 2 * m_ntrials;
            const dim3 sim_grid((sim_threads + kSimBlock - 1) / kSimBlock);
            cuda_utils::check_kernel_launch_params(sim_grid, dim3(kSimBlock));
            with_scorer([&]<typename Scorer>() {
                ScorerLaunch<Scorer>::simulate_count_items(
                    sim_grid, kSimBlock, stream,
                    thrust::raw_pointer_cast(plans.item_order_d.data()) +
                        (static_cast<SizeType>(plan.pair_offset) * m_nprobs),
                    n_items, pairs,
                    thrust::raw_pointer_cast(item_rng_idx_d.data()),
                    thrust::raw_pointer_cast(rng_state_d.data()), folds[prev],
                    ntrials_desc[prev],
                    thrust::raw_pointer_cast(m_thresholds_d.data()),
                    reinterpret_cast<uint32_t*>(
                        thrust::raw_pointer_cast(counts_d.data())),
                    thrust::raw_pointer_cast(best_h0_d.data()), params, ntrials,
                    nprobs);
            });
            cuda_utils::check_last_cuda_error("simulate_count_items_kernel");

            decide_items_kernel<<<(n_items + kDecideBlock - 1) / kDecideBlock,
                                  kDecideBlock, 0, stream>>>(
                pairs, n_items, thrust::raw_pointer_cast(item_rng_idx_d.data()),
                prev_states, thrust::raw_pointer_cast(counts_d.data()),
                thrust::raw_pointer_cast(m_thresholds_d.data()),
                thrust::raw_pointer_cast(m_probs_d.data()),
                thrust::raw_pointer_cast(cand_states_d.data()),
                thrust::raw_pointer_cast(m_cell_keys_d.data()), ntrials, nprobs,
                nbranches);
            cuda_utils::check_last_cuda_error("decide_items_kernel");

            const SizeType n_out_cells =
                static_cast<SizeType>(plan.n_beam_cur) * m_nprobs;
            if (n_out_cells > 0) {
                with_scorer([&]<typename Scorer>() {
                    static_assert(kCommitBlock ==
                                  ScorerLaunch<Scorer>::kCommitItemsBlock);
                    ScorerLaunch<Scorer>::commit_items(
                        n_out_cells, kCommitBlock, stream,
                        beams + plan.beam_cur_offset, pairs,
                        thrust::raw_pointer_cast(item_rng_idx_d.data()),
                        thrust::raw_pointer_cast(rng_state_d.data()),
                        thrust::raw_pointer_cast(counts_d.data()),
                        thrust::raw_pointer_cast(cand_states_d.data()),
                        cur_states,
                        thrust::raw_pointer_cast(m_cell_keys_d.data()),
                        folds[prev], ntrials_desc[prev],
                        thrust::raw_pointer_cast(m_thresholds_d.data()),
                        thrust::raw_pointer_cast(best_h0_d.data()), folds[cur],
                        ntrials_desc[cur],
                        thrust::raw_pointer_cast(error_flag_d.data()), params,
                        ntrials, nprobs);
                });
                cuda_utils::check_last_cuda_error("commit_items_kernel");
            }
        }
        const uint32_t error_flag = error_flag_d[0];
        if (error_flag != 0) {
            throw std::runtime_error(
                std::format("ThresholdsCuda: legacy survivor regeneration "
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
        file.createAttribute("guess", m_guess);
        file.createAttribute("seed", m_seed);
        file.createAttribute("batch_size", m_batch_size);
        file.createAttribute("thres_neigh", m_thres_neigh);
        if (is_shared_family(m_mode)) {
            file.createAttribute("min_pd", m_min_pd);
            file.createAttribute("w_terminal", m_w);
            file.createAttribute("beam_sigma", m_guide_width);
            file.createAttribute("reach_sigma", m_guide_reach);
        }

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
        file.createDataSet("beam_lo", m_beam.lo);
        file.createDataSet("beam_hi", m_beam.hi);
        if (!m_guess_pilot.empty()) {
            file.createDataSet("guess_path_pilot", m_guess_pilot);
        }
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

    uint64_t get_seed() const { return m_seed; }

    /// Mode shared: the path of the lowest C / Pd (w included) among the
    /// final cells that meet min_pd.
    detail::SearchPath selected_path(std::span<const State> states,
                                     float min_pd) const {
        return detail::search_path(
            states, m_thresholds, m_probs, m_nstages, m_nthresholds, m_nprobs,
            detail::best_terminal_cell(states, m_nstages, m_nthresholds,
                                       m_nprobs, min_pd, m_w),
            m_w);
    }

    detail::SearchPath first_pass_path(float min_pd) const {
        if (m_first_states.empty()) {
            throw std::logic_error(
                "ThresholdsCuda::first_pass_path: needs run() in mode "
                "'robust'");
        }
        return selected_path(m_first_states, min_pd);
    }

    /// The shared H1 trials of the last run(), stage-major (nstages x nh1).
    std::vector<float> shared_h1_bank() const {
        if (!m_h1_bank_host.empty()) {
            return m_h1_bank_host;
        }
        if (!is_shared_family(m_mode) || !m_h1_bank_valid) {
            throw std::logic_error(
                "ThresholdsCuda::shared_h1_bank: needs run() in mode "
                "'shared' or 'robust'");
        }
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        std::vector<float> out(m_nstages * m_nh1);
        // One leading padding row, then stage s at (s + 1) * nh1.
        cuda_utils::check_cuda_call(
            cudaMemcpy(out.data(),
                       thrust::raw_pointer_cast(m_h1_scores_d.data()) + m_nh1,
                       out.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy failed");
        return out;
    }

    /// Frees the pools allocate_pools() made; run() allocates them again.
    void release_search_memory() {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        // Mode robust's shortlist replays paths on the search's bank: keep it
        // on the host, so that solve() can run again.
        if (m_mode == DynamicThresholdMode::kRobust && m_h1_bank_valid) {
            m_h1_bank_host = shared_h1_bank();
        }
        const auto drop = [](auto& v) {
            using V = std::remove_reference_t<decltype(v)>;
            V().swap(v);
        };
        for (auto& v : m_folds_d) {
            drop(v);
        }
        for (auto& v : m_ntrials_d) {
            drop(v);
        }
        for (auto& v : m_src_d) {
            drop(v);
        }
        for (auto& v : m_idx_d) {
            drop(v);
        }
        for (auto& v : m_h1_mask_d) {
            drop(v);
        }
        for (auto& v : m_h1_folds_d) {
            drop(v);
        }
        drop(m_scores_d);
        drop(m_h1_pass_d);
        drop(m_h1_scores_d);
        drop(m_cell_keys_d);
        drop(m_states_d);
        m_h1_bank_valid = false;
    }

    std::vector<float> get_best_path_thresholds(float min_pd) const {
        if (is_shared_family(m_mode)) {
            auto path = selected_path(m_states, min_pd);
            if (!path.thresholds.empty()) {
                return std::move(path.thresholds);
            }
        }
        return detail::get_best_path_thresholds(
            std::span<const State>(m_states.data(), m_states.size()),
            m_thresholds, m_probs, m_nstages, m_nthresholds, m_nprobs, min_pd);
    }

    /// Fresh sequential Monte Carlo of one fixed schedule. With @p h0_only the
    /// H1 branch is skipped (success_h1 = 1 in the states); the H0 states are
    /// identical to the full evaluation's, whose branches use independent
    /// streams.
    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed,
                                bool h0_only) const {
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
        const uint32_t nbr       = h0_only ? 1U : 2U;
        const SizeType rows      = nbr * ntrials * m_nbins_padded;

        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        thrust::device_vector<float> buf_a(rows, 0.0F);
        thrust::device_vector<float> buf_b(rows, 0.0F);
        thrust::device_vector<float> scores(nbr * ntrials, 0.0F);
        thrust::device_vector<uint32_t> idx(nbr * ntrials);
        thrust::device_vector<uint32_t> n_sel(2, 0U);
        // counts[2 * (s + 1) + branch]: survivors after stage s; [0, 1]: start
        thrust::device_vector<uint32_t> counts(2 * (m_nstages + 1), ntrials_u);
        size_t temp_bytes = 0;
        cuda_utils::check_cuda_call(
            cub::DeviceSelect::If(
                nullptr, temp_bytes, thrust::counting_iterator<uint32_t>(0),
                thrust::raw_pointer_cast(idx.data()),
                thrust::raw_pointer_cast(n_sel.data()), ntrials_u,
                AboveThreshold{thrust::raw_pointer_cast(scores.data()), 0.0F},
                stream),
            "cub::DeviceSelect::If size query failed");
        thrust::device_vector<uint8_t> temp(temp_bytes);

        constexpr uint32_t kBlock = 256;
        {
            const dim3 grid((nbr * ntrials + kBlock - 1) / kBlock);
            cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
            // Row 0..ntrials-1 is H0; with h0_only the H1 rows are not made.
            simulate_folds_init_eval_kernel<<<grid, kBlock, 0, stream>>>(
                thrust::raw_pointer_cast(buf_a.data()),
                thrust::raw_pointer_cast(m_profile_d.data()), padded,
                m_bias_snr, 1.0F, eval_seed, ntrials_u, 0U, nbr);
            cuda_utils::check_last_cuda_error(
                "simulate_folds_init_eval_kernel");
        }

        float var_in            = 1.0F;
        constexpr float kVarAdd = 1.0F;
        float* cur_in           = thrust::raw_pointer_cast(buf_a.data());
        float* cur_out          = thrust::raw_pointer_cast(buf_b.data());
        const uint32_t nthreads = nbr * ntrials_u;
        const dim3 grid((nthreads + kBlock - 1) / kBlock);
        cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            TrialSimParams params = trial_sim_params(var_in, kVarAdd);
            params.seed           = eval_seed;
            uint32_t* n_in =
                thrust::raw_pointer_cast(counts.data()) + (2 * istage);
            uint32_t* n_out =
                thrust::raw_pointer_cast(counts.data()) + (2 * (istage + 1));
            with_scorer([&]<typename Scorer>() {
                ScorerLaunch<Scorer>::evaluate_stage(
                    grid, kBlock, stream, cur_in, n_in, cur_out,
                    thrust::raw_pointer_cast(scores.data()), params, ntrials_u,
                    static_cast<uint32_t>(istage), 0U, 0U, nthreads);
            });
            cuda_utils::check_last_cuda_error("evaluate_stage_kernel");
            for (uint32_t b = 0; b < nbr; ++b) {
                cuda_utils::check_cuda_call(
                    cub::DeviceSelect::If(
                        thrust::raw_pointer_cast(temp.data()), temp_bytes,
                        thrust::counting_iterator<uint32_t>(0),
                        thrust::raw_pointer_cast(idx.data()) + (b * ntrials),
                        thrust::raw_pointer_cast(n_sel.data()) + b, ntrials_u,
                        AboveThreshold{thrust::raw_pointer_cast(scores.data()) +
                                           (b * ntrials),
                                       thresholds[istage]},
                        stream),
                    "cub::DeviceSelect::If failed");
            }
            evaluate_gather_kernel<<<grid, kBlock, 0, stream>>>(
                cur_out, thrust::raw_pointer_cast(idx.data()),
                thrust::raw_pointer_cast(n_sel.data()), cur_in, ntrials_u,
                padded, nbr);
            cuda_utils::check_last_cuda_error("evaluate_gather_kernel");
            evaluate_finalize_kernel<kBlock><<<nbr, kBlock, 0, stream>>>(
                cur_out, thrust::raw_pointer_cast(scores.data()),
                thrust::raw_pointer_cast(n_sel.data()), cur_in, n_out,
                ntrials_u, padded);
            cuda_utils::check_last_cuda_error("evaluate_finalize_kernel");
            var_in += kVarAdd;
            // Compacted survivors were written into cur_in. The next stage
            // reads cur_in and writes cur_out, so the pointers stay as they
            // are.
        }
        std::vector<uint32_t> counts_h(counts.size());
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(counts_h.data(),
                            thrust::raw_pointer_cast(counts.data()),
                            counts_h.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");

        std::vector<State> states(m_nstages);
        State prev            = State::initial();
        const float ntrials_f = static_cast<float>(ntrials);
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const uint32_t c0 = counts_h[2 * (istage + 1)];
            const uint32_t c1 =
                h0_only ? ntrials_u : counts_h[(2 * (istage + 1)) + 1];
            const float success_h0 = static_cast<float>(c0) / ntrials_f;
            const float success_h1 = static_cast<float>(c1) / ntrials_f;
            State next = prev.gen_next(thresholds[istage], success_h0,
                                       success_h1, m_branching_pattern[istage]);
            states[istage] = next;
            prev           = next;
            if (c1 == 0) {
                break; // later stages stay empty
            }
        }
        return states;
    }

    /// evaluate()'s H1 trials for @p seed with nothing pruned: the same
    /// streams and arithmetic, H1 branch only, no compaction and no host
    /// syncs between stages. Stage-major (nstages x ntrials).
    std::vector<float> simulate_h1_scores(SizeType ntrials,
                                          uint64_t seed) const {
        if (ntrials == 0 || ntrials > (1U << 20U)) {
            throw std::invalid_argument(
                "ThresholdsCuda::simulate_h1_scores: ntrials must be in "
                "[1, 2^20]");
        }
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        thrust::device_vector<float> matrix;
        std::array<thrust::device_vector<float>, 2> folds;
        simulate_h1_scores_device(ntrials, seed, stream, matrix, folds);
        std::vector<float> h1_scores(m_nstages * ntrials);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(h1_scores.data(),
                            thrust::raw_pointer_cast(matrix.data()) + ntrials,
                            h1_scores.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        return h1_scores;
    }

    /// A signal bank of simulate_h1_scores(n, seed), kept on the device.
    std::unique_ptr<detail::SignalBank> make_signal_bank(SizeType n,
                                                         uint64_t seed) const {
        if (n == 0 || n > (1U << 20U)) {
            throw std::invalid_argument(
                "ThresholdsCuda::make_signal_bank: n must be in [1, 2^20]");
        }
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        auto bank =
            std::make_unique<DeviceSignalBank>(m_device_id, m_nstages, n);
        const CudaStream stream;
        std::array<thrust::device_vector<float>, 2> folds;
        simulate_h1_scores_device(n, seed, stream, bank->matrix(), folds);
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        return bank;
    }

    /// simulate_h1_scores with every stage's row sorted ascending (on the
    /// device).
    std::vector<float> simulate_h1_scores_sorted(SizeType ntrials,
                                                 uint64_t seed) const {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const CudaStream stream;
        thrust::device_vector<float> matrix;
        std::array<thrust::device_vector<float>, 2> folds;
        simulate_h1_scores_device(ntrials, seed, stream, matrix, folds);
        const auto items = static_cast<int64_t>(m_nstages * ntrials);
        thrust::device_vector<float> sorted(m_nstages * ntrials);
        thrust::device_vector<int64_t> offsets(m_nstages + 1);
        thrust::sequence(thrust::cuda::par.on(stream), offsets.begin(),
                         offsets.end(), int64_t{0},
                         static_cast<int64_t>(ntrials));
        const float* keys_in =
            thrust::raw_pointer_cast(matrix.data()) + ntrials;
        float* keys_out    = thrust::raw_pointer_cast(sorted.data());
        const int64_t* off = thrust::raw_pointer_cast(offsets.data());
        const auto nseg    = static_cast<int64_t>(m_nstages);
        size_t temp_bytes  = 0;
        cuda_utils::check_cuda_call(
            cub::DeviceSegmentedRadixSort::SortKeys(
                nullptr, temp_bytes, keys_in, keys_out, items, nseg, off,
                off + 1, 0, static_cast<int>(sizeof(float) * 8), stream),
            "SegmentedRadixSort failed");
        thrust::device_vector<uint8_t> temp(temp_bytes);
        cuda_utils::check_cuda_call(
            cub::DeviceSegmentedRadixSort::SortKeys(
                thrust::raw_pointer_cast(temp.data()), temp_bytes, keys_in,
                keys_out, items, nseg, off, off + 1, 0,
                static_cast<int>(sizeof(float) * 8), stream),
            "SegmentedRadixSort failed");
        std::vector<float> h1_scores(m_nstages * ntrials);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(h1_scores.data(), keys_out,
                            h1_scores.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        return h1_scores;
    }

    /// Trials per chunk of the H1 simulation: its two fold buffers (both
    /// branches) stay within kH1ChunkBytes.
    SizeType h1_chunk_trials(SizeType ntrials) const {
        const SizeType per_trial = 4 * m_nbins_padded * sizeof(float);
        const SizeType fit =
            std::bit_floor(std::max<SizeType>(kH1ChunkBytes / per_trial, 256));
        return std::min(ntrials, fit);
    }

    /// Device H1 score matrix of simulate_h1_scores, with one leading row of
    /// padding: stage s is at (s + 1) * ntrials. Enqueued on @p stream.
    /// @p matrix and @p folds are resized as needed and reused.
    void simulate_h1_scores_device(
        SizeType ntrials,
        uint64_t seed,
        cudaStream_t stream,
        thrust::device_vector<float>& matrix,
        std::array<thrust::device_vector<float>, 2>& folds) const {
        // Chunks of trials bound the fold buffers; every trial keeps the
        // random streams of its global index, so the chunk size does not
        // change the scores.
        const SizeType chunk = h1_chunk_trials(ntrials);
        const auto chunk_u   = static_cast<uint32_t>(chunk);
        const auto padded    = static_cast<uint32_t>(m_nbins_padded);
        const SizeType rows  = 2 * chunk * m_nbins_padded;
        for (auto& buf : folds) {
            if (buf.size() < rows) {
                buf.resize(rows);
            }
        }
        // One leading row of padding: stage s is at (s + 1) * ntrials.
        if (matrix.size() < (m_nstages + 1) * ntrials) {
            matrix.resize((m_nstages + 1) * ntrials);
        }
        const thrust::device_vector<uint32_t> n_in(2, chunk_u);

        constexpr uint32_t kBlock = 256;
        for (SizeType offset = 0; offset < ntrials; offset += chunk) {
            const auto offset_u = static_cast<uint32_t>(offset);
            // The last chunk may be partial: only its trials write scores.
            const auto live_u =
                static_cast<uint32_t>(std::min(chunk, ntrials - offset));
            {
                const dim3 grid((2 * chunk + kBlock - 1) / kBlock);
                cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
                simulate_folds_init_eval_kernel<<<grid, kBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(folds[0].data()),
                    thrust::raw_pointer_cast(m_profile_d.data()), padded,
                    m_bias_snr, 1.0F, seed, chunk_u, offset_u, 2U);
                cuda_utils::check_last_cuda_error(
                    "simulate_folds_init_eval_kernel");
            }
            float var_in            = 1.0F;
            constexpr float kVarAdd = 1.0F;
            float* cur_in           = thrust::raw_pointer_cast(folds[0].data());
            float* cur_out          = thrust::raw_pointer_cast(folds[1].data());
            for (SizeType istage = 0; istage < m_nstages; ++istage) {
                TrialSimParams params = trial_sim_params(var_in, kVarAdd);
                params.seed           = seed;
                const dim3 grid((chunk + kBlock - 1) / kBlock);
                cuda_utils::check_kernel_launch_params(grid, dim3(kBlock));
                // The kernel writes H1 scores at chunk + trial: shift the
                // base so they land at (istage + 1) * ntrials + offset + trial.
                float* scores = thrust::raw_pointer_cast(matrix.data()) +
                                ((istage + 1) * ntrials) + offset - chunk;
                with_scorer([&]<typename Scorer>() {
                    ScorerLaunch<Scorer>::evaluate_stage(
                        grid, kBlock, stream, cur_in,
                        thrust::raw_pointer_cast(n_in.data()), cur_out, scores,
                        params, chunk_u, static_cast<uint32_t>(istage), chunk_u,
                        offset_u, chunk_u + live_u);
                });
                cuda_utils::check_last_cuda_error("evaluate_stage_kernel");
                std::swap(cur_in, cur_out);
                var_in += kVarAdd;
            }
        }
        // n_in is freed on return: finish with it first.
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
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
    std::string m_guess;
    SizeType m_nbins_padded{};
    // Shared mode: shared H1 trials (8 x ntrials) and their 32-bit words
    SizeType m_nh1{};
    SizeType m_h1_words{};
    SizeType m_nprobs{};
    SizeType m_nstages{};
    SizeType m_nthresholds{};
    // Shared mode: target Pd of the grid and weight of the last stage's
    // survivors in the cost (0 in every other mode).
    float m_min_pd{0.0F};
    double m_w{0.0};
    SizeType m_thres_neigh{0}; // of the last run
    float m_bias_snr{};

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    std::vector<SizeType> m_box_score_widths;
    std::vector<float> m_guess_path;
    detail::BeamBounds m_beam;
    // "window" and guided beams: run() ignores thres_neigh (the guided beam
    // then limits transitions by within_reach()).
    bool m_all_transitions{false};
    // "window" beam guided by a guess (empty m_score_sigma: the whole
    // window): feasibility window, pilot guess, signal score std per stage,
    // beam half-width and transition reach (both in sigma), and whether pass
    // 2 recentres the guess on pass 1's path.
    detail::BeamBounds m_window;
    std::vector<float> m_guess_pilot;
    std::vector<float> m_score_sigma;
    float m_guide_width{0.0F};
    float m_guide_reach{0.0F};
    // "window" beam: cost to go per stage of the second pass (empty: rank
    // cell candidates by complexity_cumul).
    std::vector<float> m_cost_to_go;
    std::vector<State> m_states;
    // Shared mode: the states after the first pass (the second pass
    // overwrites m_states), for the shortlist.
    std::vector<State> m_first_states;
    // Shared mode: the H1 matrix of this run() is simulated.
    bool m_h1_bank_valid{false};
    // Mode robust: host copy of the bank after release_search_memory().
    std::vector<float> m_h1_bank_host;

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
    // Shared mode: survivor bitmasks per cell (ping-pong) and the pass masks
    // of the current beam.
    std::array<thrust::device_vector<uint32_t>, 2> m_h1_mask_d;
    thrust::device_vector<uint32_t> m_h1_pass_d;
    // Shared mode: the H1 score matrix and its simulation's fold buffers,
    // allocated by the first run() and reused.
    thrust::device_vector<float> m_h1_scores_d;
    std::array<thrust::device_vector<float>, 2> m_h1_folds_d;
    SizeType m_max_beam{};

    // Allocated on the first run() and kept for later runs.
    void allocate_pools() {
        if (!m_states_d.empty()) {
            return;
        }
        const SizeType cells     = m_max_beam * m_nprobs;
        const SizeType pool_rows = pool_rows_count();
        for (auto& v : m_folds_d) {
            v.resize(pool_rows * m_nbins_padded);
        }
        for (auto& v : m_ntrials_d) {
            v.resize(cells * 2);
        }
        if (m_mode != DynamicThresholdMode::kLegacy) {
            for (auto& v : m_src_d) {
                v.resize(cells);
            }
            for (auto& v : m_idx_d) {
                v.resize(pool_rows);
            }
            m_scores_d.resize(pool_rows);
        }
        if (is_shared_family(m_mode)) {
            for (auto& v : m_h1_mask_d) {
                v.resize(cells * m_h1_words);
            }
            m_h1_pass_d.resize(m_max_beam * m_h1_words);
        }
        m_cell_keys_d.resize(m_nthresholds * m_nprobs);
        m_states_d.resize(m_states.size());
        spdlog::info("ThresholdsCuda ({} mode): {} max beam "
                     "thresholds x {} prob bins, {:.2f} GiB of device memory",
                     mode_to_string(m_mode), m_max_beam, m_nprobs,
                     utils::to_gib(run_memory_bytes()));
    }

    // Bit-level check: host code is built with -ffast-math, under which
    // std::isfinite (and NaN comparisons) may be folded away.
    static bool is_finite(float x) noexcept {
        return (std::bit_cast<uint32_t>(x) & 0x7F800000U) != 0x7F800000U;
    }

    void validate_inputs(SizeType nprobs,
                         float prob_min,
                         float snr_final,
                         SizeType nthresholds) const {
        detail::validate_scheme_inputs(
            {.branching_pattern = m_branching_pattern,
             .ref_ducy          = m_ref_ducy,
             .ducy_max          = m_ducy_max,
             .wtsp              = m_wtsp,
             .beam_width        = m_beam_width,
             .prob_min          = prob_min,
             .snr_final         = snr_final,
             .nbins             = m_nbins,
             .ntrials           = m_ntrials,
             .nprobs            = nprobs,
             .nthresholds       = nthresholds,
             .batch_size        = m_batch_size},
            "ThresholdsCuda");
    }

    void warn_empty_stages() const {
        detail::warn_empty_stages(m_states, m_nstages, m_nthresholds, m_nprobs,
                                  "ThresholdsCuda");
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
                         std::optional<uint64_t> seed,
                         std::string_view guess,
                         float min_pd,
                         float w_terminal,
                         float beam_sigma,
                         float reach_sigma)
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
                 seed,
                 guess,
                 min_pd,
                 w_terminal,
                 beam_sigma,
                 reach_sigma) {}

    std::vector<SizeType>
    get_current_thresholds_idx(SizeType istage) const override {
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
    std::vector<float> get_probs() const override { return m_impl.get_probs(); }
    SizeType get_nstages() const override { return m_impl.get_nstages(); }
    SizeType get_nthresholds() const override {
        return m_impl.get_nthresholds();
    }
    SizeType get_nprobs() const override { return m_impl.get_nprobs(); }
    std::vector<SizeType> get_box_score_widths() const override {
        return m_impl.get_box_score_widths();
    }
    std::vector<State> get_states() const override {
        return m_impl.get_states();
    }
    SizeType run_memory_bytes() const override {
        return m_impl.run_memory_bytes();
    }
    uint64_t get_seed() const override { return m_impl.get_seed(); }
    detail::SearchPath first_pass_path(float min_pd) const override {
        return m_impl.first_pass_path(min_pd);
    }
    std::vector<float> shared_h1_bank() const override {
        return m_impl.shared_h1_bank();
    }
    void release_search_memory() override { m_impl.release_search_memory(); }
    void run(SizeType thres_neigh) override { m_impl.run(thres_neigh); }
    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed,
                                bool h0_only) const override {
        return m_impl.evaluate(thresholds, ntrials, seed, h0_only);
    }
    std::vector<float> simulate_h1_scores(SizeType ntrials,
                                          uint64_t seed) const override {
        return m_impl.simulate_h1_scores(ntrials, seed);
    }
    std::unique_ptr<detail::SignalBank>
    make_signal_bank(SizeType n, uint64_t seed) const override {
        return m_impl.make_signal_bank(n, seed);
    }
    std::string save(const std::string& outdir) const override {
        return m_impl.save(outdir);
    }
    std::vector<float>
    get_best_path_thresholds(float min_pd = 0.1F) const override {
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
                    std::optional<uint64_t> seed,
                    std::string_view guess,
                    float min_pd,
                    float w_terminal,
                    float beam_sigma,
                    float reach_sigma) {
    return std::make_unique<ThresholdsCudaEngine>(
        branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
        snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start, mode,
        batch_size, device_id, seed, guess, min_pd, w_terminal, beam_sigma,
        reach_sigma);
}
} // namespace detail

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)
