#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/circular.hpp"
#include "lib/detail/psr_utils.hpp"
#include "lib/detail/utils.hpp"

// Replays the widths of a circular tree through the branch and transform
// kernels, for circ_branch_bound_t.cpp and its CUDA twin. Branching widths
// depend only on a leaf's frequency and the level, which is what lets
// BranchingForecast::max_children bound the branch workspace.

namespace loki::test {

inline constexpr SizeType kCircParams = 5;
inline constexpr SizeType kCircStride = (kCircParams + 2) * 2;
// Parents branched per kernel call.
inline constexpr SizeType kCircBatch = 64;
// Child slots per parent: well above any child count of the band below, so
// the kernels never throw and the true counts are seen.
inline constexpr SizeType kCircSlots = 512;

// A short circular band whose crackle splits at several of its 32 levels,
// where the split sets the worst parent with either tile.
inline search::PulsarSearchConfig make_circ_cfg(bool conservative) {
    constexpr SizeType kNsamps = 1U << 22U;
    const std::vector<ParamLimit> limits{
        {.min = -3e-3, .max = 3e-3},   // crackle
        {.min = -0.01, .max = 0.01},   // snap
        {.min = -2.0, .max = 2.0},     // jerk
        {.min = -500.0, .max = 500.0}, // accel
        {.min = 120.0, .max = 240.0},  // freq
    };
    return {kNsamps,
            64e-6,
            /*nbins=*/64,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.2,
            /*use_fourier=*/true,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/8.0,
            /*region_growth_factor=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/std::nullopt,
            /*nsegments=*/32,
            /*snr_min=*/6.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/kCircParams,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0,
            /*m_p_min=*/1.4,
            /*propagator_significance=*/2.0,
            /*validation_significance=*/5.0,
            /*use_conservative_tile=*/conservative};
}

// Seeds one parent per frequency of the plan's last FFA level, with the
// widths poly_taylor_seed gives it. At every level, each parent gets values
// that put every child in the crackle hole (significant jerk, zero snap and
// accel) and is branched by
//   branch(tree, n_parents, level, coord_cur, children, origins) -> n_children
// which fills `children` (kCircStride doubles per child) and `origins`. One
// child's widths then go through the transform to the next level; zero
// values keep it on the Taylor path. Returns the most children of a parent.
template <typename BranchFn>
SizeType replay_circ_widths(const search::PulsarSearchConfig& cfg,
                            const plans::FFAPlan<ComplexType>& plan,
                            BranchFn&& branch) {
    const SizeType n_levels = plan.get_n_levels();
    const auto& dparams     = plan.get_dparams_actual()[n_levels - 1];
    const auto& lim_freq    = cfg.get_param_limits()[kCircParams - 1];
    const SizeType n_freqs  = plan.get_param_counts()[n_levels - 1].back();
    const SizeType nseg     = plan.get_nsegments_active();
    const psr_utils::MiddleOutScheme scheme(nseg, 0, cfg.get_tseg_ffa());

    std::vector<double> f0(n_freqs);
    std::vector<double> widths(n_freqs * kCircParams);
    for (SizeType i = 0; i < n_freqs; ++i) {
        f0[i] = psr_utils::get_param_val_at_idx(lim_freq, n_freqs, i);
        for (SizeType j = 0; j + 1 < kCircParams; ++j) {
            widths[(i * kCircParams) + j] = dparams[j];
        }
        widths[(i * kCircParams) + kCircParams - 1] =
            dparams[kCircParams - 1] * (utils::kCval / f0[i]);
    }

    std::vector<double> tree(kCircBatch * kCircStride);
    std::vector<double> children(kCircBatch * kCircSlots * kCircStride);
    std::vector<SizeType> origins(kCircBatch * kCircSlots);
    std::vector<SizeType> first_child(kCircBatch);
    std::vector<SizeType> n_children(kCircBatch);
    std::vector<SizeType> indices(kCircBatch);
    SizeType max_children = 0;

    for (SizeType level = 1; level < nseg; ++level) {
        const auto coord_next = scheme.get_coord(level);
        const auto coord_cur  = scheme.get_current_coord(level);
        for (SizeType b0 = 0; b0 < n_freqs; b0 += kCircBatch) {
            const SizeType nb = std::min(kCircBatch, n_freqs - b0);
            tree.assign(nb * kCircStride, 0.0);
            for (SizeType i = 0; i < nb; ++i) {
                double* leaf = tree.data() + (i * kCircStride);
                for (SizeType j = 0; j < kCircParams; ++j) {
                    leaf[(2 * j) + 1] = widths[((b0 + i) * kCircParams) + j];
                }
                leaf[4]  = 1e3 * leaf[5]; // jerk far above its width
                leaf[12] = f0[b0 + i];
            }
            const SizeType n_out = branch(std::span<const double>(tree), nb,
                                          level, coord_cur, children, origins);
            std::ranges::fill(n_children, 0);
            for (SizeType c = n_out; c-- > 0;) {
                first_child[origins[c]] = c;
                ++n_children[origins[c]];
            }
            for (SizeType i = 0; i < nb; ++i) {
                max_children = std::max(max_children, n_children[i]);
            }
            tree.assign(nb * kCircStride, 0.0);
            for (SizeType i = 0; i < nb; ++i) {
                const double* child =
                    children.data() + (first_child[i] * kCircStride);
                double* leaf = tree.data() + (i * kCircStride);
                for (SizeType j = 0; j < kCircParams; ++j) {
                    leaf[(2 * j) + 1] = child[(2 * j) + 1];
                }
                leaf[12]   = f0[b0 + i];
                indices[i] = i;
            }
            core::circ_taylor_transform_batch(
                tree, indices, coord_next, coord_cur, nb,
                cfg.get_use_conservative_tile(),
                cfg.get_propagator_significance());
            for (SizeType i = 0; i < nb; ++i) {
                for (SizeType j = 0; j < kCircParams; ++j) {
                    widths[((b0 + i) * kCircParams) + j] =
                        tree[(i * kCircStride) + (2 * j) + 1];
                }
            }
        }
    }
    return max_children;
}

} // namespace loki::test
