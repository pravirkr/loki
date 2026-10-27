#include "lib/core/dynamic.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/chebyshev.hpp"
#include "lib/core/circular.hpp"
#include "lib/core/kernels.hpp"
#include "lib/core/taylor.hpp"
#include "lib/cpu/boxcar_kernels.hpp"
#include "lib/cpu/lane_fft.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/kadane.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::core {

namespace detail {

/// Leaves per fused scoring tile: one scorer group, one lane FFT call.
constexpr SizeType kEPScoreTile = lane_fft::kLanes;
static_assert(kEPScoreTile == detection::detail::BoxcarScorer3D::kLaneStride);

/**
 * Scoring state of one prune functor: the exact SIMD boxcar scorer, the
 * Kadane cache, the lane FFT for Fourier folds, and one tile of scratch.
 * Every profile is scored on its own (one SIMD lane, one transform lane), so
 * a score does not depend on how the leaves are cut into tiles.
 */
template <SupportedFoldType FoldType> class EPScorer {
public:
    static constexpr bool kComplex = std::is_same_v<FoldType, ComplexType>;

    explicit EPScorer(const search::PulsarSearchConfig& cfg)
        : m_nbins(cfg.get_nbins()),
          m_stride(2 * (kComplex ? cfg.get_nbins_f() : cfg.get_nbins())),
          m_widths(cfg.get_scoring_widths()),
          m_boxcar(m_nbins, m_widths.data(), m_widths.size()),
          m_use_kadane(cfg.get_use_boxcar_kadane()),
          m_kadane(cfg.get_boxcar_kadane_biases(), m_nbins),
          m_tile(kEPScoreTile * m_stride),
          m_kadane_idx(kEPScoreTile) {
        if constexpr (kComplex) {
            m_transform.emplace(m_nbins);
            if (m_use_kadane) {
                m_rows.resize(kEPScoreTile * 2 * m_nbins);
            }
        }
    }

    [[nodiscard]] FoldType* tile() noexcept { return m_tile.data(); }
    [[nodiscard]] SizeType stride() const noexcept { return m_stride; }

    /// Boxcar max S/N of m <= kEPScoreTile folds [m][stride].
    void boxcar(const FoldType* folds, SizeType m, float* scores) {
        if constexpr (kComplex) {
            m_transform->inverse(folds, m);
            m_boxcar.score_lanes_max(m_transform->e_lanes(),
                                     m_transform->v_lanes(), m, scores);
        } else {
            m_boxcar.score_max(folds, m, scores);
        }
    }

    /// Winning boxcar width of m <= kEPScoreTile folds; the first configured
    /// width wins an exact tie.
    void widths(const FoldType* folds, SizeType m, std::uint32_t* out) {
        std::array<float, kEPScoreTile> best;
        best.fill(std::numeric_limits<float>::lowest());
        std::fill_n(out, m, static_cast<std::uint32_t>(m_widths.front()));
        const auto emit = [&](SizeType p, SizeType iw, float snr) {
            if (snr > best[p]) {
                best[p] = snr;
                out[p]  = static_cast<std::uint32_t>(m_widths[iw]);
            }
        };
        if constexpr (kComplex) {
            m_transform->inverse(folds, m);
            m_boxcar.score_lanes(m_transform->e_lanes(), m_transform->v_lanes(),
                                 m, emit);
        } else {
            m_boxcar.score(folds, m, emit);
        }
    }

    /// The configured score (boxcar or Kadane) of m <= kEPScoreTile folds.
    void score(const FoldType* folds, SizeType m, float* scores) {
        if (!m_use_kadane) {
            boxcar(folds, m, scores);
            return;
        }
        const float* rows = nullptr;
        if constexpr (kComplex) {
            m_transform->inverse(folds, m);
            lanes_to_rows(m);
            rows = m_rows.data();
        } else {
            rows = folds;
        }
        detection::score_and_filter_max_kadane_with_cache(
            std::span(rows, m * 2 * m_nbins), std::span(scores, m),
            std::span(m_kadane_idx), std::numeric_limits<float>::lowest(), m,
            m_nbins, m_kadane);
    }

private:
    SizeType m_nbins;
    SizeType m_stride;
    std::vector<SizeType> m_widths;
    detection::detail::BoxcarScorer3D m_boxcar;
    bool m_use_kadane;
    detection::BoxcarKadaneCache m_kadane;
    std::optional<lane_fft::EVTransform> m_transform;
    utils::UninitVector<FoldType> m_tile;
    std::vector<float> m_rows; // Kadane on Fourier folds: [tile][2][nbins]
    std::vector<SizeType> m_kadane_idx;

    void lanes_to_rows(SizeType m) noexcept {
        const float* e = m_transform->e_lanes();
        const float* v = m_transform->v_lanes();
        for (SizeType p = 0; p < m; ++p) {
            float* row = m_rows.data() + (p * 2 * m_nbins);
            for (SizeType j = 0; j < m_nbins; ++j) {
                row[j]           = e[(j * lane_fft::kLanes) + p];
                row[m_nbins + j] = v[(j * lane_fft::kLanes) + p];
            }
        }
    }
};

} // namespace detail

template <SupportedFoldType FoldType>
BasePruneDPFuncts<FoldType>::BasePruneDPFuncts(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : m_param_grid_count_init(param_grid_count_init.begin(),
                              param_grid_count_init.end()),
      m_dparams_init(dparams_init.begin(), dparams_init.end()),
      m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max),
      m_scorer(std::make_unique<detail::EPScorer<FoldType>>(m_cfg)) {
    SizeType n_coords_init = 1;
    for (const auto count : m_param_grid_count_init) {
        n_coords_init *= count;
    }
    m_n_coords_init = n_coords_init;
}

template <SupportedFoldType FoldType>
BasePruneDPFuncts<FoldType>::~BasePruneDPFuncts() = default;

template <SupportedFoldType FoldType>
void BasePruneDPFuncts<FoldType>::final_widths(
    std::span<const FoldType> folds, std::span<std::uint32_t> widths) {
    const auto stride = m_scorer->stride();
    error_check::check_equal(folds.size(), widths.size() * stride,
                             "final_widths: folds size mismatch");
    for (SizeType p = 0; p < widths.size(); p += detail::kEPScoreTile) {
        const auto count = std::min(detail::kEPScoreTile, widths.size() - p);
        m_scorer->widths(folds.data() + p * stride, count, widths.data() + p);
    }
}

template <SupportedFoldType FoldType>
void BasePruneDPFuncts<FoldType>::score_folds(std::span<const FoldType> folds,
                                              std::span<float> scores,
                                              SizeType n_leaves) {
    const auto stride = m_scorer->stride();
    error_check::check_greater_equal(folds.size(), n_leaves * stride,
                                     "score_folds: folds size mismatch");
    error_check::check_greater_equal(scores.size(), n_leaves,
                                     "score_folds: scores size mismatch");
    for (SizeType p0 = 0; p0 < n_leaves; p0 += detail::kEPScoreTile) {
        const SizeType m = std::min(detail::kEPScoreTile, n_leaves - p0);
        m_scorer->boxcar(folds.data() + (p0 * stride), m, scores.data() + p0);
    }
}

template <SupportedFoldType FoldType>
std::span<const FoldType>
BasePruneDPFuncts<FoldType>::load_segment(std::span<const FoldType> ffa_fold,
                                          SizeType seg_idx) const {
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        return ffa_fold.subspan(seg_idx * m_n_coords_init * 2 * nbins_f,
                                m_n_coords_init * 2 * nbins_f);
    } else {
        return ffa_fold.subspan(seg_idx * m_n_coords_init * 2 * nbins,
                                m_n_coords_init * 2 * nbins);
    }
}

template <SupportedFoldType FoldType>
SizeType
BasePruneDPFuncts<FoldType>::validate(std::span<double> /*leaves_branch*/,
                                      std::span<SizeType> /*leaves_origins*/,
                                      std::pair<double, double> /*coord_cur*/,
                                      SizeType n_leaves) const {
    return n_leaves;
}

template <SupportedFoldType FoldType>
std::tuple<std::vector<double>, std::vector<double>, double>
BasePruneDPFuncts<FoldType>::get_validation_params(
    std::pair<double, double> /*coord_add*/) const {
    // Return empty validation parameters for Taylor variant
    std::vector<double> const empty_arr(0);
    return std::make_tuple(empty_arr, empty_arr, 0.0);
}

template <SupportedFoldType FoldType>
void BasePruneDPFuncts<FoldType>::shift_add(
    std::span<const FoldType> folds_tree,
    std::span<const SizeType> indices_tree,
    std::span<const FoldType> folds_ffa,
    std::span<const SizeType> indices_ffa,
    std::span<const float> phase_shift,
    std::span<FoldType> folds_out,
    SizeType n_leaves,
    SizeType physical_start_idx,
    SizeType capacity) noexcept {
    if constexpr (std::is_same_v<FoldType, float>) {
        core::shift_add_linear_batch(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_cfg.get_nbins(), n_leaves, physical_start_idx, capacity);
    } else {
        core::shift_add_linear_complex_batch(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_leaves,
            physical_start_idx, capacity);
    }
}

template <SupportedFoldType FoldType>
SizeType BasePruneDPFuncts<FoldType>::score_and_filter(
    std::span<const FoldType> folds_tree,
    std::span<float> scores_tree,
    std::span<SizeType> indices_tree,
    float threshold,
    SizeType n_leaves) {
    const auto stride  = m_scorer->stride();
    SizeType n_passing = 0;
    for (SizeType p0 = 0; p0 < n_leaves; p0 += detail::kEPScoreTile) {
        const SizeType m = std::min(detail::kEPScoreTile, n_leaves - p0);
        m_scorer->score(folds_tree.data() + (p0 * stride), m,
                        scores_tree.data() + p0);
        for (SizeType i = p0; i < p0 + m; ++i) {
            if (scores_tree[i] >= threshold) {
                indices_tree[n_passing++] = i;
            }
        }
    }
    return n_passing;
}

template <SupportedFoldType FoldType>
SizeType BasePruneDPFuncts<FoldType>::shift_add_score(
    std::span<const FoldType> folds_tree,
    std::span<SizeType> indices_tree,
    std::span<const FoldType> folds_ffa,
    std::span<const SizeType> indices_ffa,
    std::span<const float> phase_shift,
    PassingFolds<FoldType> folds_out,
    std::span<float> scores_out,
    float threshold,
    SizeType n_leaves,
    SizeType physical_start_idx,
    SizeType capacity) {
    const auto stride  = m_scorer->stride();
    FoldType* tile     = m_scorer->tile();
    SizeType n_passing = 0;
    for (SizeType p0 = 0; p0 < n_leaves; p0 += detail::kEPScoreTile) {
        const SizeType m = std::min(detail::kEPScoreTile, n_leaves - p0);
        // Reads the origins of leaves [p0, p0 + m) before any passing index
        // (always < p0 + m) is written over them.
        shift_add(folds_tree, indices_tree.subspan(p0, m), folds_ffa,
                  indices_ffa.subspan(p0, m), phase_shift.subspan(p0, m),
                  std::span(tile, m * stride), m, physical_start_idx, capacity);
        m_scorer->score(tile, m, scores_out.data() + p0);
        for (SizeType i = 0; i < m; ++i) {
            if (scores_out[p0 + i] >= threshold) {
                SizeType slot = p0 + i;
                if (folds_out.ring_capacity != 0) {
                    slot = folds_out.ring_start + n_passing;
                    if (slot >= folds_out.ring_capacity) {
                        slot -= folds_out.ring_capacity;
                    }
                }
                std::copy_n(tile + (i * stride), stride,
                            folds_out.data + (slot * stride));
                indices_tree[n_passing++] = p0 + i;
            }
        }
    }
    return n_passing;
}

template <SupportedFoldType FoldType>
std::vector<double> BasePruneDPFuncts<FoldType>::get_transform_matrix(
    std::pair<double, double> /*coord_cur*/,
    std::pair<double, double> /*coord_prev*/) const {
    // Return identity matrix for Taylor variant
    // Size should match parameter dimensions (typically 4x4 for poly_order=4)
    const SizeType size = m_cfg.get_nparams() + 1;
    std::vector<double> identity(size * size, 0.0);
    for (SizeType i = 0; i < size; ++i) {
        identity[(i * size) + i] = 1.0;
    }
    return identity;
}

template <SupportedFoldType FoldType>
void BasePruneDPFuncts<FoldType>::pack(std::span<const FoldType> data,
                                       std::span<FoldType> out) const noexcept {
    // Placeholder for future implementation
    std::copy(data.begin(), data.end(), out.begin());
}

// Intermediate implementation for Taylor basis
template <SupportedFoldType FoldType>
void BaseTaylorPruneDPFuncts<FoldType>::seed(
    std::span<const FoldType> fold_segment,
    std::span<double> seed_leaves,
    std::span<float> seed_scores,
    std::pair<double, double> coord_init) {

    const auto n_leaves =
        poly_taylor_seed(this->m_param_grid_count_init, this->m_dparams_init,
                         this->m_cfg.get_param_limits(), seed_leaves,
                         coord_init, this->m_cfg.get_nparams());
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins = this->m_cfg.get_nbins();
    error_check::check_greater_equal(seed_scores.size(), n_leaves,
                                     "seed_scores size mismatch");

    // Calculate scores
    const auto stride = std::is_same_v<FoldType, ComplexType>
                            ? 2 * this->m_cfg.get_nbins_f()
                            : 2 * nbins;
    error_check::check_equal(fold_segment.size(), n_leaves * stride,
                             "fold_segment size mismatch");
    this->score_folds(fold_segment, seed_scores, n_leaves);
}

// Intermediate implementation for Chebyshev basis
template <SupportedFoldType FoldType>
void BaseChebyshevPruneDPFuncts<FoldType>::seed(
    std::span<const FoldType> fold_segment,
    std::span<double> seed_leaves,
    std::span<float> seed_scores,
    std::pair<double, double> coord_init) {
    // First seed in Taylor basis
    const auto n_leaves =
        poly_taylor_seed(this->m_param_grid_count_init, this->m_dparams_init,
                         this->m_cfg.get_param_limits(), seed_leaves,
                         coord_init, this->m_cfg.get_nparams());
    // Then convert to Chebyshev basis
    poly_taylor_to_cheby_batch(seed_leaves, coord_init, n_leaves,
                               this->m_cfg.get_nparams());
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins = this->m_cfg.get_nbins();
    error_check::check_greater_equal(seed_scores.size(), n_leaves,
                                     "seed_scores size mismatch");

    // Calculate scores
    const auto stride = std::is_same_v<FoldType, ComplexType>
                            ? 2 * this->m_cfg.get_nbins_f()
                            : 2 * nbins;
    error_check::check_equal(fold_segment.size(), n_leaves * stride,
                             "fold_segment size mismatch");
    this->score_folds(fold_segment, seed_scores, n_leaves);
}

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldType FoldType>
PrunePolyTaylorDPFuncts<FoldType>::PrunePolyTaylorDPFuncts(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldType FoldType>
SizeType PrunePolyTaylorDPFuncts<FoldType>::branch(
    std::span<double> leaves_tree,
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    SizeType n_leaves,
    memory::BranchingWorkspace& branch_ws) const {
    return poly_taylor_branch_batch(
        leaves_tree, leaves_branch, leaves_origins, coord_cur,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(), this->m_branch_max,
        n_leaves, this->m_cfg.get_nparams(), branch_ws);
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_branch,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init[n_params - 1];
    poly_taylor_resolve_batch(leaves_branch, param_indices, phase_shift,
                              this->m_cfg.get_param_limits(), coord_add,
                              coord_cur, coord_init, n_accel_init, n_freq_init,
                              this->m_cfg.get_nbins(), n_leaves,
                              this->m_cfg.get_nparams());
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_tree,
    std::span<SizeType> indices_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves) const {
    poly_taylor_transform_batch(leaves_tree, indices_tree, coord_next,
                                coord_cur, n_leaves, this->m_cfg.get_nparams(),
                                this->m_cfg.get_use_conservative_tile());
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::ascend(
    std::span<const FoldType> folds_ffa,
    std::span<const double> leaves_tree,
    std::span<FoldType> folds_tree,
    std::span<float> scores_tree,
    std::span<float> scores_ep_tree,
    std::span<const uint32_t> idx_segments,
    std::span<const std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    std::span<SizeType> scratch_param_indices,
    std::span<float> scratch_phase_shift,
    SizeType n_leaves) {
    const auto n_params      = this->m_cfg.get_nparams();
    const auto n_accel_init  = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init   = this->m_param_grid_count_init[n_params - 1];
    const auto nbins         = this->m_cfg.get_nbins();
    const auto nbins_f       = this->m_cfg.get_nbins_f();
    const auto n_segments    = idx_segments.size();
    const auto n_coords_init = this->m_n_coords_init;

    poly_taylor_ascend_resolve_batch(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        this->m_cfg.get_param_limits(), coord_segments, coord_cur, n_accel_init,
        n_freq_init, nbins, n_leaves, n_params, n_segments);

    // Copy scores to scores_ep for future backup
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        scores_ep_tree[ileaf] = scores_tree[ileaf];
    }

    // Calculate scores
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_complex_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments);
    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins, n_coords_init,
            n_leaves, n_segments);
    }
    this->score_folds(folds_tree, scores_tree, n_leaves);
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::report(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves) const {
    poly_taylor_report_batch(leaves_tree, coord_report, n_leaves,
                             this->m_cfg.get_nparams());
}

// Specialized implementation for Polynomial searches in Chebyshev Basis
template <SupportedFoldType FoldType>
PrunePolyChebyshevDPFuncts<FoldType>::PrunePolyChebyshevDPFuncts(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldType FoldType>
SizeType PrunePolyChebyshevDPFuncts<FoldType>::branch(
    std::span<double> leaves_tree,
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_prev,
    SizeType n_leaves,
    memory::BranchingWorkspace& branch_ws) const {
    return poly_chebyshev_branch_batch(
        leaves_tree, leaves_branch, leaves_origins, coord_cur, coord_prev,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(), this->m_branch_max,
        n_leaves, this->m_cfg.get_nparams(), branch_ws);
}

template <SupportedFoldType FoldType>
void PrunePolyChebyshevDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_branch,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init[n_params - 1];
    poly_chebyshev_resolve_batch(leaves_branch, param_indices, phase_shift,
                                 this->m_cfg.get_param_limits(), coord_add,
                                 coord_cur, coord_init, n_accel_init,
                                 n_freq_init, this->m_cfg.get_nbins(), n_leaves,
                                 this->m_cfg.get_nparams());
}

template <SupportedFoldType FoldType>
void PrunePolyChebyshevDPFuncts<FoldType>::transform(
    std::span<double> leaves_tree,
    std::span<SizeType> indices_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves) const {
    if (this->m_cfg.get_use_conservative_tile()) {
        throw std::logic_error(
            "Conservative tile not implemented for Chebyshev basis");
    }
    poly_chebyshev_transform_batch(leaves_tree, indices_tree, coord_next,
                                   coord_cur, n_leaves,
                                   this->m_cfg.get_nparams());
}

template <SupportedFoldType FoldType>
void PrunePolyChebyshevDPFuncts<FoldType>::ascend(
    std::span<const FoldType> folds_ffa,
    std::span<const double> leaves_tree,
    std::span<FoldType> folds_tree,
    std::span<float> scores_tree,
    std::span<float> scores_ep_tree,
    std::span<const uint32_t> idx_segments,
    std::span<const std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    std::span<SizeType> scratch_param_indices,
    std::span<float> scratch_phase_shift,
    SizeType n_leaves) {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init[n_params - 1];
    const auto nbins        = this->m_cfg.get_nbins();
    const auto nbins_f      = this->m_cfg.get_nbins_f();
    const auto n_segments   = idx_segments.size();

    poly_chebyshev_ascend_resolve_batch(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        this->m_cfg.get_param_limits(), coord_segments, coord_cur, n_accel_init,
        n_freq_init, this->m_cfg.get_nbins(), n_leaves,
        this->m_cfg.get_nparams(), n_segments);

    // Copy scores to scores_ep for future backup
    std::ranges::copy(scores_tree, scores_ep_tree.begin());

    // Calculate scores
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_complex_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            this->m_n_coords_init, n_leaves, n_segments);
    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins,
            this->m_n_coords_init, n_leaves, n_segments);
    }
    this->score_folds(folds_tree, scores_tree, n_leaves);
}

template <SupportedFoldType FoldType>
void PrunePolyChebyshevDPFuncts<FoldType>::report(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves) const {
    // First convert the Chebyshev leaves to Taylor leaves
    poly_cheby_to_taylor_batch(leaves_tree, coord_report, n_leaves,
                               this->m_cfg.get_nparams());
    // Now call the Taylor report batch
    poly_taylor_report_batch(leaves_tree, coord_report, n_leaves,
                             this->m_cfg.get_nparams());
}

// Specialized implementation for Circular orbit search in Taylor basis
template <SupportedFoldType FoldType>
PruneCircTaylorDPFuncts<FoldType>::PruneCircTaylorDPFuncts(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldType FoldType>
SizeType PruneCircTaylorDPFuncts<FoldType>::branch(
    std::span<double> leaves_tree,
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    SizeType n_leaves,
    memory::BranchingWorkspace& branch_ws) const {
    return circ_taylor_branch_batch(
        leaves_tree, leaves_branch, leaves_origins, coord_cur,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(), this->m_branch_max,
        n_leaves, this->m_cfg.get_propagator_significance(), branch_ws);
}

template <SupportedFoldType FoldType>
SizeType PruneCircTaylorDPFuncts<FoldType>::validate(
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves) const {
    return circ_taylor_validate_batch(
        leaves_branch, leaves_origins, n_leaves, this->m_cfg.get_p_orb_min(),
        this->m_cfg.get_x_mass_const(),
        this->m_cfg.get_validation_significance());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_branch,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init[n_params - 1];
    circ_taylor_resolve_batch(leaves_branch, param_indices, phase_shift,
                              this->m_cfg.get_param_limits(), coord_add,
                              coord_cur, coord_init, n_accel_init, n_freq_init,
                              this->m_cfg.get_nbins(), n_leaves,
                              this->m_cfg.get_propagator_significance());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_tree,
    std::span<SizeType> indices_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves) const {
    circ_taylor_transform_batch(leaves_tree, indices_tree, coord_next,
                                coord_cur, n_leaves,
                                this->m_cfg.get_use_conservative_tile(),
                                this->m_cfg.get_propagator_significance());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::ascend(
    std::span<const FoldType> folds_ffa,
    std::span<const double> leaves_tree,
    std::span<FoldType> folds_tree,
    std::span<float> scores_tree,
    std::span<float> scores_ep_tree,
    std::span<const uint32_t> idx_segments,
    std::span<const std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    std::span<SizeType> scratch_param_indices,
    std::span<float> scratch_phase_shift,
    SizeType n_leaves) {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init[n_params - 1];
    const auto nbins        = this->m_cfg.get_nbins();
    const auto nbins_f      = this->m_cfg.get_nbins_f();
    const auto n_segments   = idx_segments.size();

    circ_taylor_ascend_resolve_batch(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        this->m_cfg.get_param_limits(), coord_segments, coord_cur, n_accel_init,
        n_freq_init, this->m_cfg.get_nbins(), n_leaves, n_segments,
        this->m_cfg.get_propagator_significance());

    // Copy scores to scores_ep for future backup
    std::ranges::copy(scores_tree, scores_ep_tree.begin());

    // Calculate scores
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_complex_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            this->m_n_coords_init, n_leaves, n_segments);
    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins,
            this->m_n_coords_init, n_leaves, n_segments);
    }
    this->score_folds(folds_tree, scores_tree, n_leaves);
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::report(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves) const {
    poly_taylor_report_batch(leaves_tree, coord_report, n_leaves,
                             this->m_cfg.get_nparams());
}

// Factory function to create the correct implementation based on the kind
template <SupportedFoldType FoldType>
std::unique_ptr<PruneDPFuncts<FoldType>>
create_prune_dp_functs(std::string_view poly_basis,
                       std::span<const SizeType> param_grid_count_init,
                       std::span<const double> dparams_init,
                       SizeType nseg_ffa,
                       double tseg_ffa,
                       search::PulsarSearchConfig cfg,
                       SizeType batch_size,
                       SizeType branch_max) {
    const auto n_params = cfg.get_nparams();
    if (poly_basis == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFuncts<FoldType>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    if (poly_basis == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFuncts<FoldType>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    if (poly_basis == "chebyshev" && n_params <= 4) {
        return std::make_unique<PrunePolyChebyshevDPFuncts<FoldType>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    throw std::runtime_error(std::format(
        "Unknown poly_basis: '{}'. Valid options: 'taylor', 'chebyshev'",
        poly_basis));
}

// Explicit template instantiations
// Base classes need explicit instantiation for linker
template class BasePruneDPFuncts<float>;
template class BasePruneDPFuncts<ComplexType>;

template class BaseTaylorPruneDPFuncts<float>;
template class BaseTaylorPruneDPFuncts<ComplexType>;

template class BaseChebyshevPruneDPFuncts<float>;
template class BaseChebyshevPruneDPFuncts<ComplexType>;
// Leaf classes
template class PrunePolyTaylorDPFuncts<float>;
template class PrunePolyTaylorDPFuncts<ComplexType>;
template class PruneCircTaylorDPFuncts<float>;
template class PruneCircTaylorDPFuncts<ComplexType>;
template class PrunePolyChebyshevDPFuncts<float>;
template class PrunePolyChebyshevDPFuncts<ComplexType>;

// Factory function instantiations
template std::unique_ptr<PruneDPFuncts<float>>
create_prune_dp_functs<float>(std::string_view,
                              std::span<const SizeType>,
                              std::span<const double>,
                              SizeType,
                              double,
                              search::PulsarSearchConfig,
                              SizeType,
                              SizeType);
template std::unique_ptr<PruneDPFuncts<ComplexType>>
create_prune_dp_functs<ComplexType>(std::string_view,
                                    std::span<const SizeType>,
                                    std::span<const double>,
                                    SizeType,
                                    double,
                                    search::PulsarSearchConfig,
                                    SizeType,
                                    SizeType);

} // namespace loki::core
