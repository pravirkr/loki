#include "lib/search/prune_mask.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/psr_utils.hpp"
#include "lib/detail/utils.hpp"

namespace loki::search {

namespace {

/// Inclusive cell range covered by [lo, hi] on a uniform grid, or nullopt if
/// the interval is disjoint from the grid. Values are clamped to the grid
/// limits before snapping so that infinities/large values are handled.
std::optional<std::pair<SizeType, SizeType>>
cell_range(double lo, double hi, const ParamLimit& lim, SizeType count) {
    if (utils::is_nan(lo) || utils::is_nan(hi) || hi < lim.min ||
        lo > lim.max) {
        return std::nullopt;
    }
    const double lo_c = std::clamp(lo, lim.min, lim.max);
    const double hi_c = std::clamp(hi, lim.min, lim.max);
    const auto i_lo   = psr_utils::get_nearest_idx_analytical(lo_c, lim, count);
    const auto i_hi   = psr_utils::get_nearest_idx_analytical(hi_c, lim, count);
    return std::make_pair(i_lo, std::max(i_lo, i_hi));
}

} // namespace

GridMask::GridMask(ParamLimit lim_accel,
                   SizeType n_accel,
                   ParamLimit lim_freq,
                   SizeType n_freq)
    : m_lim_accel(lim_accel),
      m_lim_freq(lim_freq),
      m_n_accel(n_accel),
      m_n_freq(n_freq) {
    error_check::check_greater(n_accel, SizeType{0},
                               "GridMask: n_accel must be positive");
    error_check::check_greater(n_freq, SizeType{0},
                               "GridMask: n_freq must be positive");
    error_check::check_less_equal(lim_accel.min, lim_accel.max,
                                  "GridMask: invalid acceleration limits");
    error_check::check_less_equal(lim_freq.min, lim_freq.max,
                                  "GridMask: invalid frequency limits");
    const auto n_cells = get_n_cells();
    m_bits.assign((n_cells + 63U) / 64U, 0ULL);
}

float GridMask::get_memory_usage_gib() const noexcept {
    return static_cast<float>(m_bits.size() * sizeof(uint64_t)) /
           static_cast<float>(utils::kBytesPerGB);
}

void GridMask::set_range(SizeType lo, SizeType hi) noexcept {
    // Inclusive [lo, hi]; caller guarantees hi < n_cells.
    const SizeType word_lo = lo >> 6U;
    const SizeType word_hi = hi >> 6U;
    for (SizeType w = word_lo; w <= word_hi; ++w) {
        const SizeType bit_lo = (w == word_lo) ? (lo & 63U) : 0U;
        const SizeType bit_hi = (w == word_hi) ? (hi & 63U) : 63U;
        // Mask with bits [bit_lo, bit_hi] set.
        const uint64_t upper =
            (bit_hi == 63U) ? ~0ULL : ((1ULL << (bit_hi + 1U)) - 1ULL);
        const uint64_t lower = (1ULL << bit_lo) - 1ULL;
        const uint64_t mask  = upper & ~lower;
        const uint64_t added = mask & ~m_bits[w];
        m_n_set += static_cast<SizeType>(std::popcount(added));
        m_bits[w] |= mask;
    }
}

void GridMask::add_window_single(double f_lo,
                                 double f_hi,
                                 double a_lo,
                                 double a_hi) noexcept {
    const auto f_range = cell_range(f_lo, f_hi, m_lim_freq, m_n_freq);
    const auto a_range = cell_range(a_lo, a_hi, m_lim_accel, m_n_accel);
    if (!f_range || !a_range) {
        return;
    }
    const auto [if_lo, if_hi] = *f_range;
    const auto [ia_lo, ia_hi] = *a_range;
    for (SizeType ia = ia_lo; ia <= ia_hi; ++ia) {
        const auto base = ia * m_n_freq;
        set_range(base + if_lo, base + if_hi);
    }
    ++m_n_windows;
}

void GridMask::add_window(const algorithms::ParamWindow& window,
                          SizeType n_harmonics,
                          bool fractional_harmonics) {
    error_check::check(!utils::is_nan(window.f_lo) &&
                           !utils::is_nan(window.f_hi),
                       "GridMask::add_window: frequency bounds must not be "
                       "NaN");
    error_check::check_less_equal(window.f_lo, window.f_hi,
                                  "GridMask::add_window: requires f_lo <= "
                                  "f_hi");
    error_check::check_less_equal(window.a_lo, window.a_hi,
                                  "GridMask::add_window: requires a_lo <= "
                                  "a_hi");
    add_window_single(window.f_lo, window.f_hi, window.a_lo, window.a_hi);
    if (fractional_harmonics) {
        for (SizeType q = 1; q <= n_harmonics; ++q) {
            for (SizeType p = 1; p <= n_harmonics; ++p) {
                if (p == q || std::gcd(p, q) != 1) {
                    continue;
                }
                // (f * p) / q: the same operations as the integer
                // expansion when p or q is 1.
                const auto pd = static_cast<double>(p);
                const auto qd = static_cast<double>(q);
                add_window_single((window.f_lo * pd) / qd,
                                  (window.f_hi * pd) / qd, window.a_lo,
                                  window.a_hi);
            }
        }
        return;
    }
    for (SizeType k = 2; k <= n_harmonics; ++k) {
        const auto kd = static_cast<double>(k);
        add_window_single(window.f_lo * kd, window.f_hi * kd, window.a_lo,
                          window.a_hi);
        add_window_single(window.f_lo / kd, window.f_hi / kd, window.a_lo,
                          window.a_hi);
    }
}

bool GridMask::is_masked_at(double accel, double freq) const noexcept {
    if (empty() || utils::is_nan(accel) || utils::is_nan(freq)) {
        return false;
    }
    // A point reported just outside the grid takes the nearest edge cell.
    const double a_c   = std::clamp(accel, m_lim_accel.min, m_lim_accel.max);
    const double f_c   = std::clamp(freq, m_lim_freq.min, m_lim_freq.max);
    const auto f_range = cell_range(f_c, f_c, m_lim_freq, m_n_freq);
    const auto a_range = cell_range(a_c, a_c, m_lim_accel, m_n_accel);
    if (!f_range || !a_range) {
        return false;
    }
    return is_masked((a_range->first * m_n_freq) + f_range->first);
}

void GridMask::add_windows(std::span<const algorithms::ParamWindow> windows,
                           SizeType n_harmonics,
                           bool fractional_harmonics) {
    for (const auto& w : windows) {
        add_window(w, n_harmonics, fractional_harmonics);
    }
}

void GridMask::assign(const GridMask& base) {
    if (this == &base) {
        return;
    }
    m_lim_accel = base.m_lim_accel;
    m_lim_freq  = base.m_lim_freq;
    m_n_accel   = base.m_n_accel;
    m_n_freq    = base.m_n_freq;
    m_n_set     = base.m_n_set;
    m_n_windows = base.m_n_windows;
    // Reuses the existing allocation when sizes match.
    m_bits.assign(base.m_bits.begin(), base.m_bits.end());
}

void GridMask::clear() noexcept {
    std::ranges::fill(m_bits, 0ULL);
    m_n_set     = 0;
    m_n_windows = 0;
}

SizeType GridMask::filter_resolved(std::span<double> leaves,
                                   std::span<SizeType> origins,
                                   std::span<SizeType> param_idx,
                                   std::span<float> phase_shift,
                                   SizeType leaves_stride,
                                   SizeType n_leaves) const {
    error_check::check_greater_equal(leaves.size(), n_leaves * leaves_stride,
                                     "GridMask::filter_resolved: leaves size "
                                     "mismatch");
    error_check::check_greater_equal(origins.size(), n_leaves,
                                     "GridMask::filter_resolved: origins size "
                                     "mismatch");
    error_check::check_greater_equal(param_idx.size(), n_leaves,
                                     "GridMask::filter_resolved: param_idx "
                                     "size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "GridMask::filter_resolved: phase_shift "
                                     "size mismatch");
    if (empty()) {
        return n_leaves;
    }
    const auto n_cells                   = get_n_cells();
    double* __restrict__ leaves_ptr      = leaves.data();
    SizeType* __restrict__ origins_ptr   = origins.data();
    SizeType* __restrict__ param_idx_ptr = param_idx.data();
    float* __restrict__ phase_ptr        = phase_shift.data();

    SizeType write = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto idx = param_idx_ptr[i];
        if (idx < n_cells && is_masked(idx)) {
            continue;
        }
        if (write != i) {
            std::memcpy(leaves_ptr + (write * leaves_stride),
                        leaves_ptr + (i * leaves_stride),
                        leaves_stride * sizeof(double));
            origins_ptr[write]   = origins_ptr[i];
            param_idx_ptr[write] = idx;
            phase_ptr[write]     = phase_ptr[i];
        }
        ++write;
    }
    return write;
}

SizeType GridMask::select_seeds(std::span<SizeType> keep_indices,
                                SizeType n_seeds) const {
    error_check::check_greater_equal(keep_indices.size(), n_seeds,
                                     "GridMask::select_seeds: keep_indices "
                                     "size mismatch");
    error_check::check_less_equal(n_seeds, get_n_cells(),
                                  "GridMask::select_seeds: n_seeds exceeds "
                                  "the number of grid cells");
    SizeType n_keep = 0;
    for (SizeType i = 0; i < n_seeds; ++i) {
        if (!is_masked(i)) {
            keep_indices[n_keep++] = i;
        }
    }
    return n_keep;
}

std::vector<SizeType> even_seed_slice_edges(SizeType n_slices,
                                            SizeType n_freq) {
    const auto k =
        std::clamp<SizeType>(n_slices, 1, std::max<SizeType>(n_freq, 1));
    std::vector<SizeType> edges(k + 1);
    for (SizeType s = 0; s <= k; ++s) {
        edges[s] = (s * n_freq) / k;
    }
    return edges;
}

void check_seed_slice_edges(std::span<const SizeType> edges, SizeType n_freq) {
    if (edges.empty()) {
        return;
    }
    bool ok = edges.size() >= 2 && edges.front() == 0 && edges.back() == n_freq;
    for (SizeType i = 1; ok && i < edges.size(); ++i) {
        ok = edges[i] > edges[i - 1];
    }
    if (!ok) {
        throw std::invalid_argument(fmt::format(
            "Seed slice edges must rise strictly from 0 to the {} frequency "
            "cells, got [{}]",
            n_freq, fmt::join(edges, ", ")));
    }
}

SizeType select_slice_seeds(std::span<SizeType> keep_indices,
                            SizeType n_accel,
                            SizeType n_freq,
                            SeedSlice slice,
                            const GridMask& mask) {
    error_check::check(slice.f_begin <= slice.f_end && slice.f_end <= n_freq,
                       "select_slice_seeds: slice outside the grid");
    error_check::check_greater_equal(
        keep_indices.size(), n_accel * (slice.f_end - slice.f_begin),
        "select_slice_seeds: keep_indices too small");
    const bool masked = !mask.empty();
    if (masked) {
        error_check::check(mask.get_n_accel() == n_accel &&
                               mask.get_n_freq() == n_freq,
                           "select_slice_seeds: mask grid does not match");
    }
    SizeType n_keep = 0;
    for (SizeType ia = 0; ia < n_accel; ++ia) {
        for (SizeType i = (ia * n_freq) + slice.f_begin;
             i < (ia * n_freq) + slice.f_end; ++i) {
            if (!masked || !mask.is_masked(i)) {
                keep_indices[n_keep++] = i;
            }
        }
    }
    return n_keep;
}

} // namespace loki::search
