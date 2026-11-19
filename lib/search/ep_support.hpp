#pragma once

/**
 * @file ep_support.hpp
 * @brief The searches EP supports. Internal.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::search::detail {

/// Largest EP fold: the fused CUDA scorers transform on chip up to here.
inline constexpr SizeType kEPMaxNbins = 1024;

/// EP prunes over at least one derivative and folds at most kEPMaxNbins bins.
/// PulsarSearchConfig also configures FFA searches, which keep their own
/// limits, so every EP entry point checks this before it allocates.
inline void check_ep_supported(const PulsarSearchConfig& cfg,
                               std::string_view caller) {
    if (cfg.get_nparams() < 2) {
        throw std::invalid_argument(fmt::format(
            "{}: frequency-only EP is unsupported; use FFA", caller));
    }
    if (cfg.get_nbins_max() > kEPMaxNbins) {
        throw std::invalid_argument(
            fmt::format("{}: EP nbins_max must be <= {} (got {})", caller,
                        kEPMaxNbins, cfg.get_nbins_max()));
    }
}

/// EP seeds pruning on the FFA's acceleration x frequency grid and branches
/// every higher derivative (jerk, snap, crackle) in pruning, so the FFA's
/// final grid (@p param_counts, frequency last) must hold one point of each.
/// A derivative whose range spans more than one segment-level step gets
/// several; more segments shorten the step's span until it fits.
inline void check_ep_seed_grid(std::span<const SizeType> param_counts,
                               std::string_view caller) {
    constexpr std::array<std::string_view, 3> kNames = {"jerk", "snap",
                                                        "crackle"};
    const SizeType n_counts                          = param_counts.size();
    for (SizeType i = 0; i + 2 < n_counts; ++i) {
        if (param_counts[i] <= 1) {
            continue;
        }
        const SizeType order = n_counts - 3 - i;
        throw std::invalid_argument(fmt::format(
            "{}: EP seeds pruning on an acceleration x frequency FFA grid, "
            "but the FFA grids {} over {} points; use more segments",
            caller, order < kNames.size() ? kNames[order] : "a derivative",
            param_counts[i]));
    }
}

/// The number of samples an EP search analyses of inputs of @p n_e and
/// @p n_v samples: cfg.get_nsamps(), from a series of either that length or
/// the length the config was built for (an EP config drops the tail that
/// does not fill a whole segment).
inline SizeType ep_input_nsamps(const FFASearchConfig& cfg,
                                SizeType n_e,
                                SizeType n_v,
                                std::string_view caller) {
    const SizeType nsamps = cfg.get_nsamps();
    if (n_e != n_v || (n_e != nsamps && n_e != cfg.get_nsamps_input())) {
        throw std::invalid_argument(fmt::format(
            "{}: ts_e and ts_v must have {} samples (or the {} the config was "
            "built for), got {} and {}",
            caller, nsamps, cfg.get_nsamps_input(), n_e, n_v));
    }
    if (n_e > nsamps) {
        spdlog::info("{}: EP analyses the first {} of {} samples ({} segments "
                     "of size {}); the last {} samples fill no brute segment",
                     caller, nsamps, n_e, cfg.get_nsegments(),
                     cfg.get_bseg_ffa(), n_e - nsamps);
    }
    return nsamps;
}

/// The samples an EP search analyses (see ep_input_nsamps()).
inline std::pair<std::span<const float>, std::span<const float>>
ep_input_series(const FFASearchConfig& cfg,
                std::span<const float> ts_e,
                std::span<const float> ts_v,
                std::string_view caller) {
    const SizeType nsamps =
        ep_input_nsamps(cfg, ts_e.size(), ts_v.size(), caller);
    return {ts_e.first(nsamps), ts_v.first(nsamps)};
}

/// EP reports survivors at the centre of the pruned interval, which must be
/// the centre of the analysed data: an EP config keeps whole segments only,
/// so the last pruning level covers exactly get_tobs().
inline void check_report_epoch(const FFASearchConfig& cfg,
                               double t_ref_report,
                               std::string_view caller) {
    const double t_mid = 0.5 * cfg.get_tobs();
    if (std::abs(t_ref_report - t_mid) > 1e-9 * std::max(1.0, t_mid)) {
        throw std::logic_error(fmt::format(
            "{}: survivors are referenced at t = {} s, not at the data "
            "midpoint {} s",
            caller, t_ref_report, t_mid));
    }
}

} // namespace loki::search::detail
