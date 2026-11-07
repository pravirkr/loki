#pragma once

/**
 * @file ep_support.hpp
 * @brief The searches EP supports. Internal.
 */

#include <algorithm>
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

/// The samples an EP search analyses: the leading cfg.get_nsamps() of a
/// series of either that length or the length the config was built for
/// (an EP config drops the tail that does not fill a whole segment).
inline std::pair<std::span<const float>, std::span<const float>>
ep_input_series(const FFASearchConfig& cfg,
                std::span<const float> ts_e,
                std::span<const float> ts_v,
                std::string_view caller) {
    const SizeType nsamps = cfg.get_nsamps();
    if (ts_e.size() != ts_v.size() ||
        (ts_e.size() != nsamps && ts_e.size() != cfg.get_nsamps_input())) {
        throw std::invalid_argument(fmt::format(
            "{}: ts_e and ts_v must have {} samples (or the {} the config was "
            "built for), got {} and {}",
            caller, nsamps, cfg.get_nsamps_input(), ts_e.size(), ts_v.size()));
    }
    if (ts_e.size() > nsamps) {
        spdlog::info("{}: EP analyses the first {} of {} samples ({} segments "
                     "of size {}); the last {} samples fill no brute segment",
                     caller, nsamps, ts_e.size(), cfg.get_nsegments(),
                     cfg.get_bseg_ffa(), ts_e.size() - nsamps);
    }
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
