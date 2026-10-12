#pragma once

/**
 * @file ep_support.hpp
 * @brief The searches EP supports. Internal.
 */

#include <format>
#include <stdexcept>
#include <string_view>

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
        throw std::invalid_argument(std::format(
            "{}: frequency-only EP is unsupported; use FFA", caller));
    }
    if (cfg.get_nbins_max() > kEPMaxNbins) {
        throw std::invalid_argument(
            std::format("{}: EP nbins_max must be <= {} (got {})", caller,
                        kEPMaxNbins, cfg.get_nbins_max()));
    }
}

} // namespace loki::search::detail
