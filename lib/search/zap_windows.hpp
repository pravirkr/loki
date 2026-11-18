#pragma once

/**
 * @file zap_windows.hpp
 * @brief The `in_zap_window` test of candidate rows. Internal.
 */

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "loki/search/candidates.hpp"

namespace loki::search::detail {

/// Zapped frequency windows, sorted and merged, and the test of the
/// `in_zap_window` column (CandidateOptions::zap_windows).
class ZapWindowIndex {
public:
    explicit ZapWindowIndex(std::span<const FrequencyWindow> windows)
        : m_windows(windows.begin(), windows.end()) {
        std::ranges::sort(m_windows, {}, &FrequencyWindow::lo);
        std::vector<FrequencyWindow> merged;
        for (const auto& w : m_windows) {
            if (!merged.empty() && w.lo <= merged.back().hi) {
                merged.back().hi = std::max(merged.back().hi, w.hi);
            } else {
                merged.push_back(w);
            }
        }
        m_windows = std::move(merged);
    }

    [[nodiscard]] bool empty() const noexcept { return m_windows.empty(); }

    /// @p f lies in a window.
    [[nodiscard]] bool contains(double f) const noexcept {
        // The first window that ends at or above f.
        const auto it =
            std::ranges::lower_bound(m_windows, f, {}, &FrequencyWindow::hi);
        return it != m_windows.end() && it->lo <= f;
    }

    /// @p f0 * p / q lies in a window for some p, q in [1, ratio_max].
    [[nodiscard]] bool flags(double f0) const noexcept {
        constexpr auto kMax = CandidateOptions::kZapWindowRatioMax;
        if (m_windows.empty()) {
            return false;
        }
        for (std::uint32_t p = 1; p <= kMax; ++p) {
            for (std::uint32_t q = 1; q <= kMax; ++q) {
                if (contains(f0 * static_cast<double>(p) /
                             static_cast<double>(q))) {
                    return true;
                }
            }
        }
        return false;
    }

    [[nodiscard]] std::span<const FrequencyWindow> windows() const noexcept {
        return m_windows;
    }

private:
    std::vector<FrequencyWindow> m_windows;
};

} // namespace loki::search::detail
