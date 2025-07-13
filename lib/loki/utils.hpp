#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval = 299792458.0;

// Return the next power of two greater than or equal to n
SizeType next_power_of_two(SizeType n) noexcept;

// return max(x[i] - y[i])
float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size);

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

// return index of nearest value in sorted array
SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val);

std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

// Factory function for a standard ProgressBar
inline indicators::ProgressBar make_standard_bar(std::string_view prefix) {
    return indicators::ProgressBar{
        indicators::option::BarWidth{50},
        indicators::option::Start{" "},
        indicators::option::Fill{"\u2501"},
        indicators::option::Lead{"\u2501"},
        indicators::option::Remainder{" "},
        indicators::option::End{" "},
        indicators::option::ForegroundColor{indicators::Color::red},
        indicators::option::PrefixText{prefix},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::Stream{std::cerr}};
}

class ProgressGuard {
    bool m_show;
    // This class is used to hide the console cursor during progress bar updates
    // and restore it after the progress bar is done.
public:
    explicit ProgressGuard(bool show) : m_show(show) {
        if (m_show) {
            indicators::show_console_cursor(false);
        }
    }
    ~ProgressGuard() {
        if (m_show) {
            indicators::show_console_cursor(true);
        }
    }
    ProgressGuard(const ProgressGuard&)            = delete;
    ProgressGuard& operator=(const ProgressGuard&) = delete;
    ProgressGuard(ProgressGuard&&)                 = delete;
    ProgressGuard& operator=(ProgressGuard&&)      = delete;
};

void debug_tensor(const xt::xtensor<double, 3>& leaf_batch,
                  SizeType n_slices = 5);
void debug_tensor(const xt::xtensor<double, 2>& leaf_batch,
                  SizeType n_slices = 5);

} // namespace loki::utils
