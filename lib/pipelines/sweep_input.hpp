#pragma once

/**
 * @file sweep_input.hpp
 * @brief The folding inputs a sweep engine searches. Internal.
 */

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/search_series.hpp"

namespace loki::pipelines::detail {

/// Host arrays (the span overloads of execute()), or device views of a
/// resident io::SearchSeries (CUDA only).
struct SweepInput {
    std::span<const float> host_e;
    std::span<const float> host_v;
    DeviceSpan<const float> device_e;
    DeviceSpan<const float> device_v;

    [[nodiscard]] static SweepInput host(std::span<const float> ts_e,
                                         std::span<const float> ts_v) {
        return {.host_e = ts_e, .host_v = ts_v, .device_e = {}, .device_v = {}};
    }

    [[nodiscard]] bool resident() const noexcept {
        return device_e.data() != nullptr;
    }

    /// The host arrays; engines without device input call this.
    [[nodiscard]] std::pair<std::span<const float>, std::span<const float>>
    host_spans(std::string_view what) const {
        if (resident()) {
            throw std::logic_error(std::string(what) +
                                   ": this engine takes host input only");
        }
        return {host_e, host_v};
    }
};

/// The input of @p series for a sweep built for @p exec. The series must be
/// on the same backend and device.
[[nodiscard]] SweepInput sweep_input(const io::SearchSeries& series,
                                     const Exec& exec,
                                     std::string_view what);

} // namespace loki::pipelines::detail
