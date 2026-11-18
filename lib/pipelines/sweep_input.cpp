#include "lib/pipelines/sweep_input.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

#include "loki/common/backend.hpp"
#include "loki/io/search_series.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/io/search_series_impl.hpp"

namespace loki::pipelines::detail {

SweepInput sweep_input(const io::SearchSeries& series,
                       const Exec& exec,
                       std::string_view what) {
    if (series.empty()) {
        throw std::invalid_argument(std::string(what) + ": empty SearchSeries");
    }
    loki::detail::check_same_exec(series.exec(), exec, what);
    if (exec.backend == Backend::kCPU) {
        const auto [e, v] = io::detail::host_series(series, what);
        return SweepInput::host(e, v);
    }
    const auto [e, v] = io::detail::device_series(series, what);
    return {.host_e = {}, .host_v = {}, .device_e = e, .device_v = v};
}

} // namespace loki::pipelines::detail
