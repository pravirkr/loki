#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#include "loki/common/backend.hpp"

#include "bindings/bind.hpp"

namespace loki {

namespace py = pybind11;

PYBIND11_MODULE(libloki, m) {
    m.doc() = "Python bindings for the loki library";

    py::add_ostream_redirect(m, "ostream_redirect");

    m.def(
        "available_backends",
        [] {
            std::vector<std::string> names;
            for (const auto b : loki::available_backends()) {
                names.emplace_back(loki::to_string(b));
            }
            return names;
        },
        "Backends compiled into this build, e.g. ['cpu'] or ['cpu', 'cuda'].");

    m.def(
        "set_log_level",
        [](std::string_view level) {
            const auto parsed = spdlog::level::from_str(std::string(level));
            // from_str maps unknown names to off.
            if (parsed == spdlog::level::off && level != "off") {
                throw std::invalid_argument(
                    "level must be one of trace, debug, info, warning, "
                    "error, critical, off");
            }
            spdlog::set_level(parsed);
        },
        py::arg("level"),
        "Set the level of loki's log messages, e.g. 'warning' to hide the "
        "per-chunk and per-run info lines.");
    m.def(
        "get_log_level",
        [] {
            const auto name =
                spdlog::level::to_string_view(spdlog::get_level());
            return std::string(name.data(), name.size());
        },
        "The current level of loki's log messages.");

    bind_scores(m);
    bind_thresholds(m);
    bind_fold(m);
    bind_io(m);
    bind_configs(m);
    bind_plans(m);
    bind_ffa(m);
    bind_psr_utils(m);
    bind_prune(m);
    bind_simulation(m);
}

} // namespace loki
