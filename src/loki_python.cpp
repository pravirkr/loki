#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#include "loki/common/backend.hpp"
#include "loki/common/interrupt.hpp"
#include "loki/common/progress.hpp"

#include "bindings/bind.hpp"

namespace loki {

namespace py = pybind11;

PYBIND11_MODULE(libloki, m) {
    m.doc() = "Python bindings for the loki library";

    py::add_ostream_redirect(m, "ostream_redirect");
    // A long call stopped by Ctrl-C (see LongCall in pybind_utils.hpp).
    // NOLINTNEXTLINE(performance-unnecessary-value-param): pybind11's signature
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const loki::Interrupted&) {
            // NOLINTNEXTLINE(misc-include-cleaner): Python.h via pybind11
            PyErr_SetNone(PyExc_KeyboardInterrupt);
        }
    });

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
        "Set the level of loki's log messages: 'info' logs about one line "
        "per phase, 'debug' adds the per-chunk and per-run detail, "
        "'warning' keeps only problems.");
    m.def(
        "get_log_level",
        [] {
            const auto name =
                spdlog::level::to_string_view(spdlog::get_level());
            return std::string(name.data(), name.size());
        },
        "The current level of loki's log messages.");
    m.def(
        "set_progress_mode",
        [](std::string_view mode) {
            loki::set_progress_mode(loki::parse_progress_mode(mode));
        },
        py::arg("mode"),
        "How progress bars are drawn: 'auto' (default: live bars on an "
        "interactive terminal, a few plain lines elsewhere, e.g. in Jupyter "
        "or a log file), 'rich', 'plain' or 'off'. With 'auto' the "
        "environment variable LOKI_PROGRESS can set the mode. show_progress="
        "False never draws, whatever the mode.");
    m.def(
        "get_progress_mode",
        [] { return std::string(loki::to_string(loki::get_progress_mode())); },
        "The progress mode last set with set_progress_mode().");

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
