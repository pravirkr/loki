#include "bindings/bind.hpp"

#include <string>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/common/backend.hpp"

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

    bind_scores(m);
    bind_thresholds(m);
    bind_fold(m);
    bind_configs(m);
    bind_plans(m);
    bind_ffa(m);
    bind_psr_utils(m);
    bind_prune(m);
    bind_io(m);
    bind_simulation(m);
}

} // namespace loki
