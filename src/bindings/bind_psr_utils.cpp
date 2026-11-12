#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/loki.hpp"
#include "loki/utils/psr_utils.hpp"

#include "bindings/bind.hpp"
#include "pybind_utils.hpp"

namespace loki {
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_psr_utils(py::module_& m) {
    auto m_psr_utils = m.def_submodule("psr_utils", "PSR utils submodule");
    m_psr_utils.def(
        "shift_taylor_params_d_f",
        [](const PyArrayT<double>& pset_cur, double delta_t) {
            auto [pset_prev, delay] = psr_utils::shift_taylor_params_d_f(
                to_span<const double>(pset_cur), delta_t);
            return std::make_tuple(as_pyarray_ref(pset_prev), delay);
        },
        py::arg("pset_cur"), py::arg("delta_t"));
    m_psr_utils.def(
        "get_phase_idx",
        [](double delta_t, double freq, SizeType nbins, double delay) {
            return psr_utils::phase_index(delta_t, freq, nbins, delay);
        },
        py::arg("delta_t"), py::arg("freq"), py::arg("nbins"),
        py::arg("delay"));
    m_psr_utils.def(
        "leaf_phase",
        [](const PyArrayT<double>& leaf, const PyArrayT<double>& times,
           double t_ref, double propagator_significance) {
            if (leaf.ndim() != 2 || leaf.shape(1) != 2 || leaf.shape(0) < 3) {
                throw std::invalid_argument(
                    "leaf must have shape (n_params + 2, 2), n_params >= 1");
            }
            const auto n_params = static_cast<SizeType>(leaf.shape(0)) - 2;
            std::vector<double> phase(static_cast<SizeType>(times.size()));
            const auto model = psr_utils::leaf_phase(
                to_span<const double>(leaf), n_params,
                to_span<const double>(times), t_ref, std::span<double>(phase),
                propagator_significance);
            const auto* name = model == psr_utils::LeafModel::kCircularSnap
                                   ? "circular_snap"
                               : model == psr_utils::LeafModel::kCircularCrackle
                                   ? "circular_crackle"
                                   : "taylor";
            return std::make_tuple(as_pyarray(std::move(phase)),
                                   std::string(name));
        },
        py::arg("leaf"), py::arg("times"), py::arg("t_ref"), py::kw_only(),
        py::arg("propagator_significance") = 2.0,
        R"doc(Pulse phase, in cycles, of an EP leaf at each time.

leaf is one row of an EP result's param_sets, shape (n_params + 2, 2), with
its parameters at t_ref (the middle of the searched span). The phase follows
the model the search propagated the leaf with. Returns (phase, model), the
model one of "taylor", "circular_snap" or "circular_crackle".)doc");
}

} // namespace loki
