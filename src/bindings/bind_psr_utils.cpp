#include "bindings/bind.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/loki.hpp"
#include "loki/utils/psr_utils.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using algorithms::EPFreqSweep;
using algorithms::FFAFreqSweep;
using detection::MatchedFilter;
using plans::FFAPlanBase;
using regions::EPChunkConfig;
using regions::EPChunkStats;
using regions::EPRegionStats;
using regions::FFARegionStats;
using search::FFASearchConfig;
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
        [](double delta_t, double period, SizeType nbins, double delay) {
            return psr_utils::phase_index(delta_t, period, nbins, delay);
        },
        py::arg("delta_t"), py::arg("period"), py::arg("nbins"),
        py::arg("delay"));
}

} // namespace loki
