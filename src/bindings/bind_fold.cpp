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

#include "bindings/bind.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using algorithms::EPChunkConfig;
using algorithms::EPChunkStats;
using algorithms::EPRegionStats;
using algorithms::FFARegionStats;
using detection::MatchedFilter;
using pipelines::EPFreqSweep;
using pipelines::FFAFreqSweep;
using plans::FFAPlanBase;
using search::FFASearchConfig;
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_fold(py::module_& m) {
    auto m_fold = m.def_submodule("fold", "Fold submodule");
    m_fold.def(
        "compute_brute_fold_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads,
           std::string_view backend, int device) {
            return as_pyarray(algorithms::compute_brute_fold<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, make_exec(backend, device, nthreads)));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0);
    m_fold.def(
        "compute_brute_fold_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads,
           std::string_view backend, int device) {
            return as_pyarray(algorithms::compute_brute_fold<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, make_exec(backend, device, nthreads)));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0);
}

} // namespace loki
