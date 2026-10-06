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

void bind_ffa(py::module_& m) {
    auto m_ffa = m.def_submodule("ffa", "FFA submodule");
    bind_ffa_class<float>(m_ffa, "FFATime");
    bind_ffa_class<ComplexType>(m_ffa, "FFAFourier");

    m_ffa.def(
        "compute_ffa_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress,
           std::string_view backend, int device) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress, make_exec(backend, device));
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false,
        py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);

    m_ffa.def(
        "compute_ffa_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress,
           std::string_view backend, int device) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress, make_exec(backend, device));
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false,
        py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);

    m_ffa.def(
        "compute_ffa_fourier_return_to_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress,
           std::string_view backend, int device) {
            auto [fold, ffa_plan] =
                algorithms::compute_ffa_fourier_return_to_time(
                    to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                    quiet, show_progress,
                    make_exec(backend, device));
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false,
        py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);

    m_ffa.def(
        "compute_ffa_scores",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress,
           std::string_view backend, int device) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_scores(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress, make_exec(backend, device));
            return std::make_tuple(as_pyarray(std::move(scores)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false,
        py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);

    py::class_<FFAFreqSweep>(m_ffa, "FFAFreqSweep")
        .def(
            py::init([](const FFASearchConfig& cfg, bool show_progress,
                        std::string_view backend, int device) {
                return std::make_unique<FFAFreqSweep>(
                    cfg, show_progress, make_exec(backend, device));
            }),
            py::arg("cfg"), py::arg("show_progress") = true,
            py::kw_only(), py::arg("backend") = "cpu",
            py::arg("device") = 0)
        .def(
            "execute",
            [](FFAFreqSweep& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
            py::arg("file_prefix") = "test");
}

} // namespace loki
