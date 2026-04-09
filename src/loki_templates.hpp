#pragma once

#include "pybind_utils.hpp"

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"

namespace loki {
using algorithms::EPMultiPass;
using algorithms::FFA;
using plans::FFAPlan;
using plans::FFAPlanBase;
using regions::FFARegionPlanner;
using search::PulsarSearchConfig;

namespace py = pybind11;

// Template function to bind FFAPlanMetadata<T>
template <SupportedFoldType FoldType>
void bind_ffa_plan(py::module& m, const std::string& name) {
    py::class_<FFAPlan<FoldType>, FFAPlanBase>(m, name.c_str())
        .def(py::init<PulsarSearchConfig>(), py::arg("cfg"))
        .def_property_readonly("fold_shapes",
                               [](const FFAPlan<FoldType>& self) {
                                   return as_listof_pyarray(
                                       self.get_fold_shapes());
                               })
        .def_property_readonly("brute_fold_size",
                               &FFAPlan<FoldType>::get_brute_fold_size)
        .def_property_readonly("fold_size", &FFAPlan<FoldType>::get_fold_size)
        .def_property_readonly("fold_size_time",
                               &FFAPlan<FoldType>::get_fold_size_time)
        .def_property_readonly("buffer_size",
                               &FFAPlan<FoldType>::get_buffer_size)
        .def_property_readonly("buffer_size_time",
                               &FFAPlan<FoldType>::get_buffer_size_time)
        .def_property_readonly("buffer_memory_usage",
                               &FFAPlan<FoldType>::get_buffer_memory_usage);
}

// Template function to bind FFA<T>
template <SupportedFoldType FoldType>
void bind_ffa_class(py::module& m, const std::string& name) {
    auto cls = py::class_<FFA<FoldType>>(m, name.c_str())
                   .def(py::init<PulsarSearchConfig, bool>(), py::arg("cfg"),
                        py::arg("show_progress") = true)
                   .def_property_readonly("plan", &FFA<FoldType>::get_plan);

    // Standard execute
    cls.def(
        "execute",
        [](FFA<FoldType>& self, const PyArrayT<float>& ts_e,
           const PyArrayT<float>& ts_v, PyArrayT<FoldType>& fold) {
            self.execute(to_span<const float>(ts_e), to_span<const float>(ts_v),
                         to_span<FoldType>(fold));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    // Specialized execute for ComplexType (return to time domain)
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        cls.def(
            "execute",
            [](FFA<FoldType>& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));
    }
}

// Template function to bind FFARegionPlanner<T>
template <typename T>
void bind_ffa_region_planner(py::module& m, const std::string& name) {
    py::class_<FFARegionPlanner<T>>(m, name.c_str())
        .def(py::init<PulsarSearchConfig>(), py::arg("cfg"))
        .def_property_readonly("cfgs", &FFARegionPlanner<T>::get_cfgs)
        .def_property_readonly("nregions", &FFARegionPlanner<T>::get_nregions)
        .def_property_readonly("stats", &FFARegionPlanner<T>::get_stats);
}

// Template function to bind EPMultiPass<T>
template <SupportedFoldType FoldType>
void bind_ep_multi_pass(py::module& m, const std::string& name) {
    auto cls =
        py::class_<EPMultiPass<FoldType>>(m, name.c_str())
            .def(py::init<const PulsarSearchConfig&, const std::vector<float>&,
                          std::optional<SizeType>,
                          std::optional<std::vector<SizeType>>, SizeType,
                          SizeType, std::string_view, bool>(),
                 py::arg("cfg"), py::arg("threshold_scheme"),
                 py::arg("n_runs")   = std::nullopt,
                 py::arg("ref_segs") = std::nullopt,
                 py::arg("max_sugg") = 1U << 18U, py::arg("batch_size") = 1024U,
                 py::arg("poly_basis")    = "taylor",
                 py::arg("show_progress") = true);

    // Standard execute
    cls.def(
        "execute",
        [](EPMultiPass<FoldType>& self, const PyArrayT<float>& ts_e,
           const PyArrayT<float>& ts_v, std::string_view outdir,
           std::string_view file_prefix) {
            self.execute(to_span<const float>(ts_e), to_span<const float>(ts_v),
                         outdir, file_prefix);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
        py::arg("file_prefix"));
}

} // namespace loki