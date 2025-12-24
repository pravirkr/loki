#pragma once

#include "pybind_utils.hpp"

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"

namespace loki {
using algorithms::FFA;
using algorithms::Prune;
using plans::FFAPlan;
using plans::FFAPlanBase;
using plans::FFARegionPlanner;
using plans::FFARegionStats;
using search::PulsarSearchConfig;

namespace py = pybind11;

// Template function to bind FFAPlan<T>
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

// Template function to bind FFARegionStats<T>
template <typename T>
void bind_ffa_region_stats(py::module& m, const std::string& name) {
    py::class_<FFARegionStats<T>>(m, name.c_str())
        .def(py::init<SizeType, SizeType, SizeType, SizeType, SizeType>(),
             py::arg("max_buffer_size"), py::arg("max_coord_size"),
             py::arg("max_ncoords"), py::arg("n_widths"), py::arg("n_params"))
        .def_property_readonly("max_buffer_size",
                               &FFARegionStats<T>::get_max_buffer_size)
        .def_property_readonly("max_coord_size",
                               &FFARegionStats<T>::get_max_coord_size)
        .def_property_readonly("max_ncoords",
                               &FFARegionStats<T>::get_max_ncoords)
        .def_property_readonly("max_buffer_size_time",
                               &FFARegionStats<T>::get_max_buffer_size_time)
        .def_property_readonly("max_scores_size",
                               &FFARegionStats<T>::get_max_scores_size)
        .def_property_readonly("max_param_sets_size",
                               &FFARegionStats<T>::get_max_param_sets_size)
        .def_property_readonly("buffer_memory_usage",
                               &FFARegionStats<T>::get_buffer_memory_usage)
        .def_property_readonly("coord_memory_usage",
                               &FFARegionStats<T>::get_coord_memory_usage)
        .def_property_readonly("extra_memory_usage",
                               &FFARegionStats<T>::get_extra_memory_usage)
        .def_property_readonly("manager_memory_usage",
                               &FFARegionStats<T>::get_manager_memory_usage);
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

// Template function to bind Prune<T>
template <typename FoldType>
void bind_prune_class(py::module& m, const std::string& name) {
    py::class_<Prune<FoldType>>(m, name.c_str())
        .def(py::init<const FFAPlan<FoldType>&, const PulsarSearchConfig&,
                      const PyArrayT<float>&, SizeType, SizeType,
                      std::string_view>(),
             py::arg("ffa_plan"), py::arg("cfg"), py::arg("threshold_scheme"),
             py::arg("max_sugg") = 1U << 18U, py::arg("batch_size") = 1024U,
             py::arg("kind") = "taylor")
        .def_property_readonly("memory_usage",
                               &Prune<FoldType>::get_memory_usage)
        .def(
            "execute",
            [](Prune<FoldType>& self, const PyArrayT<FoldType>& ffa_fold,
               SizeType ref_seg, const std::string& outdir,
               const std::string& file_prefix, const std::string& kind) {
                self.execute(to_span<const FoldType>(ffa_fold), ref_seg, outdir,
                             file_prefix, kind);
            },
            py::arg("ffa_fold"), py::arg("ref_seg"), py::arg("outdir"),
            py::arg("file_prefix"), py::arg("kind"));
}

} // namespace loki