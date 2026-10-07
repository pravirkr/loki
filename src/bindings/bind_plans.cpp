#include <cstddef>
#include <string>
#include <string_view>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/common/plans.hpp"
#include "loki/loki.hpp"

#include "bindings/bind.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using algorithms::FFARegionStats;
using plans::FFAPlanBase;
using search::FFASearchConfig;
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_plans(py::module_& m) {
    auto m_plans = m.def_submodule("plans", "Plans submodule");
    PYBIND11_NUMPY_DTYPE(coord::FFACoord, i_tail, shift_tail, i_head,
                         shift_head);
    PYBIND11_NUMPY_DTYPE(coord::FFACoordFreq, idx, shift);
    PYBIND11_NUMPY_DTYPE(coord::FFARegion, f_start, f_end, nbins);

    // Bind FFAPlanBase
    py::class_<FFAPlanBase>(m_plans, "FFAPlanBase")
        .def(py::init<FFASearchConfig>(), py::arg("cfg"))
        .def_property_readonly("n_params", &FFAPlanBase::get_n_params)
        .def_property_readonly("n_levels", &FFAPlanBase::get_n_levels)
        .def_property_readonly("segment_lens",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(
                                       self.get_segment_lens());
                               })
        .def_property_readonly("nsegments",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(self.get_nsegments());
                               })
        .def_property_readonly("tsegments",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(self.get_tsegments());
                               })
        .def_property_readonly("ncoords",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(self.get_ncoords());
                               })
        .def_property_readonly("ncoords_lb",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(self.get_ncoords_lb());
                               })
        .def_property_readonly("ncoords_offsets",
                               [](const FFAPlanBase& self) {
                                   return as_pyarray_ref(
                                       self.get_ncoords_offsets());
                               })
        .def_property_readonly("param_counts",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(
                                       self.get_param_counts());
                               })
        .def_property_readonly("param_cart_strides",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(
                                       self.get_param_cart_strides());
                               })
        .def_property_readonly("dparams",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(self.get_dparams());
                               })
        .def_property_readonly("dparams_actual",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(
                                       self.get_dparams_actual());
                               })
        .def_property_readonly("config", &FFAPlanBase::get_config)
        .def_property_readonly("coord_size", &FFAPlanBase::get_coord_size)
        .def_property_readonly("coord_memory_usage",
                               &FFAPlanBase::get_coord_memory_usage)
        .def("resolve_coordinates",
             [](FFAPlanBase& self) {
                 return as_listof_pyarray(self.resolve_coordinates());
             })
        .def("resolve_coordinates_freq",
             [](FFAPlanBase& self) {
                 return as_listof_pyarray(self.resolve_coordinates_freq());
             })
        .def_property_readonly("param_grid",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(
                                       self.compute_param_grid_full());
                               })
        .def_property_readonly("params_dict",
                               [](const FFAPlanBase& self) {
                                   const auto params_map =
                                       self.get_params_dict();
                                   py::dict result;
                                   for (const auto& [key, value] : params_map) {
                                       result[py::str(key)] =
                                           as_pyarray_ref(value);
                                   }
                                   return result;
                               })
        .def("compute_param_grid",
             [](FFAPlanBase& self, SizeType ffa_level) {
                 return as_listof_pyarray(self.compute_param_grid(ffa_level));
             })
        .def(
            "get_branching_pattern_approx",
            [](FFAPlanBase& self, std::string_view poly_basis, SizeType ref_seg,
               IndexType isuggest) {
                return as_pyarray_ref(self.get_branching_pattern_approx(
                    poly_basis, ref_seg, isuggest));
            },
            py::arg("poly_basis") = "taylor", py::arg("ref_seg") = 0,
            py::arg("isuggest") = 0)
        .def(
            "get_branching_pattern",
            [](FFAPlanBase& self, std::string_view poly_basis,
               SizeType ref_seg) {
                return as_pyarray_ref(
                    self.get_branching_pattern(poly_basis, ref_seg));
            },
            py::arg("poly_basis") = "taylor", py::arg("ref_seg") = 0);

    // Bind FFARegionPlanner
    py::class_<FFARegionStats>(m_plans, "FFARegionStats")
        .def(py::init<SizeType, SizeType, SizeType, SizeType, SizeType,
                      SizeType, SizeType, SizeType, bool, bool>(),
             py::arg("max_buffer_size"), py::arg("max_coord_size"),
             py::arg("max_ncoords"), py::arg("max_ffa_levels"),
             py::arg("max_scores_scratch"), py::arg("n_params"),
             py::arg("n_samps"), py::arg("max_passing_candidates"),
             py::arg("use_fourier"), py::arg("use_gpu") = false)
        .def_property_readonly("max_buffer_size",
                               &FFARegionStats::get_max_buffer_size)
        .def_property_readonly("max_coord_size",
                               &FFARegionStats::get_max_coord_size)
        .def_property_readonly("max_ncoords", &FFARegionStats::get_max_ncoords)
        .def_property_readonly("max_ffa_levels",
                               &FFARegionStats::get_max_ffa_levels)
        .def_property_readonly("max_buffer_size_time",
                               &FFARegionStats::get_max_buffer_size_time)
        .def_property_readonly("max_scores_scratch_size",
                               &FFARegionStats::get_max_scores_scratch_size)
        .def_property_readonly("max_candidates",
                               &FFARegionStats::get_max_candidates)
        .def_property_readonly("write_param_sets_size",
                               &FFARegionStats::get_write_param_sets_size)
        .def_property_readonly("buffer_memory_usage",
                               &FFARegionStats::get_buffer_memory_usage)
        .def_property_readonly("coord_memory_usage",
                               &FFARegionStats::get_coord_memory_usage)
        .def_property_readonly("extra_memory_usage",
                               &FFARegionStats::get_extra_memory_usage)
        .def_property_readonly("device_extra_memory_usage",
                               &FFARegionStats::get_device_extra_memory_usage)
        .def_property_readonly("freq_sweep_memory_usage",
                               &FFARegionStats::get_freq_sweep_memory_usage);

    bind_ffa_plan<float>(m_plans, "FFAPlanTime");
    bind_ffa_plan<ComplexType>(m_plans, "FFAPlanFourier");
    bind_ffa_region_planner<float>(m_plans, "FFARegionPlannerTime");
    bind_ffa_region_planner<ComplexType>(m_plans, "FFARegionPlannerFourier");
    m_plans.def("generate_ffa_regions", &algorithms::generate_ffa_regions,
                py::arg("p_min"), py::arg("p_max"), py::arg("tsamp"),
                py::arg("nbins_min"), py::arg("eta_min"),
                py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024);

    // FFA submodule
}

} // namespace loki
