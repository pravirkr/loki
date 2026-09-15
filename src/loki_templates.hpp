#pragma once

#include "pybind_utils.hpp"

#include <format>
#include <limits>

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

// Bind the RFI-control configuration types used by EPMultiPass
inline void bind_prune_rfi(py::module& m) {
    using algorithms::kBirdieAccelPad;
    using algorithms::kHarvestDisabled;
    using algorithms::make_birdie_window;
    using algorithms::make_default_harvest_scheme;
    using algorithms::make_harvest_window;
    using algorithms::make_pulsar_window;
    using algorithms::ParamWindow;
    using algorithms::PruneRFIConfig;

    m.attr("HARVEST_DISABLED") = kHarvestDisabled;
    m.attr("BIRDIE_ACCEL_PAD") = kBirdieAccelPad;

    py::class_<ParamWindow>(m, "ParamWindow",
                            "Exclusion window in physical units (Hz, m/s^2). "
                            "The default acceleration range covers the whole "
                            "grid.")
        .def(py::init([](double f_lo, double f_hi, double a_lo, double a_hi) {
                 return ParamWindow{
                     .f_lo = f_lo, .f_hi = f_hi, .a_lo = a_lo, .a_hi = a_hi,};
             }),
             py::arg("f_lo"), py::arg("f_hi"),
             py::arg("a_lo") = std::numeric_limits<double>::lowest(),
             py::arg("a_hi") = std::numeric_limits<double>::max())
        .def_readwrite("f_lo", &ParamWindow::f_lo)
        .def_readwrite("f_hi", &ParamWindow::f_hi)
        .def_readwrite("a_lo", &ParamWindow::a_lo)
        .def_readwrite("a_hi", &ParamWindow::a_hi)
        .def("__repr__", [](const ParamWindow& w) {
            return std::format(
                "ParamWindow(f_lo={}, f_hi={}, a_lo={}, a_hi={})", w.f_lo,
                w.f_hi, w.a_lo, w.a_hi);
        });

    py::class_<PruneRFIConfig>(m, "PruneRFIConfig",
                               "RFI-control configuration for the EP pruning "
                               "search (pulsar mask, early harvesting, "
                               "stage-consistency veto). All mechanisms are "
                               "opt-in.")
        .def(py::init([](std::vector<ParamWindow> pulsar_mask,
                         SizeType n_harmonics, std::vector<float> harvest_scheme,
                         double harvest_mask_ntiles, SizeType max_harvests,
                         bool harvest_store_folds, bool impulsive_veto,
                         double impulsive_kappa, SizeType impulsive_min_level,
                         float impulsive_min_snr) {
                 PruneRFIConfig cfg;
                 cfg.pulsar_mask         = std::move(pulsar_mask);
                 cfg.n_harmonics         = n_harmonics;
                 cfg.harvest_scheme      = std::move(harvest_scheme);
                 cfg.harvest_mask_ntiles = harvest_mask_ntiles;
                 cfg.max_harvests        = max_harvests;
                 cfg.harvest_store_folds = harvest_store_folds;
                 cfg.impulsive_veto      = impulsive_veto;
                 cfg.impulsive_kappa     = impulsive_kappa;
                 cfg.impulsive_min_level = impulsive_min_level;
                 cfg.impulsive_min_snr   = impulsive_min_snr;
                 return cfg;
             }),
             py::kw_only(),
             py::arg("pulsar_mask")         = std::vector<ParamWindow>(),
             py::arg("n_harmonics")         = 0U,
             py::arg("harvest_scheme")      = std::vector<float>(),
             py::arg("harvest_mask_ntiles") = 4.0,
             py::arg("max_harvests")        = 4096U,
             py::arg("harvest_store_folds") = true,
             py::arg("impulsive_veto")      = false,
             py::arg("impulsive_kappa")     = 6.0,
             py::arg("impulsive_min_level") = 6U,
             py::arg("impulsive_min_snr")   = 8.0F)
        .def_readwrite("pulsar_mask", &PruneRFIConfig::pulsar_mask)
        .def_readwrite("n_harmonics", &PruneRFIConfig::n_harmonics)
        .def_readwrite("harvest_scheme", &PruneRFIConfig::harvest_scheme)
        .def_readwrite("harvest_mask_ntiles",
                       &PruneRFIConfig::harvest_mask_ntiles)
        .def_readwrite("max_harvests", &PruneRFIConfig::max_harvests)
        .def_readwrite("harvest_store_folds",
                       &PruneRFIConfig::harvest_store_folds)
        .def_readwrite("impulsive_veto", &PruneRFIConfig::impulsive_veto)
        .def_readwrite("impulsive_kappa", &PruneRFIConfig::impulsive_kappa)
        .def_readwrite("impulsive_min_level",
                       &PruneRFIConfig::impulsive_min_level)
        .def_readwrite("impulsive_min_snr", &PruneRFIConfig::impulsive_min_snr)
        .def("validate", &PruneRFIConfig::validate, py::arg("nsegments"))
        .def_property_readonly("is_active", &PruneRFIConfig::is_active)
        .def_property_readonly("has_harvest", &PruneRFIConfig::has_harvest);

    m.def("make_pulsar_window", &make_pulsar_window, py::arg("f"), py::arg("a"),
          py::arg("tobs"), py::arg("f_pad") = 0.0,
          py::arg("a_pad") = std::numeric_limits<double>::max(),
          "Exclusion window covering the full Doppler sweep of a source (f, a) "
          "over an observation of length tobs, padded by f_pad (Hz) and a_pad "
          "(m/s^2; default masks all accelerations).");

    m.def("make_birdie_window", &make_birdie_window, py::arg("f"),
          py::arg("f_pad"), py::arg("a_pad") = kBirdieAccelPad,
          "Exclusion window for a terrestrial (zero-acceleration) birdie.");

    m.def("make_harvest_window", &make_harvest_window, py::arg("f"),
          py::arg("a"), py::arg("df"), py::arg("da"), py::arg("t_ref"),
          py::arg("tobs"), py::arg("ntiles"),
          "Exclusion window around a harvested candidate.");

    m.def(
        "make_default_harvest_scheme",
        [](const std::vector<float>& threshold_scheme, SizeType min_level,
           float offset, float min_snr) {
            return make_default_harvest_scheme(threshold_scheme, min_level,
                                               offset, min_snr);
        },
        py::arg("threshold_scheme"), py::arg("min_level") = 10U,
        py::arg("offset") = 10.0F, py::arg("min_snr") = 15.0F,
        "Conservative harvest scheme derived from the threshold scheme.");
}

// Template function to bind EPMultiPass<T>
template <SupportedFoldType FoldType>
void bind_ep_multi_pass(py::module& m, const std::string& name) {
    auto cls =
        py::class_<EPMultiPass<FoldType>>(m, name.c_str())
            .def(py::init<const PulsarSearchConfig&, const std::vector<float>&,
                          std::optional<SizeType>,
                          std::optional<std::vector<SizeType>>,
                          const std::vector<SizeType>&, SizeType, SizeType,
                          std::string_view, bool, algorithms::PruneRFIConfig>(),
                 py::arg("cfg"), py::arg("threshold_scheme"),
                 py::arg("n_runs")        = std::nullopt,
                 py::arg("ref_segs")      = std::nullopt,
                 py::arg("ascend_levels") = std::vector<SizeType>(),
                 py::arg("max_sugg") = 1U << 18U, py::arg("batch_size") = 1024U,
                 py::arg("poly_basis")    = "taylor",
                 py::arg("show_progress") = true,
                 py::arg("rfi_config")    = algorithms::PruneRFIConfig());

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