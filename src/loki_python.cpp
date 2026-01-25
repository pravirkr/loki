#include "loki/loki.hpp"

#include "loki_templates.hpp"
#include "pybind_utils.hpp"

#include <cstddef>
#include <span>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/common/types.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"

namespace loki {
using algorithms::FFAManager;
using algorithms::PruneFourier;
using algorithms::PruneTime;
using algorithms::PruningManager;
using detection::MatchedFilter;
using plans::FFAPlanBase;
using search::PulsarSearchConfig;

namespace py = pybind11;

PYBIND11_MODULE(libloki, m) {
    m.doc() = "Python bindings for the loki library";

    py::add_ostream_redirect(m, "ostream_redirect");

    auto m_scores = m.def_submodule("scores", "Scores submodule");
    py::class_<MatchedFilter>(m_scores, "MatchedFilter")
        .def(py::init(
                 [](const py::array_t<size_t, py::array::c_style>& widths_arr,
                    size_t nprofiles, size_t nbins, const std::string& shape) {
                     auto widths = widths_arr.cast<std::vector<size_t>>();
                     return MatchedFilter(widths, nprofiles, nbins, shape);
                 }),
             py::arg("widths_arr"), py::arg("nprofiles"), py::arg("nbins"),
             py::arg("shape") = "boxcar")
        .def_property_readonly("ntemplates", &MatchedFilter::get_ntemplates)
        .def_property_readonly("nbins", &MatchedFilter::get_nbins)
        .def_property_readonly("templates",
                               [](MatchedFilter& mf) {
                                   // return a py::array_t<float> from a
                                   // std::vector<float> also reshape the array
                                   // to 2d using ntemplates and nbins
                                   return as_pyarray(mf.get_templates());
                               })
        .def("compute", [](MatchedFilter& self,
                           const py::array_t<float, py::array::c_style>& arr) {
            if (arr.ndim() != 1) {
                throw std::runtime_error(
                    "Input and output arrays must be 1-dimensional");
            }
            if (arr.size() != static_cast<ssize_t>(self.get_nbins())) {
                throw std::runtime_error("Input array size must match nbins");
            }
            const auto nprofiles  = arr.shape(0);
            const auto ntemplates = self.get_ntemplates();

            auto snr = py::array_t<float, py::array::c_style>(
                py::array::ShapeContainer(
                    {nprofiles, static_cast<ssize_t>(ntemplates)}));
            self.compute(std::span<const float>(arr.data(), arr.size()),
                         std::span<float>(snr.mutable_data(), snr.size()));
            return snr;
        });
    m_scores.def(
        "generate_box_width_trials",
        [](SizeType nbins, double ducy_max, double wtsp) {
            auto trials =
                detection::generate_box_width_trials(nbins, ducy_max, wtsp);
            return as_pyarray_ref(trials);
        },
        py::arg("nbins"), py::arg("ducy_max") = 0.3, py::arg("wtsp") = 1.5);

    m_scores.def(
        "snr_boxcar_1d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise) {
            if (arr.size() == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            auto out = PyArrayT<float>(widths.size());
            detection::snr_boxcar_1d(to_span<const float>(arr),
                                     to_span<const SizeType>(widths),
                                     to_span<float>(out), stdnoise);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F);
    m_scores.def(
        "snr_boxcar_2d_max",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise, int nthreads) {
            if (arr.ndim() != 2 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 2-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(1);

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_2d_max(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins, stdnoise, nthreads);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F,
        py::arg("nthreads") = 1);
    m_scores.def(
        "snr_boxcar_3d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           int nthreads) {
            if (arr.ndim() != 3 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 3-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(2);

            auto out = PyArrayT<float>({nprofiles, widths.size()});
            detection::snr_boxcar_3d(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins, nthreads);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("nthreads") = 1);
    m_scores.def(
        "snr_boxcar_3d_max",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           int nthreads) {
            if (arr.ndim() != 3 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 3-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(2);

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_3d_max(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins, nthreads);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("nthreads") = 1);

    auto m_thresholds = m.def_submodule("thresholds", "Thresholds submodule");
    using detection::DynamicThresholdScheme;

    PYBIND11_NUMPY_DTYPE(detection::State, success_h0, success_h1, complexity,
                         complexity_cumul, success_h1_cumul, nbranches,
                         threshold, cost, threshold_prev, success_h1_cumul_prev,
                         is_empty);
    py::class_<DynamicThresholdScheme>(m_thresholds, "DynamicThresholdScheme")
        .def(py::init([](const py::array_t<float>& branching_pattern,
                         float ref_ducy, SizeType nbins, SizeType ntrials,
                         SizeType nprobs, float prob_min, float snr_final,
                         SizeType nthresholds, float ducy_max, float wtsp,
                         float beam_width, SizeType trials_start,
                         bool use_lut_rng, int nthreads) {
                 return std::make_unique<DynamicThresholdScheme>(
                     std::span<const float>(branching_pattern.data(),
                                            branching_pattern.size()),
                     ref_ducy, nbins, ntrials, nprobs, prob_min, snr_final,
                     nthresholds, ducy_max, wtsp, beam_width, trials_start,
                     use_lut_rng, nthreads);
             }),
             py::arg("branching_pattern"), py::arg("ref_ducy"),
             py::arg("nbins") = 64, py::arg("ntrials") = 1024,
             py::arg("nprobs") = 10, py::arg("prob_min") = 0.05F,
             py::arg("snr_final") = 8.0F, py::arg("nthresholds") = 100,
             py::arg("ducy_max") = 0.3F, py::arg("wtsp") = 1.0F,
             py::arg("beam_width") = 0.7F, py::arg("trials_start") = 1,
             py::arg("use_lut_rng") = false, py::arg("nthreads") = 1)
        .def("run", &DynamicThresholdScheme::run, py::arg("thres_neigh") = 10)
        .def("save", &DynamicThresholdScheme::save, py::arg("outdir") = "./")
        .def_property_readonly("nstages", &DynamicThresholdScheme::get_nstages)
        .def_property_readonly("nthresholds",
                               &DynamicThresholdScheme::get_nthresholds)
        .def_property_readonly("nprobs", &DynamicThresholdScheme::get_nprobs)
        .def_property_readonly("branching_pattern",
                               [](DynamicThresholdScheme& self) {
                                   return as_pyarray(
                                       self.get_branching_pattern());
                               })
        .def_property_readonly("profile",
                               [](DynamicThresholdScheme& self) {
                                   return as_pyarray(self.get_profile());
                               })
        .def_property_readonly("thresholds",
                               [](DynamicThresholdScheme& self) {
                                   return as_pyarray(self.get_thresholds());
                               })
        .def_property_readonly("probs",
                               [](DynamicThresholdScheme& self) {
                                   return as_pyarray(self.get_probs());
                               })
        .def("get_states", [](DynamicThresholdScheme& self) {
            return as_pyarray(self.get_states());
        });
    m_thresholds.def(
        "evaluate_scheme",
        [](const PyArrayT<float>& thresholds,
           const PyArrayT<float>& branching_pattern, float ref_ducy,
           SizeType nbins, SizeType ntrials, float snr_final, float ducy_max,
           float wtsp) {
            return as_pyarray(detection::evaluate_scheme(
                to_span<const float>(thresholds),
                to_span<const float>(branching_pattern), ref_ducy, nbins,
                ntrials, snr_final, ducy_max, wtsp));
        },
        py::arg("thresholds"), py::arg("branching_pattern"),
        py::arg("ref_ducy"), py::arg("nbins") = 64, py::arg("ntrials") = 1024,
        py::arg("snr_final") = 8.0F, py::arg("ducy_max") = 0.3F,
        py::arg("wtsp") = 1.0F);
    m_thresholds.def(
        "determine_scheme",
        [](const PyArrayT<float>& survive_probs,
           const PyArrayT<float>& branching_pattern, float ref_ducy,
           SizeType nbins, SizeType ntrials, float snr_final, float ducy_max,
           float wtsp) {
            return as_pyarray(detection::determine_scheme(
                to_span<const float>(survive_probs),
                to_span<const float>(branching_pattern), ref_ducy, nbins,
                ntrials, snr_final, ducy_max, wtsp));
        },
        py::arg("survive_probs"), py::arg("branching_pattern"),
        py::arg("ref_ducy"), py::arg("nbins") = 64, py::arg("ntrials") = 1024,
        py::arg("snr_final") = 8.0F, py::arg("ducy_max") = 0.3F,
        py::arg("wtsp") = 1.0F);

    auto m_fold = m.def_submodule("fold", "Fold submodule");
    m_fold.def(
        "compute_brute_fold_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads) {
            return as_pyarray(algorithms::compute_brute_fold<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, nthreads));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1);
    m_fold.def(
        "compute_brute_fold_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads) {
            return as_pyarray(algorithms::compute_brute_fold<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, nthreads));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1);

    auto m_configs = m.def_submodule("configs", "Configs submodule");
    py::class_<PulsarSearchConfig>(m_configs, "PulsarSearchConfig")
        .def(py::init<SizeType, double, SizeType, double,
                      const std::vector<ParamLimitType>&, double, double, bool,
                      int, double, double, SizeType, SizeType,
                      std::optional<SizeType>, std::optional<SizeType>, double,
                      SizeType, double, double, bool>(),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("eta"), py::arg("param_limits"), py::arg("ducy_max") = 0.2,
             py::arg("wtsp") = 1.5, py::arg("use_fourier") = true,
             py::arg("nthreads") = 1, py::arg("max_process_memory_gb") = 8.0,
             py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024,
             py::arg("nbins_min_lossy_bf") = 64,
             py::arg("bseg_brute")         = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("snr_min") = 5.0,
             py::arg("prune_poly_order") = 3, py::arg("p_orb_min") = 1e-5,
             py::arg("snap_activation_threshold") = 5.0,
             py::arg("use_conservative_grid")     = false)
        .def_property_readonly("nsamps", &PulsarSearchConfig::get_nsamps)
        .def_property_readonly("tsamp", &PulsarSearchConfig::get_tsamp)
        .def_property_readonly("tobs", &PulsarSearchConfig::get_tobs)
        .def_property_readonly("nbins", &PulsarSearchConfig::get_nbins)
        .def_property_readonly("nbins_f", &PulsarSearchConfig::get_nbins_f)
        .def_property_readonly("eta", &PulsarSearchConfig::get_eta)
        .def_property_readonly("param_limits",
                               &PulsarSearchConfig::get_param_limits)
        .def_property_readonly("ducy_max", &PulsarSearchConfig::get_ducy_max)
        .def_property_readonly("wtsp", &PulsarSearchConfig::get_wtsp)
        .def_property_readonly("prune_poly_order",
                               &PulsarSearchConfig::get_prune_poly_order)
        .def_property_readonly("bseg_brute",
                               &PulsarSearchConfig::get_bseg_brute)
        .def_property_readonly("bseg_ffa", &PulsarSearchConfig::get_bseg_ffa)
        .def_property_readonly("p_orb_min", &PulsarSearchConfig::get_p_orb_min)
        .def_property_readonly("minimum_snap_cells",
                               &PulsarSearchConfig::get_minimum_snap_cells)
        .def_property_readonly("use_fourier",
                               &PulsarSearchConfig::get_use_fourier)
        .def_property_readonly("use_conservative_tile",
                               &PulsarSearchConfig::get_use_conservative_tile)
        .def_property_readonly("nthreads", &PulsarSearchConfig::get_nthreads)
        .def_property_readonly("tseg_brute",
                               &PulsarSearchConfig::get_tseg_brute)
        .def_property_readonly("tseg_ffa", &PulsarSearchConfig::get_tseg_ffa)
        .def_property_readonly("niters_ffa",
                               &PulsarSearchConfig::get_niters_ffa)
        .def_property_readonly("nparams", &PulsarSearchConfig::get_nparams)
        .def_property_readonly("param_names",
                               &PulsarSearchConfig::get_param_names)
        .def_property_readonly("f_min", &PulsarSearchConfig::get_f_min)
        .def_property_readonly("f_max", &PulsarSearchConfig::get_f_max)
        .def_property_readonly("score_widths",
                               [](const PulsarSearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_scoring_widths());
                               })
        .def_property_readonly("n_scoring_widths",
                               &PulsarSearchConfig::get_n_scoring_widths)
        .def("dparams_f", &PulsarSearchConfig::get_dparams_f,
             py::arg("tseg_cur"))
        .def("dparams", &PulsarSearchConfig::get_dparams, py::arg("tseg_cur"))
        .def("dparams_lim", &PulsarSearchConfig::get_dparams_lim,
             py::arg("tseg_cur"));

    // Plans submodule
    auto m_plans = m.def_submodule("plans", "Plans submodule");
    PYBIND11_NUMPY_DTYPE(plans::FFACoord, i_tail, shift_tail, i_head,
                         shift_head);
    PYBIND11_NUMPY_DTYPE(plans::FFACoordFreq, idx, shift);
    PYBIND11_NUMPY_DTYPE(plans::FFARegion, f_start, f_end, nbins);

    // Bind FFAPlanBase
    py::class_<FFAPlanBase>(m_plans, "FFAPlanBase")
        .def(py::init<PulsarSearchConfig>(), py::arg("cfg"))
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
        .def_property_readonly("params",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(self.get_params());
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
        .def_property_readonly("dparams_lim",
                               [](const FFAPlanBase& self) {
                                   return as_listof_pyarray(
                                       self.get_dparams_lim());
                               })
        .def_property_readonly("config", &FFAPlanBase::get_config)
        .def_property_readonly("coord_size", &FFAPlanBase::get_coord_size)
        .def_property_readonly("coord_memory_usage",
                               &FFAPlanBase::get_coord_memory_usage)
        .def_property_readonly("params_dict",
                               [](const FFAPlanBase& self) {
                                   auto params_map = self.get_params_dict();
                                   py::dict result;
                                   for (const auto& [key, value] : params_map) {
                                       result[py::str(key)] =
                                           as_pyarray_ref(value);
                                   }
                                   return result;
                               })
        .def("resolve_coordinates",
             [](FFAPlanBase& self) {
                 return as_listof_pyarray(self.resolve_coordinates());
             })
        .def("resolve_coordinates_freq",
             [](FFAPlanBase& self) {
                 return as_listof_pyarray(self.resolve_coordinates_freq());
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

    bind_ffa_plan<float>(m_plans, "FFAPlanTime");
    bind_ffa_plan<ComplexType>(m_plans, "FFAPlanFourier");
    bind_ffa_region_stats<float>(m_plans, "FFARegionStatsTime");
    bind_ffa_region_stats<ComplexType>(m_plans, "FFARegionStatsFourier");
    bind_ffa_region_planner<float>(m_plans, "FFARegionPlannerTime");
    bind_ffa_region_planner<ComplexType>(m_plans, "FFARegionPlannerFourier");
    m_plans.def("generate_ffa_regions", &plans::generate_ffa_regions,
                py::arg("p_min"), py::arg("p_max"), py::arg("tsamp"),
                py::arg("nbins_min"), py::arg("eta_min"),
                py::arg("growth_factor") = 2.0,
                py::arg("nbins_max")     = std::nullopt);

    // FFA submodule
    auto m_ffa = m.def_submodule("ffa", "FFA submodule");
    bind_ffa_class<float>(m_ffa, "FFATime");
    bind_ffa_class<ComplexType>(m_ffa, "FFAFourier");

    m_ffa.def(
        "compute_ffa_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_fourier_return_to_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] =
                algorithms::compute_ffa_fourier_return_to_time(
                    to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                    quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_scores",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool quiet, bool show_progress) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_scores(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    py::class_<FFAManager>(m_ffa, "FFAManager")
        .def(py::init<const PulsarSearchConfig&, bool>(), py::arg("cfg"),
             py::arg("show_progress") = true)
        .def(
            "execute",
            [](FFAManager& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
            py::arg("file_prefix") = "test");

    auto m_psr_utils = m.def_submodule("psr_utils", "PSR utils submodule");
    m_psr_utils.def(
        "shift_taylor_params_d_f",
        [](const PyArrayT<double>& pset_cur, double delta_t) {
            auto [pset_prev, delay] = transforms::shift_taylor_params_d_f(
                to_span<const double>(pset_cur), delta_t);
            return std::make_tuple(as_pyarray_ref(pset_prev), delay);
        },
        py::arg("pset_cur"), py::arg("delta_t"));
    m_psr_utils.def(
        "get_phase_idx",
        [](double delta_t, double period, SizeType nbins, double delay) {
            return psr_utils::get_phase_idx(delta_t, period, nbins, delay);
        },
        py::arg("delta_t"), py::arg("period"), py::arg("nbins"),
        py::arg("delay"));
    m_psr_utils.def(
        "ffa_taylor_resolve",
        [](const PyArrayT<double>& pset_cur,
           const std::vector<PyArrayT<double>>& param_arr, SizeType ffa_level,
           SizeType latter, double tseg_brute, SizeType nbins) {
            std::vector<std::vector<double>> param_vecs;
            param_vecs.reserve(param_arr.size());
            for (const auto& arr : param_arr) {
                param_vecs.emplace_back(arr.data(), arr.data() + arr.size());
            }
            auto [pindex_prev, relative_phase] = core::ffa_taylor_resolve(
                to_span<const double>(pset_cur), param_vecs, ffa_level, latter,
                tseg_brute, nbins);
            return std::make_tuple(as_pyarray_ref(pindex_prev), relative_phase);
        },
        py::arg("pset_cur"), py::arg("param_arr"), py::arg("ffa_level"),
        py::arg("latter"), py::arg("tseg_brute"), py::arg("nbins"));

    auto m_prune = m.def_submodule("prune", "Pruning submodule");

    py::class_<utils::WorldTree<float>>(m_prune, "WorldTreeFloat")
        .def(py::init<SizeType, SizeType, SizeType, SizeType>(),
             py::arg("capacity"), py::arg("nparams"), py::arg("nbins"),
             py::arg("max_batch_size"))
        .def_property_readonly("leaves",
                               [](const utils::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_leaves());
                               })
        .def_property_readonly("folds",
                               [](const utils::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_folds());
                               })
        .def_property_readonly("scores",
                               [](const utils::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_scores());
                               });

    py::class_<PruningManager>(m_prune, "PruningManager")
        .def(py::init<const PulsarSearchConfig&, const std::vector<float>&,
                      std::optional<SizeType>,
                      std::optional<std::vector<SizeType>>, SizeType,
                      SizeType>(),
             py::arg("cfg"), py::arg("threshold_scheme"),
             py::arg("n_runs")   = std::nullopt,
             py::arg("ref_segs") = std::nullopt,
             py::arg("max_sugg") = 1U << 18U, py::arg("batch_size") = 1024U)
        .def(
            "execute",
            [](PruningManager& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, std::string_view outdir,
               std::string_view file_prefix, std::string_view poly_basis,
               bool show_progress) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix,
                             poly_basis, show_progress);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
            py::arg("file_prefix"), py::arg("poly_basis"),
            py::arg("show_progress") = true);
}
} // namespace loki