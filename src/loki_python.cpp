#include "loki/loki.hpp"

#include "loki_templates.hpp"
#include "pybind_utils.hpp"

#include <cstddef>
#include <span>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"

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
                         std::string_view mode, int nthreads,
                         std::optional<uint64_t> seed) {
                 return std::make_unique<DynamicThresholdScheme>(
                     std::span<const float>(branching_pattern.data(),
                                            branching_pattern.size()),
                     ref_ducy, nbins, ntrials, nprobs, prob_min, snr_final,
                     nthresholds, ducy_max, wtsp, beam_width, trials_start,
                     mode, nthreads, seed);
             }),
             py::arg("branching_pattern"), py::arg("ref_ducy"),
             py::arg("nbins") = 64, py::arg("ntrials") = 1024,
             py::arg("nprobs") = 10, py::arg("prob_min") = 0.05F,
             py::arg("snr_final") = 8.0F, py::arg("nthresholds") = 100,
             py::arg("ducy_max") = 0.3F, py::arg("wtsp") = 1.0F,
             py::arg("beam_width") = 0.7F, py::arg("trials_start") = 1,
             py::arg("mode") = "legacy", py::arg("nthreads") = 1,
             py::arg("seed") = py::none())
        .def("run", &DynamicThresholdScheme::run, py::arg("thres_neigh") = 10,
             "Operational search. The path from get_best_path_thresholds is "
             "used immediately. In-run cost and success_h1_cumul are "
             "optimistic Monte Carlo estimates.")
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
        .def("get_states",
             [](DynamicThresholdScheme& self) {
                 return as_pyarray(self.get_states());
             })
        .def("get_best_path_thresholds",
             &DynamicThresholdScheme::get_best_path_thresholds,
             py::arg("min_pd") = 0.1,
             "Thresholds the live search should use. Not a re-scored path.")
        .def("evaluate",
             [](const DynamicThresholdScheme& self,
                const PyArrayT<float>& thresholds, SizeType ntrials,
                std::optional<uint64_t> seed) {
                 return as_pyarray(self.evaluate(
                     to_span<const float>(thresholds), ntrials, seed));
             },
             py::arg("thresholds"), py::arg("ntrials"),
             py::arg("seed") = py::none(),
             "Reporting only. Does not change the path and is not part of "
             "the on-the-fly pipeline. Pass a seed different from run().");
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
    PYBIND11_NUMPY_DTYPE(ParamLimit, min, max);

    py::class_<FFASearchConfig>(m_configs, "FFASearchConfig")
        .def(py::init(
                 [](SizeType nsamps, double tsamp, SizeType nbins, double eta,
                    const PyArrayT<double>& param_limits, double ducy_max,
                    double wtsp, bool use_fourier, int nthreads,
                    double max_process_memory_gb, double octave_scale,
                    SizeType nbins_max, SizeType nbins_min_lossy_bf,
                    std::optional<SizeType> bseg_brute,
                    std::optional<SizeType> bseg_ffa, double snr_min,
                    SizeType max_passing_candidates,
                    bool use_boxcar_kadane) {
                     if (param_limits.ndim() != 2 ||
                         param_limits.shape(1) != 2) {
                         throw std::invalid_argument(
                             "param_limits must be a 2D NumPy array with shape "
                             "(n_params, 2)");
                     }

                     const auto n_params =
                         static_cast<SizeType>(param_limits.shape(0));

                     std::vector<ParamLimit> limits(n_params);
                     for (SizeType i = 0; i < n_params; ++i) {
                         limits[i] = {.min = *param_limits.data(i, 0),
                                      .max = *param_limits.data(i, 1)};
                     }
                     return std::make_unique<FFASearchConfig>(
                         nsamps, tsamp, nbins, eta, limits, ducy_max, wtsp,
                         use_fourier, nthreads, max_process_memory_gb,
                         octave_scale, nbins_max, nbins_min_lossy_bf,
                         bseg_brute, bseg_ffa, snr_min, max_passing_candidates,
                         use_boxcar_kadane);
                 }),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("eta"), py::arg("param_limits"), py::arg("ducy_max") = 0.2,
             py::arg("wtsp") = 1.5, py::arg("use_fourier") = true,
             py::arg("nthreads") = 1, py::arg("max_process_memory_gb") = 8.0,
             py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024,
             py::arg("nbins_min_lossy_bf") = 64,
             py::arg("bseg_brute")         = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("snr_min") = 5.0,
             py::arg("max_passing_candidates") = 1U << 22U, // 4M
             py::arg("use_boxcar_kadane")       = false)
        .def_property_readonly("nsamps", &FFASearchConfig::get_nsamps)
        .def_property_readonly("tsamp", &FFASearchConfig::get_tsamp)
        .def_property_readonly("tobs", &FFASearchConfig::get_tobs)
        .def_property_readonly("nbins", &FFASearchConfig::get_nbins)
        .def_property_readonly("nbins_f", &FFASearchConfig::get_nbins_f)
        .def_property_readonly("eta", &FFASearchConfig::get_eta)
        .def_property_readonly("param_limits",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_param_limits());
                               })
        .def_property_readonly("ducy_max", &FFASearchConfig::get_ducy_max)
        .def_property_readonly("wtsp", &FFASearchConfig::get_wtsp)
        .def_property_readonly("bseg_brute",
                               &FFASearchConfig::get_bseg_brute)
        .def_property_readonly("bseg_ffa", &FFASearchConfig::get_bseg_ffa)
        .def_property_readonly("use_fourier",
                               &FFASearchConfig::get_use_fourier)
        .def_property_readonly("use_conservative_tile",
                               &FFASearchConfig::get_use_conservative_tile)
        .def_property_readonly("nthreads", &FFASearchConfig::get_nthreads)
        .def_property_readonly("tseg_brute",
                               &FFASearchConfig::get_tseg_brute)
        .def_property_readonly("tseg_ffa", &FFASearchConfig::get_tseg_ffa)
        .def_property_readonly("niters_ffa",
                               &FFASearchConfig::get_niters_ffa)
        .def_property_readonly("nparams", &FFASearchConfig::get_nparams)
        .def_property_readonly("param_names",
                               &FFASearchConfig::get_param_names)
        .def_property_readonly("f_min", &FFASearchConfig::get_f_min)
        .def_property_readonly("f_max", &FFASearchConfig::get_f_max)
        .def_property_readonly("score_widths",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_scoring_widths());
                               })
        .def_property_readonly("n_scoring_widths",
                               &FFASearchConfig::get_n_scoring_widths)
        .def_property_readonly("boxcar_kadane_biases",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_boxcar_kadane_biases());
                               })
        .def_property_readonly("n_boxcar_kadane_biases",
                               &FFASearchConfig::get_n_boxcar_kadane_biases)
        .def("dparams_f", &FFASearchConfig::get_dparams_f,
             py::arg("tseg_cur"))
        .def("dparams", &FFASearchConfig::get_dparams, py::arg("tseg_cur"))
        .def("dparams_actual", &FFASearchConfig::get_dparams_actual,
             py::arg("tseg_cur"));

    py::class_<PulsarSearchConfig, FFASearchConfig>(m_configs, "PulsarSearchConfig")
        .def(py::init(
                 [](SizeType nsamps, double tsamp, SizeType nbins, double eta,
                    const PyArrayT<double>& param_limits, double ducy_max,
                    double wtsp, bool use_fourier, int nthreads,
                    double max_process_memory_gb, double octave_scale,
                    SizeType nbins_max, SizeType nbins_min_lossy_bf,
                    std::optional<SizeType> bseg_brute,
                    std::optional<SizeType> bseg_ffa, double snr_min,
                    SizeType max_passing_candidates, SizeType prune_poly_order,
                    double p_orb_min, double m_c_max, double m_p_min,
                    double propagator_significance,
                    double validation_significance, bool use_conservative_tile,
                    bool use_boxcar_kadane) {
                     if (param_limits.ndim() != 2 ||
                         param_limits.shape(1) != 2) {
                         throw std::invalid_argument(
                             "param_limits must be a 2D NumPy array with shape "
                             "(n_params, 2)");
                     }

                     const auto n_params =
                         static_cast<SizeType>(param_limits.shape(0));

                     std::vector<ParamLimit> limits(n_params);
                     for (SizeType i = 0; i < n_params; ++i) {
                         limits[i] = {.min = *param_limits.data(i, 0),
                                      .max = *param_limits.data(i, 1)};
                     }
                     return std::make_unique<PulsarSearchConfig>(
                         nsamps, tsamp, nbins, eta, limits, ducy_max, wtsp,
                         use_fourier, nthreads, max_process_memory_gb,
                         octave_scale, nbins_max, nbins_min_lossy_bf,
                         bseg_brute, bseg_ffa, snr_min, max_passing_candidates,
                         prune_poly_order, p_orb_min, m_c_max, m_p_min,
                         propagator_significance, validation_significance,
                         use_conservative_tile, use_boxcar_kadane);
                 }),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("eta"), py::arg("param_limits"), py::arg("ducy_max") = 0.2,
             py::arg("wtsp") = 1.5, py::arg("use_fourier") = true,
             py::arg("nthreads") = 1, py::arg("max_process_memory_gb") = 8.0,
             py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024,
             py::arg("nbins_min_lossy_bf") = 64,
             py::arg("bseg_brute")         = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("snr_min") = 5.0,
             py::arg("max_passing_candidates") = 1U << 22U, // 4M
             py::arg("prune_poly_order") = 3, py::arg("p_orb_min") = 1e-5,
             py::arg("m_c_max") = 10.0, py::arg("m_p_min") = 1.4,
             py::arg("propagator_significance") = 2.0,
             py::arg("validation_significance") = 5.0,
             py::arg("use_conservative_tile")   = false,
             py::arg("use_boxcar_kadane")       = false)
        .def_property_readonly("prune_poly_order",
                               &PulsarSearchConfig::get_prune_poly_order)
        .def_property_readonly("p_orb_min", &PulsarSearchConfig::get_p_orb_min)
        .def_property_readonly("m_c_max", &PulsarSearchConfig::get_m_c_max)
        .def_property_readonly("m_p_min", &PulsarSearchConfig::get_m_p_min)
        .def_property_readonly("propagator_significance",
                               &PulsarSearchConfig::get_propagator_significance)
        .def_property_readonly("validation_significance",
                               &PulsarSearchConfig::get_validation_significance)
        .def_property_readonly("x_mass_const",
                               &PulsarSearchConfig::get_x_mass_const);

    m_configs.attr("EPSearchConfig") = m_configs.attr("PulsarSearchConfig");

    // Plans submodule
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
                                   auto params_map = self.get_params_dict();
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
    m_plans.def("generate_ffa_regions", &regions::generate_ffa_regions,
                py::arg("p_min"), py::arg("p_max"), py::arg("tsamp"),
                py::arg("nbins_min"), py::arg("eta_min"),
                py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024);

    // FFA submodule
    auto m_ffa = m.def_submodule("ffa", "FFA submodule");
    bind_ffa_class<float>(m_ffa, "FFATime");
    bind_ffa_class<ComplexType>(m_ffa, "FFAFourier");

    m_ffa.def(
        "compute_ffa_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] = algorithms::compute_ffa<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_fourier_return_to_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress) {
            auto [fold, ffa_plan] =
                algorithms::compute_ffa_fourier_return_to_time(
                    to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                    quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(fold)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    m_ffa.def(
        "compute_ffa_scores",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const FFASearchConfig& cfg, bool quiet, bool show_progress) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_scores(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                quiet, show_progress);
            return std::make_tuple(as_pyarray(std::move(scores)),
                                   std::move(ffa_plan));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("quiet") = false, py::arg("show_progress") = false);

    py::class_<FFAFreqSweep>(m_ffa, "FFAFreqSweep")
        .def(py::init<const FFASearchConfig&, bool>(), py::arg("cfg"),
             py::arg("show_progress") = true)
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
           const PyArrayT<SizeType>& param_grid_count_prev,
           const PyArrayT<ParamLimit>& param_limits, SizeType ffa_level,
           SizeType latter, double tseg_brute, SizeType nbins) {
            auto [pindex_prev, relative_phase] =
                core::ffa_taylor_resolve_generic(
                    to_span<const double>(pset_cur),
                    to_span<const SizeType>(param_grid_count_prev),
                    to_span<const ParamLimit>(param_limits), ffa_level, latter,
                    tseg_brute, nbins);
            return std::make_tuple(as_pyarray_ref(pindex_prev), relative_phase);
        },
        py::arg("pset_cur"), py::arg("param_grid_count_prev"),
        py::arg("param_limits"), py::arg("ffa_level"), py::arg("latter"),
        py::arg("tseg_brute"), py::arg("nbins"));

    auto m_prune = m.def_submodule("prune", "Pruning submodule");

    py::class_<memory::WorldTree<float>>(m_prune, "WorldTreeFloat")
        .def(py::init<SizeType, SizeType, SizeType, SizeType>(),
             py::arg("capacity"), py::arg("nparams"), py::arg("nbins"),
             py::arg("max_batch_size"))
        .def_property_readonly("leaves",
                               [](const memory::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_leaves());
                               })
        .def_property_readonly("folds",
                               [](const memory::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_folds());
                               })
        .def_property_readonly("scores",
                               [](const memory::WorldTree<float>& self) {
                                   return as_pyarray_ref(self.get_scores());
                               });

    // EP submodule
    bind_prune_rfi(m_prune);
    bind_ep_multi_pass<float>(m_prune, "EPMultiPassTime");
    bind_ep_multi_pass<ComplexType>(m_prune, "EPMultiPassFourier");

    py::class_<EPChunkConfig>(m_prune, "EPChunkConfig")
        .def_readonly("cfg", &EPChunkConfig::cfg)
        .def_readonly("threshold_scheme", &EPChunkConfig::threshold_scheme)
        .def_readonly("branching_pattern", &EPChunkConfig::branching_pattern)
        .def_readonly("max_sugg", &EPChunkConfig::max_sugg)
        .def_readonly("branch_max", &EPChunkConfig::branch_max)
        .def_readonly("nominal_f_start", &EPChunkConfig::nominal_f_start)
        .def_readonly("nominal_f_end", &EPChunkConfig::nominal_f_end)
        .def_readonly("actual_f_start", &EPChunkConfig::actual_f_start)
        .def_readonly("actual_f_end", &EPChunkConfig::actual_f_end)
        .def_readonly("peak_complexity", &EPChunkConfig::peak_complexity)
        .def_readonly("chunk_memory_gb", &EPChunkConfig::chunk_memory_gb)
        .def_readonly("nsegments", &EPChunkConfig::nsegments)
        .def_readonly("ncoords", &EPChunkConfig::ncoords)
        .def_readonly("buffer_size", &EPChunkConfig::buffer_size)
        .def_readonly("coord_size", &EPChunkConfig::coord_size)
        .def_readonly("fold_size", &EPChunkConfig::fold_size);

    py::class_<EPChunkStats>(m_prune, "EPChunkStats")
        .def_readonly("chunk_id", &EPChunkStats::chunk_id)
        .def_readonly("nominal_f_start", &EPChunkStats::nominal_f_start)
        .def_readonly("nominal_f_end", &EPChunkStats::nominal_f_end)
        .def_readonly("actual_f_start", &EPChunkStats::actual_f_start)
        .def_readonly("actual_f_end", &EPChunkStats::actual_f_end)
        .def_readonly("nominal_width", &EPChunkStats::nominal_width)
        .def_readonly("actual_width", &EPChunkStats::actual_width)
        .def_readonly("nbins", &EPChunkStats::nbins)
        .def_readonly("eta", &EPChunkStats::eta)
        .def_readonly("ncoords", &EPChunkStats::ncoords)
        .def_readonly("max_sugg", &EPChunkStats::max_sugg)
        .def_readonly("branch_max", &EPChunkStats::branch_max)
        .def_readonly("peak_complexity", &EPChunkStats::peak_complexity)
        .def_readonly("memory_gb", &EPChunkStats::memory_gb)
        .def_readonly("overlap_fraction", &EPChunkStats::overlap_fraction);

    py::class_<EPRegionStats>(m_prune, "EPRegionStats")
        .def_property_readonly("max_sugg", &EPRegionStats::get_max_sugg)
        .def_property_readonly("max_ncoords", &EPRegionStats::get_max_ncoords)
        .def_property_readonly("max_branch_max",
                               &EPRegionStats::get_max_branch_max)
        .def_property_readonly("max_memory_gb",
                               &EPRegionStats::get_max_memory_gb)
        .def_property_readonly("max_buffer_size",
                               &EPRegionStats::get_max_buffer_size)
        .def_property_readonly("max_coord_size",
                               &EPRegionStats::get_max_coord_size)
        .def_property_readonly("max_fold_size",
                               &EPRegionStats::get_max_fold_size)
        .def_property_readonly("nchunks", &EPRegionStats::get_nchunks)
        .def_property_readonly("chunk_stats", &EPRegionStats::get_chunk_stats);

    bind_ep_region_planner<float>(m_prune, "EPRegionPlannerTime");
    bind_ep_region_planner<ComplexType>(m_prune, "EPRegionPlannerFourier");

    py::class_<EPFreqSweep>(m_prune, "EPFreqSweep")
        .def(py::init<const PulsarSearchConfig&, bool, float, std::string_view,
                      float, algorithms::PruneRFIConfig,
                      const std::optional<std::filesystem::path>&,
                      std::optional<SizeType>,
                      std::optional<std::vector<SizeType>>>(),
             py::arg("cfg"), py::arg("show_progress") = true,
             py::arg("min_pd") = 0.1F, py::arg("poly_basis") = "taylor",
             py::arg("ref_ducy")        = 0.1F,
             py::arg("rfi_config")      = algorithms::PruneRFIConfig(),
             py::arg("plan_cache_file") = std::nullopt,
             py::arg("n_runs")          = std::nullopt,
             py::arg("ref_segs")        = std::nullopt)
        .def(
            "execute",
            [](EPFreqSweep& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir") = "./",
            py::arg("file_prefix") = "test");

    auto m_io = m.def_submodule("io", "Timeseries I/O");
    py::enum_<LocMethod>(m_io, "LocMethod")
        .value("Mean", LocMethod::kMean)
        .value("Median", LocMethod::kMedian)
        .value("None", LocMethod::kNone);
    py::enum_<ScaleMethod>(m_io, "ScaleMethod")
        .value("Std", ScaleMethod::kStd)
        .value("Iqr", ScaleMethod::kIqr)
        .value("Mad", ScaleMethod::kMad)
        .value("DoubleMad", ScaleMethod::kDoubleMad)
        .value("None", ScaleMethod::kNone);
    py::class_<io::ReadOptions>(m_io, "ReadOptions")
        .def(py::init<>())
        .def_readwrite("preprocess", &io::ReadOptions::preprocess)
        .def_readwrite("filter_window", &io::ReadOptions::filter_window)
        .def_readwrite("loc", &io::ReadOptions::loc)
        .def_readwrite("scale", &io::ReadOptions::scale)
        .def_readwrite("fast_median", &io::ReadOptions::fast_median)
        .def_readwrite("fast_median_min_points",
                       &io::ReadOptions::fast_median_min_points)
        .def_readwrite("nthreads", &io::ReadOptions::nthreads);
    auto timeseries_view = [](py::object self, std::span<float> data) {
        return py::array_t<float>(
            py::array::ShapeContainer{static_cast<py::ssize_t>(data.size())},
            py::array::StridesContainer{
                static_cast<py::ssize_t>(sizeof(float))},
            data.data(), self);
    };
    py::class_<io::TimeSeries>(m_io, "TimeSeries")
        .def(py::init([](const PyArrayT<float>& ts_e,
                         const PyArrayT<float>& ts_v, double dt) {
                 const auto intensity = to_span<const float>(ts_e);
                 const auto variance  = to_span<const float>(ts_v);
                 return io::TimeSeries(
                     std::vector<float>(intensity.begin(), intensity.end()),
                     std::vector<float>(variance.begin(), variance.end()), dt);
             }),
             py::arg("ts_e"), py::arg("ts_v"), py::arg("dt"))
        .def_static("read", &io::TimeSeries::read, py::arg("path"),
                    py::arg("options") = io::ReadOptions{})
        .def("write", &io::TimeSeries::write, py::arg("path"))
        .def_property_readonly("nsamps", &io::TimeSeries::get_nsamps)
        .def_property_readonly("dt", &io::TimeSeries::get_dt)
        .def_property_readonly("tobs", &io::TimeSeries::get_tobs)
        .def_property_readonly("ts_e",
                               [timeseries_view](py::object self) {
                                   return timeseries_view(
                                       self,
                                       self.cast<io::TimeSeries&>().get_ts_e());
                               })
        .def_property_readonly("ts_v", [timeseries_view](py::object self) {
            return timeseries_view(self,
                                   self.cast<io::TimeSeries&>().get_ts_v());
        });

    auto m_sim = m.def_submodule("simulation", "Pulse simulation");
    py::class_<simulation::DerivativeTerms>(m_sim, "DerivativeTerms")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::DerivativeTerms::shift)
        .def_readwrite("vel", &simulation::DerivativeTerms::vel)
        .def_readwrite("acc", &simulation::DerivativeTerms::acc)
        .def_readwrite("jerk", &simulation::DerivativeTerms::jerk)
        .def_readwrite("snap", &simulation::DerivativeTerms::snap);
    py::class_<simulation::DerivativeSeries>(m_sim, "DerivativeSeries")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::DerivativeSeries::shift)
        .def_readwrite("vel", &simulation::DerivativeSeries::vel)
        .def_readwrite("acc", &simulation::DerivativeSeries::acc)
        .def_readwrite("jerk", &simulation::DerivativeSeries::jerk)
        .def_readwrite("snap", &simulation::DerivativeSeries::snap)
        .def_readwrite("crackle", &simulation::DerivativeSeries::crackle);
    py::class_<simulation::GaugeDerivatives>(m_sim, "GaugeDerivatives")
        .def(py::init<>())
        .def_readwrite("freq", &simulation::GaugeDerivatives::freq)
        .def_readwrite("vel", &simulation::GaugeDerivatives::vel)
        .def_readwrite("acc", &simulation::GaugeDerivatives::acc)
        .def_readwrite("jerk", &simulation::GaugeDerivatives::jerk)
        .def_readwrite("snap", &simulation::GaugeDerivatives::snap)
        .def_readwrite("crackle", &simulation::GaugeDerivatives::crackle);
    py::class_<simulation::CircularOrbit>(m_sim, "CircularOrbit")
        .def(py::init<>())
        .def_readwrite("p_orb", &simulation::CircularOrbit::p_orb)
        .def_readwrite("psi", &simulation::CircularOrbit::psi)
        .def_readwrite("x_orb", &simulation::CircularOrbit::x_orb);
    py::class_<simulation::ModulatorParams>(m_sim, "ModulatorParams")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::ModulatorParams::shift)
        .def_readwrite("vel", &simulation::ModulatorParams::vel)
        .def_readwrite("acc", &simulation::ModulatorParams::acc)
        .def_readwrite("jerk", &simulation::ModulatorParams::jerk)
        .def_readwrite("snap", &simulation::ModulatorParams::snap)
        .def_readwrite("coeffs", &simulation::ModulatorParams::coeffs)
        .def_readwrite("p_orb", &simulation::ModulatorParams::p_orb)
        .def_readwrite("psi", &simulation::ModulatorParams::psi)
        .def_readwrite("x_orb", &simulation::ModulatorParams::x_orb)
        .def_readwrite("m_c", &simulation::ModulatorParams::m_c)
        .def_readwrite("m_p", &simulation::ModulatorParams::m_p)
        .def_readwrite("sin_i", &simulation::ModulatorParams::sin_i)
        .def_readwrite("a", &simulation::ModulatorParams::a)
        .def_readwrite("t0", &simulation::ModulatorParams::t0);

    auto modulator_generate = [](const simulation::Modulator& modulator,
                                 const PyArrayT<double>& time, double t_ref) {
        const auto samples = to_span<const double>(time);
        std::vector<double> proper(samples.size());
        modulator.generate(samples, t_ref, proper);
        return as_pyarray(std::move(proper));
    };
    py::class_<simulation::Modulator, std::unique_ptr<simulation::Modulator>>(
        m_sim, "Modulator")
        .def("generate", modulator_generate, py::arg("time"),
             py::arg("t_ref") = 0.0);
    py::class_<simulation::DerivativeModulator, simulation::Modulator>(
        m_sim, "DerivativeModulator")
        .def(py::init<simulation::DerivativeTerms>(),
             py::arg("terms") = simulation::DerivativeTerms{})
        .def("to_circular", &simulation::DerivativeModulator::to_circular);
    py::class_<simulation::DerivativeSeriesModulator, simulation::Modulator>(
        m_sim, "DerivativeSeriesModulator")
        .def(py::init<std::vector<double>>(), py::arg("coeffs"))
        .def("to_circular",
             &simulation::DerivativeSeriesModulator::to_circular);
    py::class_<simulation::CircularModulator, simulation::Modulator>(
        m_sim, "CircularModulator")
        .def(py::init<double, double, std::optional<double>,
                      std::optional<double>, double, double>(),
             py::arg("p_orb"), py::arg("psi"), py::arg("x_orb") = py::none(),
             py::arg("m_c") = py::none(), py::arg("m_p") = 1.4,
             py::arg("sin_i") = 1.0)
        .def("to_derivatives", &simulation::CircularModulator::to_derivatives)
        .def("to_derivatives_gauge",
             &simulation::CircularModulator::to_derivatives_gauge,
             py::arg("f_ref"))
        .def("to_derivatives_series",
             &simulation::CircularModulator::to_derivatives_series,
             py::arg("n"))
        .def_property_readonly("p_orb", &simulation::CircularModulator::p_orb)
        .def_property_readonly("psi", &simulation::CircularModulator::psi)
        .def_property_readonly("x_orb", &simulation::CircularModulator::x_orb);
    py::class_<simulation::CircularT0Modulator, simulation::Modulator>(
        m_sim, "CircularT0Modulator")
        .def(py::init<double, double, double>(), py::arg("a"), py::arg("p_orb"),
             py::arg("t0"));
    m_sim.def("make_modulator", &simulation::make_modulator, py::arg("type"),
              py::arg("params") = simulation::ModulatorParams{});
    py::class_<simulation::PulseSignalConfig>(m_sim, "PulseSignalConfig")
        .def(py::init<double, double, SizeType, double, double, std::string,
                      simulation::ModulatorParams, std::optional<double>,
                      std::optional<std::uint64_t>>(),
             py::arg("period"), py::arg("dt"),
             py::arg("nsamps") = (SizeType{1} << 21U), py::arg("snr") = 100.0,
             py::arg("ducy") = 0.1, py::arg("mod_type") = "derivative",
             py::arg("mod")      = simulation::ModulatorParams{},
             py::arg("mod_tref") = py::none(), py::arg("seed") = py::none())
        .def("generate", &simulation::PulseSignalConfig::generate,
             py::arg("shape") = "gaussian", py::arg("phi0") = 0.5,
             py::arg("max_iter") = 5, py::arg("tol") = 1e-2)
        .def("generate_noise", &simulation::PulseSignalConfig::generate_noise)
        .def_property_readonly("period", &simulation::PulseSignalConfig::period)
        .def_property_readonly("dt", &simulation::PulseSignalConfig::dt)
        .def_property_readonly("nsamps", &simulation::PulseSignalConfig::nsamps)
        .def_property_readonly("snr", &simulation::PulseSignalConfig::snr)
        .def_property_readonly("ducy", &simulation::PulseSignalConfig::ducy)
        .def_property_readonly("tobs", &simulation::PulseSignalConfig::tobs)
        .def_property_readonly("freq", &simulation::PulseSignalConfig::freq);
}
} // namespace loki