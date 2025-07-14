#include "loki/algorithms/prune.hpp"
#include "loki/common/types.hpp"
#include "pybind_utils.hpp"

#include <cstddef>
#include <span>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"
#include "loki/psr_utils.hpp"

namespace loki {
using algorithms::FFA;
using algorithms::FFACOMPLEX;
using algorithms::PruneComplex;
using algorithms::PruneFloat;
using algorithms::PruningManagerComplex;
using algorithms::PruningManagerFloat;
using detection::MatchedFilter;
using plans::FFAPlan;
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

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_2d_max(to_span<const float>(arr), nprofiles,
                                         to_span<const SizeType>(widths),
                                         to_span<float>(out), stdnoise,
                                         nthreads);
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

            auto out = PyArrayT<float>({nprofiles, widths.size()});
            detection::snr_boxcar_3d(to_span<const float>(arr), nprofiles,
                                     to_span<const SizeType>(widths),
                                     to_span<float>(out), nthreads);
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

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_3d_max(to_span<const float>(arr), nprofiles,
                                         to_span<const SizeType>(widths),
                                         to_span<float>(out), nthreads);
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
        "compute_brute_fold",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads) {
            return as_pyarray(algorithms::compute_brute_fold(
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
                      const std::vector<ParamLimitType>&, double, double,
                      SizeType, SizeType, std::optional<SizeType>,
                      std::optional<SizeType>, bool, SizeType, SizeType>(),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("tol_bins"), py::arg("param_limits"),
             py::arg("ducy_max") = 0.2, py::arg("wtsp") = 1.5,
             py::arg("prune_poly_order") = 3, py::arg("prune_n_derivs") = 3,
             py::arg("bseg_brute")     = std::nullopt,
             py::arg("bseg_ffa")       = std::nullopt,
             py::arg("use_fft_shifts") = true, py::arg("branch_max") = 16,
             py::arg("nthreads") = 1)
        .def_property_readonly("nsamps", &PulsarSearchConfig::get_nsamps)
        .def_property_readonly("tsamp", &PulsarSearchConfig::get_tsamp)
        .def_property_readonly("nbins", &PulsarSearchConfig::get_nbins)
        .def_property_readonly("nbins_f", &PulsarSearchConfig::get_nbins_f)
        .def_property_readonly("tol_bins", &PulsarSearchConfig::get_tol_bins)
        .def_property_readonly("param_limits",
                               &PulsarSearchConfig::get_param_limits)
        .def_property_readonly("ducy_max", &PulsarSearchConfig::get_ducy_max)
        .def_property_readonly("wtsp", &PulsarSearchConfig::get_wtsp)
        .def_property_readonly("prune_poly_order",
                               &PulsarSearchConfig::get_prune_poly_order)
        .def_property_readonly("prune_n_derivs",
                               &PulsarSearchConfig::get_prune_n_derivs)
        .def_property_readonly("bseg_brute",
                               &PulsarSearchConfig::get_bseg_brute)
        .def_property_readonly("bseg_ffa", &PulsarSearchConfig::get_bseg_ffa)
        .def_property_readonly("use_fft_shifts",
                               &PulsarSearchConfig::get_use_fft_shifts)
        .def_property_readonly("branch_max",
                               &PulsarSearchConfig::get_branch_max)
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
                                       self.get_score_widths());
                               })
        .def("dparams_f", &PulsarSearchConfig::get_dparams_f,
             py::arg("tseg_cur"))
        .def("dparams", &PulsarSearchConfig::get_dparams, py::arg("tseg_cur"))
        .def("dparams_lim", &PulsarSearchConfig::get_dparams_lim,
             py::arg("tseg_cur"));

    auto m_ffa = m.def_submodule("ffa", "FFA submodule");
    PYBIND11_NUMPY_DTYPE(plans::FFACoord, i_tail, shift_tail, i_head,
                         shift_head);
    py::class_<FFAPlan>(m_ffa, "FFAPlan")
        .def_property_readonly("segment_lens",
                               [](const FFAPlan& self) {
                                   return as_pyarray_ref(self.segment_lens);
                               })
        .def_property_readonly(
            "nsegments",
            [](const FFAPlan& self) { return as_pyarray_ref(self.nsegments); })
        .def_property_readonly(
            "tsegments",
            [](const FFAPlan& self) { return as_pyarray_ref(self.tsegments); })
        .def_property_readonly(
            "ncoords",
            [](const FFAPlan& self) { return as_pyarray_ref(self.ncoords); })
        .def_property_readonly(
            "params",
            [](const FFAPlan& self) { return as_listof_pyarray(self.params); })
        .def_property_readonly(
            "dparams",
            [](const FFAPlan& self) { return as_listof_pyarray(self.dparams); })
        .def_property_readonly("fold_shapes",
                               [](const FFAPlan& self) {
                                   return as_listof_pyarray(self.fold_shapes);
                               })
        .def_property_readonly("fold_shapes_complex",
                               [](const FFAPlan& self) {
                                   return as_listof_pyarray(
                                       self.fold_shapes_complex);
                               })
        .def_property_readonly("coordinates",
                               [](const FFAPlan& self) {
                                   return as_listof_pyarray(self.coordinates);
                               })
        .def_property_readonly("memory_usage", &FFAPlan::get_memory_usage)
        .def_property_readonly("buffer_size", &FFAPlan::get_buffer_size)
        .def_property_readonly("fold_size", &FFAPlan::get_fold_size)
        .def_property_readonly("fold_size_complex",
                               &FFAPlan::get_fold_size_complex)
        .def_property_readonly("buffer_size_complex",
                               &FFAPlan::get_buffer_size_complex)
        .def_property_readonly("params_dict", [](const FFAPlan& self) {
            auto params_map = self.get_params_dict();
            py::dict result;
            for (const auto& [key, value] : params_map) {
                result[py::str(key)] = as_pyarray_ref(value);
            }
            return result;
        });

    py::class_<FFA>(m_ffa, "FFA")
        .def(py::init<PulsarSearchConfig, bool>(), py::arg("cfg"),
             py::arg("show_progress") = true)
        .def_property_readonly("plan", &FFA::get_plan)
        .def(
            "execute",
            [](FFA& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    py::class_<FFACOMPLEX>(m_ffa, "FFACOMPLEX")
        .def(py::init<PulsarSearchConfig, bool>(), py::arg("cfg"),
             py::arg("show_progress") = true)
        .def_property_readonly("plan", &FFACOMPLEX::get_plan)
        .def(
            "execute",
            [](FFACOMPLEX& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"))
        .def(
            "execute",
            [](FFACOMPLEX& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<ComplexType>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v),
                             to_span<ComplexType>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    m_ffa.def(
        "compute_ffa",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool show_progress) {
            auto [scores, ffa_plan] = algorithms::compute_ffa(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                show_progress);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("show_progress") = true);

    m_ffa.def(
        "compute_ffa_complex",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool show_progress) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_complex(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                show_progress);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("show_progress") = true);

    m_ffa.def(
        "compute_ffa_scores",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PulsarSearchConfig& cfg, bool show_progress) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_scores(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                show_progress);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("show_progress") = true);

    auto m_psr_utils = m.def_submodule("psr_utils", "PSR utils submodule");
    m_psr_utils.def(
        "shift_params",
        [](const PyArrayT<double>& pset_cur, double delta_t) {
            auto [pset_prev, delay] = psr_utils::shift_params(
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
            std::span<const std::vector<double>> param_span(param_vecs);

            auto [pindex_prev, relative_phase] = core::ffa_taylor_resolve(
                to_span<const double>(pset_cur), param_span, ffa_level, latter,
                tseg_brute, nbins);
            return std::make_tuple(as_pyarray_ref(pindex_prev), relative_phase);
        },
        py::arg("pset_cur"), py::arg("param_arr"), py::arg("ffa_level"),
        py::arg("latter"), py::arg("tseg_brute"), py::arg("nbins"));

    auto m_prune = m.def_submodule("prune", "Pruning submodule");

    py::class_<utils::SuggestionStruct<float>>(m_prune, "SuggestionStructFloat")
        .def(py::init<SizeType, SizeType, SizeType>(), py::arg("max_sugg"),
             py::arg("nparams"), py::arg("nbins"))
        .def_property_readonly("param_sets",
                               [](const utils::SuggestionStruct<float>& self) {
                                   return as_pyarray_ref(self.get_param_sets());
                               })
        .def_property_readonly("folds",
                               [](const utils::SuggestionStruct<float>& self) {
                                   return as_pyarray_ref(self.get_folds());
                               })
        .def_property_readonly("scores",
                               [](const utils::SuggestionStruct<float>& self) {
                                   return as_pyarray_ref(self.get_scores());
                               })
        .def_property_readonly("backtracks",
                               [](const utils::SuggestionStruct<float>& self) {
                                   return as_pyarray_ref(self.get_backtracks());
                               });

    py::class_<PruningManagerFloat>(m_prune, "PruningManager")
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
            [](PruningManagerFloat& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix, const std::string& kind) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix,
                             kind);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
            py::arg("file_prefix"), py::arg("kind"));

    py::class_<PruningManagerComplex>(m_prune, "PruningManagerComplex")
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
            [](PruningManagerComplex& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, const std::string& outdir,
               const std::string& file_prefix, const std::string& kind) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), outdir, file_prefix,
                             kind);
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
            py::arg("file_prefix"), py::arg("kind"));

    py::class_<PruneFloat>(m_prune, "Prune")
        .def(py::init<const FFAPlan&, const PulsarSearchConfig&,
                      const std::vector<float>&, SizeType, SizeType,
                      std::string_view>(),
             py::arg("ffa_plan"), py::arg("cfg"), py::arg("threshold_scheme"),
             py::arg("max_sugg") = 1U << 18U, py::arg("batch_size") = 1024U,
             py::arg("kind") = "taylor")
        .def(
            "execute",
            [](PruneFloat& self, const PyArrayT<float>& ffa_fold,
               SizeType ref_seg, const std::string& outdir,
               const std::string& file_prefix, const std::string& kind) {
                self.execute(to_span<const float>(ffa_fold), ref_seg, outdir,
                             file_prefix, kind);
            },
            py::arg("ffa_fold"), py::arg("ref_seg"), py::arg("outdir"),
            py::arg("file_prefix"), py::arg("kind"))
        .def_property_readonly(
            "suggestions_in",
            [](PruneFloat& self) { return self.get_suggestions_in(); })
        .def_property_readonly(
            "suggestions_out",
            [](PruneFloat& self) { return self.get_suggestions_out(); })
        .def(
            "initialize",
            [](PruneFloat& self, const PyArrayT<float>& ffa_fold,
               SizeType ref_seg, const std::string& log_file) {
                self.initialize(to_span<const float>(ffa_fold), ref_seg,
                                log_file);
            },
            py::arg("ffa_fold"), py::arg("ref_seg"), py::arg("log_file"))
        .def(
            "execute_iteration",
            [](PruneFloat& self, const PyArrayT<float>& ffa_fold,
               const std::string& log_file) {
                self.execute_iteration(to_span<const float>(ffa_fold),
                                       log_file);
            },
            py::arg("ffa_fold"), py::arg("log_file"));
}
} // namespace loki