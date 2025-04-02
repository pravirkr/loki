#include <cstddef>
#include <span>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <loki/ffa.hpp>
#include <loki/fold.hpp>
#include <loki/loki_types.hpp>
#include <loki/score.hpp>
#include <loki/thresholds.hpp>
#include <loki/utils.hpp>

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
        "generate_width_trials",
        [](size_t nbins_max, float spacing_factor) {
            auto trials =
                loki::generate_width_trials(nbins_max, spacing_factor);
            return as_pyarray_ref(trials);
        },
        py::arg("nbins_max"), py::arg("spacing_factor") = 1.5F);

    m_scores.def(
        "snr_1d",
        [](const py::array_t<float>& arr, const py::array_t<SizeType>& widths,
           float stdnoise) {
            if (arr.ndim() != 1 || widths.ndim() != 1) {
                throw std::runtime_error("Input arrays must be 1-dimensional");
            }
            if (arr.size() == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            auto out = py::array_t<float>(widths.size());
            loki::snr_1d(
                std::span<const float>(arr.data(), arr.size()),
                std::span<const SizeType>(widths.data(), widths.size()),
                std::span<float>(out.mutable_data(), out.size()), stdnoise);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F);
    m_scores.def(
        "snr_2d",
        [](const py::array_t<float, py::array::c_style>& arr,
           const py::array_t<SizeType>& widths, float stdnoise) {
            if (arr.ndim() != 2 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 2-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);

            auto out = py::array_t<float, py::array::c_style>(
                {nprofiles, widths.size()});
            loki::snr_2d(
                std::span<const float>(arr.data(), arr.size()), nprofiles,
                std::span<const SizeType>(widths.data(), widths.size()),
                std::span<float>(out.mutable_data(), out.size()), stdnoise);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F);

    auto m_thresholds = m.def_submodule("thresholds", "Thresholds submodule");

    PYBIND11_NUMPY_DTYPE(State, success_h0, success_h1, complexity,
                         complexity_cumul, success_h1_cumul, nbranches,
                         threshold, threshold_prev, success_h1_cumul_prev,
                         is_empty);
    py::class_<DynamicThresholdScheme>(m_thresholds, "DynamicThresholdScheme")
        .def(py::init([](const py::array_t<float>& branching_pattern,
                         const py::array_t<float>& profile, SizeType ntrials,
                         SizeType nprobs, float prob_min, float snr_final,
                         SizeType nthresholds, float ducy_max, float wtsp,
                         float beam_width, SizeType nthreads) {
                 return std::make_unique<DynamicThresholdScheme>(
                     std::span<const float>(branching_pattern.data(),
                                            branching_pattern.size()),
                     std::span<const float>(profile.data(), profile.size()),
                     ntrials, nprobs, prob_min, snr_final, nthresholds,
                     ducy_max, wtsp, beam_width, nthreads);
             }),
             py::arg("branching_pattern"), py::arg("profile"),
             py::arg("ntrials") = 1024, py::arg("nprobs") = 10,
             py::arg("prob_min") = 0.05F, py::arg("snr_final") = 8.0F,
             py::arg("nthresholds") = 100, py::arg("ducy_max") = 0.3F,
             py::arg("wtsp") = 1.0F, py::arg("beam_width") = 0.7F,
             py::arg("nthreads") = 1)
        .def("get_current_thresholds_idx",
             &DynamicThresholdScheme::get_current_thresholds_idx)
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
        [](const py::array_t<float>& thresholds,
           const py::array_t<float>& branching_pattern,
           const py::array_t<float>& profile, SizeType ntrials, float snr_final,
           float ducy_max, float wtsp) {
            return evaluate_scheme(
                std::span<const float>(thresholds.data(), thresholds.size()),
                std::span<const float>(branching_pattern.data(),
                                       branching_pattern.size()),
                std::span<const float>(profile.data(), profile.size()), ntrials,
                snr_final, ducy_max, wtsp);
        },
        py::arg("thresholds"), py::arg("branching_pattern"), py::arg("profile"),
        py::arg("ntrials") = 1024, py::arg("snr_final") = 8.0F,
        py::arg("ducy_max") = 0.3F, py::arg("wtsp") = 1.0F);
    m_thresholds.def(
        "determine_scheme",
        [](const py::array_t<float>& survive_probs,
           const py::array_t<float>& branching_pattern,
           const py::array_t<float>& profile, SizeType ntrials, float snr_final,
           float ducy_max, float wtsp) {
            return determine_scheme(
                std::span<const float>(survive_probs.data(),
                                       survive_probs.size()),
                std::span<const float>(branching_pattern.data(),
                                       branching_pattern.size()),
                std::span<const float>(profile.data(), profile.size()), ntrials,
                snr_final, ducy_max, wtsp);
        },
        py::arg("survive_probs"), py::arg("branching_pattern"),
        py::arg("profile"), py::arg("ntrials") = 1024,
        py::arg("snr_final") = 8.0F, py::arg("ducy_max") = 0.3F,
        py::arg("wtsp") = 1.0F);

    auto m_utils = m.def_submodule("utils", "Utils submodule");
    m_utils.def(
        "find_neighbouring_indices",
        [](const py::array_t<SizeType>& indices, SizeType target_idx,
           SizeType num) {
            return loki::find_neighbouring_indices(
                std::span<const SizeType>(indices.data(), indices.size()),
                target_idx, num);
        },
        py::arg("beam_indices"), py::arg("target_idx"), py::arg("num"));

    auto m_fold = m.def_submodule("fold", "Fold submodule");

    m_fold.def(
        "compute_brute_fold",
        [](const py::array_t<float>& ts_e, const py::array_t<float>& ts_v,
           const py::array_t<FloatType>& freq_arr, SizeType segment_len,
           SizeType nbins, SizeType nsamps, FloatType tsamp, FloatType t_ref,
           SizeType nthreads) {
            return compute_brute_fold(
                std::span<const float>(ts_e.data(), ts_e.size()),
                std::span<const float>(ts_v.data(), ts_v.size()),
                std::span<const FloatType>(freq_arr.data(), freq_arr.size()),
                segment_len, nbins, nsamps, tsamp, t_ref, nthreads);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("nsamps"),
        py::arg("tsamp"), py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1);

    auto m_configs = m.def_submodule("configs", "Configs submodule");
    py::class_<PulsarSearchConfig>(m_configs, "PulsarSearchConfig")
        .def(py::init<SizeType, FloatType, SizeType, FloatType,
                      const std::vector<ParamLimitType>&, FloatType, FloatType,
                      std::optional<SizeType>, std::optional<SizeType>,
                      SizeType>(),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("tol_bins"), py::arg("param_limits"),
             py::arg("ducy_max") = 0.2F, py::arg("wtsp") = 1.5F,
             py::arg("bseg_brute") = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("nthreads") = 1)
        .def_property_readonly("nsamps", &PulsarSearchConfig::get_nsamps)
        .def_property_readonly("tsamp", &PulsarSearchConfig::get_tsamp)
        .def_property_readonly("nbins", &PulsarSearchConfig::get_nbins)
        .def_property_readonly("tol_bins", &PulsarSearchConfig::get_tol_bins)
        .def_property_readonly("param_limits",
                               &PulsarSearchConfig::get_param_limits)
        .def_property_readonly("ducy_max", &PulsarSearchConfig::get_ducy_max)
        .def_property_readonly("wtsp", &PulsarSearchConfig::get_wtsp)
        .def_property_readonly("bseg_brute",
                               &PulsarSearchConfig::get_bseg_brute)
        .def_property_readonly("bseg_ffa", &PulsarSearchConfig::get_bseg_ffa)
        .def_property_readonly("nthreads", &PulsarSearchConfig::get_nthreads)
        .def_property_readonly("tseg_brute",
                               &PulsarSearchConfig::get_tseg_brute)
        .def_property_readonly("tseg_ffa", &PulsarSearchConfig::get_tseg_ffa)
        .def_property_readonly("niters_ffa",
                               &PulsarSearchConfig::get_niters_ffa)
        .def_property_readonly("nparams", &PulsarSearchConfig::get_nparams)
        .def_property_readonly("f_min", &PulsarSearchConfig::get_f_min)
        .def_property_readonly("f_max", &PulsarSearchConfig::get_f_max)
        .def("dparams_f", &PulsarSearchConfig::get_dparams_f,
             py::arg("tseg_cur"))
        .def("dparams", &PulsarSearchConfig::get_dparams, py::arg("tseg_cur"))
        .def("dparams_lim", &PulsarSearchConfig::get_dparams_lim,
             py::arg("tseg_cur"));

    auto m_ffa = m.def_submodule("ffa", "FFA submodule");
    py::class_<FFA>(m_ffa, "FFA")
        .def(py::init<PulsarSearchConfig>(), py::arg("cfg"))
        .def_property_readonly("plan", &FFA::get_plan)
        .def("execute", &FFA::execute, py::arg("ts_e"), py::arg("ts_v"),
             py::arg("fold"));

    m_ffa.def(
        "compute_ffa",
        [](const py::array_t<float>& ts_e, const py::array_t<float>& ts_v,
           const py::object& cfg) {
            return compute_ffa(std::span<const float>(ts_e.data(), ts_e.size()),
                               std::span<const float>(ts_v.data(), ts_v.size()),
                               cfg.cast<PulsarSearchConfig>());
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"));
}
