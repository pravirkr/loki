#include <cstddef>
#include <span>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <loki/loki_types.hpp>
#include <loki/score.hpp>
#include <loki/thresholds.hpp>
#include <loki/utils.hpp>
#include <vector>

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

    py::class_<State>(m_thresholds, "State")
        .def(py::init<>())
        .def_readwrite("success_h0", &State::success_h0)
        .def_readwrite("success_h1", &State::success_h1)
        .def_readwrite("complexity", &State::complexity)
        .def_readwrite("complexity_cumul", &State::complexity_cumul)
        .def_readwrite("success_h1_cumul", &State::success_h1_cumul)
        .def_readwrite("nbranches", &State::nbranches)
        .def_readwrite("backtrack", &State::backtrack)
        .def_property_readonly("cost", &State::cost);
    py::class_<DynamicThresholdScheme>(m_thresholds, "DynamicThresholdScheme")
        .def(
            py::init([](const py::array_t<float>& branching_pattern,
                        const py::array_t<float>& profile, SizeType nparams,
                        float snr_final, SizeType nthresholds, SizeType ntrials,
                        SizeType nprobs, float ducy_max, float beam_width) {
                return std::make_unique<DynamicThresholdScheme>(
                    std::span<const float>(branching_pattern.data(),
                                           branching_pattern.size()),
                    std::span<const float>(profile.data(), profile.size()),
                    nparams, snr_final, nthresholds, ntrials, nprobs, ducy_max,
                    beam_width);
            }),
            py::arg("branching_pattern"), py::arg("profile"),
            py::arg("nparams"), py::arg("snr_final") = 8.0F,
            py::arg("nthresholds") = 100, py::arg("ntrials") = 1024,
            py::arg("nprobs") = 10, py::arg("ducy_max") = 0.3F,
            py::arg("beam_width") = 0.7F)
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
            auto states = self.get_states();
            py::list state_list;
            for (const auto& state : states) {
                if (state) {
                    state_list.append(State(*state));
                } else {
                    state_list.append(py::none());
                }
            }
            return state_list;
        });
    m_thresholds.def(
        "evaluate_threshold_scheme",
        [](const py::array_t<float>& thresholds,
           const py::array_t<float>& branching_pattern,
           const py::array_t<float>& profile, SizeType ntrials, float snr_final,
           float ducy_max) {
            return evaluate_threshold_scheme(
                std::span<const float>(thresholds.data(), thresholds.size()),
                std::span<const float>(branching_pattern.data(),
                                       branching_pattern.size()),
                std::span<const float>(profile.data(), profile.size()), ntrials,
                snr_final, ducy_max);
        },
        py::arg("thresholds"), py::arg("branching_pattern"), py::arg("profile"),
        py::arg("ntrials") = 1024, py::arg("snr_final") = 8.0F,
        py::arg("ducy_max") = 0.3F);
    m_thresholds.def(
        "determine_threshold_scheme",
        [](const py::array_t<float>& survive_probs,
           const py::array_t<float>& branching_pattern,
           const py::array_t<float>& profile, SizeType ntrials, float snr_final,
           float ducy_max) {
            return determine_threshold_scheme(
                std::span<const float>(survive_probs.data(),
                                       survive_probs.size()),
                std::span<const float>(branching_pattern.data(),
                                       branching_pattern.size()),
                std::span<const float>(profile.data(), profile.size()), ntrials,
                snr_final, ducy_max);
        },
        py::arg("survive_probs"), py::arg("branching_pattern"),
        py::arg("profile"), py::arg("ntrials") = 1024,
        py::arg("snr_final") = 8.0F, py::arg("ducy_max") = 0.3F);
    m_thresholds.def(
        "simulate_folds",
        [](const py::array_t<float>& folds, float var_cur,
           const py::array_t<float>& profile, float bias_snr, float var_add,
           SizeType ntrials) {
            const auto nbins      = profile.size();
            const auto ntrials_in = folds.size() / nbins;
            const auto folds_in   = FoldVector(
                std::vector<float>(folds.data(), folds.data() + folds.size()),
                var_cur, ntrials_in, nbins);
            ThreadSafeRNG rng(std::random_device{}());
            auto out = simulate_folds(
                folds_in,
                std::span<const float>(profile.data(), profile.size()), rng,
                bias_snr, var_add, ntrials);
            return as_pyarray_ref(out.data);
        },
        py::arg("folds"), py::arg("var_cur"), py::arg("profile"),
        py::arg("bias_snr") = 0.0F, py::arg("var_add") = 1.0F,
        py::arg("ntrials") = 1024);
    m_thresholds.def(
        "compute_threshold_survival",
        [](const py::array_t<float>& scores, float survive_prob) {
            return compute_threshold_survival(
                std::span<const float>(scores.data(), scores.size()),
                survive_prob);
        },
        py::arg("scores"), py::arg("survive_prob"));

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
}