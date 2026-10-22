#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/loki.hpp"

#include "bindings/bind.hpp"
#include "pybind_utils.hpp"

namespace loki {
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_thresholds(py::module_& m) {
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
                         std::optional<uint64_t> seed, std::string_view backend,
                         int device, SizeType batch_size,
                         std::string_view guess, float min_pd, float w_terminal,
                         float beam_sigma, float reach_sigma) {
                 const detection::SharedOptions shared{.min_pd     = min_pd,
                                                       .w_terminal = w_terminal,
                                                       .beam_sigma = beam_sigma,
                                                       .reach_sigma =
                                                           reach_sigma};
                 return std::make_unique<DynamicThresholdScheme>(
                     std::span<const float>(branching_pattern.data(),
                                            branching_pattern.size()),
                     ref_ducy, nbins, ntrials, nprobs, prob_min, snr_final,
                     nthresholds, ducy_max, wtsp, beam_width, trials_start,
                     mode, seed, batch_size, guess, shared,
                     make_exec(backend, device, nthreads));
             }),
             py::arg("branching_pattern"), py::arg("ref_ducy"),
             py::arg("nbins") = 64, py::arg("ntrials") = 1024,
             py::arg("nprobs") = 40, py::arg("prob_min") = 0.05F,
             py::arg("snr_final") = 8.0F, py::arg("nthresholds") = 100,
             py::arg("ducy_max") = 0.3F, py::arg("wtsp") = 1.0F,
             py::arg("beam_width") = 0.7F, py::arg("trials_start") = 1,
             py::arg("mode") = "shared", py::arg("nthreads") = 1,
             py::arg("seed")    = py::none(), py::kw_only(),
             py::arg("backend") = "cpu", py::arg("device") = 0,
             py::arg("batch_size") = 256, py::arg("guess") = "analytic",
             py::arg("min_pd") = 0.1F, py::arg("w_terminal") = 1.0F,
             py::arg("beam_sigma") = 3.0F, py::arg("reach_sigma") = 1.0F)
        .def("run", &DynamicThresholdScheme::run, py::arg("thres_neigh") = 10,
             "Operational search. The path from get_best_path_thresholds is "
             "used immediately. In-run cost and success_h1_cumul are "
             "optimistic Monte Carlo estimates.")
        .def("save", &DynamicThresholdScheme::save, py::arg("outdir") = "./",
             "Writes the search (states, grids, beam) as HDF5, in every mode; "
             "after solve() also a group 'solution'.")
        .def(
            "solve",
            [](DynamicThresholdScheme& self) {
                auto sol = self.solve();
                py::dict out;
                out["thresholds"]  = as_pyarray(std::move(sol.thresholds));
                out["pd"]          = sol.pd;
                out["pd_se"]       = sol.pd_se;
                out["pd_trials"]   = sol.pd_trials;
                out["c"]           = sol.c;
                out["cost"]        = sol.cost;
                out["eval_states"] = as_pyarray(std::move(sol.eval_states));
                out["search_thresholds"] =
                    as_pyarray(std::move(sol.search_thresholds));
                out["search_pd"] = sol.search_pd;
                out["search_c"]  = sol.search_c;
                py::list shortlist;
                for (auto& path : sol.shortlist) {
                    shortlist.append(as_pyarray(std::move(path)));
                }
                out["shortlist"] = std::move(shortlist);
                py::dict diag;
                for (const auto& [name, value] : sol.diagnostics) {
                    diag[py::str(name)] = value;
                }
                out["diagnostics"] = std::move(diag);
                out["delivery"]   = detection::delivery_to_string(sol.delivery);
                out["min_pd"]     = sol.min_pd;
                out["w_terminal"] = sol.w_terminal;
                return out;
            },
            "Modes 'shared' and 'robust', after run(): the schedule to use "
            "with the Pd and cost of exactly those thresholds measured on "
            "fresh samples (tightening, selection, refinement and repair use "
            "other samples than the report). Frees the search's device "
            "memory.")
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
        .def(
            "get_current_thresholds_idx",
            [](DynamicThresholdScheme& self, SizeType istage) {
                return as_pyarray(self.get_current_thresholds_idx(istage));
            },
            py::arg("istage"), "Grid indices of the beam of stage istage.")
        .def("get_states",
             [](DynamicThresholdScheme& self) {
                 return as_pyarray(self.get_states());
             })
        .def("get_best_path_thresholds",
             &DynamicThresholdScheme::get_best_path_thresholds,
             py::arg("min_pd") = 0.1,
             "Thresholds the live search should use. Not a re-scored path.")
        .def(
            "evaluate",
            [](const DynamicThresholdScheme& self,
               const PyArrayT<float>& thresholds, SizeType ntrials,
               std::optional<uint64_t> seed) {
                return as_pyarray(self.evaluate(
                    to_span<const float>(thresholds), ntrials, seed));
            },
            py::arg("thresholds"), py::arg("ntrials"),
            py::arg("seed") = py::none(),
            "Reporting only. Does not change the path and is not part of "
            "the on-the-fly pipeline. Pass a seed different from run().")
        .def(
            "calibrate",
            [](const DynamicThresholdScheme& self, float min_pd,
               SizeType ncandidates, float candidate_frac, SizeType ntrials_cal,
               SizeType ntrials_cost, std::optional<uint64_t> seed,
               float delta_max, bool allow_raise) {
                const detection::CalibrationOptions opts{
                    .ncandidates    = ncandidates,
                    .candidate_frac = candidate_frac,
                    .ntrials_cal    = ntrials_cal,
                    .ntrials_cost   = ntrials_cost,
                    .seed           = seed,
                    .delta_max      = delta_max,
                    .allow_raise    = allow_raise,
                };
                auto res = self.calibrate(min_pd, opts);
                py::dict out;
                out["thresholds"]       = as_pyarray(std::move(res.thresholds));
                out["delta"]            = res.delta;
                out["pd_certified"]     = res.pd_certified;
                out["pd_sigma"]         = res.pd_sigma;
                out["pd_check"]         = res.pd_check;
                out["complexity_cumul"] = res.complexity_cumul;
                out["cost"]             = res.cost;
                out["pd_inrun"]         = res.pd_inrun;
                out["cost_inrun"]       = res.cost_inrun;
                out["ncandidates"]      = res.ncandidates;
                out["states"]           = as_pyarray(std::move(res.states));
                return out;
            },
            py::arg("min_pd") = 0.1F, py::kw_only(), py::arg("ncandidates") = 1,
            py::arg("candidate_frac") = 0.5F,
            py::arg("ntrials_cal")    = SizeType{1} << 15U,
            py::arg("ntrials_cost") = 4096, py::arg("seed") = py::none(),
            py::arg("delta_max") = 1.5F, py::arg("allow_raise") = true,
            "Certify and calibrate after run(): shift the cheapest candidate "
            "paths so their Pd on a fresh unpruned signal sample is min_pd, "
            "and return the cheapest. Does not change the search.")
        .def(
            "simulate_h1_scores",
            [](const DynamicThresholdScheme& self, SizeType ntrials,
               uint64_t seed) {
                const auto nstages =
                    static_cast<py::ssize_t>(self.get_nstages());
                return as_pyarray(self.simulate_h1_scores(ntrials, seed))
                    .reshape({nstages, static_cast<py::ssize_t>(ntrials)});
            },
            py::arg("ntrials"), py::arg("seed"),
            "Unpruned signal scores of every stage, shape (nstages, ntrials).");

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
}

} // namespace loki
