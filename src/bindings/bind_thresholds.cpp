#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

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
                         int device, SizeType batch_size) {
                 return std::make_unique<DynamicThresholdScheme>(
                     std::span<const float>(branching_pattern.data(),
                                            branching_pattern.size()),
                     ref_ducy, nbins, ntrials, nprobs, prob_min, snr_final,
                     nthresholds, ducy_max, wtsp, beam_width, trials_start,
                     mode, seed, batch_size,
                     make_exec(backend, device, nthreads));
             }),
             py::arg("branching_pattern"), py::arg("ref_ducy"),
             py::arg("nbins") = 64, py::arg("ntrials") = 1024,
             py::arg("nprobs") = 10, py::arg("prob_min") = 0.05F,
             py::arg("snr_final") = 8.0F, py::arg("nthresholds") = 100,
             py::arg("ducy_max") = 0.3F, py::arg("wtsp") = 1.0F,
             py::arg("beam_width") = 0.7F, py::arg("trials_start") = 1,
             py::arg("mode") = "legacy", py::arg("nthreads") = 1,
             py::arg("seed")    = py::none(), py::kw_only(),
             py::arg("backend") = "cpu", py::arg("device") = 0,
             py::arg("batch_size") = 256)
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
}

} // namespace loki
