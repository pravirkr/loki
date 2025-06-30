#include "loki/algorithms/ffa.hpp"
#include "pybind_utils.hpp"

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"

namespace loki {
using algorithms::FFACOMPLEXCUDA;
using algorithms::FFACUDA;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

PYBIND11_MODULE(libculoki, m) { // NOLINT
    m.doc() = "Python Bindings for the loki library (CUDA Backend)";

    auto m_scores = m.def_submodule("scores", "Scores submodule");
    m_scores.def(
        "snr_boxcar_3d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise, int device_id) {
            if (arr.ndim() != 3 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 3-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);

            auto out = PyArrayT<float>({nprofiles, widths.size()});
            detection::snr_boxcar_3d_cuda(to_span<const float>(arr), nprofiles,
                                          to_span<const SizeType>(widths),
                                          to_span<float>(out), stdnoise,
                                          device_id);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F,
        py::arg("device_id") = 0);

    auto m_fold = m.def_submodule("fold", "Fold submodule");
    m_fold.def(
        "compute_brute_fold_cuda",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int device_id) {
            return as_pyarray(algorithms::compute_brute_fold_cuda(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, device_id));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("device_id") = 0);

    auto m_ffa = m.def_submodule("ffa", "FFA submodule");

    py::class_<FFACUDA>(m_ffa, "FFACUDA")
        .def(py::init<search::PulsarSearchConfig, int>(), py::arg("cfg"),
             py::arg("device_id") = 0)
        .def_property_readonly("plan", &FFACUDA::get_plan)
        .def(
            "execute",
            [](FFACUDA& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    py::class_<FFACOMPLEXCUDA>(m_ffa, "FFACOMPLEXCUDA")
        .def(py::init<search::PulsarSearchConfig, int>(), py::arg("cfg"),
             py::arg("device_id") = 0)
        .def_property_readonly("plan", &FFACOMPLEXCUDA::get_plan)
        .def(
            "execute",
            [](FFACOMPLEXCUDA& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    m_ffa.def(
        "compute_ffa_cuda",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const search::PulsarSearchConfig& cfg, int device_id) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_cuda(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                device_id);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("device_id") = 0);
    m_ffa.def(
        "compute_ffa_complex_cuda",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const search::PulsarSearchConfig& cfg, int device_id) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_complex_cuda(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                device_id);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("device_id") = 0);

    m_ffa.def(
        "compute_ffa_scores_cuda",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const search::PulsarSearchConfig& cfg, int device_id) {
            auto [scores, ffa_plan] = algorithms::compute_ffa_scores_cuda(
                to_span<const float>(ts_e), to_span<const float>(ts_v), cfg,
                device_id);
            return std::make_tuple(as_pyarray(std::move(scores)), ffa_plan);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("cfg"),
        py::arg("device_id") = 0);
}

} // namespace loki