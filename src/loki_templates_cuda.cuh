#pragma once

#include "pybind_utils.hpp"

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"

namespace loki {
using algorithms::EPMultiPassCUDA;
using algorithms::FFACUDA;
using search::PulsarSearchConfig;

namespace py = pybind11;

// Template function to bind FFACUDA<T>
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void bind_ffa_cuda_class(py::module& m, const std::string& name) {
    using HostFoldT = HostFoldType<FoldTypeCUDA>;
    auto cls =
        py::class_<FFACUDA<FoldTypeCUDA>>(m, name.c_str())
            .def(py::init<PulsarSearchConfig, int>(), py::arg("cfg"),
                 py::arg("device_id") = 0)
            .def_property_readonly("plan", &FFACUDA<FoldTypeCUDA>::get_plan);

    // Standard execute
    cls.def(
        "execute",
        [](FFACUDA<FoldTypeCUDA>& self, const PyArrayT<float>& ts_e,
           const PyArrayT<float>& ts_v, PyArrayT<HostFoldT>& fold) {
            self.execute(to_span<const float>(ts_e), to_span<const float>(ts_v),
                         to_span<HostFoldT>(fold));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));

    // Specialized execute for ComplexType (return to time domain)
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        cls.def(
            "execute",
            [](FFACUDA<FoldTypeCUDA>& self, const PyArrayT<float>& ts_e,
               const PyArrayT<float>& ts_v, PyArrayT<float>& fold) {
                self.execute(to_span<const float>(ts_e),
                             to_span<const float>(ts_v), to_span<float>(fold));
            },
            py::arg("ts_e"), py::arg("ts_v"), py::arg("fold"));
    }
}

// Template function to bind EPMultiPassCUDA<T>
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void bind_ep_multi_pass_cuda_class(py::module& m, const std::string& name) {
    auto cls =
        py::class_<EPMultiPassCUDA<FoldTypeCUDA>>(m, name.c_str())
            .def(py::init<const PulsarSearchConfig&, const std::vector<float>&,
                          std::optional<SizeType>,
                          std::optional<std::vector<SizeType>>, SizeType,
                          SizeType, std::string_view, int>(),
                 py::arg("cfg"), py::arg("threshold_scheme"),
                 py::arg("n_runs")   = std::nullopt,
                 py::arg("ref_segs") = std::nullopt,
                 py::arg("max_sugg") = 1U << 20U, py::arg("batch_size") = 4096U,
                 py::arg("poly_basis") = "taylor", py::arg("device_id") = 0);
    // Standard execute
    cls.def(
        "execute",
        [](EPMultiPassCUDA<FoldTypeCUDA>& self, const PyArrayT<float>& ts_e,
           const PyArrayT<float>& ts_v, std::string_view outdir,
           std::string_view file_prefix) {
            self.execute(to_span<const float>(ts_e), to_span<const float>(ts_v),
                         outdir, file_prefix);
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("outdir"),
        py::arg("file_prefix"));
}

} // namespace loki
