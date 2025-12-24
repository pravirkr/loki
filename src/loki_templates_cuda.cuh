#pragma once

#include "pybind_utils.hpp"

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/loki.hpp"

namespace loki {
using algorithms::FFACUDA;
using search::PulsarSearchConfig;

namespace py = pybind11;

// Template function to bind FFACUDA<T>
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void bind_ffa_cuda_class(py::module& m, const std::string& name) {
    using HostFoldType = FoldTypeTraits<FoldTypeCUDA>::HostType;
    auto cls =
        py::class_<FFACUDA<FoldTypeCUDA>>(m, name.c_str())
            .def(py::init<PulsarSearchConfig, int>(), py::arg("cfg"),
                 py::arg("device_id") = 0)
            .def_property_readonly("plan", &FFACUDA<HostFoldType>::get_plan);

    // Standard execute
    cls.def(
        "execute",
        [](FFACUDA<FoldTypeCUDA>& self, const PyArrayT<float>& ts_e,
           const PyArrayT<float>& ts_v, PyArrayT<HostFoldType>& fold) {
            self.execute(to_span<const float>(ts_e), to_span<const float>(ts_v),
                         to_span<HostFoldType>(fold));
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
} // namespace loki
