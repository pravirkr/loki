#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <loki/score.hpp>

namespace py = pybind11;

PYBIND11_MODULE(libloki, mod) {
    mod.doc() = "Python bindings for the loki library";
    mod.def("snr_1d", [](const py::array_t<float, py::array::c_style>& arr,
                         const py::array_t<size_t, py::array::c_style>& widths,
                         float stdnoise) {
        // auto widths_arr = widths.cast<std::vector<size_t>>();
        auto out = py::array_t<float, py::array::c_style>(widths.size());
        loki::snr_1d(std::span<const float>(arr.data(), arr.size()),
                     std::span<const size_t>(widths.data(), widths.size()),
                     stdnoise,
                     std::span<float>(out.mutable_data(), out.size()));
        return out;
    });
    mod.def("snr_2d", [](const py::array_t<float, py::array::c_style>& arr,
                         const py::array_t<size_t, py::array::c_style>& widths,
                         float stdnoise) {
        const auto* shape    = arr.shape();
        const auto nprofiles = shape[0];
        py::array_t<float, py::array::c_style> out({nprofiles, widths.size()});
        loki::snr_2d(std::span<const float>(arr.data(), arr.size()), nprofiles,
                     std::span<const size_t>(widths.data(), widths.size()),
                     stdnoise,
                     std::span<float>(out.mutable_data(), out.size()));
        return out;
    });

    py::class_<MatchedFilter>(mod, "MatchedFilter")
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
        // returns 2d array
        .def_property_readonly("templates",
                               [](MatchedFilter& mf) {
                                   // return a py::array_t<float> from a
                                   // std::vector<float> also reshape the array
                                   // to 2d using ntemplates and nbins
                                   return as_pyarray(mf.get_templates());
                               })
        // takes 2d array as input and returns 2d array as output
        .def("compute", [](MatchedFilter& mf,
                           const py::array_t<float, py::array::c_style>& arr) {
            const auto shape     = arr.shape();
            const auto nprofiles = shape[0];
            auto snr             = py::array_t<float, py::array::c_style>(
                py::array::ShapeContainer(
                    {nprofiles, static_cast<long>(mf.get_ntemplates())}));
            mf.compute(std::span<const float>(arr.data(), arr.size()),
                       std::span<float>(snr.mutable_data(), snr.size()));
            return snr;
        });
}
