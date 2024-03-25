#include <span>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loki/scores.hpp"

namespace py = pybind11;

// helper function to avoid making a copy when returning a py::array_t
// author: https://github.com/YannickJadoul
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr
        = std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p));
    });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

PYBIND11_MODULE(libloki, mod) {
    mod.doc() = "Python bindings for the loki library";
    mod.def("snr_1d", [](const py::array_t<float, py::array::c_style>& arr,
                         const py::array_t<size_t, py::array::c_style>& widths,
                         float stdnoise) {
        auto widths_arr = widths.cast<std::vector<size_t>>();
        auto out        = py::array_t<float>(widths_arr.size());
        loki::snr_1d(
            std::span<const float>(arr.data(), arr.size()),
            std::span<const size_t>(widths_arr.data(), widths_arr.size()),
            stdnoise, std::span<float>(out.mutable_data(), out.size()));
    });
    mod.def("snr_2d", [](const py::array_t<float, py::array::c_style>& arr,
                         const py::array_t<size_t, py::array::c_style>& widths,
                         float stdnoise) {
        const auto shape     = arr.shape();
        const auto nprofiles = shape[0];
        auto widths_arr      = widths.cast<std::vector<size_t>>();
        auto out             = py::array_t<float>(py::array::ShapeContainer(
            {nprofiles, static_cast<long>(widths_arr.size())}));
        loki::snr_2d(
            std::span<const float>(arr.data(), arr.size()), nprofiles,
            std::span<const size_t>(widths_arr.data(), widths_arr.size()),
            stdnoise, std::span<float>(out.mutable_data(), out.size()));
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
        .def_property_readonly("templates", &MatchedFilter::get_templates)
        // takes 2d array as input and returns 2d array as output
        .def("compute", [](MatchedFilter& mf,
                           const py::array_t<float, py::array::c_style>& arr) {
            const auto shape = arr.shape();
            auto snr         = py::array_t<float>(py::array::ShapeContainer(
                {shape[1], static_cast<long>(mf.get_ntemplates())}));
            mf.compute(std::span<const float>(arr.data(), arr.size()),
                       std::span<float>(snr.mutable_data(), snr.size()));
            return snr;
        });
}
