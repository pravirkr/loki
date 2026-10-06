#include "bindings/bind.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/loki.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using pipelines::EPFreqSweep;
using pipelines::FFAFreqSweep;
using detection::MatchedFilter;
using plans::FFAPlanBase;
using algorithms::EPChunkConfig;
using algorithms::EPChunkStats;
using algorithms::EPRegionStats;
using algorithms::FFARegionStats;
using search::FFASearchConfig;
using search::PulsarSearchConfig;


namespace py = pybind11;

void bind_io(py::module_& m) {
    auto m_io = m.def_submodule("io", "Timeseries I/O");
    py::enum_<LocMethod>(m_io, "LocMethod")
        .value("Mean", LocMethod::kMean)
        .value("Median", LocMethod::kMedian)
        .value("None", LocMethod::kNone);
    py::enum_<ScaleMethod>(m_io, "ScaleMethod")
        .value("Std", ScaleMethod::kStd)
        .value("Iqr", ScaleMethod::kIqr)
        .value("Mad", ScaleMethod::kMad)
        .value("DoubleMad", ScaleMethod::kDoubleMad)
        .value("None", ScaleMethod::kNone);
    py::class_<io::ReadOptions>(m_io, "ReadOptions")
        .def(py::init<>())
        .def_readwrite("preprocess", &io::ReadOptions::preprocess)
        .def_readwrite("filter_window", &io::ReadOptions::filter_window)
        .def_readwrite("loc", &io::ReadOptions::loc)
        .def_readwrite("scale", &io::ReadOptions::scale)
        .def_readwrite("fast_median", &io::ReadOptions::fast_median)
        .def_readwrite("fast_median_min_points",
                       &io::ReadOptions::fast_median_min_points)
        .def_readwrite("nthreads", &io::ReadOptions::nthreads);
    auto timeseries_view = [](py::object self, std::span<float> data) {
        return py::array_t<float>(
            py::array::ShapeContainer{static_cast<py::ssize_t>(data.size())},
            py::array::StridesContainer{
                static_cast<py::ssize_t>(sizeof(float))},
            data.data(), self);
    };
    py::class_<io::TimeSeries>(m_io, "TimeSeries")
        .def(py::init([](const PyArrayT<float>& ts_e,
                         const PyArrayT<float>& ts_v, double dt) {
                 const auto intensity = to_span<const float>(ts_e);
                 const auto variance  = to_span<const float>(ts_v);
                 return io::TimeSeries(
                     std::vector<float>(intensity.begin(), intensity.end()),
                     std::vector<float>(variance.begin(), variance.end()), dt);
             }),
             py::arg("ts_e"), py::arg("ts_v"), py::arg("dt"))
        .def_static("read", &io::TimeSeries::read, py::arg("path"),
                    py::arg("options") = io::ReadOptions{})
        .def("write", &io::TimeSeries::write, py::arg("path"))
        .def_property_readonly("nsamps", &io::TimeSeries::get_nsamps)
        .def_property_readonly("dt", &io::TimeSeries::get_dt)
        .def_property_readonly("tobs", &io::TimeSeries::get_tobs)
        .def_property_readonly("ts_e",
                               [timeseries_view](py::object self) {
                                   return timeseries_view(
                                       self,
                                       self.cast<io::TimeSeries&>().get_ts_e());
                               })
        .def_property_readonly("ts_v", [timeseries_view](py::object self) {
            return timeseries_view(self,
                                   self.cast<io::TimeSeries&>().get_ts_v());
        });
}

} // namespace loki
