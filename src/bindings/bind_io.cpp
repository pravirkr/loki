#include <span>
#include <string_view>
#include <utility>
#include <vector>

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
    py::enum_<io::PreprocessMethod>(m_io, "PreprocessMethod")
        .value("Robust", io::PreprocessMethod::kRobust)
        .value("ZScore", io::PreprocessMethod::kZScore);
    py::enum_<io::GainModel>(m_io, "GainModel")
        .value("Additive", io::GainModel::kAdditive)
        .value("Multiplicative", io::GainModel::kMultiplicative);
    py::class_<io::Birdie>(m_io, "Birdie")
        .def(py::init([](double freq, double width) {
                 return io::Birdie{.freq = freq, .width = width};
             }),
             py::arg("freq"), py::arg("width") = 0.0)
        .def_readwrite("freq", &io::Birdie::freq)
        .def_readwrite("width", &io::Birdie::width);
    using PO = io::PreprocessOptions;
    py::class_<PO>(m_io, "PreprocessOptions")
        .def(py::init<>())
        .def_readwrite("method", &PO::method)
        .def_readwrite("filter_window", &PO::filter_window)
        .def_readwrite("gain_model", &PO::gain_model)
        .def_readwrite("variance_window", &PO::variance_window)
        .def_readwrite("window_blocks", &PO::window_blocks)
        .def_readwrite("n_iter", &PO::n_iter)
        .def_readwrite("block_scales", &PO::block_scales)
        .def_readwrite("block_sigma", &PO::block_sigma)
        .def_readwrite("min_good_fraction", &PO::min_good_fraction)
        .def_readwrite("clip_sigma", &PO::clip_sigma)
        .def_readwrite("zap_periodic", &PO::zap_periodic)
        .def_readwrite("zap_sigma", &PO::zap_sigma)
        .def_readwrite("zap_whiten_bins", &PO::zap_whiten_bins)
        .def_readwrite("birdies", &PO::birdies)
        .def_readwrite("loc", &PO::loc)
        .def_readwrite("scale", &PO::scale)
        .def_readwrite("fast_median", &PO::fast_median)
        .def_readwrite("fast_median_min_points", &PO::fast_median_min_points)
        .def("validate", &PO::validate);
    using PR = io::PreprocessReport;
    py::class_<PR>(m_io, "PreprocessReport")
        .def_readonly("method", &PR::method)
        .def_readonly("nsamps", &PR::nsamps)
        .def_readonly("baseline_window", &PR::baseline_window)
        .def_readonly("variance_window", &PR::variance_window)
        .def_readonly("block_size", &PR::block_size)
        .def_readonly("n_masked", &PR::n_masked)
        .def_readonly("n_clipped", &PR::n_clipped)
        .def_readonly("n_zapped", &PR::n_zapped)
        .def_readonly("longest_masked_run", &PR::longest_masked_run)
        .def_readonly("global_scale", &PR::global_scale)
        .def_readonly("norm", &PR::norm)
        .def_property_readonly(
            "block_mu", [](const PR& r) { return as_pyarray_ref(r.block_mu); })
        .def_property_readonly(
            "block_sigma",
            [](const PR& r) { return as_pyarray_ref(r.block_sigma); })
        .def_property_readonly(
            "block_good_fraction",
            [](const PR& r) { return as_pyarray_ref(r.block_good_fraction); })
        .def_property_readonly("zero_weight_fraction",
                               &PR::zero_weight_fraction);
    m_io.def(
        "preprocess",
        [](const PyArrayT<float>& raw, double tsamp, const PO& options,
           int nthreads, std::string_view backend, int device) {
            const auto in = to_span<const float>(raw);
            std::vector<float> ts_e(in.size());
            std::vector<float> ts_v(in.size());
            auto report = io::preprocess(in, tsamp, ts_e, ts_v, options,
                                         make_exec(backend, device, nthreads));
            return py::make_tuple(as_pyarray(std::move(ts_e)),
                                  as_pyarray(std::move(ts_v)),
                                  std::move(report));
        },
        py::arg("raw"), py::arg("tsamp"), py::arg("options") = PO{},
        py::arg("nthreads") = 1, py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0,
        "Build (ts_e, ts_v, report) from a raw timeseries.");
    py::class_<io::ReadOptions>(m_io, "ReadOptions")
        .def(py::init<>())
        .def_readwrite("preprocess", &io::ReadOptions::preprocess)
        .def_readwrite("preprocessing", &io::ReadOptions::preprocessing)
        .def_readwrite("nthreads", &io::ReadOptions::nthreads);
    const auto timeseries_view = [](const py::object& self,
                                    std::span<float> data) {
        return py::array_t<float>(
            py::array::ShapeContainer{static_cast<py::ssize_t>(data.size())},
            py::array::StridesContainer{
                static_cast<py::ssize_t>(sizeof(float)),
            },
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
        .def(
            "preprocess",
            [](io::TimeSeries& self, const PO& options, int nthreads) {
                return self.preprocess(options, Exec::cpu(nthreads));
            },
            py::arg("options") = PO{}, py::arg("nthreads") = 1)
        .def_property_readonly("nsamps", &io::TimeSeries::get_nsamps)
        .def_property_readonly("dt", &io::TimeSeries::get_dt)
        .def_property_readonly("tobs", &io::TimeSeries::get_tobs)
        .def_property_readonly("ts_e",
                               [timeseries_view](const py::object& self) {
                                   return timeseries_view(
                                       self,
                                       self.cast<io::TimeSeries&>().get_ts_e());
                               })
        .def_property_readonly(
            "ts_v", [timeseries_view](const py::object& self) {
                return timeseries_view(self,
                                       self.cast<io::TimeSeries&>().get_ts_v());
            });
}

} // namespace loki
