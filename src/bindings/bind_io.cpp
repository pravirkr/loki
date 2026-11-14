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
    py::class_<io::ObservationMetadata>(m_io, "ObservationMetadata")
        .def(py::init<>())
        .def_readwrite("dm", &io::ObservationMetadata::dm)
        .def_readwrite("start_mjd", &io::ObservationMetadata::start_mjd)
        .def_readwrite("barycentric", &io::ObservationMetadata::barycentric)
        .def_readwrite("source", &io::ObservationMetadata::source)
        .def_readwrite("input_path", &io::ObservationMetadata::input_path)
        .def_readwrite("original_nsamps",
                       &io::ObservationMetadata::original_nsamps)
        .def("validate", &io::ObservationMetadata::validate);

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
    py::enum_<io::ZapMode>(m_io, "ZapMode")
        .value("Noise", io::ZapMode::kNoise)
        .value("PhaseKept", io::ZapMode::kPhaseKept)
        .value("Zero", io::ZapMode::kZero);
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
        .def_readwrite("zap_mode", &PO::zap_mode)
        .def_readwrite("zap_seed", &PO::zap_seed)
        .def_readwrite("birdies", &PO::birdies)
        .def_readwrite("loc", &PO::loc)
        .def_readwrite("scale", &PO::scale)
        .def_readwrite("fast_median", &PO::fast_median)
        .def_readwrite("fast_median_min_points", &PO::fast_median_min_points)
        .def("validate", &PO::validate);
    using PR = io::PreprocessReport;
    py::class_<PR::ZapBand>(m_io, "ZapBand")
        .def_readonly("f_lo", &PR::ZapBand::f_lo)
        .def_readonly("f_hi", &PR::ZapBand::f_hi)
        .def_readonly("n_zapped", &PR::ZapBand::n_zapped)
        .def_readonly("power_ratio", &PR::ZapBand::power_ratio);
    py::class_<PR>(m_io, "PreprocessReport")
        .def_readonly("method", &PR::method)
        .def_readonly("nsamps", &PR::nsamps)
        .def_readonly("baseline_window", &PR::baseline_window)
        .def_readonly("variance_window", &PR::variance_window)
        .def_readonly("block_size", &PR::block_size)
        .def_readonly("n_masked", &PR::n_masked)
        .def_readonly("n_clipped", &PR::n_clipped)
        .def_readonly("n_zapped", &PR::n_zapped)
        .def_readonly("zap_bands", &PR::zap_bands)
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
    py::class_<io::TimeSeriesInfo>(m_io, "TimeSeriesInfo")
        .def_readonly("nsamps", &io::TimeSeriesInfo::nsamps)
        .def_readonly("dt", &io::TimeSeriesInfo::dt)
        .def_readonly("observation", &io::TimeSeriesInfo::observation);
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
        .def_static("read_info", &io::TimeSeries::read_info, py::arg("path"))
        .def("write", &io::TimeSeries::write, py::arg("path"))
        .def(
            "preprocess",
            [](io::TimeSeries& self, const PO& options, int nthreads,
               std::string_view backend, int device) {
                return self.preprocess(options,
                                       make_exec(backend, device, nthreads));
            },
            py::arg("options") = PO{}, py::arg("nthreads") = 1, py::kw_only(),
            py::arg("backend") = "cpu", py::arg("device") = 0)
        .def_property_readonly("observation", &io::TimeSeries::get_observation)
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

    using SO = io::SiftOptions;
    py::class_<SO>(m_io, "SiftOptions")
        .def(py::init<>())
        .def_readwrite("snr_min", &SO::snr_min)
        .def_readwrite("tolerance", &SO::tolerance)
        .def_readwrite("min_segments", &SO::min_segments)
        .def_readwrite("harmonic_order_max", &SO::harmonic_order_max)
        .def_readwrite("harmonic_bins", &SO::harmonic_bins);
    py::class_<io::SiftFile>(m_io, "SiftFile")
        .def_readonly("path", &io::SiftFile::path)
        .def_readonly("dm", &io::SiftFile::dm)
        .def_readonly("start_mjd", &io::SiftFile::start_mjd)
        .def_readonly("reference_mjd", &io::SiftFile::reference_mjd)
        .def_readonly("tobs", &io::SiftFile::tobs)
        .def_readonly("segment", &io::SiftFile::segment);
    py::class_<io::SiftMember>(m_io, "SiftMember")
        .def_readonly("file", &io::SiftMember::file)
        .def_readonly("id", &io::SiftMember::id)
        .def_readonly("freq", &io::SiftMember::freq)
        .def_readonly("fdot", &io::SiftMember::fdot)
        .def_readonly("snr", &io::SiftMember::snr)
        .def_readonly("duty_cycle", &io::SiftMember::duty_cycle);
    py::class_<io::SiftGroup>(m_io, "SiftGroup")
        .def_readonly("members", &io::SiftGroup::members)
        .def_readonly("dm_curve", &io::SiftGroup::dm_curve)
        .def_readonly("segments", &io::SiftGroup::segments)
        .def_readonly("edge_ratio", &io::SiftGroup::edge_ratio)
        .def_readonly("extrapolation_bins", &io::SiftGroup::extrapolation_bins)
        .def_readonly("harmonic_of", &io::SiftGroup::harmonic_of)
        .def_readonly("harmonic_numerator", &io::SiftGroup::harmonic_numerator)
        .def_readonly("harmonic_denominator",
                      &io::SiftGroup::harmonic_denominator)
        .def_readonly("harmonic_offset_bins",
                      &io::SiftGroup::harmonic_offset_bins);
    py::class_<io::SiftResult>(m_io, "SiftResult")
        .def_readonly("files", &io::SiftResult::files)
        .def_readonly("groups", &io::SiftResult::groups)
        .def_readonly("n_candidates", &io::SiftResult::n_candidates)
        .def_readonly("n_segments", &io::SiftResult::n_segments);
    m_io.def(
        "sift",
        [](const std::vector<std::filesystem::path>& files, const SO& options) {
            return io::sift(files, options);
        },
        py::arg("files"), py::arg("options") = SO{},
        "Group the candidates of published result files by frequency: S/N "
        "against DM, segments and harmonic relations (loki sift).");
}

} // namespace loki
