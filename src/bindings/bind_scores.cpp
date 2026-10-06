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
using algorithms::EPFreqSweep;
using algorithms::FFAFreqSweep;
using detection::MatchedFilter;
using plans::FFAPlanBase;
using regions::EPChunkConfig;
using regions::EPChunkStats;
using regions::EPRegionStats;
using regions::FFARegionStats;
using search::FFASearchConfig;
using search::PulsarSearchConfig;


namespace py = pybind11;

void bind_scores(py::module_& m) {
    auto m_scores = m.def_submodule("scores", "Scores submodule");
    py::class_<MatchedFilter>(m_scores, "MatchedFilter")
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
        .def_property_readonly("templates",
                               [](MatchedFilter& mf) {
                                   // return a py::array_t<float> from a
                                   // std::vector<float> also reshape the array
                                   // to 2d using ntemplates and nbins
                                   return as_pyarray(mf.get_templates());
                               })
        .def("compute", [](MatchedFilter& self,
                           const py::array_t<float, py::array::c_style>& arr) {
            if (arr.ndim() != 1) {
                throw std::runtime_error(
                    "Input and output arrays must be 1-dimensional");
            }
            if (arr.size() != static_cast<ssize_t>(self.get_nbins())) {
                throw std::runtime_error("Input array size must match nbins");
            }
            const auto nprofiles  = arr.shape(0);
            const auto ntemplates = self.get_ntemplates();

            auto snr = py::array_t<float, py::array::c_style>(
                py::array::ShapeContainer(
                    {nprofiles, static_cast<ssize_t>(ntemplates)}));
            self.compute(std::span<const float>(arr.data(), arr.size()),
                         std::span<float>(snr.mutable_data(), snr.size()));
            return snr;
        });
    m_scores.def(
        "generate_box_width_trials",
        [](SizeType nbins, double ducy_max, double wtsp) {
            auto trials =
                detection::generate_box_width_trials(nbins, ducy_max, wtsp);
            return as_pyarray_ref(trials);
        },
        py::arg("nbins"), py::arg("ducy_max") = 0.3, py::arg("wtsp") = 1.5);

    m_scores.def(
        "snr_boxcar_1d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise) {
            if (arr.size() == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            auto out = PyArrayT<float>(widths.size());
            detection::snr_boxcar_1d(to_span<const float>(arr),
                                     to_span<const SizeType>(widths),
                                     to_span<float>(out), stdnoise);
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F);
    m_scores.def(
        "snr_boxcar_2d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise, int nthreads, std::string_view backend,
           int device) {
            if (arr.ndim() != 2 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 2-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(1);

            auto out = PyArrayT<float>({nprofiles, widths.size()});
            detection::snr_boxcar_2d(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins, stdnoise,
                make_exec(backend, device, nthreads));
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F,
        py::arg("nthreads") = 1, py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);
    m_scores.def(
        "snr_boxcar_2d_max",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           float stdnoise, int nthreads, std::string_view backend,
           int device) {
            if (arr.ndim() != 2 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 2-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(1);

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_2d_max(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins, stdnoise,
                make_exec(backend, device, nthreads));
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("stdnoise") = 1.0F,
        py::arg("nthreads") = 1, py::kw_only(), py::arg("backend") = "cpu",
        py::arg("device") = 0);
    m_scores.def(
        "snr_boxcar_3d",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           int nthreads, std::string_view backend, int device) {
            if (arr.ndim() != 3 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 3-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(2);

            auto out = PyArrayT<float>({nprofiles, widths.size()});
            detection::snr_boxcar_3d(to_span<const float>(arr),
                                     to_span<const SizeType>(widths),
                                     to_span<float>(out), nprofiles, nbins,
                                     make_exec(backend, device, nthreads));
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("nthreads") = 1,
        py::kw_only(), py::arg("backend") = "cpu", py::arg("device") = 0);
    m_scores.def(
        "snr_boxcar_3d_max",
        [](const PyArrayT<float>& arr, const PyArrayT<SizeType>& widths,
           int nthreads, std::string_view backend, int device) {
            if (arr.ndim() != 3 || widths.ndim() != 1) {
                throw std::runtime_error("Input array must be 3-dimensional, "
                                         "widths must be 1-dimensional");
            }
            if (arr.shape(0) == 0 || arr.shape(1) == 0 || widths.size() == 0) {
                throw std::runtime_error("Input arrays cannot be empty");
            }
            const auto nprofiles = arr.shape(0);
            const auto nbins     = arr.shape(2);

            auto out = PyArrayT<float>(nprofiles);
            detection::snr_boxcar_3d_max(
                to_span<const float>(arr), to_span<const SizeType>(widths),
                to_span<float>(out), nprofiles, nbins,
                make_exec(backend, device, nthreads));
            return out;
        },
        py::arg("arr"), py::arg("widths"), py::arg("nthreads") = 1,
        py::kw_only(), py::arg("backend") = "cpu", py::arg("device") = 0);
}

} // namespace loki
