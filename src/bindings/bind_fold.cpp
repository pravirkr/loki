#include <stdexcept>
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

void bind_fold(py::module_& m) {
    auto m_fold = m.def_submodule("fold", "Fold submodule");
    m_fold.def(
        "compute_brute_fold_time",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads,
           std::string_view backend, int device) {
            return as_pyarray(algorithms::compute_brute_fold<float>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, make_exec(backend, device, nthreads)));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0);
    m_fold.def(
        "compute_brute_fold_fourier",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& freq_arr, SizeType segment_len,
           SizeType nbins, double tsamp, double t_ref, int nthreads,
           std::string_view backend, int device) {
            return as_pyarray(algorithms::compute_brute_fold<ComplexType>(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(freq_arr), segment_len, nbins, tsamp,
                t_ref, make_exec(backend, device, nthreads)));
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("freq_arr"),
        py::arg("segment_len"), py::arg("nbins"), py::arg("tsamp"),
        py::arg("t_ref") = 0.0F, py::arg("nthreads") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0);
    m_fold.def(
        "fold_phase",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           const PyArrayT<double>& phase, SizeType nbins) {
            auto fold = as_pyarray(algorithms::fold_phase(
                to_span<const float>(ts_e), to_span<const float>(ts_v),
                to_span<const double>(phase), nbins));
            return fold.reshape(
                {static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(nbins)});
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("phase"), py::arg("nbins"),
        R"doc(Fold a weighted series at a phase (cycles) per sample.

Each sample goes to the bin nearest frac(phase) * nbins, as in the brute fold.
Returns shape (2, nbins): the sums of ts_e, then of ts_v, per bin.)doc");
    m_fold.def(
        "refold",
        [](const PyArrayT<float>& ts_e, const PyArrayT<float>& ts_v,
           double tsamp, double t_ref, const PyArrayT<double>& leaf,
           std::string_view model, double propagator_significance,
           SizeType nbins, SizeType nsub, SizeType df_steps, double df_span_t,
           double df_step_bins, SizeType fd_steps, double fd_span,
           SizeType split, int nthreads) {
            if (leaf.ndim() != 2 || leaf.shape(1) != 2 || leaf.shape(0) < 3) {
                throw std::invalid_argument(
                    "leaf must have shape (n_params + 2, 2), n_params >= 1");
            }
            if (model != "taylor" && model != "leaf") {
                throw std::invalid_argument(
                    "model must be \"taylor\" or \"leaf\"");
            }
            const auto lv = to_span<const double>(leaf);
            const search::RefoldTarget target{
                .leaf     = {lv.begin(), lv.end()},
                .n_params = static_cast<SizeType>(leaf.shape(0)) - 2,
                .model    = model == "leaf" ? search::RefoldModel::kLeaf
                                            : search::RefoldModel::kTaylor,
                .propagator_significance = propagator_significance};
            const search::RefoldOptions options{.nbins        = nbins,
                                                .nsub         = nsub,
                                                .df_steps     = df_steps,
                                                .df_span_t    = df_span_t,
                                                .df_step_bins = df_step_bins,
                                                .fd_steps     = fd_steps,
                                                .fd_span      = fd_span,
                                                .split        = split,
                                                .nthreads     = nthreads};
            const auto r = search::refold(to_span<const float>(ts_e),
                                          to_span<const float>(ts_v), tsamp,
                                          t_ref, target, options);
            py::dict out;
            out["nbins"]     = r.nbins;
            out["model"]     = r.model;
            out["freq"]      = r.freq;
            out["fdot"]      = r.fdot;
            out["snr_in"]    = r.snr_in;
            out["width_in"]  = r.width_in;
            out["snr"]       = r.snr;
            out["width"]     = r.width;
            out["start_bin"] = r.start_bin;
            out["df"]        = r.df;
            out["dfd"]       = r.dfd;
            out["split_snr"] = as_pyarray(std::vector<float>(r.split_snr));
            out["split_start_bin"] = r.split_start_bin;
            out["profile"]         = as_pyarray(std::vector<float>(r.profile));
            return out;
        },
        py::arg("ts_e"), py::arg("ts_v"), py::arg("tsamp"), py::arg("t_ref"),
        py::arg("leaf"), py::kw_only(), py::arg("model") = "taylor",
        py::arg("propagator_significance") = 2.0, py::arg("nbins") = 0,
        py::arg("nsub") = 128, py::arg("df_steps") = 0,
        py::arg("df_span_t") = 3.0, py::arg("df_step_bins") = 0.25,
        py::arg("fd_steps") = 0, py::arg("fd_span") = 0.0, py::arg("split") = 0,
        py::arg("nthreads") = 1,
        R"doc(Refold a series at a leaf and scan a local f (and fdot) grid.

leaf has shape (n_params + 2, 2) as a row of an EP result's param_sets, with
its parameters at t_ref; sample i is at time i * tsamp. model "taylor" folds
the Taylor polynomial of its derivatives, "leaf" the search's own model
(psr_utils.leaf_phase: a circular orbit where a five-parameter search would
use one). S/N is the best boxcar (widths 1..nbins/2) of sum(signal) /
sqrt(sum(variance)). A rejection test: noise also refolds near its search
S/N. Returns a dict (see loki::search::RefoldResult); f_best = freq + df.)doc");
}

} // namespace loki
