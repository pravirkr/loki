#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loki/loki.hpp"

#include "bindings/bind.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using search::FFASearchConfig;
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_configs(py::module_& m) {
    auto m_configs = m.def_submodule("configs", "Configs submodule");
    PYBIND11_NUMPY_DTYPE(ParamLimit, min, max);

    py::class_<FFASearchConfig>(m_configs, "FFASearchConfig")
        .def(py::init([](SizeType nsamps, double tsamp, SizeType nbins,
                         double eta, const PyArrayT<double>& param_limits,
                         double ducy_max, double wtsp, bool use_fourier,
                         int nthreads, double max_process_memory_gb,
                         double octave_scale, SizeType nbins_max,
                         SizeType nbins_min_lossy_bf,
                         std::optional<SizeType> bseg_brute,
                         std::optional<SizeType> bseg_ffa, double snr_min,
                         SizeType max_passing_candidates,
                         bool use_boxcar_kadane) {
                 if (param_limits.ndim() != 2 || param_limits.shape(1) != 2) {
                     throw std::invalid_argument(
                         "param_limits must be a 2D NumPy array with shape "
                         "(n_params, 2)");
                 }

                 const auto n_params =
                     static_cast<SizeType>(param_limits.shape(0));

                 std::vector<ParamLimit> limits(n_params);
                 for (SizeType i = 0; i < n_params; ++i) {
                     limits[i] = {
                         .min = *param_limits.data(i, 0),
                         .max = *param_limits.data(i, 1),
                     };
                 }
                 return std::make_unique<FFASearchConfig>(
                     nsamps, tsamp, nbins, eta, limits, ducy_max, wtsp,
                     use_fourier, nthreads, max_process_memory_gb, octave_scale,
                     nbins_max, nbins_min_lossy_bf, bseg_brute, bseg_ffa,
                     snr_min, max_passing_candidates, use_boxcar_kadane);
             }),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("eta"), py::arg("param_limits"), py::arg("ducy_max") = 0.2,
             py::arg("wtsp") = 1.5, py::arg("use_fourier") = true,
             py::arg("nthreads") = 1, py::arg("max_process_memory_gb") = 8.0,
             py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024,
             py::arg("nbins_min_lossy_bf") = 64,
             py::arg("bseg_brute")         = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("snr_min") = 6.0,
             py::arg("max_passing_candidates") = 1U << 22U, // 4M
             py::arg("use_boxcar_kadane")      = false)
        .def_property_readonly("nsamps", &FFASearchConfig::get_nsamps)
        .def_property_readonly("tsamp", &FFASearchConfig::get_tsamp)
        .def_property_readonly("tobs", &FFASearchConfig::get_tobs)
        .def_property_readonly("nbins", &FFASearchConfig::get_nbins)
        .def_property_readonly("nbins_f", &FFASearchConfig::get_nbins_f)
        .def_property_readonly("eta", &FFASearchConfig::get_eta)
        .def_property_readonly("param_limits",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_param_limits());
                               })
        .def_property_readonly(
            "snr_min", &FFASearchConfig::get_snr_min,
            "Minimum candidate SNR for final FFA/EP HDF5 (default 6); "
            "EP output filtering is reserved for future use.")
        .def_property_readonly("ducy_max", &FFASearchConfig::get_ducy_max)
        .def_property_readonly("wtsp", &FFASearchConfig::get_wtsp)
        .def_property_readonly("bseg_brute", &FFASearchConfig::get_bseg_brute)
        .def_property_readonly("bseg_ffa", &FFASearchConfig::get_bseg_ffa)
        .def_property_readonly("use_fourier", &FFASearchConfig::get_use_fourier)
        .def_property_readonly("use_conservative_tile",
                               &FFASearchConfig::get_use_conservative_tile)
        .def_property_readonly("nthreads", &FFASearchConfig::get_nthreads)
        .def_property_readonly("tseg_brute", &FFASearchConfig::get_tseg_brute)
        .def_property_readonly("tseg_ffa", &FFASearchConfig::get_tseg_ffa)
        .def_property_readonly("niters_ffa", &FFASearchConfig::get_niters_ffa)
        .def_property_readonly("nparams", &FFASearchConfig::get_nparams)
        .def_property_readonly("param_names", &FFASearchConfig::get_param_names)
        .def_property_readonly("f_min", &FFASearchConfig::get_f_min)
        .def_property_readonly("f_max", &FFASearchConfig::get_f_max)
        .def_property_readonly("score_widths",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_scoring_widths());
                               })
        .def_property_readonly("n_scoring_widths",
                               &FFASearchConfig::get_n_scoring_widths)
        .def_property_readonly("boxcar_kadane_biases",
                               [](const FFASearchConfig& self) {
                                   return as_pyarray_ref(
                                       self.get_boxcar_kadane_biases());
                               })
        .def_property_readonly("n_boxcar_kadane_biases",
                               &FFASearchConfig::get_n_boxcar_kadane_biases)
        .def("dparams_f", &FFASearchConfig::get_dparams_f, py::arg("tseg_cur"))
        .def("dparams", &FFASearchConfig::get_dparams, py::arg("tseg_cur"))
        .def("dparams_actual", &FFASearchConfig::get_dparams_actual,
             py::arg("tseg_cur"));

    py::class_<PulsarSearchConfig, FFASearchConfig>(m_configs,
                                                    "PulsarSearchConfig")
        .def(py::init(
                 [](SizeType nsamps, double tsamp, SizeType nbins, double eta,
                    const PyArrayT<double>& param_limits, double ducy_max,
                    double wtsp, bool use_fourier, int nthreads,
                    double max_process_memory_gb, double octave_scale,
                    SizeType nbins_max, SizeType nbins_min_lossy_bf,
                    std::optional<SizeType> bseg_brute,
                    std::optional<SizeType> bseg_ffa, double snr_min,
                    SizeType max_passing_candidates, SizeType prune_poly_order,
                    double p_orb_min, double m_c_max, double m_p_min,
                    double propagator_significance,
                    double validation_significance, bool use_conservative_tile,
                    bool use_boxcar_kadane, std::optional<double> target_snr) {
                     if (param_limits.ndim() != 2 ||
                         param_limits.shape(1) != 2) {
                         throw std::invalid_argument(
                             "param_limits must be a 2D NumPy array with shape "
                             "(n_params, 2)");
                     }

                     const auto n_params =
                         static_cast<SizeType>(param_limits.shape(0));

                     std::vector<ParamLimit> limits(n_params);
                     for (SizeType i = 0; i < n_params; ++i) {
                         limits[i] = {
                             .min = *param_limits.data(i, 0),
                             .max = *param_limits.data(i, 1),
                         };
                     }
                     return std::make_unique<PulsarSearchConfig>(
                         nsamps, tsamp, nbins, eta, limits, ducy_max, wtsp,
                         use_fourier, nthreads, max_process_memory_gb,
                         octave_scale, nbins_max, nbins_min_lossy_bf,
                         bseg_brute, bseg_ffa, snr_min, max_passing_candidates,
                         prune_poly_order, p_orb_min, m_c_max, m_p_min,
                         propagator_significance, validation_significance,
                         use_conservative_tile, use_boxcar_kadane, target_snr);
                 }),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("nbins"),
             py::arg("eta"), py::arg("param_limits"), py::arg("ducy_max") = 0.2,
             py::arg("wtsp") = 1.5, py::arg("use_fourier") = true,
             py::arg("nthreads") = 1, py::arg("max_process_memory_gb") = 8.0,
             py::arg("octave_scale") = 2.0, py::arg("nbins_max") = 1024,
             py::arg("nbins_min_lossy_bf") = 64,
             py::arg("bseg_brute")         = std::nullopt,
             py::arg("bseg_ffa") = std::nullopt, py::arg("snr_min") = 6.0,
             py::arg("max_passing_candidates") = 1U << 22U, // 4M
             py::arg("prune_poly_order") = 3, py::arg("p_orb_min") = 1e-5,
             py::arg("m_c_max") = 10.0, py::arg("m_p_min") = 1.4,
             py::arg("propagator_significance") = 2.0,
             py::arg("validation_significance") = 5.0,
             py::arg("use_conservative_tile")   = false,
             py::arg("use_boxcar_kadane")       = false, py::kw_only(),
             py::arg("target_snr")              = std::nullopt)
        .def_property_readonly(
            "target_snr", &PulsarSearchConfig::get_target_snr,
            "EP entropic information limit; None selects the parameter-count "
            "default: frequency/accel 8.5, jerk 9.2, snap 9.8, circular 10.")
        .def_property_readonly("prune_poly_order",
                               &PulsarSearchConfig::get_prune_poly_order)
        .def_property_readonly("p_orb_min", &PulsarSearchConfig::get_p_orb_min)
        .def_property_readonly("m_c_max", &PulsarSearchConfig::get_m_c_max)
        .def_property_readonly("m_p_min", &PulsarSearchConfig::get_m_p_min)
        .def_property_readonly("propagator_significance",
                               &PulsarSearchConfig::get_propagator_significance)
        .def_property_readonly("validation_significance",
                               &PulsarSearchConfig::get_validation_significance)
        .def_property_readonly("x_mass_const",
                               &PulsarSearchConfig::get_x_mass_const);

    m_configs.attr("EPSearchConfig") = m_configs.attr("PulsarSearchConfig");

    m_configs.def(
        "circular_param_limits",
        [](double f_min, double f_max, SizeType poly_order, double p_orb_min,
           double m_c_max, double m_p_min) {
            const auto limits =
                search::circular_param_limits(f_min, f_max, poly_order,
                                              {.p_orb_min = p_orb_min,
                                               .m_c_max   = m_c_max,
                                               .m_p_min   = m_p_min});
            py::array_t<double> out(
                {static_cast<py::ssize_t>(limits.size()), py::ssize_t{2}});
            auto view = out.mutable_unchecked<2>();
            for (SizeType i = 0; i < limits.size(); ++i) {
                view(static_cast<py::ssize_t>(i), 0) = limits[i].min;
                view(static_cast<py::ssize_t>(i), 1) = limits[i].max;
            }
            return out;
        },
        py::arg("f_min"), py::arg("f_max"), py::kw_only(),
        py::arg("poly_order"), py::arg("p_orb_min"), py::arg("m_c_max") = 10.0,
        py::arg("m_p_min") = 1.2,
        "param_limits (rows [snap], [jerk], accel, freq) covering circular "
        "orbits with P_orb >= p_orb_min s, m_c <= m_c_max and m_p >= m_p_min; "
        "the frequency range is widened by v/c (pyloki's from_circular).");
    m_configs.def(
        "implied_circular_orbit",
        [](const PyArrayT<double>& param_limits, double m_p) -> py::object {
            if (param_limits.ndim() != 2 || param_limits.shape(1) != 2) {
                throw std::invalid_argument(
                    "param_limits must be a 2D array with shape (nparams, 2)");
            }
            std::vector<ParamLimit> limits;
            for (py::ssize_t i = 0; i < param_limits.shape(0); ++i) {
                limits.push_back({.min = *param_limits.data(i, 0),
                                  .max = *param_limits.data(i, 1)});
            }
            const auto implied = search::implied_circular_orbit(limits, m_p);
            if (!implied) {
                return py::none();
            }
            return py::make_tuple(implied->p_orb, implied->m_c_min);
        },
        py::arg("param_limits"), py::arg("m_p") = 1.4,
        "(p_orb, m_c_min) of the fastest circular orbit an accel/jerk box "
        "reaches, or None without both limits.");

    // Plans submodule
}

} // namespace loki
