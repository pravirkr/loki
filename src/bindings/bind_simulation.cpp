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

#include "bindings/bind.hpp"
#include "loki_templates.hpp"
#include "pybind_utils.hpp"

namespace loki {
using algorithms::EPChunkConfig;
using algorithms::EPChunkStats;
using algorithms::EPRegionStats;
using algorithms::FFARegionStats;
using detection::MatchedFilter;
using pipelines::EPFreqSweep;
using pipelines::FFAFreqSweep;
using plans::FFAPlanBase;
using search::FFASearchConfig;
using search::PulsarSearchConfig;

namespace py = pybind11;

void bind_simulation(py::module_& m) {
    auto m_sim = m.def_submodule("simulation", "Pulse simulation");
    py::class_<simulation::DerivativeTerms>(m_sim, "DerivativeTerms")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::DerivativeTerms::shift)
        .def_readwrite("vel", &simulation::DerivativeTerms::vel)
        .def_readwrite("acc", &simulation::DerivativeTerms::acc)
        .def_readwrite("jerk", &simulation::DerivativeTerms::jerk)
        .def_readwrite("snap", &simulation::DerivativeTerms::snap);
    py::class_<simulation::DerivativeSeries>(m_sim, "DerivativeSeries")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::DerivativeSeries::shift)
        .def_readwrite("vel", &simulation::DerivativeSeries::vel)
        .def_readwrite("acc", &simulation::DerivativeSeries::acc)
        .def_readwrite("jerk", &simulation::DerivativeSeries::jerk)
        .def_readwrite("snap", &simulation::DerivativeSeries::snap)
        .def_readwrite("crackle", &simulation::DerivativeSeries::crackle);
    py::class_<simulation::GaugeDerivatives>(m_sim, "GaugeDerivatives")
        .def(py::init<>())
        .def_readwrite("freq", &simulation::GaugeDerivatives::freq)
        .def_readwrite("vel", &simulation::GaugeDerivatives::vel)
        .def_readwrite("acc", &simulation::GaugeDerivatives::acc)
        .def_readwrite("jerk", &simulation::GaugeDerivatives::jerk)
        .def_readwrite("snap", &simulation::GaugeDerivatives::snap)
        .def_readwrite("crackle", &simulation::GaugeDerivatives::crackle);
    py::class_<simulation::CircularOrbit>(m_sim, "CircularOrbit")
        .def(py::init<>())
        .def_readwrite("p_orb", &simulation::CircularOrbit::p_orb)
        .def_readwrite("psi", &simulation::CircularOrbit::psi)
        .def_readwrite("x_orb", &simulation::CircularOrbit::x_orb);
    py::class_<simulation::ModulatorParams>(m_sim, "ModulatorParams")
        .def(py::init<>())
        .def_readwrite("shift", &simulation::ModulatorParams::shift)
        .def_readwrite("vel", &simulation::ModulatorParams::vel)
        .def_readwrite("acc", &simulation::ModulatorParams::acc)
        .def_readwrite("jerk", &simulation::ModulatorParams::jerk)
        .def_readwrite("snap", &simulation::ModulatorParams::snap)
        .def_readwrite("coeffs", &simulation::ModulatorParams::coeffs)
        .def_readwrite("p_orb", &simulation::ModulatorParams::p_orb)
        .def_readwrite("psi", &simulation::ModulatorParams::psi)
        .def_readwrite("x_orb", &simulation::ModulatorParams::x_orb)
        .def_readwrite("m_c", &simulation::ModulatorParams::m_c)
        .def_readwrite("m_p", &simulation::ModulatorParams::m_p)
        .def_readwrite("sin_i", &simulation::ModulatorParams::sin_i)
        .def_readwrite("a", &simulation::ModulatorParams::a)
        .def_readwrite("t0", &simulation::ModulatorParams::t0);

    auto modulator_generate = [](const simulation::Modulator& modulator,
                                 const PyArrayT<double>& time, double t_ref) {
        const auto samples = to_span<const double>(time);
        std::vector<double> proper(samples.size());
        modulator.generate(samples, t_ref, proper);
        return as_pyarray(std::move(proper));
    };
    py::class_<simulation::Modulator, std::unique_ptr<simulation::Modulator>>(
        m_sim, "Modulator")
        .def("generate", modulator_generate, py::arg("time"),
             py::arg("t_ref") = 0.0);
    py::class_<simulation::DerivativeModulator, simulation::Modulator>(
        m_sim, "DerivativeModulator")
        .def(py::init<simulation::DerivativeTerms>(),
             py::arg("terms") = simulation::DerivativeTerms{})
        .def("to_circular", &simulation::DerivativeModulator::to_circular);
    py::class_<simulation::DerivativeSeriesModulator, simulation::Modulator>(
        m_sim, "DerivativeSeriesModulator")
        .def(py::init<std::vector<double>>(), py::arg("coeffs"))
        .def("to_circular",
             &simulation::DerivativeSeriesModulator::to_circular);
    py::class_<simulation::CircularModulator, simulation::Modulator>(
        m_sim, "CircularModulator")
        .def(py::init<double, double, std::optional<double>,
                      std::optional<double>, double, double>(),
             py::arg("p_orb"), py::arg("psi"), py::arg("x_orb") = py::none(),
             py::arg("m_c") = py::none(), py::arg("m_p") = 1.4,
             py::arg("sin_i") = 1.0)
        .def("to_derivatives", &simulation::CircularModulator::to_derivatives)
        .def("to_derivatives_gauge",
             &simulation::CircularModulator::to_derivatives_gauge,
             py::arg("f_ref"))
        .def("to_derivatives_series",
             &simulation::CircularModulator::to_derivatives_series,
             py::arg("n"))
        .def_property_readonly("p_orb", &simulation::CircularModulator::p_orb)
        .def_property_readonly("psi", &simulation::CircularModulator::psi)
        .def_property_readonly("x_orb", &simulation::CircularModulator::x_orb);
    py::class_<simulation::CircularT0Modulator, simulation::Modulator>(
        m_sim, "CircularT0Modulator")
        .def(py::init<double, double, double>(), py::arg("a"), py::arg("p_orb"),
             py::arg("t0"));
    m_sim.def("make_modulator", &simulation::make_modulator, py::arg("type"),
              py::arg("params") = simulation::ModulatorParams{});
    py::class_<simulation::PulseSignalConfig>(m_sim, "PulseSignalConfig")
        .def(py::init<double, double, SizeType, double, double, std::string,
                      simulation::ModulatorParams, std::optional<double>,
                      std::optional<std::uint64_t>>(),
             py::arg("period"), py::arg("dt"),
             py::arg("nsamps") = (SizeType{1} << 21U), py::arg("snr") = 100.0,
             py::arg("ducy") = 0.1, py::arg("mod_type") = "derivative",
             py::arg("mod")      = simulation::ModulatorParams{},
             py::arg("mod_tref") = py::none(), py::arg("seed") = py::none())
        .def("generate", &simulation::PulseSignalConfig::generate,
             py::arg("shape") = "gaussian", py::arg("phi0") = 0.5,
             py::arg("max_iter") = 5, py::arg("tol") = 1e-2)
        .def("generate_noise", &simulation::PulseSignalConfig::generate_noise)
        .def_property_readonly("period", &simulation::PulseSignalConfig::period)
        .def_property_readonly("dt", &simulation::PulseSignalConfig::dt)
        .def_property_readonly("nsamps", &simulation::PulseSignalConfig::nsamps)
        .def_property_readonly("snr", &simulation::PulseSignalConfig::snr)
        .def_property_readonly("ducy", &simulation::PulseSignalConfig::ducy)
        .def_property_readonly("tobs", &simulation::PulseSignalConfig::tobs)
        .def_property_readonly("freq", &simulation::PulseSignalConfig::freq);
}

} // namespace loki
