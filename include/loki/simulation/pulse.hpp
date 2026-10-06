#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/simulation/modulate.hpp"

namespace loki::simulation {

/**
 * @brief Inject a periodic pulse into Gaussian noise at a target folded SNR.
 *
 * `generate` builds a wrapped pulse CDF, modulates the time axis, and scales
 * the template so the folded boxcar SNR matches `snr`. The returned
 * timeseries stores the sum in `ts_e` and a constant noise variance in `ts_v`.
 */
class PulseSignalConfig {
public:
    PulseSignalConfig(double period,
                      double dt,
                      SizeType nsamps                   = (SizeType{1} << 21U),
                      double snr                        = 100.0,
                      double ducy                       = 0.1,
                      const std::string& mod_type       = "derivative",
                      const ModulatorParams& mod        = {},
                      std::optional<double> mod_tref    = std::nullopt,
                      std::optional<std::uint64_t> seed = std::nullopt);

    ~PulseSignalConfig();
    PulseSignalConfig(const PulseSignalConfig& other);
    PulseSignalConfig& operator=(const PulseSignalConfig& other);
    PulseSignalConfig(PulseSignalConfig&&) noexcept;
    PulseSignalConfig& operator=(PulseSignalConfig&&) noexcept;

    /**
     * @brief Draw one noisy realisation.
     *
     * @param shape One of "boxcar", "gaussian", "von_mises". Width is the FWTM.
     * @param phi0 Pulse centre in phase, wrapped into [0, 1).
     * @param max_iter Damped SNR iterations.
     * @param tol Stop when the folded SNR is within this of the target.
     */
    [[nodiscard]] io::TimeSeries generate(std::string_view shape = "gaussian",
                                          double phi0            = 0.5,
                                          int max_iter           = 5,
                                          double tol             = 1e-2);

    /// Noise only, with the closed-form standard deviation used by pyloki.
    [[nodiscard]] io::TimeSeries generate_noise();

    [[nodiscard]] double period() const noexcept;
    [[nodiscard]] double dt() const noexcept;
    [[nodiscard]] SizeType nsamps() const noexcept;
    [[nodiscard]] double snr() const noexcept;
    [[nodiscard]] double ducy() const noexcept;
    [[nodiscard]] double tobs() const noexcept;
    [[nodiscard]] double freq() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::simulation
