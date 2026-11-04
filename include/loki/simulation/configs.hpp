#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "loki/common/types.hpp"
#include "loki/simulation/modulate.hpp"

namespace loki::simulation {

/**
 * @brief Plain struct representation of a simulation TOML configuration file
 * (`loki simulate -c`).
 *
 * The reader checks keys and types only: an unknown table or key, a value of
 * the wrong type, a negative count or a `nan`/`inf` is an error. Ranges
 * (`period > 0`, ...) are checked by the caller, after command-line overrides.
 */
struct SimulateTomlConfig {
    // [signal]
    double period{0.0};
    double dt{0.0};
    SizeType nsamps{SizeType{1} << 21U};
    double snr{100.0};
    double ducy{0.1};
    std::string shape{"gaussian"};
    double phi0{0.5};
    std::optional<std::uint64_t> seed;
    // [modulation]
    std::string mod_type{"derivative"};
    ModulatorParams mod;
    std::optional<double> mod_tref;
    // [output]
    std::filesystem::path output{"simulated_pulse.tim"};
    std::string format{"tim"};

    /// @brief Load from a TOML file.
    static SimulateTomlConfig load(const std::filesystem::path& path);

    /// @brief Parse from a TOML string. Missing keys keep their defaults.
    static SimulateTomlConfig from_string(std::string_view toml_content);

    /// @brief The configuration as TOML text: from_string() of it gives
    /// back this config. Unset optional settings are left out.
    [[nodiscard]] std::string to_toml_string() const;

    /// @brief Check the value ranges, after command-line overrides. Each
    /// error names the config key and its flag, e.g. "signal.ducy (--ducy)".
    /// @throws std::invalid_argument listing every value out of range.
    void validate() const;

    /// @brief Write a documented default TOML configuration to disk.
    static void write_default(const std::filesystem::path& path);

    /// @brief Get the documented default TOML configuration text.
    static std::string_view default_toml_string();
};

} // namespace loki::simulation
