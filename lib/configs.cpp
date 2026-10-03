#include "loki/search/configs.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>
#include <toml++/toml.hpp>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/utils.hpp"

namespace loki::search {

namespace {

[[nodiscard]] bool is_allowed_key(std::string_view key,
                                  std::span<const std::string_view> allowed) {
    return std::ranges::any_of(
        allowed, [&](std::string_view candidate) { return key == candidate; });
}

void collect_unknown_keys(const toml::table& table,
                          std::span<const std::string_view> allowed,
                          std::string_view table_path,
                          std::vector<std::string>& errors) {
    for (const auto& [key, _] : table) {
        if (!is_allowed_key(key.str(), allowed)) {
            errors.push_back(std::format("{}.{}: unknown key", table_path,
                                         key.str()));
        }
    }
}

void validate_ffa_toml_document(const toml::table& root) {
    static constexpr std::array kTopLevel{
        std::string_view{"input"}, std::string_view{"search"},
        std::string_view{"performance"}, std::string_view{"output"},
        std::string_view{"cuda"}};
    static constexpr std::array kInputKeys{
        std::string_view{"timeseries"}, std::string_view{"preprocess"},
        std::string_view{"filter_window"}, std::string_view{"nsamps"},
        std::string_view{"tsamp"}, std::string_view{"dt"}};
    static constexpr std::array kSearchKeys{
        std::string_view{"f_min"}, std::string_view{"f_max"},
        std::string_view{"acc_min"}, std::string_view{"acc_max"},
        std::string_view{"jerk_min"}, std::string_view{"jerk_max"},
        std::string_view{"nbins"}, std::string_view{"eta"},
        std::string_view{"ducy_max"}, std::string_view{"wtsp"},
        std::string_view{"snr_min"}, std::string_view{"use_fourier"},
        std::string_view{"use_boxcar_kadane"}};
    static constexpr std::array kPerformanceKeys{
        std::string_view{"nthreads"},
        std::string_view{"max_process_memory_gb"},
        std::string_view{"octave_scale"}, std::string_view{"nbins_max"},
        std::string_view{"nbins_min_lossy_bf"}, std::string_view{"bseg_brute"},
        std::string_view{"bseg_ffa"},
        std::string_view{"max_passing_candidates"},
        std::string_view{"use_cuda"}, std::string_view{"device_id"}};
    static constexpr std::array kOutputKeys{std::string_view{"outdir"},
                                              std::string_view{"prefix"}};
    static constexpr std::array kCudaKeys{std::string_view{"enable"},
                                          std::string_view{"device_id"}};

    std::vector<std::string> errors;
    for (const auto& [key, _] : root) {
        if (!is_allowed_key(key.str(), kTopLevel)) {
            errors.push_back(
                std::format("unknown top-level key '{}'", key.str()));
        }
    }
    if (const auto* input = root["input"].as_table()) {
        collect_unknown_keys(*input, kInputKeys, "input", errors);
    }
    if (const auto* search = root["search"].as_table()) {
        collect_unknown_keys(*search, kSearchKeys, "search", errors);
    }
    if (const auto* perf = root["performance"].as_table()) {
        collect_unknown_keys(*perf, kPerformanceKeys, "performance", errors);
    }
    if (const auto* output = root["output"].as_table()) {
        collect_unknown_keys(*output, kOutputKeys, "output", errors);
    }
    if (const auto* cuda = root["cuda"].as_table()) {
        collect_unknown_keys(*cuda, kCudaKeys, "cuda", errors);
    }
    if (!errors.empty()) {
        std::string message = errors.front();
        for (std::size_t i = 1; i < errors.size(); ++i) {
            message += "; ";
            message += errors[i];
        }
        throw std::invalid_argument(
            std::format("Invalid FFA TOML config: {}", message));
    }
}

[[nodiscard]] int64_t require_non_negative_int64(int64_t value,
                                                 std::string_view path) {
    if (value < 0) {
        throw std::invalid_argument(
            std::format("{} must be non-negative (got {})", path, value));
    }
    return value;
}

} // namespace

// ==============================================================================
// FFATomlConfig Implementation
// ==============================================================================

std::string FFATomlConfig::default_toml_string() {
    return R"(# ==============================================================================
# Loki FFA Pulsar Search Configuration
# ==============================================================================

[input]
# Path to input timeseries (.tim or .dat format)
timeseries = "input.tim"

# Preprocess timeseries: subtract running median baseline and z-score normalize
preprocess = true

# Running median filter window in seconds (used if preprocess = true)
filter_window = 1.0

[search]
# Search frequency range in Hz (Period P = 1 / f)
f_min = 0.5
f_max = 100.0

# Optional line-of-sight acceleration limits in m/s^2 (omit or set equal to 0 for freq-only)
acc_min = 0.0
acc_max = 0.0

# Optional line-of-sight jerk limits in m/s^3 (omit or set equal to 0 for no jerk search)
jerk_min = 0.0
jerk_max = 0.0

# Number of profile bins at shortest search period (must be >= 2 and power of 2)
nbins = 64

# Tolerance in bins across observation duration (typically 1.0)
eta = 1.0

# Maximum pulse duty cycle (FWTM in phase) for boxcar template width trials
ducy_max = 0.2

# Boxcar width trial geometric scale factor (typically 1.2 - 1.5)
wtsp = 1.5

# Minimum signal-to-noise ratio detection threshold for candidate extraction
snr_min = 5.0

# Folding domain: true for Fourier-domain FFA (recommended), false for Time-domain FFA
use_fourier = true

# Peak detection algorithm: false for standard multi-width boxcar, true for Kadane
use_boxcar_kadane = false

[performance]
# Number of CPU OpenMP threads (clamped to hardware concurrency)
nthreads = 4

# Maximum memory cap in GB used for chunk planning
max_process_memory_gb = 8.0

# Octave scaling factor between successive search bands (2.0 = octave doubling)
octave_scale = 2.0

# Maximum number of fold bins for long-period bands
nbins_max = 1024

# Minimum number of bins for lossy brute force initial stage
nbins_min_lossy_bf = 64

# Maximum candidate buffer capacity held in memory before streaming to disk
max_passing_candidates = 4194304

# Run FFA on NVIDIA GPU (requires CUDA-enabled Loki build)
use_cuda = false

# GPU device ID
device_id = 0

[output]
# Directory where candidate HDF5 file will be saved
outdir = "./"

# Output filename prefix (results saved as <outdir>/<prefix>_ffa_results.h5)
prefix = "loki"
)";
}

void FFATomlConfig::write_default(const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(
            std::format("Could not open file for writing default config: '{}'",
                        path.string()));
    }
    file << default_toml_string();
}

FFATomlConfig FFATomlConfig::from_string(std::string_view toml_content) {
    try {
        const toml::table tbl = toml::parse(toml_content);
        validate_ffa_toml_document(tbl);
        FFATomlConfig cfg;

        // [input] table
        if (const auto* input = tbl["input"].as_table()) {
            if (auto val = (*input)["timeseries"].value<std::string>()) {
                cfg.timeseries_path = *val;
            }
            if (auto val = (*input)["preprocess"].value<bool>()) {
                cfg.preprocess = *val;
            }
            if (auto val = (*input)["filter_window"].value<double>()) {
                cfg.filter_window = *val;
            }
            if (auto val = (*input)["nsamps"].value<int64_t>()) {
                cfg.nsamps = static_cast<SizeType>(
                    require_non_negative_int64(*val, "input.nsamps"));
            }
            if (auto val = (*input)["tsamp"].value<double>()) {
                cfg.tsamp = *val;
            } else if (auto val_dt = (*input)["dt"].value<double>()) {
                cfg.tsamp = *val_dt;
            }
        }

        // [search] table
        if (const auto* search = tbl["search"].as_table()) {
            if (auto val = (*search)["f_min"].value<double>()) {
                cfg.f_min = *val;
            }
            if (auto val = (*search)["f_max"].value<double>()) {
                cfg.f_max = *val;
            }
            if (auto val = (*search)["acc_min"].value<double>()) {
                cfg.acc_min = *val;
            }
            if (auto val = (*search)["acc_max"].value<double>()) {
                cfg.acc_max = *val;
            }
            if (auto val = (*search)["jerk_min"].value<double>()) {
                cfg.jerk_min = *val;
            }
            if (auto val = (*search)["jerk_max"].value<double>()) {
                cfg.jerk_max = *val;
            }
            if (auto val = (*search)["nbins"].value<int64_t>()) {
                cfg.nbins = static_cast<SizeType>(
                    require_non_negative_int64(*val, "search.nbins"));
            }
            if (auto val = (*search)["eta"].value<double>()) {
                cfg.eta = *val;
            }
            if (auto val = (*search)["ducy_max"].value<double>()) {
                cfg.ducy_max = *val;
            }
            if (auto val = (*search)["wtsp"].value<double>()) {
                cfg.wtsp = *val;
            }
            if (auto val = (*search)["snr_min"].value<double>()) {
                cfg.snr_min = *val;
            }
            if (auto val = (*search)["use_fourier"].value<bool>()) {
                cfg.use_fourier = *val;
            }
            if (auto val = (*search)["use_boxcar_kadane"].value<bool>()) {
                cfg.use_boxcar_kadane = *val;
            }
        }

        // [performance] table
        if (const auto* perf = tbl["performance"].as_table()) {
            if (auto val = (*perf)["nthreads"].value<int64_t>()) {
                cfg.nthreads = static_cast<int>(
                    require_non_negative_int64(*val, "performance.nthreads"));
            }
            if (auto val = (*perf)["max_process_memory_gb"].value<double>()) {
                cfg.max_process_memory_gb = *val;
            }
            if (auto val = (*perf)["octave_scale"].value<double>()) {
                cfg.octave_scale = *val;
            }
            if (auto val = (*perf)["nbins_max"].value<int64_t>()) {
                cfg.nbins_max = static_cast<SizeType>(
                    require_non_negative_int64(*val, "performance.nbins_max"));
            }
            if (auto val = (*perf)["nbins_min_lossy_bf"].value<int64_t>()) {
                cfg.nbins_min_lossy_bf = static_cast<SizeType>(
                    require_non_negative_int64(*val,
                                               "performance.nbins_min_lossy_bf"));
            }
            if (auto val = (*perf)["bseg_brute"].value<int64_t>()) {
                cfg.bseg_brute = static_cast<SizeType>(
                    require_non_negative_int64(*val, "performance.bseg_brute"));
            }
            if (auto val = (*perf)["bseg_ffa"].value<int64_t>()) {
                cfg.bseg_ffa = static_cast<SizeType>(
                    require_non_negative_int64(*val, "performance.bseg_ffa"));
            }
            if (auto val = (*perf)["max_passing_candidates"].value<int64_t>()) {
                cfg.max_passing_candidates = static_cast<SizeType>(
                    require_non_negative_int64(
                        *val, "performance.max_passing_candidates"));
            }
            if (auto val = (*perf)["use_cuda"].value<bool>()) {
                cfg.use_cuda = *val;
            }
            if (auto val = (*perf)["device_id"].value<int64_t>()) {
                cfg.device_id = static_cast<int>(
                    require_non_negative_int64(*val, "performance.device_id"));
            }
        }

        // [output] table
        if (const auto* output = tbl["output"].as_table()) {
            if (auto val = (*output)["outdir"].value<std::string>()) {
                cfg.outdir = *val;
            }
            if (auto val = (*output)["prefix"].value<std::string>()) {
                cfg.prefix = *val;
            }
        }

        // [cuda] table
        if (const auto* cuda = tbl["cuda"].as_table()) {
            if (auto val = (*cuda)["enable"].value<bool>()) {
                cfg.use_cuda = *val;
            }
            if (auto val = (*cuda)["device_id"].value<int64_t>()) {
                cfg.device_id = static_cast<int>(
                    require_non_negative_int64(*val, "cuda.device_id"));
            }
        }

        return cfg;
    } catch (const toml::parse_error& err) {
        throw std::runtime_error(std::format(
            "TOML parse error at line {}, col {}: {}", err.source().begin.line,
            err.source().begin.column, err.description()));
    }
}

FFATomlConfig FFATomlConfig::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            std::format("Configuration file not found: '{}'", path.string()));
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(std::format(
            "Could not open configuration file: '{}'", path.string()));
    }
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    return from_string(content);
}

FFASearchConfig
FFATomlConfig::to_search_config(std::optional<SizeType> override_nsamps,
                                std::optional<double> override_tsamp) const {
    const SizeType final_nsamps =
        override_nsamps.value_or(nsamps.value_or(1U << 21U));
    const double final_tsamp = override_tsamp.value_or(tsamp.value_or(6.4e-5));

    if (acc_min.has_value() != acc_max.has_value()) {
        throw std::invalid_argument(
            "Both acc_min and acc_max must be specified together.");
    }
    if (jerk_min.has_value() != jerk_max.has_value()) {
        throw std::invalid_argument(
            "Both jerk_min and jerk_max must be specified together.");
    }

    std::vector<ParamLimit> limits;
    const bool has_jerk = jerk_min.has_value() && jerk_max.has_value() &&
                          (*jerk_min != 0.0 || *jerk_max != 0.0);
    const bool has_acc  = acc_min.has_value() && acc_max.has_value() &&
                          (*acc_min != 0.0 || *acc_max != 0.0);

    if (has_jerk) {
        limits.push_back({.min = *jerk_min, .max = *jerk_max});
        limits.push_back(
            {.min = acc_min.value_or(0.0), .max = acc_max.value_or(0.0)});
        limits.push_back({.min = f_min, .max = f_max});
    } else if (has_acc) {
        limits.push_back({.min = *acc_min, .max = *acc_max});
        limits.push_back({.min = f_min, .max = f_max});
    } else {
        limits.push_back({.min = f_min, .max = f_max});
    }

    if (use_boxcar_kadane) {
        throw std::invalid_argument(
            "use_boxcar_kadane is not supported for FFA frequency sweep "
            "(multi-width boxcar decoding is required)");
    }

    const int effective_nthreads =
        nthreads <= 0 ? omp_get_max_threads() : nthreads;

    return {final_nsamps,
            final_tsamp,
            nbins,
            eta,
            limits,
            ducy_max,
            wtsp,
            use_fourier,
            effective_nthreads,
            max_process_memory_gb,
            octave_scale,
            nbins_max,
            nbins_min_lossy_bf,
            bseg_brute,
            bseg_ffa,
            snr_min,
            max_passing_candidates,
            use_boxcar_kadane};
}

// ==============================================================================
// FFASearchConfig::Impl Definition
// ==============================================================================

class FFASearchConfig::Impl {
public:
    Impl(SizeType nsamps,
         double tsamp,
         SizeType nbins,
         double eta,
         std::span<const ParamLimit> param_limits,
         double ducy_max,
         double wtsp,
         bool use_fourier,
         int nthreads,
         double max_process_memory_gb,
         double octave_scale,
         SizeType nbins_max,
         SizeType nbins_min_lossy_bf,
         std::optional<SizeType> bseg_brute,
         std::optional<SizeType> bseg_ffa,
         double snr_min,
         SizeType max_passing_candidates,
         bool use_boxcar_kadane)
        : m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_nbins(nbins),
          m_eta(eta),
          m_param_limits(param_limits.begin(), param_limits.end()),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_use_fourier(use_fourier),
          m_nthreads(nthreads),
          m_max_process_memory_gb(max_process_memory_gb),
          m_octave_scale(octave_scale),
          m_nbins_max(nbins_max),
          m_nbins_min_lossy_bf(nbins_min_lossy_bf),
          m_snr_min(snr_min),
          m_max_passing_candidates(max_passing_candidates),
          m_use_boxcar_kadane(use_boxcar_kadane),
          m_bseg_brute_explicit(bseg_brute),
          m_bseg_ffa_explicit(bseg_ffa) {
        if (m_param_limits.empty()) {
            throw std::runtime_error("coord_limits must be non-empty");
        }
        m_nbins_f = (m_nbins / 2) + 1;
        m_nparams = m_param_limits.size();
        m_param_names.assign(kParamNames.end() - m_nparams, kParamNames.end());
        m_f_min      = m_param_limits[m_nparams - 1].min;
        m_f_max      = m_param_limits[m_nparams - 1].max;
        m_bseg_brute = bseg_brute.value_or(get_bseg_brute_default());
        m_bseg_ffa   = bseg_ffa.value_or(get_bseg_ffa_default());

        m_nthreads = std::clamp(m_nthreads, 1, omp_get_max_threads());
        validate();
        m_tseg_brute = static_cast<double>(m_bseg_brute) * m_tsamp;
        m_tseg_ffa   = static_cast<double>(m_bseg_ffa) * m_tsamp;
        m_niters_ffa = static_cast<SizeType>(std::countr_zero(m_bseg_ffa) -
                                             std::countr_zero(m_bseg_brute));
        m_scoring_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        m_boxcar_kadane_biases = {1.42F, 0.76F, 0.41F};

        spdlog::debug(
            "FFASearchConfig: nsamps={}, tsamp={}, nbins={}, eta={}, "
            "ducy_max={}, wtsp={}, use_fourier={}, nthreads={}, snr_min={}, "
            "bseg_brute={}, bseg_ffa={}",
            m_nsamps, m_tsamp, m_nbins, m_eta, m_ducy_max, m_wtsp,
            m_use_fourier, m_nthreads, m_snr_min, m_bseg_brute, m_bseg_ffa);
    }

    // Getters
    SizeType get_nsamps() const { return m_nsamps; }
    double get_tsamp() const { return m_tsamp; }
    double get_tobs() const { return static_cast<double>(m_nsamps) * m_tsamp; }
    SizeType get_nbins() const { return m_nbins; }
    SizeType get_nbins_f() const { return m_nbins_f; }
    double get_eta() const { return m_eta; }
    std::span<const ParamLimit> get_param_limits() const {
        return m_param_limits;
    }
    double get_ducy_max() const { return m_ducy_max; }
    double get_wtsp() const { return m_wtsp; }
    bool get_use_fourier() const { return m_use_fourier; }
    int get_nthreads() const { return m_nthreads; }
    double get_max_process_memory_gb() const { return m_max_process_memory_gb; }
    double get_octave_scale() const { return m_octave_scale; }
    SizeType get_nbins_max() const { return m_nbins_max; }
    SizeType get_nbins_min_lossy_bf() const { return m_nbins_min_lossy_bf; }
    SizeType get_bseg_brute() const { return m_bseg_brute; }
    SizeType get_bseg_ffa() const { return m_bseg_ffa; }
    double get_snr_min() const { return m_snr_min; }
    SizeType get_max_passing_candidates() const {
        return m_max_passing_candidates;
    }
    bool get_use_boxcar_kadane() const { return m_use_boxcar_kadane; }
    double get_tseg_brute() const { return m_tseg_brute; }
    double get_tseg_ffa() const { return m_tseg_ffa; }
    SizeType get_niters_ffa() const { return m_niters_ffa; }
    SizeType get_nparams() const { return m_nparams; }
    std::vector<std::string> get_param_names() const { return m_param_names; }
    double get_f_min() const { return m_f_min; }
    double get_f_max() const { return m_f_max; }
    std::vector<SizeType> get_scoring_widths() const {
        return m_scoring_widths;
    }
    SizeType get_n_scoring_widths() const { return m_scoring_widths.size(); }
    std::vector<float> get_boxcar_kadane_biases() const {
        return m_boxcar_kadane_biases;
    }
    SizeType get_n_boxcar_kadane_biases() const {
        return m_boxcar_kadane_biases.size();
    }

    void set_max_process_memory_gb(double max_process_memory_gb) {
        error_check::check_greater(max_process_memory_gb, 0,
                                   "max_process_memory_gb must be positive");
        m_max_process_memory_gb = max_process_memory_gb;
    }

    std::vector<double> get_dparams_f(double tseg_cur) const {
        const double t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
        return psr_utils::poly_taylor_step_f(m_nparams, tseg_cur, m_nbins,
                                             m_eta, t_ref);
    }

    std::vector<double> get_dparams(double tseg_cur) const {
        const double t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
        return psr_utils::poly_taylor_step_d_f(m_nparams, tseg_cur, m_nbins,
                                               m_eta, m_f_max, t_ref);
    }

    std::vector<double> get_dparams_actual(double tseg_cur) const {
        const std::vector<SizeType> param_grid_count =
            get_param_grid_count(tseg_cur);
        std::vector<double> dparams_act(m_nparams);
        for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
            dparams_act[iparam] =
                (m_param_limits[iparam].max - m_param_limits[iparam].min) /
                static_cast<double>(param_grid_count[iparam]);
        }
        return dparams_act;
    }

    std::vector<SizeType> get_param_grid_count(double tseg_cur) const {
        const std::vector<double> dparams = get_dparams(tseg_cur);
        std::vector<SizeType> count(m_nparams);
        for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
            count[iparam] = psr_utils::range_param_count(
                m_param_limits[iparam].min, m_param_limits[iparam].max,
                dparams[iparam]);
        }
        return count;
    }

    FFASearchConfig
    get_updated_config(SizeType nbins,
                       double eta,
                       std::span<const ParamLimit> param_limits) const {
        return {m_nsamps,
                m_tsamp,
                nbins,
                eta,
                param_limits,
                m_ducy_max,
                m_wtsp,
                m_use_fourier,
                m_nthreads,
                m_max_process_memory_gb,
                m_octave_scale,
                m_nbins_max,
                m_nbins_min_lossy_bf,
                m_bseg_brute_explicit,
                m_bseg_ffa_explicit,
                m_snr_min,
                m_max_passing_candidates,
                m_use_boxcar_kadane};
    }

    FFASearchConfig get_updated_config(SizeType nbins,
                                       double eta,
                                       double f_min,
                                       double f_max) const {
        std::vector<ParamLimit> param_limits(m_param_limits.begin(),
                                             m_param_limits.end());
        param_limits.back().min = f_min;
        param_limits.back().max = f_max;
        return get_updated_config(nbins, eta, param_limits);
    }

private:
    SizeType m_nsamps;
    double m_tsamp;
    SizeType m_nbins;
    SizeType m_nbins_f;
    double m_eta;
    std::vector<ParamLimit> m_param_limits;
    double m_ducy_max;
    double m_wtsp;
    bool m_use_fourier;
    int m_nthreads;
    double m_max_process_memory_gb;
    double m_octave_scale;
    SizeType m_nbins_max;
    SizeType m_nbins_min_lossy_bf;
    double m_snr_min;
    SizeType m_max_passing_candidates;
    bool m_use_boxcar_kadane;
    std::optional<SizeType> m_bseg_brute_explicit;
    std::optional<SizeType> m_bseg_ffa_explicit;

    SizeType m_bseg_brute{};
    SizeType m_bseg_ffa{};
    double m_tseg_brute{};
    double m_tseg_ffa{};
    SizeType m_niters_ffa{};
    SizeType m_nparams{};
    std::vector<std::string> m_param_names;
    double m_f_min{};
    double m_f_max{};
    std::vector<SizeType> m_scoring_widths;
    std::vector<float> m_boxcar_kadane_biases;

    void validate() const {
        error_check::check_greater(m_nsamps, 0, "nsamps must be positive");
        error_check::check_power_of_2(m_nsamps, "nsamps");
        error_check::check_greater(m_tsamp, 0, "tsamp must be positive");
        error_check::check_greater(m_eta, 0,
                                   "eta (tolerance bins) must be positive");
        error_check::check_greater(m_ducy_max, 0.0, "ducy_max must be positive");
        error_check::check_less_equal(m_ducy_max, 1.0,
                                      "ducy_max must be <= 1.0");
        error_check::check_greater(m_wtsp, 1.0, "wtsp must be > 1.0");
        error_check::check_greater(m_max_process_memory_gb, 0,
                                   "max_process_memory_gb must be positive");
        error_check::check_greater_equal(
            m_nbins_max, m_nbins,
            "nbins_max must be greater than or equal to nbins");
        error_check::check_power_of_2(m_bseg_brute, "bseg_brute");
        error_check::check_power_of_2(m_bseg_ffa, "bseg_ffa");
        error_check::check_less(m_bseg_brute, m_nsamps,
                                "bseg_brute must be less than nsamps");
        error_check::check_less_equal(
            m_bseg_ffa, m_nsamps,
            "bseg_ffa must be less than or equal to nsamps");
        error_check::check_greater_equal(
            m_bseg_ffa, m_bseg_brute,
            "bseg_ffa must be greater than or equal to bseg_brute");
        error_check::check_greater_equal(m_nparams, 1,
                                         "nparams must be at least 1");
        error_check::check_greater(m_f_min, 0.0,
                                   "Frequency f_min must be positive");
        error_check::check_greater(
            m_f_max, m_f_min, "Frequency f_max must be greater than f_min");
        for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
            const auto& param_limit = m_param_limits[iparam];
            error_check::check_greater_equal(
                param_limit.max, param_limit.min,
                std::format(
                    "param_limits[{}] must be increasing (got [{}, {}])",
                    iparam, param_limit.min, param_limit.max));
        }
    }

    SizeType get_bseg_brute_default() const {
        const auto tobs   = static_cast<double>(m_nsamps) * m_tsamp;
        const auto cycles = tobs * m_f_min;
        if (cycles <= 1.0) {
            return std::min(m_nsamps / 2, SizeType{64});
        }
        const auto levels     = static_cast<int>(std::floor(std::log2(cycles)));
        const int init_levels = (m_nparams == 1) ? 1 : 2;
        const int shift       = std::max(1, levels - init_levels);
        const SizeType bseg   = static_cast<SizeType>(m_nsamps >> shift);
        return std::clamp(bseg, SizeType{2}, m_nsamps / 2);
    }

    SizeType get_bseg_ffa_default() const { return m_nsamps; }
};

// ==============================================================================
// FFASearchConfig Implementation
// ==============================================================================

FFASearchConfig::FFASearchConfig(SizeType nsamps,
                                 double tsamp,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 double ducy_max,
                                 double wtsp,
                                 bool use_fourier,
                                 int nthreads,
                                 double max_process_memory_gb,
                                 double octave_scale,
                                 SizeType nbins_max,
                                 SizeType nbins_min_lossy_bf,
                                 std::optional<SizeType> bseg_brute,
                                 std::optional<SizeType> bseg_ffa,
                                 double snr_min,
                                 SizeType max_passing_candidates,
                                 bool use_boxcar_kadane)
    : m_impl(std::make_unique<Impl>(nsamps,
                                    tsamp,
                                    nbins,
                                    eta,
                                    param_limits,
                                    ducy_max,
                                    wtsp,
                                    use_fourier,
                                    nthreads,
                                    max_process_memory_gb,
                                    octave_scale,
                                    nbins_max,
                                    nbins_min_lossy_bf,
                                    bseg_brute,
                                    bseg_ffa,
                                    snr_min,
                                    max_passing_candidates,
                                    use_boxcar_kadane)) {}

FFASearchConfig::~FFASearchConfig()                          = default;
FFASearchConfig::FFASearchConfig(FFASearchConfig&&) noexcept = default;
FFASearchConfig&
FFASearchConfig::operator=(FFASearchConfig&&) noexcept = default;
FFASearchConfig::FFASearchConfig(const FFASearchConfig& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

FFASearchConfig& FFASearchConfig::operator=(const FFASearchConfig& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}

SizeType FFASearchConfig::get_nsamps() const noexcept {
    return m_impl->get_nsamps();
}
double FFASearchConfig::get_tsamp() const noexcept {
    return m_impl->get_tsamp();
}
double FFASearchConfig::get_tobs() const noexcept { return m_impl->get_tobs(); }
SizeType FFASearchConfig::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
SizeType FFASearchConfig::get_nbins_f() const noexcept {
    return m_impl->get_nbins_f();
}
double FFASearchConfig::get_eta() const noexcept { return m_impl->get_eta(); }
std::span<const ParamLimit> FFASearchConfig::get_param_limits() const noexcept {
    return m_impl->get_param_limits();
}
double FFASearchConfig::get_ducy_max() const noexcept {
    return m_impl->get_ducy_max();
}
double FFASearchConfig::get_wtsp() const noexcept { return m_impl->get_wtsp(); }
bool FFASearchConfig::get_use_fourier() const noexcept {
    return m_impl->get_use_fourier();
}
int FFASearchConfig::get_nthreads() const noexcept {
    return m_impl->get_nthreads();
}
double FFASearchConfig::get_max_process_memory_gb() const noexcept {
    return m_impl->get_max_process_memory_gb();
}
double FFASearchConfig::get_octave_scale() const noexcept {
    return m_impl->get_octave_scale();
}
SizeType FFASearchConfig::get_nbins_max() const noexcept {
    return m_impl->get_nbins_max();
}
SizeType FFASearchConfig::get_nbins_min_lossy_bf() const noexcept {
    return m_impl->get_nbins_min_lossy_bf();
}
SizeType FFASearchConfig::get_bseg_brute() const noexcept {
    return m_impl->get_bseg_brute();
}
SizeType FFASearchConfig::get_bseg_ffa() const noexcept {
    return m_impl->get_bseg_ffa();
}
double FFASearchConfig::get_snr_min() const noexcept {
    return m_impl->get_snr_min();
}
SizeType FFASearchConfig::get_max_passing_candidates() const noexcept {
    return m_impl->get_max_passing_candidates();
}
bool FFASearchConfig::get_use_boxcar_kadane() const noexcept {
    return m_impl->get_use_boxcar_kadane();
}
double FFASearchConfig::get_tseg_brute() const noexcept {
    return m_impl->get_tseg_brute();
}
double FFASearchConfig::get_tseg_ffa() const noexcept {
    return m_impl->get_tseg_ffa();
}
SizeType FFASearchConfig::get_niters_ffa() const noexcept {
    return m_impl->get_niters_ffa();
}
SizeType FFASearchConfig::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
std::vector<std::string> FFASearchConfig::get_param_names() const noexcept {
    return m_impl->get_param_names();
}
double FFASearchConfig::get_f_min() const noexcept {
    return m_impl->get_f_min();
}
double FFASearchConfig::get_f_max() const noexcept {
    return m_impl->get_f_max();
}
std::vector<SizeType> FFASearchConfig::get_scoring_widths() const noexcept {
    return m_impl->get_scoring_widths();
}
SizeType FFASearchConfig::get_n_scoring_widths() const noexcept {
    return m_impl->get_n_scoring_widths();
}
std::vector<float> FFASearchConfig::get_boxcar_kadane_biases() const noexcept {
    return m_impl->get_boxcar_kadane_biases();
}
SizeType FFASearchConfig::get_n_boxcar_kadane_biases() const noexcept {
    return m_impl->get_n_boxcar_kadane_biases();
}
void FFASearchConfig::set_max_process_memory_gb(
    double max_process_memory_gb) {
    m_impl->set_max_process_memory_gb(max_process_memory_gb);
}
std::vector<double>
FFASearchConfig::get_dparams_f(double tseg_cur) const noexcept {
    return m_impl->get_dparams_f(tseg_cur);
}
std::vector<double>
FFASearchConfig::get_dparams(double tseg_cur) const noexcept {
    return m_impl->get_dparams(tseg_cur);
}
std::vector<double>
FFASearchConfig::get_dparams_actual(double tseg_cur) const noexcept {
    return m_impl->get_dparams_actual(tseg_cur);
}
std::vector<SizeType>
FFASearchConfig::get_param_grid_count(double tseg_cur) const noexcept {
    return m_impl->get_param_grid_count(tseg_cur);
}
FFASearchConfig FFASearchConfig::get_updated_config(
    SizeType nbins,
    double eta,
    std::span<const ParamLimit> param_limits) const {
    return m_impl->get_updated_config(nbins, eta, param_limits);
}
FFASearchConfig FFASearchConfig::get_updated_config(SizeType nbins,
                                                    double eta,
                                                    double f_min,
                                                    double f_max) const {
    return m_impl->get_updated_config(nbins, eta, f_min, f_max);
}

FFASearchConfig FFASearchConfig::from_toml(const std::filesystem::path& path,
                                           std::optional<SizeType> nsamps,
                                           std::optional<double> tsamp) {
    const auto toml_cfg = FFATomlConfig::load(path);
    return toml_cfg.to_search_config(nsamps, tsamp);
}

FFASearchConfig
FFASearchConfig::from_toml_string(std::string_view toml_content,
                                  std::optional<SizeType> nsamps,
                                  std::optional<double> tsamp) {
    const auto toml_cfg = FFATomlConfig::from_string(toml_content);
    return toml_cfg.to_search_config(nsamps, tsamp);
}

void FFASearchConfig::write_default_toml(const std::filesystem::path& path) {
    FFATomlConfig::write_default(path);
}

std::string FFASearchConfig::default_toml_string() {
    return FFATomlConfig::default_toml_string();
}

// ==============================================================================
// EPSearchConfig::EPImpl Definition
// ==============================================================================

class EPSearchConfig::EPImpl {
public:
    EPImpl(SizeType prune_poly_order,
           double p_orb_min,
           double m_c_max,
           double m_p_min,
           double propagator_significance,
           double validation_significance,
           bool use_conservative_tile)
        : m_prune_poly_order(prune_poly_order),
          m_p_orb_min(p_orb_min),
          m_m_c_max(m_c_max),
          m_m_p_min(m_p_min),
          m_propagator_significance(propagator_significance),
          m_validation_significance(validation_significance),
          m_use_conservative_tile(use_conservative_tile) {}

    SizeType m_prune_poly_order;
    double m_p_orb_min;
    double m_m_c_max;
    double m_m_p_min;
    double m_propagator_significance;
    double m_validation_significance;
    bool m_use_conservative_tile;

    double get_x_mass_const() const noexcept {
        constexpr double safety = 1.1;
        return utils::kGMsunOneThird * safety * m_m_c_max /
               std::pow(m_m_p_min + m_m_c_max, 2.0 / 3.0);
    }
};

// ==============================================================================
// EPSearchConfig Implementation
// ==============================================================================

EPSearchConfig::EPSearchConfig(SizeType nsamps,
                               double tsamp,
                               SizeType nbins,
                               double eta,
                               std::span<const ParamLimit> param_limits,
                               double ducy_max,
                               double wtsp,
                               bool use_fourier,
                               int nthreads,
                               double max_process_memory_gb,
                               double octave_scale,
                               SizeType nbins_max,
                               SizeType nbins_min_lossy_bf,
                               std::optional<SizeType> bseg_brute,
                               std::optional<SizeType> bseg_ffa,
                               double snr_min,
                               SizeType max_passing_candidates,
                               SizeType prune_poly_order,
                               double p_orb_min,
                               double m_c_max,
                               double m_p_min,
                               double propagator_significance,
                               double validation_significance,
                               bool use_conservative_tile,
                               bool use_boxcar_kadane)
    : FFASearchConfig(nsamps,
                      tsamp,
                      nbins,
                      eta,
                      param_limits,
                      ducy_max,
                      wtsp,
                      use_fourier,
                      nthreads,
                      max_process_memory_gb,
                      octave_scale,
                      nbins_max,
                      nbins_min_lossy_bf,
                      bseg_brute,
                      bseg_ffa,
                      snr_min,
                      max_passing_candidates,
                      use_boxcar_kadane),
      m_ep_impl(std::make_unique<EPImpl>(prune_poly_order,
                                         p_orb_min,
                                         m_c_max,
                                         m_p_min,
                                         propagator_significance,
                                         validation_significance,
                                         use_conservative_tile)) {}

EPSearchConfig::EPSearchConfig(FFASearchConfig ffa_cfg,
                               SizeType prune_poly_order,
                               double p_orb_min,
                               double m_c_max,
                               double m_p_min,
                               double propagator_significance,
                               double validation_significance,
                               bool use_conservative_tile)
    : FFASearchConfig(std::move(ffa_cfg)),
      m_ep_impl(std::make_unique<EPImpl>(prune_poly_order,
                                         p_orb_min,
                                         m_c_max,
                                         m_p_min,
                                         propagator_significance,
                                         validation_significance,
                                         use_conservative_tile)) {}

EPSearchConfig::~EPSearchConfig()                                    = default;
EPSearchConfig::EPSearchConfig(EPSearchConfig&&) noexcept            = default;
EPSearchConfig& EPSearchConfig::operator=(EPSearchConfig&&) noexcept = default;
EPSearchConfig::EPSearchConfig(const EPSearchConfig& other)
    : FFASearchConfig(other),
      m_ep_impl(std::make_unique<EPImpl>(*other.m_ep_impl)) {}

EPSearchConfig& EPSearchConfig::operator=(const EPSearchConfig& other) {
    if (this != &other) {
        FFASearchConfig::operator=(other);
        m_ep_impl = std::make_unique<EPImpl>(*other.m_ep_impl);
    }
    return *this;
}

SizeType EPSearchConfig::get_prune_poly_order() const noexcept {
    return m_ep_impl->m_prune_poly_order;
}
double EPSearchConfig::get_p_orb_min() const noexcept {
    return m_ep_impl->m_p_orb_min;
}
double EPSearchConfig::get_m_c_max() const noexcept {
    return m_ep_impl->m_m_c_max;
}
double EPSearchConfig::get_m_p_min() const noexcept {
    return m_ep_impl->m_m_p_min;
}
double EPSearchConfig::get_propagator_significance() const noexcept {
    return m_ep_impl->m_propagator_significance;
}
double EPSearchConfig::get_validation_significance() const noexcept {
    return m_ep_impl->m_validation_significance;
}
bool EPSearchConfig::get_use_conservative_tile() const noexcept {
    return m_ep_impl->m_use_conservative_tile;
}
double EPSearchConfig::get_x_mass_const() const noexcept {
    return m_ep_impl->get_x_mass_const();
}

EPSearchConfig EPSearchConfig::get_updated_config(
    SizeType nbins,
    double eta,
    std::span<const ParamLimit> param_limits) const {
    auto ffa = FFASearchConfig::get_updated_config(nbins, eta, param_limits);
    return EPSearchConfig(std::move(ffa), m_ep_impl->m_prune_poly_order,
                          m_ep_impl->m_p_orb_min, m_ep_impl->m_m_c_max,
                          m_ep_impl->m_m_p_min,
                          m_ep_impl->m_propagator_significance,
                          m_ep_impl->m_validation_significance,
                          m_ep_impl->m_use_conservative_tile);
}

EPSearchConfig EPSearchConfig::get_updated_config(SizeType nbins,
                                                  double eta,
                                                  double f_min,
                                                  double f_max) const {
    auto ffa = FFASearchConfig::get_updated_config(nbins, eta, f_min, f_max);
    return EPSearchConfig(std::move(ffa), m_ep_impl->m_prune_poly_order,
                          m_ep_impl->m_p_orb_min, m_ep_impl->m_m_c_max,
                          m_ep_impl->m_m_p_min,
                          m_ep_impl->m_propagator_significance,
                          m_ep_impl->m_validation_significance,
                          m_ep_impl->m_use_conservative_tile);
}

} // namespace loki::search
