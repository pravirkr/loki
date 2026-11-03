#include "loki/search/configs.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>
#include <toml++/toml.hpp> // NOLINT(misc-include-cleaner) -- toml::table, parse, parse_error

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/orbit.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/psr_utils.hpp"
#include "lib/detail/toml_schema.hpp"
#include "lib/detail/toml_utils.hpp"
#include "lib/detail/utils.hpp"
#include "lib/search/config_keys.hpp"
#include "lib/search/ep_support.hpp"

namespace loki::search {

namespace {

using toml_utils::array_number;
using toml_utils::read_number;
using toml_utils::read_string;
using toml_utils::reject_nonfinite_floats;

/// Keys removed from the schema, with what replaces them.
[[nodiscard]] std::string_view removed_key_hint(std::string_view path) {
    if (path == "rfi" || path.starts_with("rfi.") ||
        path.find("harvest") != std::string_view::npos ||
        path.find("pulsar_mask") != std::string_view::npos ||
        path.find("impulsive") != std::string_view::npos) {
        return "RFI options (pulsar mask, harvesting, impulsive veto) are not "
               "available in the EP search yet";
    }
    if (path == "cuda" || path.starts_with("cuda.")) {
        return "the [cuda] table was removed; use [performance] backend = "
               "\"cuda\" and device = <id>";
    }
    if (path == "input.preprocess" || path == "input.filter_window" ||
        path == "input.fast_median" || path == "input.fast_median_min_points") {
        return "moved to the [preprocessing] table";
    }
    if (path == "performance.use_cuda") {
        return "use performance.backend = \"cuda\"";
    }
    if (path == "performance.device_id") {
        return "use performance.device";
    }
    return {};
}

/// The order of the tables in a written config, as in the templates.
constexpr std::array kTableOrder{
    std::string_view{"input"},       std::string_view{"preprocessing"},
    std::string_view{"search"},      std::string_view{"ep"},
    std::string_view{"performance"}, std::string_view{"output"},
    std::string_view{"candidates"},
};

using Orbit = FFATomlConfig::OrbitLimits;

const toml_utils::Schema<Orbit>& orbit_schema() {
    using toml_utils::field;
    static const toml_utils::Schema<Orbit> schema{
        toml_utils::required(
            field<Orbit>("", "order", [](auto& o) -> auto& { return o.order; }),
            R"("accel", "jerk", "snap" or "crackle")"),
        field<Orbit>("", "p_orb_min",
                     [](auto& o) -> auto& { return o.p_orb_min; }),
        field<Orbit>("", "p_orb_min_tobs",
                     [](auto& o) -> auto& { return o.p_orb_min_tobs; }),
        field<Orbit>("", "m_c_max", [](auto& o) -> auto& { return o.m_c_max; }),
        field<Orbit>("", "m_p_min", [](auto& o) -> auto& { return o.m_p_min; }),
    };
    return schema;
}

/// The keys of an FFA config, shared by the EP config.
template <typename Cfg> toml_utils::Schema<Cfg> ffa_fields() {
    using toml_utils::choice;
    using toml_utils::custom;
    using toml_utils::field;
    constexpr std::array kMethods{
        std::pair{std::string_view{"robust"}, io::PreprocessMethod::kRobust},
        std::pair{std::string_view{"zscore"}, io::PreprocessMethod::kZScore},
    };
    constexpr std::array kGainModels{
        std::pair{std::string_view{"additive"}, io::GainModel::kAdditive},
        std::pair{std::string_view{"multiplicative"},
                  io::GainModel::kMultiplicative},
    };
    return {
        // [input]
        field<Cfg>("input", "timeseries",
                   [](auto& c) -> auto& { return c.timeseries_path; }),
        field<Cfg>("input", "nsamps",
                   [](auto& c) -> auto& { return c.nsamps; }),
        field<Cfg>("input", "tsamp", [](auto& c) -> auto& { return c.tsamp; }),
        // Alias of tsamp, read only when tsamp is not set.
        custom<Cfg>(
            "input", "dt",
            [](const toml::table& t, const std::string& path, Cfg& c) {
                if (!c.tsamp) {
                    c.tsamp = read_number(t, "dt", path);
                }
            },
            [](const Cfg& /*c*/, toml::table& /*t*/) {}),
        field<Cfg>("input", "dm",
                   [](auto& c) -> auto& { return c.observation.dm; }),
        field<Cfg>("input", "start_mjd",
                   [](auto& c) -> auto& { return c.observation.start_mjd; }),
        field<Cfg>("input", "barycentric",
                   [](auto& c) -> auto& { return c.observation.barycentric; }),
        // [preprocessing]
        field<Cfg>("preprocessing", "preprocess",
                   [](auto& c) -> auto& { return c.preprocess; }),
        choice<Cfg>(
            "preprocessing", "method",
            [](auto& c) -> auto& { return c.preprocessing.method; }, kMethods),
        choice<Cfg>(
            "preprocessing", "gain_model",
            [](auto& c) -> auto& { return c.preprocessing.gain_model; },
            kGainModels),
        field<Cfg>(
            "preprocessing", "filter_window",
            [](auto& c) -> auto& { return c.preprocessing.filter_window; }),
        field<Cfg>("preprocessing", "min_window_periods",
                   [](auto& c) -> auto& { return c.min_window_periods; }),
        field<Cfg>(
            "preprocessing", "variance_window",
            [](auto& c) -> auto& { return c.preprocessing.variance_window; }),
        field<Cfg>(
            "preprocessing", "window_blocks",
            [](auto& c) -> auto& { return c.preprocessing.window_blocks; }),
        field<Cfg>(
            "preprocessing", "fast_median",
            [](auto& c) -> auto& { return c.preprocessing.fast_median; }),
        field<Cfg>("preprocessing", "fast_median_min_points",
                   [](auto& c) -> auto& {
                       return c.preprocessing.fast_median_min_points;
                   }),
        field<Cfg>("preprocessing", "n_iter",
                   [](auto& c) -> auto& { return c.preprocessing.n_iter; }),
        field<Cfg>(
            "preprocessing", "block_scales",
            [](auto& c) -> auto& { return c.preprocessing.block_scales; }),
        field<Cfg>(
            "preprocessing", "block_sigma",
            [](auto& c) -> auto& { return c.preprocessing.block_sigma; }),
        field<Cfg>(
            "preprocessing", "min_good_fraction",
            [](auto& c) -> auto& { return c.preprocessing.min_good_fraction; }),
        field<Cfg>("preprocessing", "clip_sigma",
                   [](auto& c) -> auto& { return c.preprocessing.clip_sigma; }),
        field<Cfg>(
            "preprocessing", "zap_periodic",
            [](auto& c) -> auto& { return c.preprocessing.zap_periodic; }),
        field<Cfg>("preprocessing", "zap_sigma",
                   [](auto& c) -> auto& { return c.preprocessing.zap_sigma; }),
        field<Cfg>(
            "preprocessing", "zap_whiten_bins",
            [](auto& c) -> auto& { return c.preprocessing.zap_whiten_bins; }),
        custom<Cfg>(
            "preprocessing", "birdies",
            [](const toml::table& t, const std::string& path, Cfg& c) {
                const auto* arr = t["birdies"].as_array();
                if (arr == nullptr) {
                    throw std::invalid_argument(std::format(
                        "{} must be an array of [freq_hz, width_hz]", path));
                }
                auto& birdies = c.preprocessing.birdies;
                birdies.clear();
                for (const auto& el : *arr) {
                    const auto* pair = el.as_array();
                    if (pair == nullptr || pair->size() != 2) {
                        throw std::invalid_argument(std::format(
                            "{} entries must be [freq_hz, width_hz]", path));
                    }
                    birdies.push_back(io::Birdie{
                        .freq  = array_number((*pair)[0], path),
                        .width = array_number((*pair)[1], path),
                    });
                }
            },
            [](const Cfg& c, toml::table& t) {
                toml::array birdies;
                for (const auto& birdie : c.preprocessing.birdies) {
                    birdies.push_back(toml::array{birdie.freq, birdie.width});
                }
                t.insert("birdies", std::move(birdies));
            }),
        // [search]
        field<Cfg>("search", "f_min", [](auto& c) -> auto& { return c.f_min; }),
        field<Cfg>("search", "f_max", [](auto& c) -> auto& { return c.f_max; }),
        field<Cfg>("search", "acc_min",
                   [](auto& c) -> auto& { return c.acc_min; }),
        field<Cfg>("search", "acc_max",
                   [](auto& c) -> auto& { return c.acc_max; }),
        field<Cfg>("search", "jerk_min",
                   [](auto& c) -> auto& { return c.jerk_min; }),
        field<Cfg>("search", "jerk_max",
                   [](auto& c) -> auto& { return c.jerk_max; }),
        toml_utils::sub_table<Cfg>(
            "search", "orbit", [](auto& c) -> auto& { return c.orbit; },
            orbit_schema()),
        field<Cfg>("search", "nbins", [](auto& c) -> auto& { return c.nbins; }),
        field<Cfg>("search", "eta", [](auto& c) -> auto& { return c.eta; }),
        field<Cfg>("search", "ducy_max",
                   [](auto& c) -> auto& { return c.ducy_max; }),
        field<Cfg>("search", "wtsp", [](auto& c) -> auto& { return c.wtsp; }),
        field<Cfg>("search", "snr_min",
                   [](auto& c) -> auto& { return c.snr_min; }),
        field<Cfg>("search", "use_fourier",
                   [](auto& c) -> auto& { return c.use_fourier; }),
        field<Cfg>("search", "use_boxcar_kadane",
                   [](auto& c) -> auto& { return c.use_boxcar_kadane; }),
        // [performance]
        field<Cfg>("performance", "nthreads",
                   [](auto& c) -> auto& { return c.nthreads; }),
        custom<Cfg>(
            "performance", "backend",
            [](const toml::table& t, const std::string& path, Cfg& c) {
                c.backend = parse_backend(*read_string(t, "backend", path));
            },
            [](const Cfg& c, toml::table& t) {
                t.insert("backend", std::string(to_string(c.backend)));
            }),
        field<Cfg>("performance", "device",
                   [](auto& c) -> auto& { return c.device; }),
        field<Cfg>("performance", "max_process_memory_gb",
                   [](auto& c) -> auto& { return c.max_process_memory_gb; }),
        field<Cfg>("performance", "octave_scale",
                   [](auto& c) -> auto& { return c.octave_scale; }),
        field<Cfg>("performance", "nbins_max",
                   [](auto& c) -> auto& { return c.nbins_max; }),
        field<Cfg>("performance", "nbins_min_lossy_bf",
                   [](auto& c) -> auto& { return c.nbins_min_lossy_bf; }),
        field<Cfg>("performance", "bseg_brute",
                   [](auto& c) -> auto& { return c.bseg_brute; }),
        field<Cfg>("performance", "bseg_ffa",
                   [](auto& c) -> auto& { return c.bseg_ffa; }),
        field<Cfg>("performance", "max_passing_candidates",
                   [](auto& c) -> auto& { return c.max_passing_candidates; }),
        // [output]
        field<Cfg>("output", "outdir",
                   [](auto& c) -> auto& { return c.outdir; }),
        field<Cfg>("output", "prefix",
                   [](auto& c) -> auto& { return c.prefix; }),
        // [candidates]
        field<Cfg>("candidates", "enabled",
                   [](auto& c) -> auto& { return c.candidates.enabled; }),
        field<Cfg>("candidates", "cluster",
                   [](auto& c) -> auto& { return c.candidates.cluster; }),
        field<Cfg>("candidates", "grouping",
                   [](auto& c) -> auto& { return c.candidates.grouping; }),
        field<Cfg>("candidates", "radius",
                   [](auto& c) -> auto& { return c.candidates.radius; }),
        field<Cfg>("candidates", "phase_radius",
                   [](auto& c) -> auto& { return c.candidates.phase_radius; }),
        field<Cfg>(
            "candidates", "retain_detections",
            [](auto& c) -> auto& { return c.candidates.retain_detections; }),
        field<Cfg>(
            "candidates", "flag_harmonics",
            [](auto& c) -> auto& { return c.candidates.flag_harmonics; }),
        field<Cfg>(
            "candidates", "remove_harmonics",
            [](auto& c) -> auto& { return c.candidates.remove_harmonics; }),
        field<Cfg>("candidates", "harmonic_denominator_max",
                   [](auto& c) -> auto& {
                       return c.candidates.harmonic_denominator_max;
                   }),
        field<Cfg>("candidates", "harmonic_phase_distance",
                   [](auto& c) -> auto& {
                       return c.candidates.harmonic_phase_distance;
                   }),
        field<Cfg>("candidates", "harmonic_snr_distance",
                   [](auto& c) -> auto& {
                       return c.candidates.harmonic_snr_distance;
                   }),
        field<Cfg>(
            "candidates", "max_candidates",
            [](auto& c) -> auto& { return c.candidates.max_candidates; }),
        field<Cfg>("candidates", "memory_gb",
                   [](auto& c) -> auto& { return c.candidates.memory_gb; }),
        field<Cfg>(
            "candidates", "temporary_directory",
            [](auto& c) -> auto& { return c.candidates.temporary_directory; }),
    };
}

const toml_utils::Schema<FFATomlConfig>& ffa_schema() {
    static const auto schema = ffa_fields<FFATomlConfig>();
    return schema;
}

/// The FFA keys and the [ep] table, with the plan cache in [output].
const toml_utils::Schema<EPTomlConfig>& ep_schema() {
    using toml_utils::field;
    using E = EPTomlConfig;
    // An orbit prior of [ep] is not written when [search.orbit] sets it.
    const auto orbit_prior = [](toml_utils::Field<E> f) {
        f.write = [write = std::move(f.write)](const E& c, toml::table& t) {
            if (!c.orbit) {
                write(c, t);
            }
        };
        return f;
    };
    static const auto schema = [&] {
        auto fields = ffa_fields<E>();
        const std::array extra{
            field<E>("ep", "target_snr",
                     [](auto& c) -> auto& { return c.target_snr; }),
            field<E>("ep", "poly_basis",
                     [](auto& c) -> auto& { return c.poly_basis; }),
            field<E>("ep", "min_pd", [](auto& c) -> auto& { return c.min_pd; }),
            field<E>("ep", "ref_ducy",
                     [](auto& c) -> auto& { return c.ref_ducy; }),
            field<E>("ep", "prune_poly_order",
                     [](auto& c) -> auto& { return c.prune_poly_order; }),
            field<E>("ep", "n_runs", [](auto& c) -> auto& { return c.n_runs; }),
            field<E>("ep", "ref_segs",
                     [](auto& c) -> auto& { return c.ref_segs; }),
            orbit_prior(field<E>("ep", "p_orb_min",
                                 [](auto& c) -> auto& { return c.p_orb_min; })),
            orbit_prior(field<E>("ep", "m_c_max",
                                 [](auto& c) -> auto& { return c.m_c_max; })),
            orbit_prior(field<E>("ep", "m_p_min",
                                 [](auto& c) -> auto& { return c.m_p_min; })),
            field<E>(
                "ep", "propagator_significance",
                [](auto& c) -> auto& { return c.propagator_significance; }),
            field<E>(
                "ep", "validation_significance",
                [](auto& c) -> auto& { return c.validation_significance; }),
            field<E>("ep", "use_conservative_tile",
                     [](auto& c) -> auto& { return c.use_conservative_tile; }),
            field<E>("output", "plan_cache",
                     [](auto& c) -> auto& { return c.plan_cache; }),
            field<E>("output", "use_plan_cache",
                     [](auto& c) -> auto& { return c.use_plan_cache; }),
        };
        fields.insert(fields.end(), extra.begin(), extra.end());
        return fields;
    }();
    return schema;
}

/// Checks of an FFA config that span several keys or need the whole
/// [preprocessing] table.
void check_ffa_fields(const FFATomlConfig& cfg) {
    if (!utils::is_finite(cfg.min_window_periods) ||
        cfg.min_window_periods < 0.0) {
        throw std::invalid_argument(
            "preprocessing.min_window_periods must be >= 0");
    }
    try {
        cfg.preprocessing.validate();
    } catch (const std::invalid_argument& err) {
        throw std::invalid_argument(
            std::format("Invalid [preprocessing] table: {}", err.what()));
    }
    if (cfg.candidates.harmonic_denominator_max > 10000) {
        throw std::invalid_argument(
            "candidates.harmonic_denominator_max must be at most 10000");
    }
    cfg.candidates.validate();
    cfg.observation.validate();
}

void require_positive(double value, std::string_view path) {
    if (!utils::is_finite(value) || value <= 0.0) {
        throw std::invalid_argument(
            std::format("{} must be a positive number (got {})", path, value));
    }
}

} // namespace

// ==============================================================================
// FFATomlConfig Implementation
// ==============================================================================

std::string_view FFATomlConfig::default_toml_string() {
    return R"(# ==============================================================================
# Loki FFA Pulsar Search Configuration
# ==============================================================================

[input]
# Path to input timeseries (.tim or .dat format)
timeseries = "input.tim"

# Sample count and sample interval, for a config used without data (e.g.
# FFASearchConfig.from_toml). A search takes them from the timeseries.
# nsamps = 2097152
# tsamp = 0.0000640

# Observation metadata stored with the candidates; each overrides the
# timeseries header (or --dm, --start-mjd, --barycentric on the command line).
# dm = 0.0              # dispersion measure, pc cm^-3
# start_mjd = 60000.0   # MJD of the first sample
# barycentric = false   # true: barycentric, false: topocentric sample times

[preprocessing]
# Build the folding inputs from the raw timeseries (see docs/preprocessing.md)
preprocess = true

# "robust": local baseline and variance, RFI masking, inverse-variance weights.
# "zscore": legacy running-median detrend + global z-score (unit weights).
method = "robust"

# Robust: noise gain model, "additive" (g = 1) or "multiplicative" (g = baseline)
gain_model = "additive"

# Running-median baseline window in seconds. The effective window is at least
# min_window_periods / f_min, so slow pulsars are not subtracted (0 disables).
filter_window = 1.0
min_window_periods = 10.0

# Robust: local variance window in seconds (0 = the effective baseline window)
variance_window = 0.0

# Robust: statistics blocks per baseline window
window_blocks = 101

# Zscore: approximate long running-median windows by block averaging. Set false
# for the exact sliding median. fast_median_min_points is the short-series width.
fast_median = true
fast_median_min_points = 101

# Robust: rounds of (statistics -> bad-block flags)
n_iter = 2

# Robust: bad-block time scales in seconds ([] disables block masking), the
# robust z-score threshold for a block mean or variance, and the minimum valid
# fraction of a block
block_scales = [0.016, 0.065, 0.26, 1.05]
block_sigma = 6.0
min_good_fraction = 0.3

# Robust: zero samples beyond this many local sigmas (0 disables)
clip_sigma = 6.0

# Robust: zap periodic RFI in the whitened power spectrum. Off by default:
# a bright pulsar's harmonics can exceed the threshold.
zap_periodic = false
zap_sigma = 8.0
zap_whiten_bins = 1001

# Robust: known RFI lines, always zapped, as [[freq_hz, width_hz], ...]
birdies = []

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

# Number of profile bins at shortest search period (must be >= 2)
nbins = 64

# Tolerance in bins across observation duration (typically 1.0)
eta = 1.0

# Maximum pulse duty cycle (FWTM in phase) for boxcar template width trials
ducy_max = 0.2

# Boxcar width trial geometric scale factor (typically 1.2 - 1.5)
wtsp = 1.5

# Minimum candidate SNR written to final HDF5 (EP filtering reserved for later)
snr_min = 7.0

# Folding domain: true for Fourier-domain FFA (recommended), false for Time-domain FFA
use_fourier = true

# Peak detection must be the multi-width boxcar; Kadane is not supported
use_boxcar_kadane = false

# Alternative to acc_* and jerk_*: limits of physical circular orbits. The
# acceleration, jerk (snap, crackle) limits and a Doppler-widened frequency
# range are derived from them (f_min and f_max are then intrinsic spin
# frequencies). Exactly one of p_orb_min (seconds) and p_orb_min_tobs (in units
# of the observation length) is required. Coherent Taylor searches commonly
# use p_orb_min_tobs = 10 (accel), 5 (jerk) or 3 (snap).
# [search.orbit]
# order = "jerk"          # "accel" | "jerk" | "snap" | "crackle" (EP only)
# p_orb_min_tobs = 5.0
# p_orb_min = 100.0      # seconds, instead of p_orb_min_tobs
# m_c_max = 10.0          # companion mass, M_sun
# m_p_min = 1.2           # pulsar mass, M_sun

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

# Execution backend: "cpu", or "cuda" (requires a CUDA-enabled Loki build)
backend = "cpu"

# GPU device ordinal, used when backend = "cuda"
device = 0

# Segment lengths in samples, powers of 2 (default: chosen by the planner)
# bseg_brute = 1024     # brute-force fold segment
# bseg_ffa = 2097152    # FFA segment (default: nsamps)

[output]
# Directory where candidate HDF5 file will be saved
outdir = "./"

# Output filename prefix (results saved as <outdir>/<prefix>_ffa_results.h5)
prefix = "loki"

[candidates]
# false writes the flat detection file of earlier versions: no grouping or
# selection, and no host memory reserved for them.
enabled = true
# Phase-track friends-of-friends; radius in narrower pulse widths.
cluster = true
grouping = "transitive" # or "anchored" for strongest-representative suppression
radius = 1.0
# Minimum separation in phase cycles. Set 0 for pulse-width-only clustering.
phase_radius = 0.05
retain_detections = true
flag_harmonics = false
remove_harmonics = false
harmonic_denominator_max = 100
harmonic_phase_distance = 1.0
harmonic_snr_distance = 3.0
max_candidates = 0 # unlimited; applied after harmonic filtering
memory_gb = 1.0 # additional host budget, reserved within CPU search budget
# temporary_directory = "/scratch/loki"
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

namespace {

[[nodiscard]] std::vector<std::string>
with_orbit_keys(std::vector<std::string> keys) {
    for (const auto& f : orbit_schema()) {
        keys.push_back(std::format("search.orbit.{}", f.key));
    }
    return keys;
}

} // namespace

std::vector<std::string> ffa_config_keys() {
    return with_orbit_keys(toml_utils::schema_keys(ffa_schema()));
}

std::vector<std::string> ep_config_keys() {
    return with_orbit_keys(toml_utils::schema_keys(ep_schema()));
}

std::string FFATomlConfig::to_toml_string() const {
    return toml_utils::to_toml_text(
        toml_utils::write_fields(ffa_schema(), *this), kTableOrder);
}

FFATomlConfig FFATomlConfig::from_string(std::string_view toml_content) {
    const toml::table tbl = toml_utils::parse(toml_content);
    reject_nonfinite_floats(tbl, toml_content, "");
    toml_utils::check_document(ffa_schema(), tbl, "Invalid FFA TOML config",
                               removed_key_hint);
    FFATomlConfig cfg;
    toml_utils::read_fields(ffa_schema(), tbl, cfg);
    check_ffa_fields(cfg);
    return cfg;
}

FFATomlConfig FFATomlConfig::load(const std::filesystem::path& path) {
    return toml_utils::load_file(path, &FFATomlConfig::from_string);
}

io::PreprocessOptions FFATomlConfig::to_preprocess_options() const {
    io::PreprocessOptions o = preprocessing;
    // filter_window = 0 already means one baseline for the whole series.
    if (min_window_periods > 0.0 && f_min > 0.0 && o.filter_window > 0.0) {
        const double floor = min_window_periods / f_min;
        if (floor > o.filter_window) {
            spdlog::info("Preprocessing: baseline window raised from {:.3g} s "
                         "to {:.3g} s ({} periods of f_min = {} Hz)",
                         o.filter_window, floor, min_window_periods, f_min);
            o.filter_window = floor;
        }
    }
    return o;
}

namespace {

/// [search.orbit]: the parameter limits of the circular orbits it allows.
std::vector<ParamLimit>
orbit_param_limits(const FFATomlConfig::OrbitLimits& orbit,
                   double f_min,
                   double f_max,
                   double tobs,
                   bool has_acc,
                   bool has_jerk) {
    if (has_acc || has_jerk) {
        throw std::invalid_argument(
            "search.orbit sets the parameter limits: remove the non-zero "
            "search.acc_* and search.jerk_* (or the orbit table)");
    }
    SizeType poly_order = 0;
    if (orbit.order == "accel") {
        poly_order = 2;
    } else if (orbit.order == "jerk") {
        poly_order = 3;
    } else if (orbit.order == "snap") {
        poly_order = 4;
    } else if (orbit.order == "crackle") {
        poly_order = 5;
    } else {
        throw std::invalid_argument(std::format(
            "search.orbit.order must be \"accel\", \"jerk\", \"snap\" or "
            "\"crackle\", got \"{}\"",
            orbit.order));
    }
    if (orbit.p_orb_min.has_value() == orbit.p_orb_min_tobs.has_value()) {
        throw std::invalid_argument(
            "search.orbit needs exactly one of p_orb_min (seconds) and "
            "p_orb_min_tobs (units of the observation length)");
    }
    if (orbit.p_orb_min_tobs.has_value() &&
        (!utils::is_finite(*orbit.p_orb_min_tobs) ||
         *orbit.p_orb_min_tobs <= 0.0)) {
        throw std::invalid_argument(
            std::format("search.orbit.p_orb_min_tobs must be positive, got {}",
                        *orbit.p_orb_min_tobs));
    }
    const double p_orb_min =
        orbit.p_orb_min.value_or(orbit.p_orb_min_tobs.value_or(0.0) * tobs);
    auto limits = circular_param_limits(f_min, f_max, poly_order,
                                        {.p_orb_min = p_orb_min,
                                         .m_c_max   = orbit.m_c_max,
                                         .m_p_min   = orbit.m_p_min});
    if (p_orb_min < tobs) {
        spdlog::warn("search.orbit: p_orb_min = {:.4g} s is shorter than the "
                     "observation ({:.4g} s); a Taylor phase model of order "
                     "{} is not coherent over a whole orbit",
                     p_orb_min, tobs, poly_order);
    }
    static constexpr std::array kNames{"crackle", "snap", "jerk", "accel"};
    std::string derived;
    for (SizeType i = 0; i + 1 < limits.size(); ++i) {
        derived += std::format("{} +/- {:.4g}, ",
                               kNames[kNames.size() - (limits.size() - 1) + i],
                               limits[i].max);
    }
    spdlog::info("search.orbit ({}, P_orb >= {:.4g} s = {:.3g} T_obs, m_c <= "
                 "{:.3g}, m_p >= {:.3g}): {}f = [{:.6g}, {:.6g}] Hz",
                 orbit.order, p_orb_min, p_orb_min / tobs, orbit.m_c_max,
                 orbit.m_p_min, derived, limits.back().min, limits.back().max);
    return limits;
}

/// Logs the circular orbit an explicit box with jerk implies, and warns when
/// it is faster than the observation.
void report_implied_orbit(std::span<const ParamLimit> limits, double tobs) {
    const auto implied = implied_circular_orbit(limits);
    if (!implied) {
        return;
    }
    const double ratio = implied->p_orb / tobs;
    if (ratio < 1.0) {
        spdlog::warn(
            "search: the acc/jerk limits reach circular orbits of "
            "P_orb = {:.4g} s ({:.2g} T_obs) with a companion of at "
            "least {:.3g} M_sun (m_p 1.4); a Taylor phase model is not "
            "coherent over a whole orbit. Consider [search.orbit].",
            implied->p_orb, ratio, implied->m_c_min);
    } else {
        spdlog::info("search: the acc/jerk limits reach circular orbits of "
                     "P_orb = {:.4g} s ({:.2g} T_obs) with a companion of at "
                     "least {:.3g} M_sun (m_p 1.4)",
                     implied->p_orb, ratio, implied->m_c_min);
    }
}

} // namespace

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

    const bool has_jerk = jerk_min.has_value() && jerk_max.has_value() &&
                          (*jerk_min != 0.0 || *jerk_max != 0.0);
    const bool has_acc  = acc_min.has_value() && acc_max.has_value() &&
                          (*acc_min != 0.0 || *acc_max != 0.0);
    const double tobs   = static_cast<double>(final_nsamps) * final_tsamp;

    std::vector<ParamLimit> limits;
    if (orbit.has_value()) {
        limits =
            orbit_param_limits(*orbit, f_min, f_max, tobs, has_acc, has_jerk);
    } else if (has_jerk) {
        limits.push_back({.min = *jerk_min, .max = *jerk_max});
        limits.push_back(
            {.min = acc_min.value_or(0.0), .max = acc_max.value_or(0.0)});
        limits.push_back({.min = f_min, .max = f_max});
        report_implied_orbit(limits, tobs);
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

    // Entry point: 0 (or less) means "all threads". Below this layer the
    // count is only clamped from below (docs/architecture.md, Numerics).
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

namespace {

/**
 * @brief Relative cost weights for the direct-DFT bseg_brute selector.
 *
 * Only the direct-DFT Fourier brute fold (nbins <= nbins_min_lossy_bf) picks
 * its segment length by cost: there the brute fold is a (sample, frequency,
 * Fourier bin) triple loop and dominates the run time. The time-domain and
 * lossy-Fourier paths use a cycles-at-f_max target instead (see
 * bseg_target_cycles). Weights are in units of one time-domain brute-fold
 * gather-add, fitted with bench/scripts/ffa/bseg_brute_sweep.py +
 * fit_cost_model.py (Apple M1 Pro, 8 threads). The optimum is flat, so they
 * only need to be roughly right; use an explicit `bseg_brute` to override on
 * unusual hardware or backends.
 *
 * - brute_op: cost per (sample, frequency, Fourier bin) of the SIMD DFT
 *             (~0.27 gather-adds).
 * - merge_op: cost per complex merge element-op, i.e. per
 *             2 * ncoords * nsegments * nbins_f (~4.9 gather-adds).
 */
struct DirectDftCostWeights {
    double brute_op;
    double merge_op;
};

constexpr DirectDftCostWeights kDirectDftWeights{
    .brute_op = 0.27,
    .merge_op = 4.9,
};

} // namespace

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
        // Before the defaults below, which depend on nbins.
        error_check::check_greater_equal(m_nbins, 2, "nbins must be >= 2");
        m_nbins_f = (m_nbins / 2) + 1;
        m_nparams = m_param_limits.size();
        m_param_names.assign(kParamNames.end() - m_nparams, kParamNames.end());
        m_f_min      = m_param_limits[m_nparams - 1].min;
        m_f_max      = m_param_limits[m_nparams - 1].max;
        m_bseg_brute = bseg_brute.value_or(get_bseg_brute_default());
        m_bseg_ffa   = bseg_ffa.value_or(get_bseg_ffa_default());

        m_nthreads = std::max(m_nthreads, 1);
        validate();
        m_tseg_brute = static_cast<double>(m_bseg_brute) * m_tsamp;
        m_tseg_ffa   = static_cast<double>(m_bseg_ffa) * m_tsamp;
        m_niters_ffa = static_cast<SizeType>(std::countr_zero(m_bseg_ffa) -
                                             std::countr_zero(m_bseg_brute));
        m_scoring_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        m_boxcar_kadane_biases = {1.42F, 0.76F, 0.41F};

        spdlog::trace(
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
        error_check::check_greater(m_ducy_max, 0.0,
                                   "ducy_max must be positive");
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

    /**
     * @brief Default brute-fold segment length (the FFA entry level).
     *
     * Chosen by a cycles-at-f_max target, or by a cost model for the
     * direct-DFT Fourier brute fold (see select_bseg_brute). The previous
     * "two cycles at f_min" heuristic is kept as a fallback for degenerate
     * configurations (e.g. nsamps that is not a power of two, which validate()
     * rejects anyway, or parameter ranges the grid-count helpers refuse).
     */
    SizeType get_bseg_brute_default() const {
        try {
            if (const auto bseg = select_bseg_brute()) {
                return *bseg;
            }
        } catch (const std::exception& ex) {
            spdlog::debug("bseg_brute cost model unavailable ({}); using "
                          "legacy heuristic",
                          ex.what());
        }
        return get_bseg_brute_legacy();
    }

    /**
     * @brief Largest k such that B = 2^k is a valid brute segment length, or
     * std::nullopt when no power-of-two choice exists (nsamps that is not a
     * power of two, which validate() rejects anyway).
     *
     * B must stay below nsamps (validate) and below the FFA segment cap.
     */
    [[nodiscard]] std::optional<SizeType> bseg_brute_k_max() const {
        if (m_nsamps < 4 || !std::has_single_bit(m_nsamps)) {
            return std::nullopt;
        }
        const SizeType bseg_ffa_cap = m_bseg_ffa_explicit.value_or(m_nsamps);
        if (bseg_ffa_cap < 2 || !std::has_single_bit(bseg_ffa_cap) ||
            bseg_ffa_cap > m_nsamps) {
            return std::nullopt;
        }
        return std::min(static_cast<SizeType>(std::countr_zero(bseg_ffa_cap)),
                        static_cast<SizeType>(std::countr_zero(m_nsamps)) - 1);
    }

    /// validate_plan(): level 0 (segments of 2^k samples) may only have a
    /// frequency grid, i.e. a single trial for every higher-order parameter.
    [[nodiscard]] bool level0_is_frequency_only(SizeType k) const {
        const auto c0 = get_param_grid_count(
            static_cast<double>(SizeType{1} << k) * m_tsamp);
        return std::all_of(c0.begin(), c0.end() - 1,
                           [](SizeType count) { return count == 1; });
    }

    /**
     * @brief Pick the default brute segment length B (a power of two), or
     * std::nullopt if no valid choice exists.
     *
     * - Direct-DFT Fourier: minimise a cost model (select_bseg_direct_dft).
     * - Time domain and lossy Fourier: a cycles-at-f_max target
     *   (select_bseg_by_cycles).
     */
    [[nodiscard]] std::optional<SizeType> select_bseg_brute() const {
        const bool direct_dft =
            m_use_fourier && m_nbins <= m_nbins_min_lossy_bf;
        return direct_dft ? select_bseg_direct_dft() : select_bseg_by_cycles();
    }

    /**
     * @brief B from a target number of cycles per segment at f_max.
     *
     * The run-length brute cost per frequency is ~nbins * ceil(B / P): flat
     * up to one period at f_max and then growing with the cycles per
     * segment. The accuracy side is the number of merge levels, each of which
     * rounds the shifts to whole bins and loses S/N (measured with
     * bench/scripts/ffa/bseg_brute_sensitivity.py: about 1% per halving of B
     * below ~8 cycles at nbins 32, with the loss falling as nbins grows). So
     * B is set from a target number of cycles at f_max, the fast end of the
     * chunk, which the planner keeps within octave_scale of f_min.
     *
     * Takes the largest valid B at or below the power of two nearest to the
     * target; the smallest valid B if all candidates are above it.
     */
    [[nodiscard]] std::optional<SizeType> select_bseg_by_cycles() const {
        const auto k_max = bseg_brute_k_max();
        if (!k_max.has_value()) {
            return std::nullopt;
        }
        const double target_cycles  = bseg_target_cycles();
        const double target_samples = target_cycles / (m_tsamp * m_f_max);
        const auto desired_k        = static_cast<SizeType>(std::max(
            1L, std::lround(std::log2(std::max(target_samples, 2.0)))));
        std::optional<SizeType> largest_below_target;
        std::optional<SizeType> smallest_valid;
        for (SizeType k = 1; k <= *k_max; ++k) {
            if (!level0_is_frequency_only(k)) {
                continue;
            }
            const auto bseg_k = SizeType{1} << k;
            if (!smallest_valid.has_value()) {
                smallest_valid = bseg_k;
            }
            if (k <= desired_k) {
                largest_below_target = bseg_k;
            }
        }
        const auto chosen = largest_below_target.has_value()
                                ? largest_below_target
                                : smallest_valid;
        if (chosen.has_value()) {
            spdlog::trace("bseg_brute cycle target: nbins={}, nparams={}, "
                          "f=[{:.4g}, {:.4g}] Hz -> bseg_brute={} (target "
                          "{:.2f} cycles at f_max, got {:.2f})",
                          m_nbins, m_nparams, m_f_min, m_f_max, *chosen,
                          target_cycles,
                          static_cast<double>(*chosen) * m_tsamp * m_f_max);
        }
        return chosen;
    }

    /**
     * @brief B minimising W_brute * brute_ops(B) + W_merge * merge_ops(B) for
     * the direct-DFT Fourier brute fold.
     *
     * - Brute fold at segment length B costs ~2 * nsamps * F(B) * nbins_f
     *   (sample, frequency, bin) triples, F(B) being the level-0 frequency
     *   grid size (proportional to B).
     * - Every FFA merge level costs ~2 * ncoords * nsegments * nbins_f
     *   element ops. For a frequency-only search F * N is constant, so each
     *   level costs the same.
     * - Halving B halves the brute cost and adds one merge level, which pays
     *   off while B > ~2 * W_merge/W_brute * nbins_f.
     *
     * Level counts come from the real parameter grids, so higher-order
     * searches and the F>=1 floor are handled, and candidates that violate
     * the "level 0 has a single higher-order trial" plan constraint are
     * skipped.
     */
    [[nodiscard]] std::optional<SizeType> select_bseg_direct_dft() const {
        const auto k_max_opt = bseg_brute_k_max();
        if (!k_max_opt.has_value()) {
            return std::nullopt;
        }
        const SizeType k_max = *k_max_opt;
        const auto width     = static_cast<double>(m_nbins_f);
        const auto nsamps    = static_cast<double>(m_nsamps);
        const auto& w        = kDirectDftWeights;

        // Level-0 frequency counts and merge work per segment length 2^k.
        std::vector<SizeType> nfreqs0(k_max + 1, 0);
        std::vector<double> merge_elems(k_max + 1, 0.0);
        for (SizeType k = 1; k <= k_max; ++k) {
            const double tseg = static_cast<double>(SizeType{1} << k) * m_tsamp;
            const auto counts = get_param_grid_count(tseg);
            nfreqs0[k]        = counts.back();
            double ncoords    = 1.0;
            for (const auto c : counts) {
                ncoords *= static_cast<double>(c);
            }
            const double nsegments = nsamps / static_cast<double>(1ULL << k);
            merge_elems[k]         = ncoords * nsegments * 2.0 * width;
        }
        // Suffix sums: merge work of all levels above k.
        std::vector<double> merge_above(k_max + 2, 0.0);
        for (SizeType k = k_max; k >= 1; --k) {
            merge_above[k - 1] = merge_above[k] + merge_elems[k];
        }

        // Beyond ~16 bins' worth of samples a larger B can no longer pay for
        // itself. Without this guard a very narrow chunk, where F(B) floors
        // at 1, would pick an enormous B and make its brute-fold plan shape
        // larger than that of the wider chunks around it, breaking the
        // planner's assumption that narrowing a chunk never increases its
        // memory. Fall back to the unrestricted scan only if the guard
        // leaves no valid candidate.
        constexpr SizeType kMaxBsegPerBin = 16;
        const SizeType bseg_guard         = kMaxBsegPerBin * m_nbins;

        std::optional<SizeType> best;
        double best_cost = 0.0;
        for (const bool guarded : {true, false}) {
            for (SizeType k = 1; k <= k_max; ++k) {
                const auto bseg_k = SizeType{1} << k;
                if (guarded && bseg_k > bseg_guard) {
                    break;
                }
                if (!level0_is_frequency_only(k)) {
                    continue;
                }
                const double brute_ops =
                    2.0 * nsamps * static_cast<double>(nfreqs0[k]) * width;
                const double cost =
                    (w.brute_op * brute_ops) + (w.merge_op * merge_above[k]);
                if (!best.has_value() || cost <= best_cost) {
                    best      = bseg_k;
                    best_cost = cost;
                }
            }
            if (best.has_value()) {
                break;
            }
        }
        if (best.has_value()) {
            spdlog::trace("bseg_brute direct-DFT cost model: nbins={}, "
                          "f=[{:.4g}, {:.4g}] Hz -> bseg_brute={} (model cost "
                          "{:.3e})",
                          m_nbins, m_f_min, m_f_max, *best, best_cost);
        }
        return best;
    }

    /**
     * @brief Target cycles per brute segment at f_max for the run-length
     * (time-domain and lossy-Fourier) fold.
     *
     * Fitted from injection sweeps (2^21 samples, eta = 1, chunk ratio 1.5):
     * the per-level shift rounding costs S/N roughly in proportion to
     * 1/nbins^2 (pulse width in bins), so wide-profile chunks need fewer
     * cycles. Acceleration searches have more merge dimensions and need
     * about twice as many. The chosen B is the power of two nearest to the
     * target (never above it by more than the rounding).
     */
    [[nodiscard]] double bseg_target_cycles() const {
        double cycles = 1.5;
        if (m_nbins <= 64) {
            cycles = 4.0;
        } else if (m_nbins <= 128) {
            cycles = 2.0;
        }
        return (m_nparams > 1) ? 2.0 * cycles : cycles;
    }

    /// Legacy heuristic: roughly two cycles at f_min (four for nparams > 1).
    SizeType get_bseg_brute_legacy() const {
        const auto tobs   = static_cast<double>(m_nsamps) * m_tsamp;
        const auto cycles = tobs * m_f_min;
        if (cycles <= 1.0) {
            return std::min(m_nsamps / 2, SizeType{64});
        }
        const auto levels     = static_cast<int>(std::floor(std::log2(cycles)));
        const int init_levels = (m_nparams == 1) ? 1 : 2;
        const int shift       = std::max(1, levels - init_levels);
        const auto bseg =
            static_cast<SizeType>(m_nsamps >> static_cast<unsigned>(shift));
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
std::vector<std::string> FFASearchConfig::get_param_names() const {
    return m_impl->get_param_names();
}
double FFASearchConfig::get_f_min() const noexcept {
    return m_impl->get_f_min();
}
double FFASearchConfig::get_f_max() const noexcept {
    return m_impl->get_f_max();
}
std::vector<SizeType> FFASearchConfig::get_scoring_widths() const {
    return m_impl->get_scoring_widths();
}
SizeType FFASearchConfig::get_n_scoring_widths() const noexcept {
    return m_impl->get_n_scoring_widths();
}
std::vector<float> FFASearchConfig::get_boxcar_kadane_biases() const {
    return m_impl->get_boxcar_kadane_biases();
}
SizeType FFASearchConfig::get_n_boxcar_kadane_biases() const noexcept {
    return m_impl->get_n_boxcar_kadane_biases();
}
void FFASearchConfig::set_max_process_memory_gb(double max_process_memory_gb) {
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
std::vector<double> FFASearchConfig::get_dparams_actual(double tseg_cur) const {
    return m_impl->get_dparams_actual(tseg_cur);
}
std::vector<SizeType>
FFASearchConfig::get_param_grid_count(double tseg_cur) const {
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

std::string_view FFASearchConfig::default_toml_string() {
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
           bool use_conservative_tile,
           SizeType nparams,
           std::optional<double> target_snr)
        : m_prune_poly_order(prune_poly_order),
          m_p_orb_min(p_orb_min),
          m_m_c_max(m_c_max),
          m_m_p_min(m_p_min),
          m_propagator_significance(propagator_significance),
          m_validation_significance(validation_significance),
          m_use_conservative_tile(use_conservative_tile),
          m_target_snr_override(target_snr),
          m_target_snr(target_snr.value_or(nparams <= 2   ? 8.5
                                           : nparams == 3 ? 9.2
                                           : nparams == 4 ? 9.8
                                                          : 10.0)) {
        if (!utils::is_finite(m_target_snr) || m_target_snr <= 0.0) {
            throw std::invalid_argument(
                "target_snr must be finite and positive");
        }
    }

    SizeType m_prune_poly_order;
    double m_p_orb_min;
    double m_m_c_max;
    double m_m_p_min;
    double m_propagator_significance;
    double m_validation_significance;
    bool m_use_conservative_tile;
    std::optional<double> m_target_snr_override;
    double m_target_snr;

    double get_x_mass_const() const noexcept {
        constexpr double kSafety = 1.1;
        return utils::kGMsunOneThird * kSafety * m_m_c_max /
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
                               bool use_boxcar_kadane,
                               std::optional<double> target_snr)
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
                                         use_conservative_tile,
                                         get_nparams(),
                                         target_snr)) {}

EPSearchConfig::EPSearchConfig(FFASearchConfig ffa_cfg,
                               SizeType prune_poly_order,
                               double p_orb_min,
                               double m_c_max,
                               double m_p_min,
                               double propagator_significance,
                               double validation_significance,
                               bool use_conservative_tile,
                               std::optional<double> target_snr)
    : FFASearchConfig(std::move(ffa_cfg)),
      m_ep_impl(std::make_unique<EPImpl>(prune_poly_order,
                                         p_orb_min,
                                         m_c_max,
                                         m_p_min,
                                         propagator_significance,
                                         validation_significance,
                                         use_conservative_tile,
                                         get_nparams(),
                                         target_snr)) {}

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
double EPSearchConfig::get_target_snr() const noexcept {
    return m_ep_impl->m_target_snr;
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

EPSearchConfig EPSearchConfig::get_updated_ep_config(
    SizeType nbins,
    double eta,
    std::span<const ParamLimit> param_limits) const {
    auto ffa = get_updated_config(nbins, eta, param_limits);
    return EPSearchConfig(
        std::move(ffa), m_ep_impl->m_prune_poly_order, m_ep_impl->m_p_orb_min,
        m_ep_impl->m_m_c_max, m_ep_impl->m_m_p_min,
        m_ep_impl->m_propagator_significance,
        m_ep_impl->m_validation_significance,
        m_ep_impl->m_use_conservative_tile, m_ep_impl->m_target_snr_override);
}

EPSearchConfig EPSearchConfig::get_updated_ep_config(SizeType nbins,
                                                     double eta,
                                                     double f_min,
                                                     double f_max) const {
    auto ffa = get_updated_config(nbins, eta, f_min, f_max);
    return EPSearchConfig(
        std::move(ffa), m_ep_impl->m_prune_poly_order, m_ep_impl->m_p_orb_min,
        m_ep_impl->m_m_c_max, m_ep_impl->m_m_p_min,
        m_ep_impl->m_propagator_significance,
        m_ep_impl->m_validation_significance,
        m_ep_impl->m_use_conservative_tile, m_ep_impl->m_target_snr_override);
}

// ==============================================================================
// EPTomlConfig Implementation
// ==============================================================================

namespace {

/// Range checks that the types cannot express. Used on parse and before a
/// config is turned into a search config.
void validate_ep_config(const EPTomlConfig& cfg) {
    if (cfg.nbins_max > detail::kEPMaxNbins ||
        cfg.nbins > detail::kEPMaxNbins) {
        throw std::invalid_argument(std::format(
            "EP nbins and nbins_max must be <= {}", detail::kEPMaxNbins));
    }
    if (cfg.target_snr &&
        (!utils::is_finite(*cfg.target_snr) || *cfg.target_snr <= 0.0)) {
        throw std::invalid_argument(
            "ep.target_snr must be finite and positive");
    }
    if (cfg.poly_basis != "taylor" && cfg.poly_basis != "chebyshev") {
        throw std::invalid_argument(std::format(
            "ep.poly_basis must be 'taylor' or 'chebyshev' (got '{}')",
            cfg.poly_basis));
    }
    if (!utils::is_finite(cfg.min_pd) || cfg.min_pd <= 0.0F ||
        cfg.min_pd > 1.0F) {
        throw std::invalid_argument(
            std::format("ep.min_pd must be in (0, 1] (got {})", cfg.min_pd));
    }
    // The threshold scheme needs a pulse narrower than the period.
    if (!utils::is_finite(cfg.ref_ducy) || cfg.ref_ducy <= 0.0F ||
        cfg.ref_ducy >= 1.0F) {
        throw std::invalid_argument(std::format(
            "ep.ref_ducy must be in (0, 1) (got {})", cfg.ref_ducy));
    }
    if (cfg.prune_poly_order < 1) {
        throw std::invalid_argument(std::format(
            "ep.prune_poly_order must be >= 1 (got {})", cfg.prune_poly_order));
    }
    if (cfg.n_runs.has_value() && *cfg.n_runs < 1) {
        throw std::invalid_argument("ep.n_runs must be >= 1");
    }
    if (cfg.ref_segs.has_value()) {
        if (cfg.ref_segs->empty()) {
            throw std::invalid_argument(
                "ep.ref_segs must not be empty (omit it to prune every run)");
        }
        auto segs = *cfg.ref_segs;
        std::ranges::sort(segs);
        if (std::ranges::adjacent_find(segs) != segs.end()) {
            throw std::invalid_argument("ep.ref_segs must not repeat a run");
        }
    }
    require_positive(cfg.p_orb_min, "ep.p_orb_min");
    require_positive(cfg.m_c_max, "ep.m_c_max");
    require_positive(cfg.m_p_min, "ep.m_p_min");
    require_positive(cfg.propagator_significance, "ep.propagator_significance");
    require_positive(cfg.validation_significance, "ep.validation_significance");
}

} // namespace

std::string_view EPTomlConfig::default_toml_string() {
    return R"(# ==============================================================================
# Loki EP Pulsar Search Configuration (Extreme Pruning)
# ==============================================================================
# Every [input], [preprocessing], [search], [performance] and [output] key of
# the FFA file has the same meaning here. This file adds the [ep] table and
# output.plan_cache.

[input]
# Path to input timeseries (.tim or .dat format), read by `search ep`
timeseries = "input.tim"

# Sample count and sample interval. `search ep --plan-only` reads no data, so
# these are required there. A search takes them from the timeseries.
nsamps = 2097152
tsamp = 0.0000640

# Observation metadata stored with the candidates; each overrides the
# timeseries header (or --dm, --start-mjd, --barycentric on the command line).
# dm = 0.0              # dispersion measure, pc cm^-3
# start_mjd = 60000.0   # MJD of the first sample
# barycentric = false   # true: barycentric, false: topocentric sample times

[preprocessing]
# Build the folding inputs from the raw timeseries (see docs/preprocessing.md)
preprocess = true

# "robust": local baseline and variance, RFI masking, inverse-variance weights.
# "zscore": legacy running-median detrend + global z-score (unit weights).
method = "robust"

# Robust: noise gain model, "additive" (g = 1) or "multiplicative" (g = baseline)
gain_model = "additive"

# Running-median baseline window in seconds. The effective window is at least
# min_window_periods / f_min, so slow pulsars are not subtracted (0 disables).
filter_window = 1.0
min_window_periods = 10.0

# Robust: local variance window in seconds (0 = the effective baseline window)
variance_window = 0.0

# Robust: statistics blocks per baseline window
window_blocks = 101

# Zscore: approximate long running-median windows by block averaging. Set false
# for the exact sliding median. fast_median_min_points is the short-series width.
fast_median = true
fast_median_min_points = 101

# Robust: rounds of (statistics -> bad-block flags)
n_iter = 2

# Robust: bad-block time scales in seconds ([] disables block masking), the
# robust z-score threshold for a block mean or variance, and the minimum valid
# fraction of a block
block_scales = [0.016, 0.065, 0.26, 1.05]
block_sigma = 6.0
min_good_fraction = 0.3

# Robust: zero samples beyond this many local sigmas (0 disables)
clip_sigma = 6.0

# Robust: zap periodic RFI in the whitened power spectrum. Off by default:
# a bright pulsar's harmonics can exceed the threshold.
zap_periodic = false
zap_sigma = 8.0
zap_whiten_bins = 1001

# Robust: known RFI lines, always zapped, as [[freq_hz, width_hz], ...]
birdies = []

[search]
# Search frequency range in Hz (Period P = 1 / f)
f_min = 0.5
f_max = 100.0

# Line-of-sight acceleration limits in m/s^2: set a physical search range.
# Frequency-only EP is unsupported; use the FFA pipeline for that search.
# Edit these zero placeholders before executing this template.
acc_min = 0.0
acc_max = 0.0

# Optional line-of-sight jerk limits in m/s^3 (omit or set equal to 0 for no jerk search)
jerk_min = 0.0
jerk_max = 0.0

# Number of profile bins at shortest search period (must be >= 2)
nbins = 64

# Tolerance in bins across observation duration (typically 1.0)
eta = 1.0

# Maximum pulse duty cycle (FWTM in phase) for boxcar template width trials
ducy_max = 0.2

# Boxcar width trial geometric scale factor (typically 1.2 - 1.5)
wtsp = 1.5

# Minimum candidate SNR written to final HDF5 (EP filtering reserved for later)
snr_min = 7.0

# Folding domain: true for Fourier-domain FFA (recommended), false for Time-domain FFA
use_fourier = true

# Peak detection must be the multi-width boxcar; Kadane is not supported
use_boxcar_kadane = false

[ep]
# Entropic information limit for EP multipass/pipeline threshold design only.
# Omit to use the search default (from the parameter count):
# frequency/accel 8.5, jerk 9.2, snap 9.8, circular 10.0. Independent of search.snr_min.
# target_snr = 9.2

# Polynomial basis of the pruning: "taylor" or "chebyshev"
poly_basis = "taylor"

# Requested detection probability cutoff, in (0, 1].
# Shortfalls during selection or evaluation warn and are accepted temporarily.
min_pd = 0.1

# Reference duty cycle of the pruning, in (0, 1]
ref_ducy = 0.1

# Polynomial order of the pruning (>= 1)
prune_poly_order = 3

# Number of runs (reference segments) to prune. Set either n_runs or ref_segs,
# not both: the pruning needs the runs it prunes.
n_runs = 16

# Explicit reference segments, instead of n_runs (exclusive with n_runs)
# ref_segs = [0, 1]

# Orbit and mass priors of the circular (crackle) pruning. A [search.orbit]
# table sets them instead.
# p_orb_min = 0.00001
# m_c_max = 10.0
# m_p_min = 1.4

# Significance levels of the propagator and the validation
propagator_significance = 2.0
validation_significance = 5.0

# Use the conservative tile of the Fourier search
use_conservative_tile = false

# Alternative to acc_* and jerk_*: limits of physical circular orbits. The
# acceleration, jerk (snap, crackle) limits and a Doppler-widened frequency
# range are derived from them (f_min and f_max are then intrinsic spin
# frequencies). Exactly one of p_orb_min (seconds) and p_orb_min_tobs (in units
# of the observation length) is required. Coherent Taylor searches commonly
# use p_orb_min_tobs = 10 (accel), 5 (jerk) or 3 (snap).
# [search.orbit]
# order = "jerk"          # "accel" | "jerk" | "snap" | "crackle" (EP only)
# p_orb_min_tobs = 5.0
# p_orb_min = 100.0      # seconds, instead of p_orb_min_tobs
# m_c_max = 10.0          # companion mass, M_sun
# m_p_min = 1.2           # pulsar mass, M_sun

[performance]
# Number of CPU OpenMP threads (0 = all hardware threads)
nthreads = 4

# Maximum memory cap in GB: process RSS on CPU, device memory on CUDA
max_process_memory_gb = 8.0

# Octave scaling factor between successive search bands (2.0 = octave doubling)
octave_scale = 2.0

# Maximum number of EP fold bins for long-period bands (cannot exceed 1024)
nbins_max = 1024

# Minimum number of bins for lossy brute force initial stage
nbins_min_lossy_bf = 64

# Maximum candidate buffer capacity held in memory before streaming to disk
max_passing_candidates = 4194304

# Execution backend: "cpu", or "cuda" (requires a CUDA-enabled Loki build)
backend = "cpu"

# GPU device ordinal, used when backend = "cuda"
device = 0

# FFA segment length in samples: a power of 2 below nsamps, since the pruning
# needs at least one FFA level. Use nsamps / 128 for a long series.
bseg_ffa = 16384
# Brute-force fold segment in samples, a power of 2 (default: the planner's)
# bseg_brute = 1024

[candidates]
# Postprocess the merged final Taylor runs after all pruning has finished.
# Other polynomial bases/models currently retain diagnostic-only output.
# false writes the run diagnostics of earlier versions only.
enabled = true
cluster = true
grouping = "transitive" # or "anchored"
radius = 1.0
phase_radius = 0.05
retain_detections = true
flag_harmonics = false
remove_harmonics = false
harmonic_denominator_max = 100
harmonic_phase_distance = 1.0
harmonic_snr_distance = 3.0
max_candidates = 0
# Separate host candidate budget; reserved before CPU chunk planning.
memory_gb = 1.0
# temporary_directory = "/scratch/loki"

[output]
# Directory where the results file is saved
outdir = "./"

# Output filename prefix (results saved as <outdir>/<prefix>_ep_results.h5)
prefix = "loki"

# Plan cache (HDF5). `search ep --plan-only` writes <outdir>/<prefix>_ep_plan.h5
# when this is omitted. Searches also cache here by default.
# Disable automatic plan caching (explicit plan_cache paths still apply).
use_plan_cache = true
# plan_cache = "./loki_ep_plan.h5"
)";
}

void EPTomlConfig::write_default(const std::filesystem::path& path) {
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

std::string EPTomlConfig::to_toml_string() const {
    return toml_utils::to_toml_text(
        toml_utils::write_fields(ep_schema(), *this), kTableOrder);
}

EPTomlConfig EPTomlConfig::from_string(std::string_view toml_content) {
    const toml::table tbl = toml_utils::parse(toml_content);
    reject_nonfinite_floats(tbl, toml_content, "");
    toml_utils::check_document(ep_schema(), tbl, "Invalid EP TOML config",
                               removed_key_hint);
    EPTomlConfig cfg;
    toml_utils::read_fields(ep_schema(), tbl, cfg);
    check_ffa_fields(cfg);
    // [search.orbit] sets the orbit of the pruning too: one source.
    if (cfg.orbit.has_value()) {
        if (const auto* ep = tbl["ep"].as_table()) {
            for (const auto key : {"p_orb_min", "m_c_max", "m_p_min"}) {
                if (ep->contains(key)) {
                    throw std::invalid_argument(std::format(
                        "ep.{} is set by [search.orbit]: remove it", key));
                }
            }
        }
    }
    validate_ep_config(cfg);
    return cfg;
}

EPTomlConfig EPTomlConfig::load(const std::filesystem::path& path) {
    return toml_utils::load_file(path, &EPTomlConfig::from_string);
}

EPSearchConfig
EPTomlConfig::to_ep_search_config(std::optional<SizeType> override_nsamps,
                                  std::optional<double> override_tsamp) const {
    validate_ep_config(*this);
    // Checked here, not on parse: the command line may still set the runs.
    if (n_runs.has_value() && ref_segs.has_value()) {
        throw std::invalid_argument(
            "ep.n_runs and ep.ref_segs are exclusive: set one of them");
    }
    if (!n_runs.has_value() && !ref_segs.has_value()) {
        throw std::invalid_argument(
            "ep.n_runs or ep.ref_segs must be set: the pruning needs the runs "
            "it prunes");
    }
    auto ffa = to_search_config(override_nsamps, override_tsamp);
    if (ffa.get_nparams() < 2) {
        throw std::invalid_argument(
            "frequency-only EP is unsupported: set a search range in "
            "search.acc_min/acc_max (or --acc-min/--acc-max), or a "
            "[search.orbit] table; use 'loki search ffa' for a frequency-only "
            "search");
    }
    // The pruning needs at least one FFA level above the brute-force segment,
    // so the FFA segment must be shorter than the series.
    if (ffa.get_bseg_ffa() >= ffa.get_nsamps()) {
        throw std::invalid_argument(std::format(
            "performance.bseg_ffa ({}) must be below nsamps ({}): the EP "
            "pruning needs at least one FFA level",
            ffa.get_bseg_ffa(), ffa.get_nsamps()));
    }
    // The pruning checks its leaves against the orbit of [search.orbit].
    double orbit_p_min = p_orb_min;
    double orbit_m_c   = m_c_max;
    double orbit_m_p   = m_p_min;
    if (orbit.has_value()) {
        orbit_p_min = orbit->p_orb_min.value_or(
            orbit->p_orb_min_tobs.value_or(0.0) * ffa.get_tobs());
        orbit_m_c = orbit->m_c_max;
        orbit_m_p = orbit->m_p_min;
    }
    return EPSearchConfig(std::move(ffa), prune_poly_order, orbit_p_min,
                          orbit_m_c, orbit_m_p, propagator_significance,
                          validation_significance, use_conservative_tile,
                          target_snr);
}

} // namespace loki::search
