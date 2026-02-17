#include "loki/algorithms/regions.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils.hpp"

namespace loki::regions {

std::vector<coord::FFARegion> generate_ffa_regions(double p_min,
                                                   double p_max,
                                                   double tsamp,
                                                   SizeType nbins_min,
                                                   double eta_min,
                                                   double octave_scale,
                                                   SizeType nbins_max) {
    error_check::check_greater(p_min, 0.0, "p_min must be positive.");
    error_check::check_greater(p_max, p_min, "p_max must be > p_min.");
    error_check::check_greater(tsamp, 0.0, "tsamp must be positive.");
    error_check::check_greater_equal(nbins_min, 2, "nbins_min must be >= 2.");
    error_check::check_greater(eta_min, 0.0, "eta_min must be positive.");
    error_check::check_greater_equal(octave_scale, 1.0,
                                     "octave_scale must be >= 1.0.");
    error_check::check_greater_equal(nbins_max, nbins_min,
                                     "nbins_max must be >= nbins_min.");

    std::vector<coord::FFARegion> regions;
    SizeType nbins_cur =
        std::min(nbins_min, static_cast<SizeType>(p_min / tsamp));
    double rho = eta_min / static_cast<double>(nbins_cur);

    // Core invariant: physical time width of a single folding bin (in
    // seconds). Core invariant: duty-cycle resolution (rho) in bins.
    const double bin_width = p_min / static_cast<double>(nbins_cur);
    auto p_cur_low         = p_min;
    while (p_cur_low < p_max) {
        auto nbins_k = nbins_cur;
        if (nbins_k >= nbins_max) {
            double eta_k = std::round(rho * static_cast<double>(nbins_max));
            regions.push_back({1.0 / p_max, 1.0 / p_cur_low, nbins_max, eta_k});
            break;
        }
        const double p_cur_high = std::min(p_cur_low * octave_scale, p_max);
        double eta_k            = rho * static_cast<double>(nbins_k);
        regions.push_back({1.0 / p_cur_high, 1.0 / p_cur_low, nbins_k, eta_k});
        p_cur_low = p_cur_high;
        nbins_cur = static_cast<SizeType>(p_cur_low / bin_width);
    }
    return regions;
}

template <SupportedFoldType FoldType> struct FFARegionStats<FoldType>::Impl {
    SizeType max_buffer_size{};
    SizeType max_coord_size{};
    SizeType max_ncoords{}; // maximum number of coordinates in the last level
    SizeType max_ffa_levels{};
    SizeType n_widths{};
    SizeType n_params{};
    SizeType n_samps{}; // ts_e.size()
    SizeType m_max_passing_candidates{};
    bool use_gpu{};

    Impl(SizeType max_buffer_size,
         SizeType max_coord_size,
         SizeType max_ncoords,
         SizeType max_ffa_levels,
         SizeType n_widths,
         SizeType n_params,
         SizeType n_samps,
         SizeType max_passing_candidates,
         bool use_gpu)
        : max_buffer_size(max_buffer_size),
          max_coord_size(max_coord_size),
          max_ncoords(max_ncoords),
          max_ffa_levels(max_ffa_levels),
          n_widths(n_widths),
          n_params(n_params),
          n_samps(n_samps),
          m_max_passing_candidates(max_passing_candidates),
          use_gpu(use_gpu) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = default;
    Impl& operator=(const Impl&) = default;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    SizeType get_max_buffer_size_time() const noexcept {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            return 2 * max_buffer_size;
        } else {
            return max_buffer_size;
        }
    }
    SizeType get_max_ffa_levels() const noexcept { return max_ffa_levels; }
    SizeType get_max_scores_size() const noexcept {
        return std::max(max_ncoords * n_widths, m_max_passing_candidates);
    }
    SizeType get_write_param_sets_size() const noexcept {
        return kFFAFreqSweepWriteBatchSize * (n_params + 1); // includes width
    }
    float get_buffer_memory_usage() const noexcept {
        return static_cast<float>(2 * max_buffer_size * sizeof(FoldType)) /
               static_cast<float>(1ULL << 30U);
    }
    float get_coord_memory_usage() const noexcept {
        SizeType coord_size;
        if (n_params == 1) {
            coord_size = max_coord_size * sizeof(coord::FFACoordFreq);
        } else {
            coord_size = max_coord_size * sizeof(coord::FFACoord);
        }
        return static_cast<float>(coord_size) / static_cast<float>(1ULL << 30U);
    }
    float get_extra_memory_usage() const noexcept {
        return static_cast<float>(
                   (get_write_param_sets_size() * sizeof(double)) +
                   (get_max_scores_size() * sizeof(uint32_t)) +
                   (get_max_scores_size() * sizeof(float))) /
               static_cast<float>(1ULL << 30U);
    }
    float get_cpu_memory_usage() const noexcept {
        return get_buffer_memory_usage() + get_coord_memory_usage() +
               get_extra_memory_usage();
    }
    // Use this for GPU memory usage calculation
    float get_device_memory_usage() const noexcept {
        // ts_e_d + ts_v_d + scores_d (widths_d is negligible)
        const float device_extra_gb =
            ((2 * n_samps + get_max_scores_size()) * sizeof(float) +
             (get_max_scores_size() * sizeof(uint32_t))) /
            static_cast<float>(1ULL << 30U);
        // m_fold_d_time
        const float fold_d_time_gb = get_max_buffer_size_time() *
                                     sizeof(float) /
                                     static_cast<float>(1ULL << 30U);
        return device_extra_gb + fold_d_time_gb + get_buffer_memory_usage() +
               get_coord_memory_usage();
    }
    float get_freq_sweep_memory_usage() const noexcept {
        if (use_gpu) {
            return get_device_memory_usage();
        }
        return get_cpu_memory_usage();
    }
}; // End FFARegionStats::Impl definition

template <SupportedFoldType FoldType> class FFARegionPlanner<FoldType>::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg, bool use_gpu)
        : m_base_cfg(std::move(cfg)),
          m_use_gpu(use_gpu) {
        plan_regions();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const std::vector<search::PulsarSearchConfig>& get_cfgs() const noexcept {
        return m_cfgs;
    }
    SizeType get_nregions() const noexcept { return m_cfgs.size(); }
    const FFARegionStats<FoldType>& get_stats() const noexcept {
        return m_stats;
    }
    const std::vector<coord::FFAChunkStats>& get_chunk_stats() const noexcept {
        return m_chunk_stats;
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    bool m_use_gpu;

    std::vector<search::PulsarSearchConfig> m_cfgs;
    FFARegionStats<FoldType> m_stats{0, 0, 0, 0, 0, 0, 0, 0, m_use_gpu};
    std::vector<coord::FFAChunkStats> m_chunk_stats;

    double calculate_max_drift(const search::PulsarSearchConfig& cfg) const {
        if (cfg.get_nparams() <= 1) {
            return 0.0;
        }

        auto param_limits = cfg.get_param_limits();
        // Drift from center to edge
        const auto t_half = cfg.get_tobs() / 2.0;
        if (cfg.get_nparams() == 2) {
            const auto max_accel = std::max(std::abs(param_limits[0].min),
                                            std::abs(param_limits[0].max));
            const auto drift     = max_accel * t_half;
            return drift / utils::kCval;
        }
        if (cfg.get_nparams() == 3) {
            const auto max_jerk  = std::max(std::abs(param_limits[0].min),
                                            std::abs(param_limits[0].max));
            const auto max_accel = std::max(std::abs(param_limits[1].min),
                                            std::abs(param_limits[1].max));
            const auto drift =
                (max_accel * t_half) + (max_jerk * t_half * t_half / 2.0);
            return drift / utils::kCval;
        }
        throw std::runtime_error(
            "Unsupported number of parameters for drift calculation");
    }

    void plan_regions() {
        // Step 1: Generate FFA regions based on frequency-dependent
        // nbins/eta
        const double f_min     = m_base_cfg.get_f_min();
        const double f_max     = m_base_cfg.get_f_max();
        const double p_min     = 1.0 / f_max;
        const double p_max     = 1.0 / f_min;
        const auto ffa_regions = generate_ffa_regions(
            p_min, p_max, m_base_cfg.get_tsamp(), m_base_cfg.get_nbins(),
            m_base_cfg.get_eta(), m_base_cfg.get_octave_scale(),
            m_base_cfg.get_nbins_max());
        // Print region info
        spdlog::info("FFARegionPlanner - planned regions:");
        for (const auto& region : ffa_regions) {
            spdlog::info("Region: f=[{:08.3f}, {:08.3f}] Hz, nbins={:04d}, "
                         "eta={:04.1f}",
                         region.f_start, region.f_end, region.nbins,
                         region.eta);
        }

        // Step 2: For each region, subdivide in frequency if it doesn't fit
        // in memory
        const auto max_drift = calculate_max_drift(m_base_cfg);
        // Log drift information
        if (max_drift > 0 && m_base_cfg.get_nparams() > 1) {
            spdlog::info("Drift-aware chunking: max_drift={:.6f} ({:.4f}%) "
                         "for tobs={:.1f}s, n_params={}",
                         max_drift, max_drift * 100.0, m_base_cfg.get_tobs(),
                         m_base_cfg.get_nparams());
        }
        SizeType max_buffer_size{};
        SizeType max_coord_size{};
        SizeType max_ncoords{};
        SizeType max_ffa_levels{};
        for (const auto& region : ffa_regions) {
            subdivide_region_by_memory(region.f_start, region.f_end,
                                       region.nbins, region.eta, max_drift,
                                       max_buffer_size, max_coord_size,
                                       max_ncoords, max_ffa_levels);
        }
        m_stats = FFARegionStats<FoldType>(
            max_buffer_size, max_coord_size, max_ncoords, max_ffa_levels,
            m_base_cfg.get_n_scoring_widths(), m_base_cfg.get_nparams(),
            m_base_cfg.get_nsamps(), m_base_cfg.get_max_passing_candidates(),
            m_use_gpu);

        // Log summary statistics
        log_planning_summary();
    }

    void subdivide_region_by_memory(double f_start,
                                    double f_end,
                                    SizeType nbins,
                                    double eta,
                                    double max_drift,
                                    SizeType& max_buffer_size,
                                    SizeType& max_coord_size,
                                    SizeType& max_ncoords,
                                    SizeType& max_ffa_levels) {
        // For GPU, this is the device memory limit. For CPU, this is the
        // process memory limit.
        const auto max_memory_gb = m_base_cfg.get_max_process_memory_gb();
        // Reserve some headroom for OS, Python interpreter, and rounding
        // errors
        constexpr double kSafetyMarginGB = 0.5; // 500 MB safety margin
        const auto effective_limit_gb    = max_memory_gb - kSafetyMarginGB;

        constexpr double kMinViableRange  = 0.1; // Minimum possible range in Hz
        constexpr double kMinChunkSize    = 0.01; // Merge threshold (Hz)
        constexpr double kSearchTolerance = 0.01; // Binary search stop (Hz)
        constexpr SizeType kMaxProbes     = 20;   // Avoid infinite loops

        // Pre-flight check: can we fit minimum viable chunk at worst case
        // (high freq)?
        {
            const double min_f_start =
                (f_end - kMinViableRange) * (1.0 - max_drift);
            const double min_f_end = f_end * (1.0 + max_drift);
            auto check_cfg         = m_base_cfg.get_updated_config(
                nbins, eta, min_f_start, min_f_end);
            plans::FFAPlan<FoldType> check_plan(std::move(check_cfg));

            FFARegionStats<FoldType> min_stats{
                check_plan.get_buffer_size(),
                check_plan.get_coord_size(),
                check_plan.get_ncoords().back(),
                check_plan.get_n_levels(),
                m_base_cfg.get_n_scoring_widths(),
                m_base_cfg.get_nparams(),
                m_base_cfg.get_nsamps(),
                m_base_cfg.get_max_passing_candidates(),
                m_use_gpu};

            if (min_stats.get_freq_sweep_memory_usage() > effective_limit_gb) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: Cannot fit minimum viable chunk at "
                    "highest frequency.\n"
                    "  Nominal range: {:.3f} Hz, with drift: {:.3f} Hz\n"
                    "  Required memory: {:.2f} GB, Available: {:.2f} GB\n"
                    "  Suggestion: Increase max_memory_gb or reduce "
                    "parameter "
                    "searchranges.",
                    kMinViableRange, min_f_end - min_f_start,
                    min_stats.get_freq_sweep_memory_usage(),
                    effective_limit_gb));
            }
        }

        // Reverse iteration: go from high frequency to low (f_end →
        // f_start)
        double current_f_end = f_end;
        while (current_f_end > f_start) {
            // Early exit for tiny remaining ranges
            const double range_width = current_f_end - f_start;
            if (range_width < kMinViableRange) {
                spdlog::debug(
                    "Skipping remaining range [{:08.3f}, {:08.3f}] Hz "
                    "(only {:.3f} Hz wide, below minimum {:.3f} Hz)",
                    f_start, current_f_end, range_width, kMinViableRange);
                break;
            }
            // Binary search for largest chunk that fits (working backwards
            // from current_f_end)
            double f_low        = f_start;
            double f_high       = current_f_end;
            double best_f_start = current_f_end; // Start from the top

            // Start with a Heuristic probe (10% of remaining range or 1 Hz)
            double f_probe = std::max(
                current_f_end - std::max(0.1 * range_width, 1.0), f_start);
            SizeType probe_count = 0;

            while (probe_count < kMaxProbes &&
                   (f_high - f_low) > kSearchTolerance) {
                // Create test config for this chunk
                auto test_cfg = m_base_cfg.get_updated_config(
                    nbins, eta, f_probe * (1.0 - max_drift),
                    current_f_end * (1.0 + max_drift));
                // Simulate the chunk
                plans::FFAPlan<FoldType> test_plan(std::move(test_cfg));
                FFARegionStats<FoldType> sim_stats{
                    std::max(max_buffer_size, test_plan.get_buffer_size()),
                    std::max(max_coord_size, test_plan.get_coord_size()),
                    std::max(max_ncoords, test_plan.get_ncoords().back()),
                    std::max(max_ffa_levels, test_plan.get_n_levels()),
                    m_base_cfg.get_n_scoring_widths(),
                    m_base_cfg.get_nparams(),
                    m_base_cfg.get_nsamps(),
                    m_base_cfg.get_max_passing_candidates(),
                    m_use_gpu};
                if (sim_stats.get_freq_sweep_memory_usage() <=
                    effective_limit_gb) {
                    // Fits! Try to include more (go lower in frequency)
                    best_f_start = f_probe;
                    f_high       = f_probe;

                    if (f_probe <= f_start + kSearchTolerance) {
                        best_f_start = f_start; // Close enough, use f_start
                        break;
                    }
                    // Probe lower
                    f_probe = std::max(std::midpoint(f_low, f_high), f_start);
                } else {
                    // Doesn't fit, need smaller chunk (go higher in
                    // frequency)
                    f_low   = f_probe;
                    f_probe = std::midpoint(f_low, f_high);
                }

                probe_count++;
            }
            // Check if remaining range is too small to warrant separate
            // chunk
            const double remaining_range = best_f_start - f_start;
            if (remaining_range > 0 && remaining_range < kMinChunkSize &&
                best_f_start < current_f_end) {
                // Extend current chunk to include small remainder
                best_f_start = f_start;
            }
            if (best_f_start >= current_f_end) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: Cannot fit any chunk in range "
                    "[{:08.3f}, {:08.3f}] Hz with nbins={} into memory "
                    "limit "
                    "{:.2f} GB.\n"
                    "  Suggestion: Increase max_memory_gb or reduce "
                    "parameter "
                    "searchranges.",
                    f_start, current_f_end, nbins, max_memory_gb));
            }

            // Finalize this chunk
            const double nominal_start = best_f_start;
            const double nominal_end   = current_f_end;
            const double nominal_width = nominal_end - nominal_start;
            const double actual_start  = nominal_start * (1.0 - max_drift);
            const double actual_end    = nominal_end * (1.0 + max_drift);
            const double actual_width  = actual_end - actual_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            auto chunk_cfg = m_base_cfg.get_updated_config(
                nbins, eta, actual_start, actual_end);
            plans::FFAPlan<FoldType> chunk_plan(chunk_cfg);
            const double chunk_memory_gb =
                chunk_plan.get_buffer_memory_usage() +
                chunk_plan.get_coord_memory_usage();

            // Store config and stats
            m_cfgs.push_back(chunk_cfg);
            m_chunk_stats.push_back(
                coord::FFAChunkStats{.nominal_f_start  = nominal_start,
                                     .nominal_f_end    = nominal_end,
                                     .actual_f_start   = actual_start,
                                     .actual_f_end     = actual_end,
                                     .nominal_width    = nominal_width,
                                     .actual_width     = actual_width,
                                     .total_memory_gb  = chunk_memory_gb,
                                     .overlap_fraction = overlap_fraction});
            spdlog::debug("Chunk: nominal=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "actual=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "overlap={:.1f}%, mem={:.2f} GB",
                          nominal_start, nominal_end, nominal_width,
                          actual_start, actual_end, actual_width,
                          overlap_fraction * 100.0, chunk_memory_gb);
            max_buffer_size =
                std::max(max_buffer_size, chunk_plan.get_buffer_size());
            max_coord_size =
                std::max(max_coord_size, chunk_plan.get_coord_size());
            max_ncoords =
                std::max(max_ncoords, chunk_plan.get_ncoords().back());
            max_ffa_levels =
                std::max(max_ffa_levels, chunk_plan.get_n_levels());
            // Move to next chunk (going backwards in frequency)
            current_f_end = best_f_start;
        }
    }

    void log_planning_summary() const {
        if (m_chunk_stats.empty()) {
            return;
        }

        // Compute aggregate statistics
        double total_nominal_width = 0.0;
        double total_actual_width  = 0.0;
        double max_memory          = 0.0;
        double avg_memory          = 0.0;
        double max_overlap_pct     = 0.0;
        double avg_overlap_pct     = 0.0;

        for (const auto& stat : m_chunk_stats) {
            total_nominal_width += stat.nominal_width;
            total_actual_width += stat.actual_width;
            max_memory = std::max(max_memory, stat.total_memory_gb);
            avg_memory += stat.total_memory_gb;
            max_overlap_pct =
                std::max(max_overlap_pct, stat.overlap_fraction * 100.0);
            avg_overlap_pct += stat.overlap_fraction * 100.0;
        }
        avg_memory /= static_cast<float>(m_chunk_stats.size());
        avg_overlap_pct /= static_cast<double>(m_chunk_stats.size());

        const double redundancy_factor =
            total_actual_width / total_nominal_width;
        if (m_use_gpu) {
            spdlog::info("FFARegionPlanner - GPU Summary:");
        } else {
            spdlog::info("FFARegionPlanner - CPU Summary:");
        }
        spdlog::info("  Total chunks: {}", m_chunk_stats.size());
        spdlog::info(
            "  Nominal coverage: {:.3f} Hz, Actual coverage: {:.3f} Hz",
            total_nominal_width, total_actual_width);
        spdlog::info("  Redundancy factor: {:.3f}x ({:.1f}% extra computation)",
                     redundancy_factor, (redundancy_factor - 1.0) * 100.0);
        spdlog::info("  Overlap: avg={:.1f}%, max={:.1f}%", avg_overlap_pct,
                     max_overlap_pct);
        spdlog::info("  Memory per chunk: avg={:.2f} GB, max={:.2f} GB",
                     avg_memory, max_memory);
        spdlog::info("  Freq Sweep allocated: {:.2f} GB (limit: {:.2f} GB)",
                     m_stats.get_freq_sweep_memory_usage(),
                     m_base_cfg.get_max_process_memory_gb());
    }

}; // End FFARegionPlanner::Impl definition

// --- Definitions for FFARegionStats ---
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(SizeType max_buffer_size,
                                         SizeType max_coord_size,
                                         SizeType max_ncoords,
                                         SizeType max_ffa_levels,
                                         SizeType n_widths,
                                         SizeType n_params,
                                         SizeType n_samps,
                                         SizeType max_passing_candidates,
                                         bool use_gpu)
    : m_impl(std::make_unique<Impl>(max_buffer_size,
                                    max_coord_size,
                                    max_ncoords,
                                    max_ffa_levels,
                                    n_widths,
                                    n_params,
                                    n_samps,
                                    max_passing_candidates,
                                    use_gpu)) {}
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::~FFARegionStats() = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(FFARegionStats&&) noexcept = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>&
FFARegionStats<FoldType>::operator=(FFARegionStats&&) noexcept = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(const FFARegionStats& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>&
FFARegionStats<FoldType>::operator=(const FFARegionStats& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_buffer_size() const noexcept {
    return m_impl->max_buffer_size;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_coord_size() const noexcept {
    return m_impl->max_coord_size;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_ncoords() const noexcept {
    return m_impl->max_ncoords;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_ffa_levels() const noexcept {
    return m_impl->get_max_ffa_levels();
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_buffer_size_time() const noexcept {
    return m_impl->get_max_buffer_size_time();
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_scores_size() const noexcept {
    return m_impl->get_max_scores_size();
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_write_param_sets_size() const noexcept {
    return m_impl->get_write_param_sets_size();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_buffer_memory_usage() const noexcept {
    return m_impl->get_buffer_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_coord_memory_usage() const noexcept {
    return m_impl->get_coord_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_extra_memory_usage() const noexcept {
    return m_impl->get_extra_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_freq_sweep_memory_usage() const noexcept {
    return m_impl->get_freq_sweep_memory_usage();
}

// --- Definitions for FFARegionPlanner ---
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::FFARegionPlanner(
    const search::PulsarSearchConfig& cfg, bool use_gpu)
    : m_impl(std::make_unique<Impl>(cfg, use_gpu)) {}
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::~FFARegionPlanner() = default;
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::FFARegionPlanner(FFARegionPlanner&&) noexcept =
    default;
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>& FFARegionPlanner<FoldType>::operator=(
    FFARegionPlanner<FoldType>&&) noexcept = default;
template <SupportedFoldType FoldType>
const std::vector<search::PulsarSearchConfig>&
FFARegionPlanner<FoldType>::get_cfgs() const noexcept {
    return m_impl->get_cfgs();
}
template <SupportedFoldType FoldType>
SizeType FFARegionPlanner<FoldType>::get_nregions() const noexcept {
    return m_impl->get_nregions();
}
template <SupportedFoldType FoldType>
const FFARegionStats<FoldType>&
FFARegionPlanner<FoldType>::get_stats() const noexcept {
    return m_impl->get_stats();
}

template class FFARegionStats<float>;
template class FFARegionStats<ComplexType>;
template class FFARegionPlanner<float>;
template class FFARegionPlanner<ComplexType>;

} // namespace loki::regions