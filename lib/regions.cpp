#include "loki/algorithms/regions.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils.hpp"

namespace loki::regions {

namespace {
// Cached evaluation of a candidate chunk. Holds the plan-derived sizes
// so the chunk can be finalized without re-simulating.
struct ChunkEval {
    search::PulsarSearchConfig cfg;
    SizeType buffer_size;
    SizeType coord_size;
    SizeType ncoords;
    SizeType ffa_levels;
    SizeType scores_scratch;     // ncoords * n_widths for THIS chunk's nbins
    double chunk_only_memory_gb; // For logging/stats.
    double allocated_memory_gb;  // With accumulated maxima
};

// Running maxima over all accepted chunks. The shared workspace and scratch
// buffers are sized from these, so every probe must be tested against them
// rather than against the chunk in isolation.
struct PlanMaxima {
    SizeType buffer_size{};
    SizeType coord_size{};
    SizeType ncoords{};
    SizeType ffa_levels{};
    SizeType scores_scratch{};

    void absorb(const PlanMaxima& other) noexcept {
        buffer_size    = std::max(buffer_size, other.buffer_size);
        coord_size     = std::max(coord_size, other.coord_size);
        ncoords        = std::max(ncoords, other.ncoords);
        ffa_levels     = std::max(ffa_levels, other.ffa_levels);
        scores_scratch = std::max(scores_scratch, other.scores_scratch);
    }
};

PlanMaxima to_maxima(const ChunkEval& eval) noexcept {
    return {
        .buffer_size    = eval.buffer_size,
        .coord_size     = eval.coord_size,
        .ncoords        = eval.ncoords,
        .ffa_levels     = eval.ffa_levels,
        .scores_scratch = eval.scores_scratch,
    };
}
} // namespace

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
    error_check::check_greater_equal(nbins_min, SizeType{2},
                                     "nbins_min must be >= 2.");
    error_check::check_greater(eta_min, 0.0, "eta_min must be positive.");
    error_check::check_greater(
        static_cast<double>(nbins_min), eta_min,
        "eta_min must be < nbins_min (duty cycle rho < 1).");
    error_check::check_greater(octave_scale, 1.0,
                               "octave_scale must be strictly > 1.0.");
    error_check::check_greater_equal(nbins_max, nbins_min,
                                     "nbins_max must be >= nbins_min.");
    error_check::check_greater(
        p_min, 2.0 * tsamp,
        std::format("p_min ({:.9f} s) must exceed 2*tsamp ({:.9f} s); the "
                    "requested f_max = {:.6f} Hz meets or exceeds the "
                    "Nyquist frequency ({:.6f} Hz) for tsamp={:.9f} s. "
                    "Lower f_max (raise p_min) or use a finer tsamp.",
                    p_min, 2.0 * tsamp, 1.0 / p_min, 1.0 / (2.0 * tsamp),
                    tsamp));

    // t_W = max(P_min / b_min, t_s). Finest physical bin width in the plan.
    // rho = eta_0 / b_min. Held fixed; eta_k = rho * N_{b,k}.
    const double t_w = std::max(p_min / static_cast<double>(nbins_min), tsamp);
    const double rho = eta_min / static_cast<double>(nbins_min);

    std::vector<coord::FFARegion> regions;
    double p_cur_low = p_min;
    while (p_cur_low < p_max) {
        auto nbins_k = static_cast<SizeType>(std::round(
            std::min(p_cur_low / t_w, static_cast<double>(nbins_max))));
        nbins_k      = std::clamp(nbins_k, SizeType{2}, nbins_max);
        if (nbins_k >= nbins_max) {
            regions.push_back({
                .f_start = 1.0 / p_max,
                .f_end   = 1.0 / p_cur_low,
                .nbins   = nbins_max,
                .eta     = rho * static_cast<double>(nbins_max),
            });
            break;
        }
        const double p_cur_high = std::min(p_cur_low * octave_scale, p_max);
        error_check::check_greater(
            p_cur_high, p_cur_low,
            std::format("Period grid stalled at P={:.17g} s with "
                        "octave_scale={:.17g}.",
                        p_cur_low, octave_scale));
        regions.push_back({
            .f_start = 1.0 / p_cur_high,
            .f_end   = 1.0 / p_cur_low,
            .nbins   = nbins_k,
            .eta     = rho * static_cast<double>(nbins_k),
        });
        p_cur_low = p_cur_high;
    }
    return regions;
}

// --- FFARegionStats Implementation ---

FFARegionStats::FFARegionStats(SizeType max_buffer_size,
                               SizeType max_coord_size,
                               SizeType max_ncoords,
                               SizeType max_ffa_levels,
                               SizeType max_scores_scratch,
                               SizeType n_params,
                               SizeType n_samps,
                               SizeType max_passing_candidates,
                               bool use_fourier,
                               bool use_gpu)
    : m_max_buffer_size(max_buffer_size),
      m_max_coord_size(max_coord_size),
      m_max_ncoords(max_ncoords),
      m_max_ffa_levels(max_ffa_levels),
      m_max_scores_scratch(max_scores_scratch),
      m_n_params(n_params),
      m_n_samps(n_samps),
      m_max_passing_candidates(max_passing_candidates),
      m_use_fourier(use_fourier),
      m_use_gpu(use_gpu) {}

SizeType FFARegionStats::get_max_buffer_size_time() const noexcept {
    return m_use_fourier ? 2 * m_max_buffer_size : m_max_buffer_size;
}
SizeType FFARegionStats::get_max_scores_scratch_size() const noexcept {
    return m_max_scores_scratch;
}
SizeType FFARegionStats::get_max_candidates() const noexcept {
    return m_max_passing_candidates;
}
SizeType FFARegionStats::get_write_param_sets_size() const noexcept {
    return kFFAFreqSweepWriteBatchSize * (m_n_params + 1); // includes width
}
float FFARegionStats::get_buffer_memory_usage() const noexcept {
    const auto fold_bytes = m_use_fourier ? sizeof(ComplexType) : sizeof(float);
    return static_cast<float>(2 * m_max_buffer_size * fold_bytes) /
           static_cast<float>(1ULL << 30U);
}
float FFARegionStats::get_coord_memory_usage() const noexcept {
    SizeType coord_size;
    if (m_n_params == 1) {
        coord_size = m_max_coord_size * sizeof(coord::FFACoordFreq);
    } else {
        coord_size = m_max_coord_size * sizeof(coord::FFACoord);
    }
    return static_cast<float>(coord_size) / static_cast<float>(1ULL << 30U);
}
float FFARegionStats::get_extra_memory_usage() const noexcept {
    // Per-chunk raw score scratch, plus the fixed-capacity candidate
    // accumulator (score + local index + region id), plus the write staging.
    constexpr SizeType kBytesPerCandidate =
        sizeof(float) + (2 * sizeof(uint32_t));
    return static_cast<float>((get_write_param_sets_size() * sizeof(double)) +
                              (get_max_scores_scratch_size() * sizeof(float)) +
                              (get_max_candidates() * kBytesPerCandidate)) /
           static_cast<float>(1ULL << 30U);
}
float FFARegionStats::get_device_extra_memory_usage() const noexcept {
    // ts_e_d + ts_v_d + per-chunk scores_d and passing_indices_d scratch
    // (widths_d is negligible). The candidate accumulator lives on the host.
    return static_cast<float>(
               (((2 * m_n_samps) + get_max_scores_scratch_size()) *
                sizeof(float)) +
               (get_max_scores_scratch_size() * sizeof(uint32_t))) /
           static_cast<float>(1ULL << 30U);
}
float FFARegionStats::get_cpu_memory_usage() const noexcept {
    return get_buffer_memory_usage() + get_coord_memory_usage() +
           get_extra_memory_usage();
}
// Use this for GPU memory usage calculation
float FFARegionStats::get_device_memory_usage() const noexcept {
    return get_device_extra_memory_usage() + get_buffer_memory_usage() +
           get_coord_memory_usage();
}
float FFARegionStats::get_freq_sweep_memory_usage() const noexcept {
    if (m_use_gpu) {
        return get_device_memory_usage();
    }
    return get_cpu_memory_usage();
}

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
    const FFARegionStats& get_stats() const noexcept { return m_stats; }
    const std::vector<coord::FFAChunkStats>& get_chunk_stats() const noexcept {
        return m_chunk_stats;
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    bool m_use_gpu;

    std::vector<search::PulsarSearchConfig> m_cfgs;
    FFARegionStats m_stats{
        0, 0, 0, 0, 0, 0, 0, 0, m_base_cfg.get_use_fourier(), m_use_gpu};
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
        m_cfgs.clear();
        m_chunk_stats.clear();

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
        if (max_drift < 0.0 || max_drift >= 1.0) {
            throw std::runtime_error(std::format(
                "FFARegionPlanner: max_drift must be in [0, 1); got {:.6f}. "
                "This typically indicates a parameter range that is too large "
                "for the observation duration.",
                max_drift));
        }
        if (max_drift > 0 && m_base_cfg.get_nparams() > 1) {
            spdlog::info("Drift-aware chunking: max_drift={:.6f} "
                         "({:.4f}%) for tobs={:.1f}s, n_params={}",
                         max_drift, max_drift * 100.0, m_base_cfg.get_tobs(),
                         m_base_cfg.get_nparams());
        }
        PlanMaxima maxima;
        for (const auto& region : ffa_regions) {
            subdivide_region_by_memory(region.f_start, region.f_end,
                                       region.nbins, region.eta, max_drift,
                                       maxima);
        }
        m_stats = FFARegionStats(
            maxima.buffer_size, maxima.coord_size, maxima.ncoords,
            maxima.ffa_levels, maxima.scores_scratch, m_base_cfg.get_nparams(),
            m_base_cfg.get_nsamps(), m_base_cfg.get_max_passing_candidates(),
            m_base_cfg.get_use_fourier(), m_use_gpu);

        // Log summary statistics
        log_planning_summary();
    }

    void subdivide_region_by_memory(double f_start,
                                    double f_end,
                                    SizeType nbins,
                                    double eta,
                                    double max_drift,
                                    PlanMaxima& maxima) {
        if (f_end <= f_start) {
            return; // Empty or inverted region; nothing to do.
        }
        if (f_start * (1.0 - max_drift) <= 0.0) {
            throw std::runtime_error(std::format(
                "FFARegionPlanner: drift expansion would push f_start "
                "({:.6f} Hz) to a non-positive frequency (drift={:.6f}). "
                "Reduce parameter ranges or raise f_min.",
                f_start, max_drift));
        }
        // For GPU, this is the device memory limit. For CPU, this is the
        // process memory limit.
        const auto max_memory_gb = m_base_cfg.get_max_process_memory_gb();
        // Reserve some headroom for OS, Python interpreter, and rounding
        // errors
        constexpr double kSafetyMarginGB = 0.5; // 500 MB safety margin
        const auto effective_limit_gb    = max_memory_gb - kSafetyMarginGB;
        if (effective_limit_gb <= 0.0) {
            throw std::runtime_error(std::format(
                "FFARegionPlanner: max_process_memory_gb ({:.2f} GB) must "
                "exceed the safety margin ({:.2f} GB).",
                max_memory_gb, kSafetyMarginGB));
        }

        constexpr double kRelativeTolerance = 1.0e-4;
        // Convergence tolerance for the frequency-width *bisection search*
        // (Hz). This only controls how precisely we locate the largest
        // fitting chunk width.
        constexpr double kBisectionToleranceHz = 1.0e-2;
        // The minimal chunk size (Hz) below which the planner will never
        // subdivide further.
        constexpr double kMinChunkWidthHz     = 1.0e-2;
        constexpr SizeType kMaxBisectionSteps = 50;

        const double region_span = f_end - f_start;
        const double boundary_tolerance =
            std::max(kBisectionToleranceHz, kRelativeTolerance * region_span);

        auto stats_for = [&](const PlanMaxima& m) {
            return FFARegionStats{m.buffer_size,
                                  m.coord_size,
                                  m.ncoords,
                                  m.ffa_levels,
                                  m.scores_scratch,
                                  m_base_cfg.get_nparams(),
                                  m_base_cfg.get_nsamps(),
                                  m_base_cfg.get_max_passing_candidates(),
                                  m_base_cfg.get_use_fourier(),
                                  m_use_gpu};
        };

        // The single point of contact with the (currently cheap, future
        // expensive) memory model. Future surrogate-model work plugs in here.
        auto evaluate_chunk = [&](double nominal_start,
                                  double nominal_end) -> ChunkEval {
            if (nominal_end <= nominal_start) {
                throw std::runtime_error(
                    std::format("FFARegionPlanner: zero/negative-width chunk "
                                "[{:.8f}, {:.8f}] Hz.",
                                nominal_start, nominal_end));
            }
            const double actual_start = nominal_start * (1.0 - max_drift);
            const double actual_end   = nominal_end * (1.0 + max_drift);
            if (actual_start <= 0.0 || actual_end <= actual_start) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: invalid drift-expanded range "
                    "[{:.8f}, {:.8f}] Hz from nominal [{:.8f}, {:.8f}] Hz.",
                    actual_start, actual_end, nominal_start, nominal_end));
            }

            auto cfg = m_base_cfg.get_updated_config(nbins, eta, actual_start,
                                                     actual_end);
            plans::FFAPlan<FoldType> plan(cfg);

            const SizeType buf   = plan.get_buffer_size();
            const SizeType coord = plan.get_coord_size();
            const SizeType nc    = plan.get_ncoords().back();
            const SizeType lv    = plan.get_n_levels();
            // Width count follows nbins, which is fixed within a region but
            // differs between regions, so it must come from the chunk config.
            const SizeType scratch = nc * cfg.get_n_scoring_widths();

            const PlanMaxima chunk_only{
                .buffer_size    = buf,
                .coord_size     = coord,
                .ncoords        = nc,
                .ffa_levels     = lv,
                .scores_scratch = scratch,
            };
            const double chunk_only_gb =
                stats_for(chunk_only).get_freq_sweep_memory_usage();

            // Workspace is sized by max over all chunks; fit must use that.
            PlanMaxima allocated = maxima;
            allocated.absorb(chunk_only);
            const double allocated_gb =
                stats_for(allocated).get_freq_sweep_memory_usage();

            return ChunkEval{
                .cfg                  = std::move(cfg),
                .buffer_size          = buf,
                .coord_size           = coord,
                .ncoords              = nc,
                .ffa_levels           = lv,
                .scores_scratch       = scratch,
                .chunk_only_memory_gb = chunk_only_gb,
                .allocated_memory_gb  = allocated_gb,
            };
        };

        auto fits = [&](const ChunkEval& e) {
            return e.allocated_memory_gb <= effective_limit_gb;
        };

        // Bisect over chunk *width* (anchored at current_f_end) to find the
        // largest fitting width.
        auto find_boundary_bisect =
            [&](double current_f_end, double min_width, ChunkEval min_eval,
                double max_width) -> std::pair<double, ChunkEval> {
            double lo_width = min_width; // known fits
            double hi_width = max_width; // known fails
            ChunkEval best  = std::move(min_eval);

            for (SizeType step = 0; step < kMaxBisectionSteps &&
                                    (hi_width - lo_width) > boundary_tolerance;
                 ++step) {

                const double mid_width   = std::midpoint(lo_width, hi_width);
                const double probe_start = current_f_end - mid_width;
                auto probe = evaluate_chunk(probe_start, current_f_end);

                if (fits(probe)) {
                    lo_width = mid_width;
                    best     = std::move(probe);
                } else {
                    hi_width = mid_width;
                }
            }
            return {current_f_end - lo_width, std::move(best)};
        };

        // Find the largest fitting chunk ending at `current_f_end`.
        auto find_largest_fitting_chunk =
            [&](double current_f_end) -> std::pair<double, ChunkEval> {
            const double remaining_width = current_f_end - f_start;

            // Fast path: the entire remaining range fits.
            auto full_eval = evaluate_chunk(f_start, current_f_end);
            if (fits(full_eval)) {
                return {f_start, std::move(full_eval)};
            }
            // Minimum-viable chunk width: the smallest chunk we will ever
            // produce. If even this doesn't fit, the search is too memory-bound
            // for FFA to be a valid algorithm here.
            const double min_chunk_width =
                std::min(remaining_width, kMinChunkWidthHz);
            const double min_probe_start = current_f_end - min_chunk_width;
            auto min_eval = evaluate_chunk(min_probe_start, current_f_end);
            if (!fits(min_eval)) {
                const auto drift_span = (current_f_end * (1.0 + max_drift)) -
                                        (min_probe_start * (1.0 - max_drift));
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: Cannot fit minimum viable chunk at "
                    "the top of [{:08.3f}, {:08.3f}] Hz.\n"
                    "  Nominal range: {:.3f} Hz, drift-expanded span: {:.3f} "
                    "Hz\n"
                    "  Required memory: {:.2f} GB, Available: {:.2f} GB\n"
                    "  Suggestion: Increase max_memory_gb or reduce "
                    "parameter search ranges.",
                    f_start, current_f_end, min_chunk_width, drift_span,
                    min_eval.allocated_memory_gb, effective_limit_gb));
            }

            return find_boundary_bisect(current_f_end, min_chunk_width,
                                        std::move(min_eval), remaining_width);
        };

        // Sliver merge: if the bisection produced a tiny low-frequency
        // remainder, attempt to absorb it into the current chunk.
        constexpr double kSliverWidthFactor = 4.0; // sliver up to 4x tolerance
        auto try_absorb_sliver =
            [&](double current_f_end, double nominal_start,
                ChunkEval bisect_eval) -> std::pair<double, ChunkEval> {
            const double remainder_width = nominal_start - f_start;
            const double sliver_threshold =
                kSliverWidthFactor * boundary_tolerance;

            if (remainder_width <= 0.0 || remainder_width > sliver_threshold) {
                return {nominal_start, std::move(bisect_eval)};
            }

            // Evaluate the merged chunk. If it fits, prefer it.
            auto merged_eval = evaluate_chunk(f_start, current_f_end);
            if (fits(merged_eval)) {
                return {f_start, std::move(merged_eval)};
            }
            return {nominal_start, std::move(bisect_eval)};
        };

        // Main covering loop
        double current_f_end = f_end;
        while (current_f_end > f_start) {
            auto [nominal_start, eval] =
                find_largest_fitting_chunk(current_f_end);
            std::tie(nominal_start, eval) = try_absorb_sliver(
                current_f_end, nominal_start, std::move(eval));
            // Defensive: bisection is guaranteed to make progress because the
            // minimum-width probe fits. If this triggers, there's a logic bug.
            if (nominal_start >= current_f_end) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: no progress in [{:08.3f}, {:08.3f}] Hz "
                    "with nbins={}. This is a planner bug.",
                    f_start, current_f_end, nbins));
            }

            const double nominal_end   = current_f_end;
            const double nominal_width = nominal_end - nominal_start;
            const double actual_start  = nominal_start * (1.0 - max_drift);
            const double actual_end    = nominal_end * (1.0 + max_drift);
            const double actual_width  = actual_end - actual_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            m_cfgs.push_back(eval.cfg);
            m_chunk_stats.push_back(coord::FFAChunkStats{
                .nominal_f_start  = nominal_start,
                .nominal_f_end    = nominal_end,
                .actual_f_start   = actual_start,
                .actual_f_end     = actual_end,
                .nominal_width    = nominal_width,
                .actual_width     = actual_width,
                .total_memory_gb  = eval.chunk_only_memory_gb,
                .overlap_fraction = overlap_fraction,
            });

            spdlog::debug("Chunk: nominal=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "actual=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "overlap={:.1f}%, mem={:.2f} GB",
                          nominal_start, nominal_end, nominal_width,
                          actual_start, actual_end, actual_width,
                          overlap_fraction * 100.0, eval.chunk_only_memory_gb);

            maxima.absorb(to_maxima(eval));

            current_f_end = nominal_start;
        }
    }

    void log_planning_summary() const {
        if (m_chunk_stats.empty()) {
            return;
        }

        // Aggregate metrics. We need both bulk-behavior summaries (median,
        // p95) and outlier callouts (max, count of slivers) because a single
        // sliver can skew mean/max and mislead readers.
        const SizeType n = m_chunk_stats.size();
        std::vector<double> overlaps_pct;
        std::vector<double> memories_gb;
        overlaps_pct.reserve(n);
        memories_gb.reserve(n);

        double total_nominal_width = 0.0;
        double total_actual_width  = 0.0;

        for (const auto& s : m_chunk_stats) {
            overlaps_pct.push_back(s.overlap_fraction * 100.0);
            memories_gb.push_back(s.total_memory_gb);
            total_nominal_width += s.nominal_width;
            total_actual_width += s.actual_width;
        }

        auto percentile = [](std::vector<double>& v, double p) {
            // Linear-interpolation percentile on a sorted copy. p in [0, 100].
            if (v.empty()) {
                return 0.0;
            }
            std::ranges::sort(v);
            const double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
            const auto lo     = static_cast<SizeType>(std::floor(rank));
            const auto hi     = static_cast<SizeType>(std::ceil(rank));
            const double frac = rank - static_cast<double>(lo);
            return v[lo] + (frac * (v[hi] - v[lo]));
        };

        const double overlap_p50 = percentile(overlaps_pct, 50.0);
        const double overlap_p95 = percentile(overlaps_pct, 95.0);
        const double overlap_max =
            overlaps_pct.back(); // sorted by percentile()
        const double memory_p50 = percentile(memories_gb, 50.0);
        const double memory_p95 = percentile(memories_gb, 95.0);
        const double memory_max = memories_gb.back();

        const double redundancy_factor =
            total_actual_width / total_nominal_width;

        spdlog::info("FFARegionPlanner - {} Summary:",
                     m_use_gpu ? "GPU" : "CPU");
        spdlog::info("  Total chunks: {}", n);
        spdlog::info("  Coverage: nominal={:.3f} Hz, actual={:.3f} Hz "
                     "(redundancy {:.2f}x, +{:.1f}% computation)",
                     total_nominal_width, total_actual_width, redundancy_factor,
                     (redundancy_factor - 1.0) * 100.0);
        spdlog::info("  Overlap per chunk: p50={:.1f}%, p95={:.1f}%, "
                     "max={:.1f}%",
                     overlap_p50, overlap_p95, overlap_max);
        spdlog::info("  Memory per chunk: p50={:.2f} GB, p95={:.2f} GB, "
                     "max={:.2f} GB",
                     memory_p50, memory_p95, memory_max);
        spdlog::info("  Freq Sweep allocated: {:.2f} GB (limit: {:.2f} GB)",
                     m_stats.get_freq_sweep_memory_usage(),
                     m_base_cfg.get_max_process_memory_gb());
    }

}; // End FFARegionPlanner::Impl definition

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
const FFARegionStats& FFARegionPlanner<FoldType>::get_stats() const noexcept {
    return m_impl->get_stats();
}

template class FFARegionPlanner<float>;
template class FFARegionPlanner<ComplexType>;

} // namespace loki::regions