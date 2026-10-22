#include "loki/algorithms/ep_regions.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <highfive/H5File.hpp>
#include <spdlog/spdlog.h>

#include "loki/algorithms/regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_chunking.hpp"
#include "lib/algorithms/ep_memory.hpp"
#include "lib/algorithms/ep_memory_cuda.hpp"
#include "lib/algorithms/planner_memory.hpp"
#include "lib/algorithms/prune_engine.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"
#include "lib/search/ep_support.hpp"

namespace loki::algorithms {

namespace {

constexpr std::string_view kCacheVersion = "1.18.0";

/// Threshold-scheme simulation of the planner. The seeds are fixed: the
/// scheme sets the chunking, so plans of one configuration must not change
/// from run to run. CUDA results are reproducible per (seed, batch_size), so
/// the batch size is part of the cache key.
constexpr SizeType kDtsNTrials = 1024;
// Bins from min_pd to 1: the resolution of the Pd-cost trade-off (see
// DynamicThresholdScheme's constructor).
constexpr SizeType kDtsNProbs = 40;
// Mode "shared" uses the guided beam (SharedOptions defaults): kDtsGuess,
// kDtsBeamWidth and kDtsThresNeigh only fill the engine's signature.
constexpr std::string_view kDtsMode  = "shared";
constexpr std::string_view kDtsGuess = "window";
// Grid of the evaluation-only engines, which never search.
constexpr float kDtsProbMin        = 0.05F;
constexpr SizeType kDtsNThresholds = 100;
constexpr float kDtsBeamWidth      = 0.7F;
constexpr SizeType kDtsTrialsStart = 1;
constexpr SizeType kDtsThresNeigh  = 10;
constexpr uint64_t kDtsSeed        = 0x10C1;
// A schedule whose measured Pd is below this fraction of min_pd is warned
// about.
constexpr float kDtsPdTolerance = 0.9F;
// Largest Pd the threshold search can aim at.
constexpr float kDtsMaxTarget = 0.999F;
// Weight of the survivors of the last stage in the cost: they go on to the
// next processing step.
constexpr float kDtsWTerminal = 1.0F;
// Memory and runtime control of the threshold design. The search pools grow
// with cells x trials x nbins, so beyond kDtsNativeMaxBins bins (and whenever
// the native pools do not fit the limit) the search runs at a small nbins
// (kDtsNbinsSim) and its schedule is re-fitted at the true nbins with samples
// whose size is bounded by the memory limit (DynamicThresholdScheme's
// refit_scheme). The maximum over boxcar templates has a heavier noise tail at
// a larger nbins (about nbins times more independent phase positions), so a
// schedule searched at 64 bins is too permissive at 256-1024 bins: its cost
// per detection is 1.1-3.3 times a native search's there (measured with
// bench/scripts/schemes/nbins_proxy_study.py). The refit removes most of it
// (common upward shift, 1.0-1.05 times in the same tests) and always delivers
// the target Pd at the true nbins. The design is run once per plan (the plan
// cache keeps it), so runtime matters less than never running out of memory.
constexpr SizeType kDtsNativeMaxBins = 128;
constexpr SizeType kDtsNbinsSim      = 64;
// Fraction of the memory limit the refit's samples may use, and their range.
constexpr double kDtsRefitMemoryFraction = 0.25;
constexpr double kDtsRefitMemoryMaxGB    = 4.0;
constexpr SizeType kDtsRefitBankMin      = SizeType{1} << 13U;
constexpr SizeType kDtsRefitNoiseMin     = SizeType{1} << 11U;
// Search budgets tried in order until the pools fit: {trials, bins above the
// target}. The first is the validated default.
constexpr std::array<std::pair<SizeType, SizeType>, 5> kDtsSearchLadder = {
    {{1024, 40}, {512, 40}, {512, 20}, {256, 20}, {128, 16}}};
// A reference pulse must span at least this many simulated bins.
constexpr double kDtsMinPulseBins = 4.0;
// Bumped with any change of the threshold pipeline.
constexpr std::string_view kDtsAlgorithm = "shared-per-region";
constexpr SizeType kDtsBatchSize         = 256;

double calculate_max_drift(const search::PulsarSearchConfig& cfg) {
    if (cfg.get_nparams() <= 1) {
        return 0.0;
    }
    const auto param_limits = cfg.get_param_limits();
    const auto t_half       = cfg.get_tobs() / 2.0;
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
    if (cfg.get_nparams() == 4) {
        const auto max_snap  = std::max(std::abs(param_limits[0].min),
                                        std::abs(param_limits[0].max));
        const auto max_jerk  = std::max(std::abs(param_limits[1].min),
                                        std::abs(param_limits[1].max));
        const auto max_accel = std::max(std::abs(param_limits[2].min),
                                        std::abs(param_limits[2].max));
        const auto drift     = (max_accel * t_half) +
                               (max_jerk * t_half * t_half / 2.0) +
                               (max_snap * t_half * t_half * t_half / 6.0);
        return drift / utils::kCval;
    }
    throw std::runtime_error(
        "Unsupported number of parameters for drift calculation");
}

int resolve_n_workers(int nthreads, std::optional<SizeType> n_workers) {
    const int max_workers = std::max(1, nthreads);
    if (!n_workers) {
        return max_workers;
    }
    return static_cast<int>(std::clamp<SizeType>(
        *n_workers, 1, static_cast<SizeType>(max_workers)));
}

/// The memory policy of @p exec (docs/memory.md): host memory with
/// n_workers pruning threads on the CPU, device memory with one serial worker
/// on the GPU.
detail::EPMemoryContext
make_memory_context(const search::PulsarSearchConfig& cfg,
                    const PruneRFIConfig& rfi_config,
                    std::optional<SizeType> n_workers,
                    const Exec& exec) {
    search::detail::check_ep_supported(cfg, "EPRegionPlanner");
    detail::EPMemoryContext ctx{
        .nparams   = cfg.get_nparams(),
        .nsamps    = cfg.get_nsamps(),
        .n_workers = resolve_n_workers(cfg.get_nthreads(), n_workers),
        .harvest   = detail::EPHarvestBound::from(rfi_config),
    };
    if (exec.backend == Backend::kCPU) {
        return ctx;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        if (rfi_config.is_active()) {
            loki::detail::throw_unimplemented("EPRegionPlanner (rfi_config)",
                                              exec.backend);
        }
        ctx.n_workers         = 1;
        ctx.batch_size        = detail::kEPBatchSizeCuda;
        ctx.harvest           = {};
        ctx.kind              = detail::EPMemoryKind::kCuda;
        ctx.device            = exec.device;
        ctx.cub_scratch_bytes = &detail::ep_cuda_cub_scratch_bytes;
        ctx.share_ffa_fold    = true;
        return ctx;
    }
#endif
    loki::detail::throw_unavailable("EPRegionPlanner", exec.backend);
}

/// Memory the plan is fitted to. The CPU uses max_process_memory_gb. A GPU
/// sweep follows the FFA GPU sweep: the smaller of that and the free device
/// memory less detail::kDeviceReserveGB, and host memory is never checked.
double resolve_memory_limit_gb(const search::PulsarSearchConfig& cfg,
                               const Exec& exec) {
    const double user_limit = cfg.get_max_process_memory_gb();
    if (exec.backend == Backend::kCPU) {
        return user_limit;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        const double free_gb   = detail::ep_cuda_free_memory_gb(exec.device);
        const double usable_gb = free_gb - detail::kDeviceReserveGB;
        spdlog::info("EPRegionPlanner: GPU {} has {:.2f} GB free, {:.2f} GB "
                     "usable for the sweep (user limit {:.2f} GB)",
                     exec.device, free_gb, usable_gb, user_limit);
        if (usable_gb < 1.0) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: insufficient GPU memory: {:.2f} GB free, "
                "need at least {:.2f} GB",
                free_gb, 1.0 + detail::kDeviceReserveGB));
        }
        return std::min(user_limit, usable_gb);
    }
#endif
    loki::detail::throw_unavailable("EPRegionPlanner", exec.backend);
}

} // namespace

SizeType
ep_sweep_n_workers(int nthreads,
                   std::optional<SizeType> n_runs,
                   const std::optional<std::vector<SizeType>>& ref_segs) {
    return detail::compute_ep_n_workers(nthreads, n_runs, ref_segs);
}

template <SupportedFoldType FoldType> class EPRegionPlanner<FoldType>::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         float min_pd,
         std::string_view poly_basis,
         float ref_ducy,
         const std::optional<std::filesystem::path>& plan_cache_file,
         const PruneRFIConfig& rfi_config,
         std::optional<SizeType> n_workers,
         std::optional<SizeType> n_runs,
         const std::optional<std::vector<SizeType>>& ref_segs,
         const Exec& exec)
        : m_base_cfg(std::move(cfg)),
          m_min_pd(min_pd),
          m_poly_basis(poly_basis),
          m_ref_ducy(ref_ducy),
          m_exec(exec),
          m_n_runs(n_runs),
          m_ref_segs(ref_segs),
          m_memory(
              make_memory_context(m_base_cfg, rfi_config, n_workers, exec)),
          m_memory_limit_gb(resolve_memory_limit_gb(m_base_cfg, exec)) {
        if (!utils::is_finite(m_min_pd) || m_min_pd <= 0.0F ||
            m_min_pd > 1.0F) {
            throw std::invalid_argument(
                "EPRegionPlanner: min_pd must be in (0, 1]");
        }
        // Every coarse region uses the same FFA segmentation for this cfg.
        const plans::FFAPlan<FoldType> initial_plan(m_base_cfg);
        if (!m_n_runs && !m_ref_segs) {
            m_n_runs = initial_plan.get_nsegments().back();
        }
        (void)utils::determine_ref_segs(initial_plan.get_nsegments().back(),
                                        m_n_runs, m_ref_segs);
        if (plan_cache_file && std::filesystem::exists(*plan_cache_file)) {
            load_cache(*plan_cache_file);
        } else {
            plan_regions();
            if (plan_cache_file) {
                save_cache(*plan_cache_file);
            }
        }
    }

    [[nodiscard]] const std::vector<EPChunkConfig>&
    get_chunk_cfgs() const noexcept {
        return m_chunk_cfgs;
    }
    [[nodiscard]] SizeType get_nchunks() const noexcept {
        return m_chunk_cfgs.size();
    }
    [[nodiscard]] const EPRegionStats& get_stats() const noexcept {
        return m_stats;
    }

    void save_cache(const std::filesystem::path& filepath) const {
        if (filepath.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(filepath.parent_path(), ec);
        }
        HighFive::File file(filepath.string(), HighFive::File::Overwrite);

        // Header attributes for configuration validation
        file.createAttribute("ep_plan_cache_version",
                             std::string(kCacheVersion));
        file.createAttribute("ep_backend",
                             std::string(to_string(m_exec.backend)));
        file.createAttribute(
            "ep_memory_model_kind",
            std::string(m_memory.kind == detail::EPMemoryKind::kCuda ? "cuda"
                                                                     : "cpu"));
        // Recorded so a plan can be traced to the GPU it was measured on.
        // Loading does not compare it: the model does not depend on it.
        std::string device_arch = "host";
#ifdef LOKI_ENABLE_GPU
        if (m_exec.backend == loki::detail::kGPUBackend) {
            device_arch = detail::ep_cuda_device_arch(m_exec.device);
        }
#endif
        file.createAttribute("device_arch", device_arch);
        file.createAttribute("dts_batch_size", kDtsBatchSize);
        file.createAttribute("dts_algorithm", std::string(kDtsAlgorithm));
        file.createAttribute("dts_w_terminal", kDtsWTerminal);
        file.createAttribute("dts_mode", std::string(kDtsMode));
        file.createAttribute("ring_margin_z", detail::kEPRingMarginZ);
        file.createAttribute("dts_nbins_fallback", kDtsNbinsSim);
        file.createAttribute("dts_native_max_bins", kDtsNativeMaxBins);
        file.createAttribute("dts_min_pulse_bins", kDtsMinPulseBins);
        const plans::FFAPlan<FoldType> run_plan(m_base_cfg);
        file.createDataSet("ref_segs", utils::determine_ref_segs(
                                           run_plan.get_nsegments().back(),
                                           m_n_runs, m_ref_segs));
        file.createAttribute("nsamps", m_base_cfg.get_nsamps());
        file.createAttribute("tsamp", m_base_cfg.get_tsamp());
        file.createAttribute("f_min", m_base_cfg.get_f_min());
        file.createAttribute("f_max", m_base_cfg.get_f_max());
        file.createAttribute("nbins", m_base_cfg.get_nbins());
        file.createAttribute("eta", m_base_cfg.get_eta());
        file.createAttribute("nparams", m_base_cfg.get_nparams());
        file.createAttribute("ducy_max", m_base_cfg.get_ducy_max());
        file.createAttribute("wtsp", m_base_cfg.get_wtsp());
        file.createAttribute(
            "use_fourier", static_cast<uint8_t>(m_base_cfg.get_use_fourier()));
        file.createAttribute("max_process_memory_gb",
                             m_base_cfg.get_max_process_memory_gb());
        file.createAttribute("octave_scale", m_base_cfg.get_octave_scale());
        file.createAttribute("nbins_max", m_base_cfg.get_nbins_max());
        file.createAttribute("nbins_min_lossy_bf",
                             m_base_cfg.get_nbins_min_lossy_bf());
        file.createAttribute("bseg_brute", m_base_cfg.get_bseg_brute());
        file.createAttribute("bseg_ffa", m_base_cfg.get_bseg_ffa());
        file.createAttribute("prune_poly_order",
                             m_base_cfg.get_prune_poly_order());
        file.createAttribute(
            "use_conservative_tile",
            static_cast<uint8_t>(m_base_cfg.get_use_conservative_tile()));
        file.createAttribute("min_pd", m_min_pd);
        file.createAttribute("poly_basis", std::string(m_poly_basis));
        file.createAttribute("ref_ducy", m_ref_ducy);
        file.createAttribute("target_snr", m_base_cfg.get_target_snr());
        file.createAttribute("nthreads", m_base_cfg.get_nthreads());
        file.createAttribute("ep_memory_model_version",
                             detail::kEPMemoryModelVersion);
        file.createAttribute("batch_size", m_memory.batch_size);
        file.createAttribute("n_workers", m_memory.n_workers);
        file.createAttribute("harvest_enabled",
                             static_cast<uint8_t>(m_memory.harvest.enabled));
        file.createAttribute("max_harvests", m_memory.harvest.max_harvests);
        file.createAttribute(
            "harvest_store_folds",
            static_cast<uint8_t>(m_memory.harvest.store_folds));
        file.createAttribute("nchunks", m_chunk_cfgs.size());

        std::vector<double> limits_min;
        std::vector<double> limits_max;
        limits_min.reserve(m_base_cfg.get_nparams());
        limits_max.reserve(m_base_cfg.get_nparams());
        for (const auto& lim : m_base_cfg.get_param_limits()) {
            limits_min.push_back(lim.min);
            limits_max.push_back(lim.max);
        }
        file.createAttribute("param_limits_min", limits_min);
        file.createAttribute("param_limits_max", limits_max);

        auto chunks_grp = file.createGroup("chunks");
        for (SizeType i = 0; i < m_chunk_cfgs.size(); ++i) {
            const auto& chunk = m_chunk_cfgs[i];
            auto chunk_grp =
                chunks_grp.createGroup(std::format("chunk_{:04d}", i));
            chunk_grp.createAttribute("nominal_f_start", chunk.nominal_f_start);
            chunk_grp.createAttribute("nominal_f_end", chunk.nominal_f_end);
            chunk_grp.createAttribute("actual_f_start", chunk.actual_f_start);
            chunk_grp.createAttribute("actual_f_end", chunk.actual_f_end);
            chunk_grp.createAttribute("nbins", chunk.cfg.get_nbins());
            chunk_grp.createAttribute("eta", chunk.cfg.get_eta());
            chunk_grp.createAttribute("max_sugg", chunk.max_sugg);
            chunk_grp.createAttribute("branch_max", chunk.branch_max);
            chunk_grp.createAttribute("batch_size", chunk.batch_size);
            chunk_grp.createAttribute("predicted_peak", chunk.predicted_peak);
            chunk_grp.createAttribute("predicted_live", chunk.predicted_live);
            chunk_grp.createAttribute("peak_complexity", chunk.peak_complexity);
            chunk_grp.createAttribute("pd", chunk.pd);
            chunk_grp.createAttribute("pd_se", chunk.pd_se);
            chunk_grp.createAttribute("chunk_memory_gb", chunk.chunk_memory_gb);
            chunk_grp.createAttribute("nsegments", chunk.nsegments);
            chunk_grp.createAttribute("ncoords", chunk.ncoords);
            chunk_grp.createAttribute("buffer_size", chunk.buffer_size);
            chunk_grp.createAttribute("coord_size", chunk.coord_size);
            chunk_grp.createAttribute("fold_size", chunk.fold_size);

            chunk_grp.createDataSet("threshold_scheme", chunk.threshold_scheme);
            chunk_grp.createDataSet("branching_pattern",
                                    chunk.branching_pattern);
        }
        spdlog::info("EPRegionPlanner: saved {} chunks to plan cache '{}'",
                     m_chunk_cfgs.size(), filepath.string());
    }

    void load_cache(const std::filesystem::path& filepath) {
        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error(
                std::format("EPRegionPlanner: cache file '{}' does not exist",
                            filepath.string()));
        }
        const HighFive::File file(filepath.string(), HighFive::File::ReadOnly);
        if (!file.hasAttribute("ep_plan_cache_version")) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: file '{}' is not a valid EP plan cache",
                filepath.string()));
        }
        std::string file_version;
        file.getAttribute("ep_plan_cache_version").read(file_version);
        if (file_version != kCacheVersion) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache file '{}' has version {}, expected {} "
                "(stale plans carry an invalid branch_max; re-plan)",
                filepath.string(), file_version, kCacheVersion));
        }

        // A plan from the other backend is accepted. Its threshold scheme is
        // statistically equivalent, and the peak is recomputed below with this
        // backend's memory policy, so a plan that does not fit is still
        // rejected.
        std::string file_backend;
        file.getAttribute("ep_backend").read(file_backend);
        // The worker and thread counts are part of a plan from this backend
        // only. A plan from the other backend (a CUDA plan has one worker) is
        // accepted with any counts, because its peak is rechecked below.
        const bool same_backend = file_backend == to_string(m_exec.backend);
        if (!same_backend) {
            spdlog::info("EPRegionPlanner: plan cache '{}' was planned for the "
                         "{} backend and is used on {}; its peak is rechecked",
                         filepath.string(), file_backend,
                         to_string(m_exec.backend));
        }

        const auto check_attr_double = [&](const std::string& name,
                                           double val) {
            double file_val{};
            file.getAttribute(name).read(file_val);
            if (std::abs(val - file_val) >
                1e-6 * std::max(1.0, std::abs(val))) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache file '{}' mismatch for attribute "
                    "'{}': expected {:.8f}, found {:.8f}",
                    filepath.string(), name, val, file_val));
            }
        };

        const auto check_attr_size = [&](const std::string& name,
                                         SizeType val) {
            SizeType file_val{};
            file.getAttribute(name).read(file_val);
            if (val != file_val) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache file '{}' mismatch for attribute "
                    "'{}': expected {}, found {}",
                    filepath.string(), name, val, file_val));
            }
        };

        check_attr_size("nsamps", m_base_cfg.get_nsamps());
        check_attr_double("tsamp", m_base_cfg.get_tsamp());
        check_attr_double("f_min", m_base_cfg.get_f_min());
        check_attr_double("f_max", m_base_cfg.get_f_max());
        check_attr_size("nbins", m_base_cfg.get_nbins());
        check_attr_double("eta", m_base_cfg.get_eta());
        check_attr_size("nparams", m_base_cfg.get_nparams());
        check_attr_double("ducy_max", m_base_cfg.get_ducy_max());
        check_attr_double("wtsp", m_base_cfg.get_wtsp());
        // The limit is part of a plan from this backend. On the other backend
        // it means other memory (device vs process), so a plan from there is
        // accepted under this run's limit and its peak is rechecked below.
        if (same_backend) {
            check_attr_double("max_process_memory_gb",
                              m_base_cfg.get_max_process_memory_gb());
        }
        check_attr_double("octave_scale", m_base_cfg.get_octave_scale());
        check_attr_size("nbins_max", m_base_cfg.get_nbins_max());
        // The FFA segmentation sets every chunk's coordinates and buffers.
        check_attr_size("nbins_min_lossy_bf",
                        m_base_cfg.get_nbins_min_lossy_bf());
        check_attr_size("bseg_brute", m_base_cfg.get_bseg_brute());
        check_attr_size("bseg_ffa", m_base_cfg.get_bseg_ffa());
        check_attr_size("prune_poly_order", m_base_cfg.get_prune_poly_order());

        uint8_t file_use_fourier{};
        file.getAttribute("use_fourier").read(file_use_fourier);
        if (static_cast<bool>(file_use_fourier) !=
            m_base_cfg.get_use_fourier()) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache mismatch for 'use_fourier'"));
        }

        uint8_t file_cons_tile{};
        file.getAttribute("use_conservative_tile").read(file_cons_tile);
        if (static_cast<bool>(file_cons_tile) !=
            m_base_cfg.get_use_conservative_tile()) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache mismatch for 'use_conservative_tile'"));
        }

        check_attr_double("min_pd", m_min_pd);
        check_attr_double("ref_ducy", m_ref_ducy);
        check_attr_double("target_snr", m_base_cfg.get_target_snr());
        int file_nthreads{};
        file.getAttribute("nthreads").read(file_nthreads);
        // The CPU thread count is part of a CPU plan (the workers and the
        // threshold simulation). It plays no role on the GPU, and a plan from
        // the other backend is accepted whatever its count.
        if (same_backend && m_exec.backend == Backend::kCPU &&
            file_nthreads != m_base_cfg.get_nthreads()) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache mismatch for 'nthreads': expected {}, "
                "found {}",
                m_base_cfg.get_nthreads(), file_nthreads));
        }

        const auto check_attr_u64 = [&](const std::string& name,
                                        std::uint64_t val) {
            std::uint64_t file_val{};
            file.getAttribute(name).read(file_val);
            if (val != file_val) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache mismatch for '{}': expected {}, "
                    "found {}",
                    name, val, file_val));
            }
        };
        check_attr_u64("ep_memory_model_version",
                       detail::kEPMemoryModelVersion);
        // The batch sets the memory of a plan from this backend. A plan from
        // the other backend is rechecked with this backend's batch (see above).
        if (same_backend) {
            check_attr_u64("batch_size", m_memory.batch_size);
        }
        check_attr_u64("dts_batch_size", kDtsBatchSize);
        {
            // The scheme sets the chunking: the algorithm and everything that
            // changes its thresholds are part of the plan.
            std::string file_algorithm;
            file.getAttribute("dts_algorithm").read(file_algorithm);
            if (file_algorithm != kDtsAlgorithm) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache mismatch for 'dts_algorithm': "
                    "expected '{}', found '{}'",
                    kDtsAlgorithm, file_algorithm));
            }
            check_attr_double("dts_w_terminal", kDtsWTerminal);
            check_attr_double("dts_min_pulse_bins", kDtsMinPulseBins);
            check_attr_size("dts_nbins_fallback", kDtsNbinsSim);
            check_attr_size("dts_native_max_bins", kDtsNativeMaxBins);
            check_attr_double("ring_margin_z", detail::kEPRingMarginZ);
            std::string file_mode;
            file.getAttribute("dts_mode").read(file_mode);
            if (file_mode != kDtsMode) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache mismatch for 'dts_mode': "
                    "expected '{}', found '{}'",
                    kDtsMode, file_mode));
            }
        }
        std::vector<SizeType> cached_runs;
        file.getDataSet("ref_segs").read(cached_runs);
        const plans::FFAPlan<FoldType> run_plan(m_base_cfg);
        const auto requested_runs = utils::determine_ref_segs(
            run_plan.get_nsegments().back(), m_n_runs, m_ref_segs);
        for (const auto ref_seg : requested_runs) {
            if (std::ranges::find(cached_runs, ref_seg) == cached_runs.end()) {
                throw std::invalid_argument(
                    "EPRegionPlanner: cache mismatch for ref_segs");
            }
        }
        if (same_backend) {
            check_attr_u64("n_workers",
                           static_cast<std::uint64_t>(m_memory.n_workers));
        }
        check_attr_u64("harvest_enabled", m_memory.harvest.enabled ? 1U : 0U);
        check_attr_u64("max_harvests", m_memory.harvest.max_harvests);
        check_attr_u64("harvest_store_folds",
                       m_memory.harvest.store_folds ? 1U : 0U);

        std::string file_poly_basis;
        file.getAttribute("poly_basis").read(file_poly_basis);
        if (file_poly_basis != m_poly_basis) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache mismatch for 'poly_basis': expected "
                "'{}', found '{}'",
                m_poly_basis, file_poly_basis));
        }

        std::vector<double> file_lim_min;
        std::vector<double> file_lim_max;
        file.getAttribute("param_limits_min").read(file_lim_min);
        file.getAttribute("param_limits_max").read(file_lim_max);
        const auto cur_limits = m_base_cfg.get_param_limits();
        if (file_lim_min.size() != cur_limits.size()) {
            throw std::invalid_argument(
                "EPRegionPlanner: cache mismatch for param_limits size");
        }
        for (SizeType i = 0; i < cur_limits.size(); ++i) {
            if (std::abs(cur_limits[i].min - file_lim_min[i]) > 1e-6 ||
                std::abs(cur_limits[i].max - file_lim_max[i]) > 1e-6) {
                throw std::invalid_argument(std::format(
                    "EPRegionPlanner: cache mismatch for param_limits[{}]", i));
            }
        }

        SizeType nchunks{};
        file.getAttribute("nchunks").read(nchunks);

        m_chunk_cfgs.clear();
        m_chunk_cfgs.reserve(nchunks);
        std::vector<EPChunkStats> chunk_stats;
        chunk_stats.reserve(nchunks);

        SizeType max_sugg_all        = 0;
        SizeType max_ncoords_all     = 0;
        SizeType max_branch_max_all  = 0;
        SizeType max_buffer_size_all = 0;
        SizeType max_coord_size_all  = 0;
        SizeType max_fold_size_all   = 0;

        const auto chunks_grp = file.getGroup("chunks");
        for (SizeType i = 0; i < nchunks; ++i) {
            const auto chunk_grp =
                chunks_grp.getGroup(std::format("chunk_{:04d}", i));
            double nominal_f_start{};
            double nominal_f_end{};
            double actual_f_start{};
            double actual_f_end{};
            SizeType nbins{};
            double eta{};
            SizeType max_sugg{};
            SizeType branch_max{};
            SizeType batch_size{};
            SizeType predicted_peak{};
            SizeType predicted_live{};
            double peak_complexity{};
            double pd{};
            double pd_se{};
            double chunk_memory_gb{};
            SizeType nsegments{};

            chunk_grp.getAttribute("nominal_f_start").read(nominal_f_start);
            chunk_grp.getAttribute("nominal_f_end").read(nominal_f_end);
            chunk_grp.getAttribute("actual_f_start").read(actual_f_start);
            chunk_grp.getAttribute("actual_f_end").read(actual_f_end);
            chunk_grp.getAttribute("nbins").read(nbins);
            chunk_grp.getAttribute("eta").read(eta);
            chunk_grp.getAttribute("max_sugg").read(max_sugg);
            chunk_grp.getAttribute("branch_max").read(branch_max);
            chunk_grp.getAttribute("batch_size").read(batch_size);
            chunk_grp.getAttribute("predicted_peak").read(predicted_peak);
            chunk_grp.getAttribute("predicted_live").read(predicted_live);
            if (!same_backend) {
                batch_size = m_memory.batch_size;
            }
            if (batch_size == 0 || batch_size > m_memory.batch_size ||
                !std::has_single_bit(batch_size)) {
                throw std::invalid_argument(
                    "EPRegionPlanner: invalid cached batch size");
            }
            const auto required = detail::ep_tree_capacity(
                predicted_live, batch_size, branch_max, m_memory.kind);
            if (same_backend && max_sugg < required) {
                throw std::invalid_argument(
                    "EPRegionPlanner: cached capacity below forecast");
            }
            // A backend change also changes the batch allowance. Preserve
            // the forecast, rather than the other backend's capacity floor.
            max_sugg = same_backend ? std::max(max_sugg, required) : required;
            chunk_grp.getAttribute("peak_complexity").read(peak_complexity);
            chunk_grp.getAttribute("pd").read(pd);
            chunk_grp.getAttribute("pd_se").read(pd_se);
            chunk_grp.getAttribute("chunk_memory_gb").read(chunk_memory_gb);
            chunk_grp.getAttribute("nsegments").read(nsegments);

            std::vector<float> threshold_scheme;
            std::vector<float> branching_pattern;
            chunk_grp.getDataSet("threshold_scheme").read(threshold_scheme);
            chunk_grp.getDataSet("branching_pattern").read(branching_pattern);

            auto chunk_cfg = m_base_cfg.get_updated_ep_config(
                nbins, eta, actual_f_start, actual_f_end);
            const plans::FFAPlan<FoldType> plan(chunk_cfg);
            SizeType ncoords = plan.get_ncoords().back();
            if (chunk_grp.hasAttribute("ncoords")) {
                chunk_grp.getAttribute("ncoords").read(ncoords);
            }
            const SizeType buffer_size = plan.get_buffer_size();
            const SizeType coord_size  = plan.get_coord_size();
            const SizeType fold_size   = plan.get_fold_size();
            const SizeType transient =
                detail::ep_ffa_transient_bytes<FoldType>(m_memory, plan);

            m_chunk_cfgs.push_back(EPChunkConfig{
                .cfg                 = std::move(chunk_cfg),
                .threshold_scheme    = std::move(threshold_scheme),
                .branching_pattern   = std::move(branching_pattern),
                .max_sugg            = max_sugg,
                .branch_max          = branch_max,
                .batch_size          = batch_size,
                .predicted_peak      = predicted_peak,
                .predicted_live      = predicted_live,
                .nominal_f_start     = nominal_f_start,
                .nominal_f_end       = nominal_f_end,
                .actual_f_start      = actual_f_start,
                .actual_f_end        = actual_f_end,
                .peak_complexity     = peak_complexity,
                .pd                  = pd,
                .pd_se               = pd_se,
                .chunk_memory_gb     = chunk_memory_gb,
                .nsegments           = nsegments,
                .ncoords             = ncoords,
                .buffer_size         = buffer_size,
                .coord_size          = coord_size,
                .fold_size           = fold_size,
                .ffa_transient_bytes = transient,
            });

            const double nominal_width = nominal_f_end - nominal_f_start;
            const double actual_width  = actual_f_end - actual_f_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            chunk_stats.push_back(EPChunkStats{
                .chunk_id         = i,
                .nominal_f_start  = nominal_f_start,
                .nominal_f_end    = nominal_f_end,
                .actual_f_start   = actual_f_start,
                .actual_f_end     = actual_f_end,
                .nominal_width    = nominal_width,
                .actual_width     = actual_width,
                .nbins            = nbins,
                .eta              = eta,
                .ncoords          = ncoords,
                .max_sugg         = max_sugg,
                .branch_max       = branch_max,
                .batch_size       = batch_size,
                .predicted_peak   = predicted_peak,
                .predicted_live   = predicted_live,
                .peak_complexity  = peak_complexity,
                .pd               = pd,
                .pd_se            = pd_se,
                .memory_gb        = chunk_memory_gb,
                .overlap_fraction = overlap_fraction,
            });

            max_sugg_all        = std::max(max_sugg_all, max_sugg);
            max_ncoords_all     = std::max(max_ncoords_all, ncoords);
            max_branch_max_all  = std::max(max_branch_max_all, branch_max);
            max_buffer_size_all = std::max(max_buffer_size_all, buffer_size);
            max_coord_size_all  = std::max(max_coord_size_all, coord_size);
            max_fold_size_all   = std::max(max_fold_size_all, fold_size);
        }

        const double peak_memory_gb =
            m_chunk_cfgs.empty()
                ? 0.0
                : detail::ep_sweep_peak_gb<FoldType>(m_chunk_cfgs, m_memory);
        const double limit_gb =
            m_memory_limit_gb -
            detail::ep_memory_reserve_gb(m_base_cfg, m_memory.kind);
        if (peak_memory_gb > limit_gb) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: plan cache '{}' ({} plan) needs {:.2f} GB "
                "with {} worker(s) on {}, more than the limit {:.2f} GB; use "
                "fewer workers, raise max_process_memory_gb, or delete it and "
                "re-plan",
                filepath.string(), file_backend, peak_memory_gb,
                m_memory.n_workers, to_string(m_exec.backend), limit_gb));
        }
        m_stats = EPRegionStats(
            max_sugg_all, max_ncoords_all, max_branch_max_all,
            static_cast<float>(peak_memory_gb), max_buffer_size_all,
            max_coord_size_all, max_fold_size_all, std::move(chunk_stats));
        m_stats.set_memory_limit_gb(m_memory_limit_gb);
        spdlog::info("EPRegionPlanner: loaded {} chunks from plan cache '{}'",
                     m_chunk_cfgs.size(), filepath.string());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    float m_min_pd;
    std::string m_poly_basis;
    float m_ref_ducy;
    Exec m_exec;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    detail::EPMemoryContext m_memory;
    /// Memory the plan is fitted to (see resolve_memory_limit_gb).
    double m_memory_limit_gb;

    std::vector<EPChunkConfig> m_chunk_cfgs;
    EPRegionStats m_stats;

    void plan_regions() {
        m_chunk_cfgs.clear();

        const double f_min = m_base_cfg.get_f_min();
        const double f_max = m_base_cfg.get_f_max();
        const double p_min = 1.0 / f_max;
        const double p_max = 1.0 / f_min;

        const auto ffa_regions = generate_ffa_regions(
            p_min, p_max, m_base_cfg.get_tsamp(), m_base_cfg.get_nbins(),
            m_base_cfg.get_eta(), m_base_cfg.get_octave_scale(),
            m_base_cfg.get_nbins_max());

        spdlog::info("EPRegionPlanner: planned {} coarse FFA regions",
                     ffa_regions.size());

        const auto max_drift = calculate_max_drift(m_base_cfg);
        if (max_drift < 0.0 || max_drift >= 1.0) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: max_drift must be in [0, 1); got {:.6f}.",
                max_drift));
        }

        const auto max_memory_gb = m_memory_limit_gb;
        const auto reserve_gb =
            detail::ep_memory_reserve_gb(m_base_cfg, m_memory.kind);
        const auto effective_limit = max_memory_gb - reserve_gb;
        if (effective_limit <= 0.0) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: the memory limit ({:.2f} GB) must "
                "exceed the safety margin ({:.2f} GB)",
                max_memory_gb, reserve_gb));
        }

        // A design's simulation pools live only while it runs, so none is
        // left when the EP search allocates.
        std::vector<detail::RegionDesign> designs;
        designs.reserve(ffa_regions.size());
        for (const auto& region : ffa_regions) {
            if (region.f_end <= region.f_start) {
                continue;
            }
            designs.push_back(make_region_designer(region.f_start, region.f_end,
                                                   region.nbins, region.eta,
                                                   max_drift, effective_limit));
        }

        auto plan =
            detail::plan_chunks<FoldType>(m_base_cfg, m_poly_basis, designs,
                                          max_drift, effective_limit, m_memory);
        if (plan.replanned) {
            spdlog::info("EPRegionPlanner: replanned with fixed shared FFA "
                         "buffers (buffer_size={}, coord_size={})",
                         plan.buffer_size, plan.coord_size);
        }

        m_chunk_cfgs = std::move(plan.chunk_cfgs);
        report_chunks();
        const auto max_branch_max_all =
            plan.chunk_stats.empty()
                ? SizeType{0}
                : std::ranges::max_element(plan.chunk_stats, {},
                                           &EPChunkStats::branch_max)
                      ->branch_max;
        m_stats = EPRegionStats(
            plan.max_sugg, plan.max_ncoords, max_branch_max_all,
            static_cast<float>(plan.peak_memory_gb), plan.buffer_size,
            plan.coord_size, plan.fold_size, std::move(plan.chunk_stats));
        m_stats.set_memory_limit_gb(m_memory_limit_gb);

        spdlog::info(
            "EPRegionPlanner complete: {} chunks planned, max_sugg={}, "
            "max_mem={:.2f} GB (limit: {:.2f} GB, n_workers={})",
            m_chunk_cfgs.size(), utils::format_pow2(m_stats.get_max_sugg()),
            m_stats.get_max_memory_gb(), max_memory_gb, m_memory.n_workers);
    }

    /// One line per accepted chunk, and a warning for a chunk whose own
    /// thresholds measure a Pd below the delivery tolerance of min_pd (the
    /// schedule is kept; min_pd is not changed).
    void report_chunks() const {
        for (SizeType i = 0; i < m_chunk_cfgs.size(); ++i) {
            const auto& c = m_chunk_cfgs[i];
            spdlog::info("EPRegionPlanner: chunk {} f=[{:.6g}, {:.6g}] Hz "
                         "(nbins={}): Pd {:.4f} +/- {:.4f}, max_sugg={}, "
                         "peak complexity {:.4g}",
                         i, c.nominal_f_start, c.nominal_f_end,
                         c.cfg.get_nbins(), c.pd, c.pd_se,
                         utils::format_pow2(c.max_sugg), c.peak_complexity);
            if (c.pd < kDtsPdTolerance * m_min_pd) {
                spdlog::warn(
                    "EPRegionPlanner: the thresholds of chunk {} (f=[{:.6g}, "
                    "{:.6g}] Hz) reach Pd={:.4f} +/- {:.4f}, below {:.0f} % "
                    "of min_pd={:.6g}; accepting them; configured min_pd is "
                    "unchanged.",
                    i, c.nominal_f_start, c.nominal_f_end, c.pd, c.pd_se,
                    100.0 * kDtsPdTolerance, m_min_pd);
            }
        }
    }

    /// Pd the threshold search aims at. min_pd = 1 cannot be met (the grid is
    /// anchored at the target): it searches for 0.999, and the planner still
    /// warns that its own measured Pd is below min_pd.
    float scheme_target() const {
        return m_min_pd >= 1.0F ? kDtsMaxTarget : m_min_pd;
    }

    /// Fallback nbins of the threshold simulation when the pools of the true
    /// nbins do not fit (the chunk populations are still evaluated at the true
    /// nbins). The reference pulse must span at least kDtsMinPulseBins of them.
    SizeType scheme_nbins(SizeType nbins) const {
        const auto resolved =
            static_cast<SizeType>(std::ceil(kDtsMinPulseBins / m_ref_ducy));
        const SizeType sim = ((std::max(kDtsNbinsSim, resolved) + 3) / 4) * 4;
        return std::min(nbins, sim);
    }

    std::unique_ptr<detection::detail::ThresholdsEngine>
    make_threshold_engine(std::span<const float> branching_pattern,
                          SizeType nbins,
                          uint64_t seed,
                          SizeType ntrials = kDtsNTrials,
                          SizeType nprobs  = kDtsNProbs) const {
        // Shared mode builds its grid from min_pd: nprobs = bins above it.
        return detection::detail::make_thresholds_engine(
            branching_pattern, m_ref_ducy, nbins, ntrials, nprobs, kDtsProbMin,
            static_cast<float>(m_base_cfg.get_target_snr()), kDtsNThresholds,
            static_cast<float>(m_base_cfg.get_ducy_max()),
            static_cast<float>(m_base_cfg.get_wtsp()), kDtsBeamWidth,
            kDtsTrialsStart, kDtsMode, seed, kDtsBatchSize, kDtsGuess,
            detection::SharedOptions{.min_pd     = scheme_target(),
                                     .w_terminal = kDtsWTerminal},
            m_exec.backend == Backend::kCPU
                ? Exec::cpu(m_base_cfg.get_nthreads())
                : m_exec);
    }

    /// An engine for evaluate() only. The CPU engine then allocates no
    /// simulation pools (the GPU engine never does before run()).
    std::unique_ptr<detection::detail::ThresholdsEngine>
    make_evaluation_engine(std::span<const float> branching_pattern,
                           SizeType nbins,
                           uint64_t seed) const {
        if (m_exec.backend != Backend::kCPU) {
            return make_threshold_engine(branching_pattern, nbins, seed);
        }
        return detection::detail::make_thresholds_cpu(
            branching_pattern, m_ref_ducy, nbins, kDtsNTrials, kDtsNProbs,
            kDtsProbMin, static_cast<float>(m_base_cfg.get_target_snr()),
            kDtsNThresholds, static_cast<float>(m_base_cfg.get_ducy_max()),
            static_cast<float>(m_base_cfg.get_wtsp()), kDtsBeamWidth,
            kDtsTrialsStart, "improved", m_base_cfg.get_nthreads(), seed,
            /*simulate=*/false);
    }

    /// Samples of the refit at the true nbins, sized to the memory budget:
    /// the host or device buffers of a simulation are about 24 B per trial and
    /// bin (a signal bank) and 12 B (a noise evaluation).
    detection::detail::SolveBudget refit_budget(SizeType nbins,
                                                double memory_limit_gb) const {
        const double bytes = std::min(kDtsRefitMemoryFraction * memory_limit_gb,
                                      kDtsRefitMemoryMaxGB) *
                             detail::kBytesPerGiB;
        const auto floor_pow2 = [](double x, SizeType lo, SizeType hi) {
            const auto v = static_cast<SizeType>(std::max(x, 1.0));
            return std::clamp(std::bit_floor(v), lo, hi);
        };
        const double per_bin = static_cast<double>(nbins);
        return {.nbank  = floor_pow2(bytes / (24.0 * per_bin), kDtsRefitBankMin,
                                     SizeType{1} << 17U),
                .nnoise = floor_pow2(bytes / (12.0 * per_bin),
                                     kDtsRefitNoiseMin, SizeType{1} << 15U)};
    }

    /// The thresholds of a chunk's branching pattern @p bp, in two cases:
    ///  - native (nbins <= kDtsNativeMaxBins and the search pools fit): the
    ///    search and DynamicThresholdScheme's solve() at the true nbins;
    ///  - otherwise the search runs at kDtsNbinsSim bins and its schedule is
    ///    re-fitted at the true nbins on samples bounded by the memory limit.
    /// The search budget steps down (kDtsSearchLadder) until its pools fit;
    /// only a limit below the smallest rung fails, with a message. The
    /// design carries the Pd measured at the true nbins and the noise
    /// population of the final evaluation (and its trial count), which sizes
    /// the chunk. [f_start, f_end] is the chunk's band, for the logs.
    detail::ChunkDesign design_thresholds(std::span<const float> bp,
                                          double f_start,
                                          double f_end,
                                          SizeType nbins,
                                          double memory_limit_gb) const {
        const auto gb = [](const auto& engine) {
            return static_cast<double>(engine->run_memory_bytes()) /
                   detail::kBytesPerGiB;
        };
        // The pools of the search are freed before the EP search, so they are
        // checked against the limit on their own. Returns the first rung that
        // fits, or nullptr.
        const auto fit_search = [&](SizeType nbins_search)
            -> std::pair<std::unique_ptr<detection::detail::ThresholdsEngine>,
                         std::pair<SizeType, SizeType>> {
            for (const auto& rung : kDtsSearchLadder) {
                auto engine = make_threshold_engine(bp, nbins_search, kDtsSeed,
                                                    rung.first, rung.second);
                if (gb(engine) <= memory_limit_gb) {
                    return {std::move(engine), rung};
                }
            }
            return {nullptr, kDtsSearchLadder.back()};
        };
        const SizeType nbins_proxy = scheme_nbins(nbins);
        bool refit                 = nbins > nbins_proxy;
        SizeType nbins_search      = nbins;
        if (refit && nbins <= kDtsNativeMaxBins) {
            // Native when the default budget fits, else the proxy
            const auto probe = make_threshold_engine(bp, nbins, kDtsSeed);
            refit            = gb(probe) > memory_limit_gb;
        }
        if (refit) {
            nbins_search = nbins_proxy;
        }
        auto [dyn, rung] = fit_search(nbins_search);
        if (!dyn) {
            const auto smallest = make_threshold_engine(
                bp, nbins_search, kDtsSeed, kDtsSearchLadder.back().first,
                kDtsSearchLadder.back().second);
            throw std::runtime_error(std::format(
                "EPRegionPlanner: the threshold search for nbins={} needs "
                "{:.2f} GB at its smallest budget ({} trials, {} probability "
                "bins), more than the {:.2f} GB limit",
                nbins_search, gb(smallest), kDtsSearchLadder.back().first,
                kDtsSearchLadder.back().second, memory_limit_gb));
        }
        if (rung != kDtsSearchLadder.front()) {
            spdlog::debug(
                "EPRegionPlanner: the threshold search for f=[{:.6g}, "
                "{:.6g}] Hz needs a reduced budget to fit {:.2f} GB: "
                "{} trials, {} probability bins",
                f_start, f_end, memory_limit_gb, rung.first, rung.second);
        }
        dyn->run(kDtsThresNeigh);
        auto sol        = detection::detail::solve_scheme(*dyn, scheme_target(),
                                                          kDtsWTerminal,
                                                          /*robust=*/false);
        SizeType nnoise = dyn->solve_budget().nnoise;
        if (sol.thresholds.empty()) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: simulation has no surviving terminal path "
                "in f=[{:.6g}, {:.6g}] Hz (nbins={}, target_snr={:.6g})",
                f_start, f_end, nbins_search, m_base_cfg.get_target_snr()));
        }
        dyn.reset(); // free the search before the refit's samples
        if (refit) {
            const auto budget = refit_budget(nbins, memory_limit_gb);
            const auto engine = make_evaluation_engine(bp, nbins, kDtsSeed);
            const double searched_cost = sol.cost;
            sol                        = detection::detail::refit_scheme(
                *engine, sol, scheme_target(), kDtsWTerminal, kDtsSeed, budget);
            nnoise = budget.nnoise;
            spdlog::debug(
                "EPRegionPlanner: f=[{:.6g}, {:.6g}] Hz: searched at "
                "nbins={}, re-fitted at nbins={} ({} signal / {} noise "
                "trials): cost per detection {:.4g} (searched {:.4g})",
                f_start, f_end, nbins_search, nbins, budget.nbank,
                budget.nnoise, sol.cost, searched_cost);
        }
        spdlog::debug("EPRegionPlanner: f=[{:.6g}, {:.6g}] Hz, thresholds: Pd "
                      "{:.4f} +/- {:.4f} (min_pd {:.4g}), cost per detection "
                      "{:.4g}",
                      f_start, f_end, sol.pd, sol.pd_se, m_min_pd, sol.cost);
        return detail::ChunkDesign{
            .thresholds = std::move(sol.thresholds),
            .pd         = sol.pd,
            .pd_se      = sol.pd_se,
            .forecast   = population_forecast(sol.eval_states, nnoise)};
    }

    /// The noise population of an evaluation of @p ntrials trials.
    static detail::EPPopulationForecast
    population_forecast(std::span<const detection::State> states,
                        SizeType ntrials) {
        detail::EPPopulationForecast forecast{.ntrials = ntrials};
        forecast.complexity.reserve(states.size());
        forecast.pass_rate.reserve(states.size());
        for (const auto& state : states) {
            forecast.complexity.push_back(state.is_empty ? 0.0
                                                         : state.complexity);
            forecast.pass_rate.push_back(state.is_empty ? 0.0
                                                        : state.success_h0);
        }
        return forecast;
    }

    /// The designer of one coarse region: the full threshold design of a
    /// branching pattern, memoised (it is deterministic: fixed seeds).
    detail::RegionDesign make_region_designer(double f_start,
                                              double f_end,
                                              SizeType nbins,
                                              double eta,
                                              double max_drift,
                                              double memory_limit_gb) const {
        // Every chunk of a region has the FFA segmentation of the region.
        const auto rep_cfg = m_base_cfg.get_updated_ep_config(
            nbins, eta, f_start * (1.0 - max_drift), f_end * (1.0 + max_drift));
        const plans::FFAPlan<FoldType> rep_plan(rep_cfg);

        using DesignCache = std::map<std::vector<float>, detail::ChunkDesign>;
        auto design = [this, nbins, memory_limit_gb,
                       cache = std::make_shared<DesignCache>()](
                          std::span<const float> bp, double start, double end) {
            std::vector<float> key(bp.begin(), bp.end());
            if (const auto it = cache->find(key); it != cache->end()) {
                return it->second;
            }
            auto d = design_thresholds(bp, start, end, nbins, memory_limit_gb);
            cache->emplace(std::move(key), d);
            return d;
        };
        return detail::RegionDesign{
            .f_start   = f_start,
            .f_end     = f_end,
            .nbins     = nbins,
            .eta       = eta,
            .nsegments = rep_plan.get_nsegments().back(),
            .design    = std::move(design),
        };
    }
};

// --- EPRegionPlanner Template Definitions ---

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>::EPRegionPlanner(
    const search::PulsarSearchConfig& cfg,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const std::optional<std::filesystem::path>& plan_cache_file,
    const PruneRFIConfig& rfi_config,
    std::optional<SizeType> n_workers,
    Exec exec)
    : m_impl(std::make_unique<Impl>(cfg,
                                    min_pd,
                                    poly_basis,
                                    ref_ducy,
                                    plan_cache_file,
                                    rfi_config,
                                    n_workers,
                                    std::nullopt,
                                    std::nullopt,
                                    exec)) {}

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>::EPRegionPlanner(
    const search::PulsarSearchConfig& cfg,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const std::optional<std::filesystem::path>& plan_cache_file,
    const PruneRFIConfig& rfi_config,
    std::optional<SizeType> n_workers,
    std::optional<SizeType> n_runs,
    const std::optional<std::vector<SizeType>>& ref_segs,
    Exec exec)
    : m_impl(std::make_unique<Impl>(cfg,
                                    min_pd,
                                    poly_basis,
                                    ref_ducy,
                                    plan_cache_file,
                                    rfi_config,
                                    n_workers,
                                    n_runs,
                                    ref_segs,
                                    exec)) {}

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>::~EPRegionPlanner() = default;

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>::EPRegionPlanner(EPRegionPlanner&&) noexcept =
    default;

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>&
EPRegionPlanner<FoldType>::operator=(EPRegionPlanner&&) noexcept = default;

template <SupportedFoldType FoldType>
const std::vector<EPChunkConfig>&
EPRegionPlanner<FoldType>::get_chunk_cfgs() const noexcept {
    return m_impl->get_chunk_cfgs();
}

template <SupportedFoldType FoldType>
SizeType EPRegionPlanner<FoldType>::get_nchunks() const noexcept {
    return m_impl->get_nchunks();
}

template <SupportedFoldType FoldType>
const EPRegionStats& EPRegionPlanner<FoldType>::get_stats() const noexcept {
    return m_impl->get_stats();
}

template <SupportedFoldType FoldType>
void EPRegionPlanner<FoldType>::save_cache(
    const std::filesystem::path& filepath) const {
    m_impl->save_cache(filepath);
}

template <SupportedFoldType FoldType>
void EPRegionPlanner<FoldType>::load_cache(
    const std::filesystem::path& filepath) {
    m_impl->load_cache(filepath);
}

// Explicit template instantiations
template class EPRegionPlanner<float>;
template class EPRegionPlanner<ComplexType>;

} // namespace loki::algorithms
