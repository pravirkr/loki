#include "loki/algorithms/ep_regions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <optional>
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
#include "lib/common/dispatch.hpp"
#include "lib/detail/utils.hpp"

namespace loki::algorithms {

namespace {

constexpr float kSafetyMultiplier = 1.25F;
// 1.2.0: per-chunk branch_max, per-group memory model.
// 1.3.0: inputs, harvest store and worker count in the memory model; stores
// the model version, batch size, n_workers and harvest bound.
// 1.4.0: the backend of the sweep (the model and the threshold simulation
// differ between CPU and CUDA), the memory-model kind, the batch size of
// the simulation, and the device architecture (informational).
constexpr std::string_view kCacheVersion = "1.4.0";

/// Batch size of the threshold-scheme simulation. CUDA results are
/// reproducible per (seed, batch_size), so it is part of the cache key.
constexpr SizeType kDtsBatchSize = 256;

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
        ctx.harvest           = {};
        ctx.kind              = detail::EPMemoryKind::kCuda;
        ctx.device            = exec.device;
        ctx.cub_scratch_bytes = &detail::ep_cuda_cub_scratch_bytes;
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

template <SupportedFoldType FoldType> class EPRegionPlanner<FoldType>::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         float min_pd,
         std::string_view poly_basis,
         float ref_ducy,
         const std::optional<std::filesystem::path>& plan_cache_file,
         const PruneRFIConfig& rfi_config,
         std::optional<SizeType> n_workers,
         const Exec& exec)
        : m_base_cfg(std::move(cfg)),
          m_min_pd(min_pd),
          m_poly_basis(poly_basis),
          m_ref_ducy(ref_ducy),
          m_exec(exec),
          m_memory(
              make_memory_context(m_base_cfg, rfi_config, n_workers, exec)),
          m_memory_limit_gb(resolve_memory_limit_gb(m_base_cfg, exec)) {
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
        file.createAttribute("prune_poly_order",
                             m_base_cfg.get_prune_poly_order());
        file.createAttribute(
            "use_conservative_tile",
            static_cast<uint8_t>(m_base_cfg.get_use_conservative_tile()));
        file.createAttribute("min_pd", m_min_pd);
        file.createAttribute("poly_basis", std::string(m_poly_basis));
        file.createAttribute("ref_ducy", m_ref_ducy);
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
            chunk_grp.createAttribute("peak_complexity", chunk.peak_complexity);
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

        std::string file_backend;
        file.getAttribute("ep_backend").read(file_backend);
        if (file_backend != to_string(m_exec.backend)) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache file '{}' was planned for the {} "
                "backend, not {} (the memory model and the threshold "
                "simulation differ between backends; re-plan)",
                filepath.string(), file_backend, to_string(m_exec.backend)));
        }

        std::string file_kind;
        file.getAttribute("ep_memory_model_kind").read(file_kind);
        const char* const expected_kind =
            m_memory.kind == detail::EPMemoryKind::kCuda ? "cuda" : "cpu";
        if (file_kind != expected_kind) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache file '{}' uses the {} memory model, "
                "not {} (re-plan)",
                filepath.string(), file_kind, expected_kind));
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
        check_attr_double("max_process_memory_gb",
                          m_base_cfg.get_max_process_memory_gb());
        check_attr_double("octave_scale", m_base_cfg.get_octave_scale());
        check_attr_size("nbins_max", m_base_cfg.get_nbins_max());
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
        int file_nthreads{};
        file.getAttribute("nthreads").read(file_nthreads);
        // The CPU thread count is part of the CPU plan (the workers and the
        // threshold simulation); it plays no role on the GPU.
        if (m_exec.backend == Backend::kCPU &&
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
        check_attr_u64("batch_size", m_memory.batch_size);
        check_attr_u64("dts_batch_size", kDtsBatchSize);
        check_attr_u64("n_workers",
                       static_cast<std::uint64_t>(m_memory.n_workers));
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
            double peak_complexity{};
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
            chunk_grp.getAttribute("peak_complexity").read(peak_complexity);
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
                .nominal_f_start     = nominal_f_start,
                .nominal_f_end       = nominal_f_end,
                .actual_f_start      = actual_f_start,
                .actual_f_end        = actual_f_end,
                .peak_complexity     = peak_complexity,
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
                .peak_complexity  = peak_complexity,
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
            detail::effective_memory_limit_gb(m_memory_limit_gb);
        if (peak_memory_gb > limit_gb) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: plan cache '{}' needs {:.2f} GB, more than "
                "the limit {:.2f} GB; delete it and re-plan",
                filepath.string(), peak_memory_gb, limit_gb));
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
        const auto effective_limit =
            detail::effective_memory_limit_gb(max_memory_gb);
        if (effective_limit <= 0.0) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: the memory limit ({:.2f} GB) must "
                "exceed the safety margin ({:.2f} GB)",
                max_memory_gb, detail::kUnmodelledReserveGB));
        }

        // The threshold schemes are expensive: design every region once and
        // reuse the designs if a second planning pass is needed.
        std::vector<detail::RegionDesign> designs;
        designs.reserve(ffa_regions.size());
        for (const auto& region : ffa_regions) {
            if (region.f_end <= region.f_start) {
                continue;
            }
            designs.push_back(design_region(region.f_start, region.f_end,
                                            region.nbins, region.eta,
                                            max_drift));
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
            m_chunk_cfgs.size(), m_stats.get_max_sugg(),
            m_stats.get_max_memory_gb(), max_memory_gb, m_memory.n_workers);
    }

    /// Computes the branching pattern and simulates the DynamicThresholdScheme
    /// once for a coarse band.
    detail::RegionDesign design_region(double f_start,
                                       double f_end,
                                       SizeType nbins,
                                       double eta,
                                       double max_drift) const {
        const double region_actual_start = f_start * (1.0 - max_drift);
        const double region_actual_end   = f_end * (1.0 + max_drift);
        const auto rep_cfg               = m_base_cfg.get_updated_ep_config(
            nbins, eta, region_actual_start, region_actual_end);
        const plans::FFAPlan<FoldType> rep_plan(rep_cfg);
        const auto bp_double = rep_plan.get_branching_pattern(m_poly_basis);
        const std::vector<float> bp_float(bp_double.begin(), bp_double.end());
        const SizeType nsegments = rep_plan.get_nsegments().back();

        const auto snr_final = static_cast<float>(m_base_cfg.get_snr_min());
        const auto ducy_max  = static_cast<float>(m_base_cfg.get_ducy_max());
        const auto wtsp      = static_cast<float>(m_base_cfg.get_wtsp());

        constexpr SizeType kNTrials     = 1024;
        constexpr SizeType kNThresholds = 100;
        constexpr SizeType kNProbs      = 10;
        constexpr float kProbMin        = 0.05F;
        constexpr float kBeamWidth      = 0.7F;
        constexpr SizeType kTrialsStart = 1;
        // Fixed seed: the scheme sets max_sugg and the chunking, so plans of
        // one configuration must not change from run to run.
        constexpr uint64_t kPlannerSeed = 0x10C1;

        detection::DynamicThresholdScheme dyn(
            bp_float, m_ref_ducy, nbins, kNTrials, kNProbs, kProbMin, snr_final,
            kNThresholds, ducy_max, wtsp, kBeamWidth, kTrialsStart, "legacy",
            /*seed=*/kPlannerSeed, /*batch_size=*/kDtsBatchSize,
            m_exec.backend == Backend::kCPU
                ? Exec::cpu(m_base_cfg.get_nthreads())
                : m_exec);
        dyn.run();

        auto threshold_scheme = dyn.get_best_path_thresholds(m_min_pd);
        if (threshold_scheme.empty()) {
            spdlog::warn(
                "EPRegionPlanner: no path for min_pd={:.2f} in f=[{:.2f}, "
                "{:.2f}], retrying with min_pd=0.01",
                m_min_pd, f_start, f_end);
            threshold_scheme = dyn.get_best_path_thresholds(0.01F);
        }
        if (threshold_scheme.empty()) {
            threshold_scheme.resize(nsegments - 1);
            for (SizeType s = 0; s < nsegments - 1; ++s) {
                threshold_scheme[s] =
                    std::max(1.5F, snr_final * static_cast<float>(s + 1) /
                                       static_cast<float>(nsegments));
            }
        }

        const auto states = detection::evaluate_scheme(
            threshold_scheme, bp_float, m_ref_ducy, nbins, kNTrials, snr_final,
            ducy_max, wtsp);
        float peak_complexity = 1.0F;
        for (const auto& st : states) {
            if (!st.is_empty) {
                peak_complexity = std::max(peak_complexity, st.complexity);
            }
        }
        return detail::RegionDesign{
            .f_start          = f_start,
            .f_end            = f_end,
            .nbins            = nbins,
            .eta              = eta,
            .threshold_scheme = std::move(threshold_scheme),
            .bp_float         = bp_float,
            .peak_complexity  = peak_complexity,
            .safe_complexity =
                std::max(1.0F, peak_complexity) * kSafetyMultiplier,
            .nsegments = nsegments,
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
