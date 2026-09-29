#include "loki/algorithms/ep_regions.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <numeric>
#include <utility>
#include <vector>

#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils.hpp"

namespace loki::regions {

namespace {

constexpr SizeType kBatchSize          = 1024U;
constexpr double kSafetyMarginGB       = 0.5; // 500 MB headroom
constexpr float kSafetyMultiplier      = 1.25F;
constexpr double kBisectionToleranceHz = 1.0e-2;
constexpr double kMinChunkWidthHz      = 1.0e-2;
constexpr double kRelativeTolerance    = 1.0e-4;
constexpr SizeType kMaxBisectionSteps  = 50;

template <SupportedFoldType FoldType>
double calculate_ep_chunk_memory_gb(SizeType nparams,
                                    SizeType nbins,
                                    SizeType nsegments,
                                    SizeType ncoords_ffa,
                                    SizeType max_sugg,
                                    SizeType branch_max,
                                    SizeType batch_size,
                                    int nthreads,
                                    SizeType ffa_fold_size,
                                    SizeType ffa_buffer_size,
                                    SizeType ffa_coord_size) {
    constexpr bool kIsComplex = std::is_same_v<FoldType, ComplexType>;
    const SizeType nbins_f    = (nbins / 2) + 1;
    const SizeType nbins_arg  = kIsComplex ? nbins_f : nbins;
    const SizeType fold_bytes =
        kIsComplex ? sizeof(ComplexType) : sizeof(float);
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (nparams + 2) * kParamStride;

    // 1. WorldTree per thread:
    const SizeType max_batch_size = batch_size * branch_max;
    const SizeType world_tree_bytes =
        (max_sugg * leaves_stride * sizeof(double)) +
        (max_sugg * 2 * nbins_arg * fold_bytes) +
        (max_sugg * 2 * sizeof(float)) +
        ((max_sugg + max_batch_size) * sizeof(float)) +
        (max_batch_size * sizeof(SizeType)) + (max_sugg * sizeof(uint8_t));

    // 2. PruneWorkspace per thread:
    const SizeType max_branched_leaves = batch_size * branch_max;
    const SizeType max_branched_param_idx =
        std::max(max_branched_leaves, nsegments * batch_size);
    const SizeType prune_ws_bytes =
        (max_branched_leaves * leaves_stride * sizeof(double)) +
        (max_branched_leaves * 2 * nbins_arg * fold_bytes) +
        (max_branched_leaves * sizeof(float)) +
        (max_branched_leaves * sizeof(SizeType)) +
        (max_branched_param_idx * sizeof(SizeType)) +
        (max_branched_param_idx * sizeof(float)) +
        (max_branched_leaves * sizeof(float));

    // 3. BranchingWorkspace per thread:
    const SizeType branch_ws_bytes =
        (batch_size * nparams * branch_max * sizeof(double)) +
        (batch_size * nparams * sizeof(double)) +
        (batch_size * nparams * sizeof(SizeType)) +
        (batch_size * nparams * sizeof(double));

    // 4. Seed memory per thread (in EPWorkspace):
    const SizeType seed_bytes = (ncoords_ffa * leaves_stride * sizeof(double)) +
                                (ncoords_ffa * sizeof(float)) +
                                (ncoords_ffa * sizeof(SizeType));

    // 5. IRFFT scratch per thread (if complex):
    SizeType irfft_bytes = 0;
    if constexpr (kIsComplex) {
        const SizeType max_nfft =
            std::max(2 * batch_size * branch_max, 2 * ncoords_ffa);
        irfft_bytes = (max_nfft * nbins_f * sizeof(ComplexType)) +
                      (max_nfft * nbins * sizeof(float));
    }

    const SizeType per_thread_bytes = world_tree_bytes + prune_ws_bytes +
                                      branch_ws_bytes + seed_bytes +
                                      irfft_bytes;
    const SizeType all_threads_bytes =
        static_cast<SizeType>(nthreads) * per_thread_bytes;

    // 6. FFA fold output buffer (shared):
    const SizeType ffa_fold_bytes = ffa_fold_size * fold_bytes;

    // 7. FFA internal buffer during compute_ffa:
    const SizeType coord_unit_bytes =
        (nparams == 1) ? sizeof(coord::FFACoordFreq) : sizeof(coord::FFACoord);
    const SizeType ffa_ws_bytes = (2 * ffa_buffer_size * fold_bytes) +
                                  (ffa_coord_size * coord_unit_bytes);

    const SizeType total_bytes =
        all_threads_bytes + ffa_fold_bytes + ffa_ws_bytes;
    return static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30U);
}

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

struct EvaluatedChunk {
    search::PulsarSearchConfig cfg;
    SizeType ncoords{0};
    SizeType max_sugg{0};
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    double memory_gb{0.0};
};

struct RunningMaxima {
    SizeType max_sugg{0};
    SizeType ncoords{0};
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};

    void absorb(const EvaluatedChunk& c) noexcept {
        max_sugg    = std::max(max_sugg, c.max_sugg);
        ncoords     = std::max(ncoords, c.ncoords);
        fold_size   = std::max(fold_size, c.fold_size);
        buffer_size = std::max(buffer_size, c.buffer_size);
        coord_size  = std::max(coord_size, c.coord_size);
    }
};

} // namespace

template <SupportedFoldType FoldType> class EPRegionPlanner<FoldType>::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         float min_pd,
         std::string_view poly_basis,
         float ref_ducy,
         const std::optional<std::filesystem::path>& plan_cache_file)
        : m_base_cfg(std::move(cfg)),
          m_min_pd(min_pd),
          m_poly_basis(poly_basis),
          m_ref_ducy(ref_ducy) {
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
        file.createAttribute("ep_plan_cache_version", std::string("1.0.0"));
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
        HighFive::File file(filepath.string(), HighFive::File::ReadOnly);
        if (!file.hasAttribute("ep_plan_cache_version")) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: file '{}' is not a valid EP plan cache",
                filepath.string()));
        }

        auto check_attr_double = [&](const std::string& name, double val) {
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

        auto check_attr_size = [&](const std::string& name, SizeType val) {
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
        if (file_nthreads != m_base_cfg.get_nthreads()) {
            throw std::invalid_argument(std::format(
                "EPRegionPlanner: cache mismatch for 'nthreads': expected {}, "
                "found {}",
                m_base_cfg.get_nthreads(), file_nthreads));
        }

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
        float max_memory_gb_all      = 0.0F;
        SizeType max_buffer_size_all = 0;
        SizeType max_coord_size_all  = 0;
        SizeType max_fold_size_all   = 0;

        auto chunks_grp = file.getGroup("chunks");
        for (SizeType i = 0; i < nchunks; ++i) {
            auto chunk_grp =
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

            auto chunk_cfg = m_base_cfg.get_updated_config(
                nbins, eta, actual_f_start, actual_f_end);
            plans::FFAPlan<FoldType> plan(chunk_cfg);
            SizeType ncoords = plan.get_ncoords().back();
            if (chunk_grp.hasAttribute("ncoords")) {
                chunk_grp.getAttribute("ncoords").read(ncoords);
            }
            const SizeType buffer_size = plan.get_buffer_size();
            const SizeType coord_size  = plan.get_coord_size();
            const SizeType fold_size   = plan.get_fold_size();

            m_chunk_cfgs.push_back(EPChunkConfig{
                .cfg               = std::move(chunk_cfg),
                .threshold_scheme  = std::move(threshold_scheme),
                .branching_pattern = std::move(branching_pattern),
                .max_sugg          = max_sugg,
                .branch_max        = branch_max,
                .nominal_f_start   = nominal_f_start,
                .nominal_f_end     = nominal_f_end,
                .actual_f_start    = actual_f_start,
                .actual_f_end      = actual_f_end,
                .peak_complexity   = peak_complexity,
                .chunk_memory_gb   = chunk_memory_gb,
                .nsegments         = nsegments,
                .ncoords           = ncoords,
                .buffer_size       = buffer_size,
                .coord_size        = coord_size,
                .fold_size         = fold_size,
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
            max_memory_gb_all   = std::max(max_memory_gb_all,
                                           static_cast<float>(chunk_memory_gb));
            max_buffer_size_all = std::max(max_buffer_size_all, buffer_size);
            max_coord_size_all  = std::max(max_coord_size_all, coord_size);
            max_fold_size_all   = std::max(max_fold_size_all, fold_size);
        }

        m_stats = EPRegionStats(max_sugg_all, max_ncoords_all,
                                max_branch_max_all, max_memory_gb_all,
                                max_buffer_size_all, max_coord_size_all,
                                max_fold_size_all, std::move(chunk_stats));
        spdlog::info("EPRegionPlanner: loaded {} chunks from plan cache '{}'",
                     m_chunk_cfgs.size(), filepath.string());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    float m_min_pd;
    std::string m_poly_basis;
    float m_ref_ducy;

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

        const auto max_memory_gb   = m_base_cfg.get_max_process_memory_gb();
        const auto effective_limit = max_memory_gb - kSafetyMarginGB;
        if (effective_limit <= 0.0) {
            throw std::runtime_error(std::format(
                "EPRegionPlanner: max_process_memory_gb ({:.2f} GB) must "
                "exceed safety margin ({:.2f} GB)",
                max_memory_gb, kSafetyMarginGB));
        }

        RunningMaxima maxima;
        std::vector<EPChunkStats> chunk_stats;

        for (const auto& region : ffa_regions) {
            subdivide_region(region.f_start, region.f_end, region.nbins,
                             region.eta, max_drift, effective_limit, maxima,
                             chunk_stats);
        }

        m_stats = EPRegionStats(
            maxima.max_sugg, maxima.ncoords,
            chunk_stats.empty()
                ? SizeType{0}
                : std::ranges::max_element(chunk_stats, {},
                                           &EPChunkStats::branch_max)
                      ->branch_max,
            chunk_stats.empty()
                ? 0.0F
                : static_cast<float>(
                      std::ranges::max_element(chunk_stats, {},
                                               &EPChunkStats::memory_gb)
                          ->memory_gb),
            maxima.buffer_size, maxima.coord_size, maxima.fold_size,
            std::move(chunk_stats));

        spdlog::info(
            "EPRegionPlanner complete: {} chunks planned, max_sugg={}, "
            "max_mem={:.2f} GB (limit: {:.2f} GB)",
            m_chunk_cfgs.size(), m_stats.get_max_sugg(),
            m_stats.get_max_memory_gb(), max_memory_gb);
    }

    void subdivide_region(double f_start,
                          double f_end,
                          SizeType nbins,
                          double eta,
                          double max_drift,
                          double effective_limit_gb,
                          RunningMaxima& maxima,
                          std::vector<EPChunkStats>& chunk_stats) {
        if (f_end <= f_start) {
            return;
        }

        // 1. Compute branching pattern and simulate DynamicThresholdScheme
        // once for the coarse band
        const double region_actual_start = f_start * (1.0 - max_drift);
        const double region_actual_end   = f_end * (1.0 + max_drift);
        auto rep_cfg                     = m_base_cfg.get_updated_config(
            nbins, eta, region_actual_start, region_actual_end);
        plans::FFAPlan<FoldType> rep_plan(rep_cfg);
        const auto bp_double = rep_plan.get_branching_pattern(m_poly_basis);
        const std::vector<float> bp_float(bp_double.begin(), bp_double.end());
        const auto branch_max_raw = *std::ranges::max_element(bp_double);
        const SizeType branch_max = std::max(
            static_cast<SizeType>(std::ceil(branch_max_raw * 2.0)), 32UL);
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

        detection::DynamicThresholdScheme dyn(
            bp_float, m_ref_ducy, nbins, kNTrials, kNProbs, kProbMin, snr_final,
            kNThresholds, ducy_max, wtsp, kBeamWidth, kTrialsStart, "legacy",
            m_base_cfg.get_nthreads());
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

        auto states = detection::evaluate_scheme(threshold_scheme, bp_float,
                                                 m_ref_ducy, nbins, kNTrials,
                                                 snr_final, ducy_max, wtsp);
        float peak_complexity = 1.0F;
        for (const auto& s : states) {
            if (!s.is_empty) {
                peak_complexity = std::max(peak_complexity, s.complexity);
            }
        }
        const float safe_complexity =
            std::max(1.0F, peak_complexity) * kSafetyMultiplier;

        // 2. Analytic evaluator for candidate chunk
        auto evaluate_chunk = [&](double nominal_start,
                                  double nominal_end) -> EvaluatedChunk {
            const double act_start = nominal_start * (1.0 - max_drift);
            const double act_end   = nominal_end * (1.0 + max_drift);
            auto chunk_cfg =
                m_base_cfg.get_updated_config(nbins, eta, act_start, act_end);
            plans::FFAPlan<FoldType> plan(chunk_cfg);
            const SizeType ncoords = plan.get_ncoords().back();
            const SizeType max_sugg =
                std::max(SizeType{1024},
                         static_cast<SizeType>(
                             std::ceil(static_cast<double>(ncoords) *
                                       static_cast<double>(safe_complexity))));

            const double mem_gb = calculate_ep_chunk_memory_gb<FoldType>(
                m_base_cfg.get_nparams(), nbins, nsegments, ncoords, max_sugg,
                branch_max, kBatchSize, m_base_cfg.get_nthreads(),
                plan.get_fold_size(), plan.get_buffer_size(),
                plan.get_coord_size());

            return EvaluatedChunk{
                .cfg         = std::move(chunk_cfg),
                .ncoords     = ncoords,
                .max_sugg    = max_sugg,
                .fold_size   = plan.get_fold_size(),
                .buffer_size = plan.get_buffer_size(),
                .coord_size  = plan.get_coord_size(),
                .memory_gb   = mem_gb,
            };
        };

        auto fits = [&](const EvaluatedChunk& c) {
            // Must fit both in isolation and when absorbed into running maxima
            if (c.memory_gb > effective_limit_gb) {
                return false;
            }
            RunningMaxima test_max = maxima;
            test_max.absorb(c);
            const double sweep_mem_gb = calculate_ep_chunk_memory_gb<FoldType>(
                m_base_cfg.get_nparams(), nbins, nsegments, test_max.ncoords,
                test_max.max_sugg, branch_max, kBatchSize,
                m_base_cfg.get_nthreads(), test_max.fold_size,
                test_max.buffer_size, test_max.coord_size);
            return sweep_mem_gb <= effective_limit_gb;
        };

        const double region_span = f_end - f_start;
        const double boundary_tolerance =
            std::max(kBisectionToleranceHz, kRelativeTolerance * region_span);

        auto bisect_chunk =
            [&](double current_f_end, double min_width, EvaluatedChunk min_eval,
                double max_width) -> std::pair<double, EvaluatedChunk> {
            double lo_width     = min_width;
            double hi_width     = max_width;
            EvaluatedChunk best = std::move(min_eval);

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

        auto find_largest_fitting =
            [&](double current_f_end) -> std::pair<double, EvaluatedChunk> {
            const double remaining = current_f_end - f_start;
            auto full_eval         = evaluate_chunk(f_start, current_f_end);
            if (fits(full_eval)) {
                return {f_start, std::move(full_eval)};
            }

            const double min_width = std::min(remaining, kMinChunkWidthHz);
            const double min_start = current_f_end - min_width;
            auto min_eval          = evaluate_chunk(min_start, current_f_end);
            if (!fits(min_eval)) {
                throw std::runtime_error(std::format(
                    "EPRegionPlanner: Cannot fit minimum viable chunk at "
                    "[{:08.3f}, {:08.3f}] Hz.\n"
                    "  Required memory: {:.2f} GB, Available: {:.2f} GB\n"
                    "  Suggestion: Increase max_process_memory_gb.",
                    min_start, current_f_end, min_eval.memory_gb,
                    effective_limit_gb));
            }

            return bisect_chunk(current_f_end, min_width, std::move(min_eval),
                                remaining);
        };

        // Sliver absorption
        constexpr double kSliverFactor = 4.0;
        auto try_absorb_sliver =
            [&](double current_f_end, double nominal_start,
                EvaluatedChunk eval) -> std::pair<double, EvaluatedChunk> {
            const double remainder = nominal_start - f_start;
            if (remainder <= 0.0 ||
                remainder > (kSliverFactor * boundary_tolerance)) {
                return {nominal_start, std::move(eval)};
            }
            auto merged = evaluate_chunk(f_start, current_f_end);
            if (fits(merged)) {
                return {f_start, std::move(merged)};
            }
            return {nominal_start, std::move(eval)};
        };

        // Main subdivision loop
        double current_f_end = f_end;
        while (current_f_end > f_start) {
            auto [nominal_start, eval]    = find_largest_fitting(current_f_end);
            std::tie(nominal_start, eval) = try_absorb_sliver(
                current_f_end, nominal_start, std::move(eval));

            if (nominal_start >= current_f_end) {
                throw std::runtime_error(std::format(
                    "EPRegionPlanner: no progress in [{:08.3f}, {:08.3f}] Hz.",
                    f_start, current_f_end));
            }

            const double nominal_end   = current_f_end;
            const double nominal_width = nominal_end - nominal_start;
            const double actual_start  = nominal_start * (1.0 - max_drift);
            const double actual_end    = nominal_end * (1.0 + max_drift);
            const double actual_width  = actual_end - actual_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            const auto chunk_id = m_chunk_cfgs.size();
            m_chunk_cfgs.push_back(EPChunkConfig{
                .cfg               = eval.cfg,
                .threshold_scheme  = threshold_scheme,
                .branching_pattern = bp_float,
                .max_sugg          = eval.max_sugg,
                .branch_max        = branch_max,
                .nominal_f_start   = nominal_start,
                .nominal_f_end     = nominal_end,
                .actual_f_start    = actual_start,
                .actual_f_end      = actual_end,
                .peak_complexity   = peak_complexity,
                .chunk_memory_gb   = eval.memory_gb,
                .nsegments         = nsegments,
                .ncoords           = eval.ncoords,
                .buffer_size       = eval.buffer_size,
                .coord_size        = eval.coord_size,
                .fold_size         = eval.fold_size,
            });

            chunk_stats.push_back(EPChunkStats{
                .chunk_id         = chunk_id,
                .nominal_f_start  = nominal_start,
                .nominal_f_end    = nominal_end,
                .actual_f_start   = actual_start,
                .actual_f_end     = actual_end,
                .nominal_width    = nominal_width,
                .actual_width     = actual_width,
                .nbins            = nbins,
                .eta              = eta,
                .ncoords          = eval.ncoords,
                .max_sugg         = eval.max_sugg,
                .branch_max       = branch_max,
                .peak_complexity  = peak_complexity,
                .memory_gb        = eval.memory_gb,
                .overlap_fraction = overlap_fraction,
            });

            maxima.absorb(eval);
            current_f_end = nominal_start;
        }
    }
};

// --- EPRegionPlanner Template Definitions ---

template <SupportedFoldType FoldType>
EPRegionPlanner<FoldType>::EPRegionPlanner(
    const search::PulsarSearchConfig& cfg,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const std::optional<std::filesystem::path>& plan_cache_file)
    : m_impl(std::make_unique<Impl>(
          cfg, min_pd, poly_basis, ref_ducy, plan_cache_file)) {}

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

} // namespace loki::regions
