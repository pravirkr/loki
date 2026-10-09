#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

/**
 * @brief Configuration and threshold scheme for a single EP chunk.
 */
struct EPChunkConfig {
    search::PulsarSearchConfig cfg;
    std::vector<float> threshold_scheme;
    std::vector<float> branching_pattern;
    SizeType max_sugg{1U << 18U};
    /// Branching capacity of this chunk's own plan (not the region's).
    SizeType branch_max{32U};
    double nominal_f_start{0.0};
    double nominal_f_end{0.0};
    double actual_f_start{0.0};
    double actual_f_end{0.0};
    double peak_complexity{1.0};
    /// Memory of this chunk alone: its own per-worker buffers and FFA
    /// buffers plus the inputs. Not the sweep peak (see EPRegionStats).
    double chunk_memory_gb{0.0};
    SizeType nsegments{0};
    SizeType ncoords{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    SizeType fold_size{0};
    /// Transient scratch of this chunk's FFA, live while the EP workspace is
    /// (bytes; CUDA backend only, zero on the CPU). Shared across chunks as
    /// the maximum.
    SizeType ffa_transient_bytes{0};
};

/**
 * @brief Statistics for a single planned EP chunk.
 */
struct EPChunkStats {
    SizeType chunk_id{0};
    double nominal_f_start{0.0};
    double nominal_f_end{0.0};
    double actual_f_start{0.0};
    double actual_f_end{0.0};
    double nominal_width{0.0};
    double actual_width{0.0};
    SizeType nbins{0};
    double eta{0.0};
    SizeType ncoords{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    double peak_complexity{0.0};
    /// Memory of this chunk alone (EPChunkConfig::chunk_memory_gb).
    double memory_gb{0.0};
    double overlap_fraction{0.0};
};

/**
 * @brief Aggregate statistics for EP regions planned across a frequency sweep.
 */
class EPRegionStats {
public:
    EPRegionStats() = default;
    EPRegionStats(SizeType max_sugg,
                  SizeType max_ncoords,
                  SizeType max_branch_max,
                  float max_memory_gb,
                  std::vector<EPChunkStats> chunk_stats)
        : m_max_sugg(max_sugg),
          m_max_ncoords(max_ncoords),
          m_max_branch_max(max_branch_max),
          m_max_memory_gb(max_memory_gb),
          m_chunk_stats(std::move(chunk_stats)) {}

    EPRegionStats(SizeType max_sugg,
                  SizeType max_ncoords,
                  SizeType max_branch_max,
                  float max_memory_gb,
                  SizeType max_buffer_size,
                  SizeType max_coord_size,
                  SizeType max_fold_size,
                  std::vector<EPChunkStats> chunk_stats)
        : m_max_sugg(max_sugg),
          m_max_ncoords(max_ncoords),
          m_max_branch_max(max_branch_max),
          m_max_memory_gb(max_memory_gb),
          m_max_buffer_size(max_buffer_size),
          m_max_coord_size(max_coord_size),
          m_max_fold_size(max_fold_size),
          m_chunk_stats(std::move(chunk_stats)) {}

    [[nodiscard]] SizeType get_max_sugg() const noexcept { return m_max_sugg; }
    [[nodiscard]] SizeType get_max_ncoords() const noexcept {
        return m_max_ncoords;
    }
    [[nodiscard]] SizeType get_max_branch_max() const noexcept {
        return m_max_branch_max;
    }
    /// Peak memory of the sweep: the largest per-worker buffer set of any
    /// run of equal-nbins chunks plus the shared FFA buffers and the inputs.
    /// Excludes the unmodelled reserve (see docs/memory.md).
    [[nodiscard]] float get_max_memory_gb() const noexcept {
        return m_max_memory_gb;
    }
    /// Memory the plan was fitted to, before the unmodelled reserve: the
    /// config's max_process_memory_gb on the CPU, and on CUDA the smaller of
    /// that and the free device memory less a fixed device reserve.
    [[nodiscard]] double get_memory_limit_gb() const noexcept {
        return m_memory_limit_gb;
    }
    void set_memory_limit_gb(double limit_gb) noexcept {
        m_memory_limit_gb = limit_gb;
    }
    [[nodiscard]] SizeType get_max_buffer_size() const noexcept {
        return m_max_buffer_size;
    }
    [[nodiscard]] SizeType get_max_coord_size() const noexcept {
        return m_max_coord_size;
    }
    [[nodiscard]] SizeType get_max_fold_size() const noexcept {
        return m_max_fold_size;
    }
    [[nodiscard]] SizeType get_nchunks() const noexcept {
        return m_chunk_stats.size();
    }
    [[nodiscard]] const std::vector<EPChunkStats>&
    get_chunk_stats() const noexcept {
        return m_chunk_stats;
    }

private:
    SizeType m_max_sugg{0};
    SizeType m_max_ncoords{0};
    SizeType m_max_branch_max{0};
    float m_max_memory_gb{0.0F};
    double m_memory_limit_gb{0.0};
    SizeType m_max_buffer_size{0};
    SizeType m_max_coord_size{0};
    SizeType m_max_fold_size{0};
    std::vector<EPChunkStats> m_chunk_stats;
};

/**
 * @brief Workers an EPFreqSweep prunes with: min(nthreads, runs), where runs
 * is n_runs, the number of ref_segs, or nthreads when neither is given.
 *
 * Pass the same value as the n_workers of EPRegionPlanner when a plan is made
 * outside the sweep (plan-only), so that the plan cache matches the sweep.
 */
[[nodiscard]] SizeType
ep_sweep_n_workers(int nthreads,
                   std::optional<SizeType> n_runs,
                   const std::optional<std::vector<SizeType>>& ref_segs);

/**
 * @brief A planner for EP (Extreme Pruning) regions across a frequency sweep.
 *
 * Subdivides a frequency search range into optimal memory-bounded chunks.
 * Uses generate_ffa_regions() for coarse bands (nbins, eta), runs
 * DynamicThresholdScheme simulation once per coarse band, and bisects chunk
 * widths analytically to respect max_process_memory_gb.
 * The threshold scheme and max_sugg are designed per coarse band; branch_max
 * is derived per chunk from the chunk's own plan.
 *
 * Memory model (mirrors EPFreqSweep, see docs/memory.md): per-worker buffers
 * (EP workspace, irfft scratch and, when harvesting is enabled, the harvest
 * store) are sized from the maxima of each contiguous run of chunks with the
 * same nbins, while the FFA workspace, the fold buffer and the input series
 * are held for the whole sweep. The peak must stay within
 * max_process_memory_gb minus a fixed reserve for unmodelled memory. If the
 * shared size grows after an earlier run was planned, the plan is recomputed
 * with the shared size fixed (searching a smaller fixed size if needed), so
 * every run fits next to the final shared buffers.
 *
 * Supports saving and loading planned chunk configurations to/from HDF5 cache
 * files with strict validation of all search parameters.
 *
 * @param rfi_config RFI configuration of the sweep: when harvesting is
 * enabled, each worker's harvest store is budgeted at max_harvests records.
 * @param n_workers Workers pruning at the same time (EPFreqSweep uses
 * min(nthreads, number of runs)); clamped to [1, nthreads]. Defaults to
 * nthreads. Ignored on CUDA, where one worker prunes the runs in turn.
 * @param exec Backend of the sweep the plan is for. On CUDA the threshold
 * schemes are designed with the CUDA DynamicThresholdScheme and memory means
 * device memory only: max_process_memory_gb is capped by the free device
 * memory less a fixed reserve, and host memory is never checked. An active
 * @p rfi_config is rejected (not implemented on CUDA). A plan cache written on
 * one backend is accepted on the other: its threshold scheme is statistically
 * equivalent, and load_cache rechecks its peak with the current backend.
 * @tparam FoldType float for time domain, ComplexType for Fourier domain.
 */
template <SupportedFoldType FoldType> class EPRegionPlanner {
public:
    explicit EPRegionPlanner(const search::PulsarSearchConfig& cfg,
                             float min_pd                = 0.1F,
                             std::string_view poly_basis = "taylor",
                             float ref_ducy              = 0.1F,
                             const std::optional<std::filesystem::path>&
                                 plan_cache_file               = std::nullopt,
                             const PruneRFIConfig& rfi_config  = {},
                             std::optional<SizeType> n_workers = std::nullopt,
                             Exec exec                         = {});

    ~EPRegionPlanner();
    EPRegionPlanner(EPRegionPlanner&&) noexcept;
    EPRegionPlanner& operator=(EPRegionPlanner&&) noexcept;
    EPRegionPlanner(const EPRegionPlanner&)            = delete;
    EPRegionPlanner& operator=(const EPRegionPlanner&) = delete;

    [[nodiscard]] const std::vector<EPChunkConfig>&
    get_chunk_cfgs() const noexcept;
    [[nodiscard]] SizeType get_nchunks() const noexcept;
    [[nodiscard]] const EPRegionStats& get_stats() const noexcept;

    void save_cache(const std::filesystem::path& filepath) const;
    void load_cache(const std::filesystem::path& filepath);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms
