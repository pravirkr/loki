#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <unordered_map>
#include <vector>

#include "loki/search/candidates.hpp"

namespace loki::search::detail {

[[nodiscard]] bool candidate_brighter(const Candidate& a, const Candidate& b);

/// Keeps the CandidateOptions::max_per_cell brightest trials of each
/// frequency cell of each source region (an EP run or an FFA region), so that
/// a bright source in one run cannot displace its detections in another.
/// Cells are cell_bins Fourier bins or cell_log_width in ln(frequency) wide.
/// Trials arrive in any order; candidate_brighter is a total order, so the
/// kept set does not depend on it. The kept trials are charged against
/// CandidateOptions::memory_gb until take() moves them out: when they would
/// exceed it, every cell's keep is halved (max_per_cell_used()), which keeps
/// the kept set independent of the arrival order.
class CellCap {
public:
    CellCap(const CandidateOptions& options,
            double tobs,
            std::size_t reserved_bytes = 0);
    void add(const Candidate& candidate);
    /// Moves the kept trials out, in no particular order.
    [[nodiscard]] std::vector<Candidate> take();
    [[nodiscard]] std::uint64_t dropped() const noexcept { return m_dropped; }
    /// Trials held since the last take().
    [[nodiscard]] std::uint64_t kept() const noexcept { return m_kept; }
    /// Trials each cell keeps since the last take(): max_per_cell, or less
    /// when the memory budget forced it lower.
    [[nodiscard]] std::uint64_t max_per_cell_used() const noexcept {
        return m_cap;
    }
    /// Highest S/N dropped since the last take(); zero when none was.
    [[nodiscard]] float max_dropped_snr() const noexcept {
        return m_max_dropped_snr;
    }

private:
    struct Key {
        std::uint32_t region;
        std::int64_t cell;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<std::uint64_t>{}(
                (static_cast<std::uint64_t>(k.cell) * 0x9E3779B97F4A7C15ULL) ^
                k.region);
        }
    };
    void add_to_full(std::vector<Candidate>& kept, const Candidate& c);
    std::uint64_t m_max;
    std::uint64_t m_cap; ///< Current per-cell keep, at most m_max.
    bool m_log;
    double m_scale;
    double m_memory_gb;
    // Each cell is a heap whose front is its weakest kept trial.
    std::unordered_map<Key, std::vector<Candidate>, KeyHash> m_cells;
    std::uint64_t m_kept{};
    std::uint64_t m_dropped{};
    float m_max_dropped_snr{};
};

/// Second stage of the per-run reduction, after CellCap: within each source
/// region, in decreasing S/N, a trial is absorbed by a stronger kept trial
/// that explains it, and only unexplained trials are kept. A stronger trial
/// explains a weaker one, of S/N at most dominance_kappa times its own, when
/// their phase tracks stay within the suppression radius of each other, or
/// when the weaker S/N is at most dominance_kappa times what the stronger
/// trial's pulse would give along the weaker track (a boxcar model of its
/// phase difference). Only kept trials within two cell widths in frequency
/// are compared.
class DominanceReducer {
public:
    DominanceReducer(double tobs, const CandidateOptions& options);
    /// Keeps the trials that no stronger trial of their region explains, in
    /// region and then detection order; returns how many it absorbed.
    std::uint64_t reduce(std::vector<Candidate>& trials) const;

private:
    [[nodiscard]] bool explains(const Candidate& strong,
                                const Candidate& weak) const;
    double m_tobs;
    CandidateOptions m_options;
    bool m_bins;
    double m_reach; ///< Hz with cell_bins, else a fraction of frequency.
};

/// The trial of an EP final leaf: @p leaf is its spool row of
/// (nparams + 2) x 2 doubles (value, error), parameters in Loki order
/// [..., accel, freq]. Its identity is (run << 32) | index.
[[nodiscard]] Candidate ep_leaf_candidate(std::span<const double> leaf,
                                          std::size_t nparams,
                                          float score,
                                          std::uint32_t width,
                                          std::uint32_t nbins,
                                          std::uint32_t run,
                                          std::uint32_t index);

/// The per-run reduction an EP search applies before writing a run.
struct LeafReductionSpec {
    CandidateOptions options; ///< max_per_cell must be nonzero.
    double snr_min{};
    double tobs{};
    std::uint32_t nbins{}; ///< Fold bins of the run's chunk.
};

/// What the per-run reduction did to one EP run.
struct RunLeafReduction {
    std::uint64_t final_leaves{};  ///< Leaves of the run.
    std::uint64_t above_snr_min{}; ///< Of those, trials at or above snr_min.
    std::uint64_t cell_dropped{};
    std::uint64_t dominated{};
    float max_cell_dropped_snr{};
    std::uint64_t max_per_cell_used{};   ///< Below max_per_cell: memory bound.
    std::vector<std::uint32_t> retained; ///< Leaf indices kept, ascending.
};

/// Applies CellCap and, with dominance_kappa, DominanceReducer to the final
/// leaves of one run, exactly as candidate publication does for one run.
/// @p leaf(i) and @p score(i) give leaf i; @p widths has one per leaf.
[[nodiscard]] RunLeafReduction reduce_run_leaves(
    const LeafReductionSpec& spec,
    std::size_t n_leaves,
    std::size_t nparams,
    const std::function<std::span<const double>(std::size_t)>& leaf,
    const std::function<float(std::size_t)>& score,
    std::span<const std::uint32_t> widths);

/// Receives globally S/N-sorted trials. append() returns a provisional
/// identity; cluster_id() resolves it after finish(), including bridges from
/// later trials. The host index is charged conservatively against
/// CandidateOptions::memory_gb.
class CandidateClusterer {
public:
    CandidateClusterer(double tobs,
                       const CandidateOptions& options,
                       std::size_t reserved_bytes = 0);
    std::uint64_t append(const Candidate& candidate);
    void finish();
    [[nodiscard]] std::uint64_t cluster_id(std::uint64_t provisional) const;
    std::vector<CandidateCluster> clusters;

private:
    struct Node {
        Candidate representative;
        double duty{}; // Widest recorded trial: preserves every possible edge.
        std::uint64_t parent{};
        std::uint64_t member_count{1}; // Meaningful on union-find roots only.
    };
    std::uint64_t root(std::uint64_t node);
    std::uint64_t append_transitive(const Candidate& candidate);
    void finish_transitive();
    /// Sets each cluster's harmonic parent (CandidateOptions::flag_harmonics).
    void flag_harmonic_families();
    double m_tobs;
    CandidateOptions m_options;
    double m_memory_gb;
    std::multimap<double, std::uint64_t> m_index;
    std::map<std::array<double, 6>, std::uint64_t> m_tracks;
    std::vector<Node> m_nodes;
    std::vector<std::uint64_t> m_remap;
};

} // namespace loki::search::detail
