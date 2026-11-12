#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "loki/search/candidates.hpp"

namespace loki::search::detail {

[[nodiscard]] bool candidate_brighter(const Candidate& a, const Candidate& b);

/// Keeps the CandidateOptions::max_per_cell brightest trials of each cell of
/// width cell_log_width in ln(frequency). Trials arrive in any order;
/// candidate_brighter is a total order, so the kept set does not depend on
/// it. The index is charged against CandidateOptions::memory_gb.
class CellCap {
public:
    explicit CellCap(const CandidateOptions& options,
                     std::size_t reserved_bytes = 0);
    void add(const Candidate& candidate);
    /// Moves the kept trials out, in no particular order.
    [[nodiscard]] std::vector<Candidate> take();
    [[nodiscard]] std::uint64_t dropped() const noexcept { return m_dropped; }

private:
    std::uint64_t m_max;
    double m_width;
    double m_memory_gb;
    // Each cell is a heap whose front is its weakest kept trial.
    std::unordered_map<std::int64_t, std::vector<Candidate>> m_cells;
    std::uint64_t m_kept{};
    std::uint64_t m_dropped{};
};

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
    double m_tobs;
    CandidateOptions m_options;
    double m_memory_gb;
    std::multimap<double, std::uint64_t> m_index;
    std::map<std::array<double, 6>, std::uint64_t> m_tracks;
    std::vector<Node> m_nodes;
    std::vector<std::uint64_t> m_remap;
};

} // namespace loki::search::detail
