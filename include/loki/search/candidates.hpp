#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace loki::search {

/// Host-side postprocessing only; independent of the scoring backend.
struct CandidateOptions {
    /// false skips candidate processing: FFA writes the search's flat
    /// detection file and EP its run diagnostics, as before candidate
    /// products, and no host memory is reserved.
    bool enabled{true};
    bool cluster{true};
    std::string grouping{"transitive"}; ///< "transitive" or "anchored".
    double radius{1.0}; ///< Phase-residual span in narrower pulse widths.
    /// Minimum phase-span suppression radius in cycles; zero disables floor.
    double phase_radius{0.05};
    bool retain_detections{true};
    bool flag_harmonics{false};
    bool remove_harmonics{false};
    std::uint32_t harmonic_denominator_max{100};
    double harmonic_phase_distance{1.0};
    double harmonic_snr_distance{3.0};
    std::uint64_t max_candidates{0}; ///< Zero means unlimited.
    double memory_gb{1.0};           ///< Additional host memory, GB (2^30).
    std::filesystem::path temporary_directory;
    void validate() const;
};

/// Normalized Taylor trial at the midpoint of the searched interval.
/// derivatives[k] is d^k(frequency)/dt^k, in Hz/s^k. Unused entries are zero.
/// Native parameters retain Loki ordering [..., jerk, acceleration, frequency].
struct Candidate {
    std::array<double, 6> derivatives{};
    std::array<double, 6> parameters{};
    std::array<double, 6> parameter_errors{};
    bool has_parameter_errors{false};
    std::uint64_t id{};
    std::uint32_t source_region{};
    std::uint32_t source_index{};
    std::uint32_t width{};
    std::uint32_t nbins{};
    float snr{};
    [[nodiscard]] double duty_cycle() const;
    void validate() const;
};

struct CandidateCluster {
    Candidate representative;
    std::uint64_t member_count{1};
    std::int64_t parent_cluster_id{-1};
    std::uint32_t harmonic_numerator{};
    std::uint32_t harmonic_denominator{};
    bool selected{true};
    /// Maximum phase span from the representative to any member, in cycles.
    double phase_extent{};
};

struct CandidateClustering {
    std::vector<CandidateCluster> clusters;
    /// Cluster index for each input row, in the original input order.
    std::vector<std::uint64_t> membership;
};

/// Constant-phase-invariant distance over [-tobs/2, tobs/2].
[[nodiscard]] double
phase_track_distance(const Candidate& a, const Candidate& b, double tobs);

/// Convenience API for small host batches. The sweep uses the same algorithm
/// with disk-backed sorting instead of materializing the complete input.
[[nodiscard]] CandidateClustering
cluster_candidates(std::span<const Candidate> candidates,
                   double tobs,
                   const CandidateOptions& options = {});

} // namespace loki::search
