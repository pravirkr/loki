#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "loki/search/leaf_basis.hpp"

namespace loki::search {

/// Host-side postprocessing only; independent of the scoring backend.
/// A closed frequency interval [lo, hi], in Hz.
struct FrequencyWindow {
    double lo{0.0};
    double hi{0.0};
};

struct CandidateOptions {
    /// false skips candidate processing: FFA writes the search's flat
    /// detection file, as before candidate products, and reserves no host
    /// memory; EP writes its runs, reduced unless max_per_cell = 0 (see
    /// ep_reduces_runs()).
    bool enabled{true};
    bool cluster{true};
    std::string grouping{"transitive"}; ///< "transitive" or "anchored".
    double radius{1.0}; ///< Phase-residual span in narrower pulse widths.
    /// Minimum phase-span suppression radius in cycles; zero disables floor.
    double phase_radius{0.05};
    bool retain_detections{true};
    bool flag_harmonics{false};
    bool remove_harmonics{false};
    /// Largest numerator and denominator of a harmonic ratio p/q.
    std::uint32_t harmonic_denominator_max{16};
    /// Scaled phase tracks of a harmonic and its parent agree within this
    /// many pulse widths (of the faster one).
    double harmonic_phase_distance{1.0};
    /// A harmonic's S/N exceeds its parent's by at most this.
    double harmonic_snr_distance{3.0};
    /// A parent of at least harmonic_doppler_min_snr also takes a child
    /// whose frequency is within this fraction of p/q times its own, with no
    /// track test: a binary's harmonics stay in ratio while its orbit curves
    /// their tracks beyond the search's order. Zero (the default) disables
    /// it; 3e-4 covers 90 km/s. Enable it only with few bright parents: each
    /// covers about 2% of the frequencies of a 5-720 Hz search.
    double harmonic_doppler_tolerance{0.0};
    double harmonic_doppler_min_snr{20.0};
    /// Frequency windows the preprocessing zapped, in Hz. The tables'
    /// `in_zap_window` column is 1 for a row whose f0 * p / q lies in one of
    /// them for some p, q in [1, kZapWindowRatioMax]: a flag for sifting, no
    /// row is removed. Search configurations fill it from the
    /// [preprocessing] birdies (FFATomlConfig::candidate_options()).
    std::vector<FrequencyWindow> zap_windows;
    static constexpr std::uint32_t kZapWindowRatioMax = 4;
    /// Keep at most this many of the strongest trials in each frequency
    /// cell of each source (EP run or FFA region), before grouping. Zero
    /// keeps every trial.
    std::uint64_t max_per_cell{0};
    /// Width of a frequency cell in Fourier bins of the observation: a trial
    /// at frequency f is in cell floor(f * tobs / cell_bins). Exactly one of
    /// cell_bins and cell_log_width is required with max_per_cell.
    double cell_bins{0};
    /// Width of a frequency cell in natural-log frequency: a trial at
    /// frequency f is in cell floor(ln(f) / cell_log_width).
    double cell_log_width{0};
    /// Within each run's capped cells, absorb a trial that a stronger kept
    /// trial of the same run explains: one whose phase track stays within the
    /// suppression radius of it, or whose S/N is at most dominance_kappa
    /// times what the stronger trial's pulse would give along its track.
    /// Zero disables it. It runs only within the cell cap: with max_per_cell
    /// zero it has no effect.
    double dominance_kappa{0};
    std::uint64_t max_candidates{0}; ///< Zero means unlimited.
    double memory_gb{1.0};           ///< Additional host memory, GB (2^30).
    std::filesystem::path temporary_directory;
    void validate() const;
    /// Whether a polynomial Taylor EP search with these options reduces each
    /// run as it writes it (cell cap and dominance): with candidates, and
    /// without them too, so that the raw result of a search without
    /// candidates holds only the leaves publication keeps. max_per_cell = 0
    /// keeps every final leaf.
    [[nodiscard]] bool ep_reduces_runs() const noexcept {
        return max_per_cell != 0;
    }
    /// Whether such a search sets memory_gb of host memory aside (inside a
    /// CPU search's max_process_memory_gb): to publish candidates, or to
    /// reduce its runs.
    [[nodiscard]] bool ep_reserves_memory() const noexcept {
        return enabled || ep_reduces_runs();
    }
    /// Defaults of EP searches: their final leaves are reduced per run, with
    /// max_per_cell 32 in cells of 64 Fourier bins and dominance_kappa 1.5.
    /// max_per_cell = 0 keeps every leaf.
    [[nodiscard]] static CandidateOptions ep_defaults();
};

/// Normalized Taylor trial at the midpoint of the searched interval.
/// derivatives[k] is d^k(frequency)/dt^k, in Hz/s^k. Unused entries are zero.
/// Native parameters retain Loki ordering [..., jerk, acceleration, frequency].
struct Candidate {
    std::array<double, 6> derivatives{};
    std::array<double, 6> parameters{};
    std::array<double, 6> parameter_errors{};
    bool has_parameter_errors{false};
    /// EP: a parameter other than frequency lies outside its chunk's
    /// configured limits at the reported epoch (the observation midpoint).
    bool outside_prior{false};
    /// EP: the basis the leaf was last propagated in (see LeafBasis);
    /// kPolynomial except for resolved orbits of a circular-orbit search.
    LeafBasis leaf_basis{LeafBasis::kPolynomial};
    std::uint64_t id{};
    std::uint32_t source_region{};
    std::uint32_t source_index{};
    std::uint32_t width{};
    std::uint32_t nbins{};
    float snr{};
    /// EP circular-orbit searches: orbital period 2 pi / omega in seconds,
    /// with omega from the pair that resolved the orbit (snap and
    /// acceleration, or crackle and jerk) at the reported epoch. Zero when
    /// the leaf is not a resolved orbit.
    float p_orb{};
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
