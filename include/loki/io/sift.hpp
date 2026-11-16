#pragma once

/**
 * @file sift.hpp
 * @brief Group the candidates of many result files (DM trials x time
 * segments of one source) by frequency, and describe each group: its S/N
 * against DM, the segments it appears in, and harmonic relations between
 * groups.
 *
 * A pulsar peaks at a DM inside the searched trials and persists over
 * segments; interference is usually flat in DM or strongest at the lowest
 * trial. Host only (CPU); reads published result files (common schema).
 */

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::io {

struct SiftOptions {
    /// Candidates below this S/N are ignored.
    double snr_min{7.0};
    /// Candidates whose frequencies differ by at most this fraction (to the
    /// next one in frequency order) share a group. The default covers the
    /// Earth's and a modest orbital Doppler shift between segments.
    double tolerance{2.0e-4};
    /// Groups seen in fewer segments are dropped.
    SizeType min_segments{1};
    /// Harmonic relations between groups: ratios p/q with p, q at most this
    /// (0: none).
    std::uint32_t harmonic_order_max{16};
    /// In a file with members of both groups, their strongest members'
    /// frequencies agree with p/q to within this many bins of 1 / tobs:
    /// within one file a harmonic shares its parent's Doppler shift.
    double harmonic_bins{2.0};
};

/// One result file, from its `/observation` attributes.
struct SiftFile {
    std::filesystem::path path;
    std::optional<double> dm;
    std::optional<double> start_mjd;
    std::optional<double> reference_mjd;
    double tobs{};
    /// Index of its time segment: files with the same start MJD (or, without
    /// one, the same input series) share a segment. Segments are numbered by
    /// start MJD.
    SizeType segment{};
};

/// A candidate of a group.
struct SiftMember {
    SizeType file{}; ///< Index into SiftResult::files.
    std::uint64_t id{};
    double freq{}; ///< Hz, at the file's reference time.
    double fdot{}; ///< Hz/s.
    float snr{};
    double duty_cycle{}; ///< width / nbins.
};

struct SiftGroup {
    /// Members by decreasing S/N; the first is the best.
    std::vector<SiftMember> members;
    /// Best S/N per DM trial, by increasing DM (files with a DM only).
    std::vector<std::pair<double, float>> dm_curve;
    /// Segments with a member, increasing.
    std::vector<SizeType> segments;
    /// Best S/N at the lowest and highest DM of all files over the group's
    /// best S/N: near 1 for interference flat in DM, small for a pulsar that
    /// peaks inside the trials. Empty without DMs.
    std::optional<double> edge_ratio;
    /// Largest |f_b - f_a - fdot_a (t_b - t_a)| * T_b over consecutive
    /// segments' best members, from their reference MJDs: how far, in
    /// frequency bins, a segment's best member is from the previous one's
    /// Taylor extrapolation. Empty with fewer than two segments or no MJDs.
    std::optional<double> extrapolation_bins;
    /// The stronger group this one is a harmonic of (index into
    /// SiftResult::groups), with frequency ratio p/q to it; -1 if none.
    std::int64_t harmonic_of{-1};
    std::uint32_t harmonic_numerator{0};
    std::uint32_t harmonic_denominator{0};
    /// |f - p/q f_parent| * tobs of the strongest members in the shared file
    /// that confirmed the relation.
    std::optional<double> harmonic_offset_bins;
};

struct SiftResult {
    std::vector<SiftFile> files;
    /// By decreasing best S/N.
    std::vector<SiftGroup> groups;
    SizeType n_candidates{}; ///< Candidates at or above snr_min.
    SizeType n_segments{};
};

/**
 * @brief Group the candidates of @p files by frequency.
 *
 * Candidates are sorted by frequency and joined friends-of-friends while
 * consecutive frequencies differ by at most options.tolerance (as a
 * fraction). A group is a harmonic of a stronger group when its best
 * frequency is within the tolerance of p/q times the stronger group's best,
 * p/q is the only ratio of the allowed orders within it, its best DM is
 * inside the stronger group's DM range, and in some file with members of
 * both their strongest members agree with p/q to within
 * options.harmonic_bins / tobs. The tolerance alone is too wide to tell: at
 * 2e-4 the ratios of order 16 cover about 1% of all frequencies.
 *
 * Throws std::invalid_argument for a file that is not a published result
 * (a raw or spool file: publish it first) and std::runtime_error for one
 * that cannot be read.
 */
[[nodiscard]] SiftResult sift(std::span<const std::filesystem::path> files,
                              const SiftOptions& options = {});

} // namespace loki::io
