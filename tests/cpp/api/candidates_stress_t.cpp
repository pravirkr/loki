// A bright source seen in thousands of trials: the regime where the
// frequency-window prefilter of the clusterer does nothing and every pair
// must be tested. Guards the result against references computed here from
// the public distance function (transitive closure, anchored assignment,
// stored phase extents) and the running time (a return to per-pair
// root-finding would show here).
//
// Bit patterns are deliberately not asserted: under -ffast-math the spans
// differ in their last bits between compilers and -march settings
// (docs/architecture.md, Numerics), and the anchored assignment can flip at
// a link boundary on one of them. The bit-identity of the bounded clusterer
// to the previous implementation was checked by fingerprint when the fix
// was made (2026-11-09, same build: transitive 0x3949e458c8b41812, anchored
// 0x3d6a9ac603f6b912 on an M-series Mac and on GCC 11/14 x86).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/search/candidates.hpp"

namespace {
using loki::search::Candidate;
using loki::search::CandidateOptions;

// 2^25 samples of 64 us: the production jerk benchmark.
constexpr double kTobs         = 2147.483648;
constexpr std::uint32_t kNbins = 64;
constexpr double kHalf         = kTobs / 2;

// Deterministic jitter in [0, 1).
double jitter(std::uint64_t x) {
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33U;
    return static_cast<double>(x >> 11U) * 0x1.0p-53;
}

Candidate make(std::uint64_t id,
               double f,
               double fdot,
               double fddot,
               std::uint32_t width,
               float snr) {
    Candidate c;
    c.id             = id;
    c.derivatives[0] = f;
    c.derivatives[1] = fdot;
    c.derivatives[2] = fddot;
    c.parameters[0]  = f;
    c.width          = width;
    c.nbins          = kNbins;
    c.snr            = snr;
    return c;
}

// One source at 141 Hz on a (f, fdot, fddot) grid whose steps span 0.02,
// 0.08 and 0.03 cycles over the observation, so links depend on the width
// (limits from 0.05 to 5/64 cycles), plus scattered noise trials.
std::vector<Candidate> bright_source_detections() {
    std::vector<Candidate> out;
    std::uint64_t id = 0;
    for (int i = -30; i <= 30; ++i) {
        for (int j = -6; j <= 6; ++j) {
            for (int k = -2; k <= 2; ++k) {
                const double f     = 141.0 + (i * 0.02 / kTobs);
                const double fdot  = j * 0.08 * 2 / (kHalf * kHalf);
                const double fddot = k * 0.015 * 6 / (kHalf * kHalf * kHalf);
                const auto width   = static_cast<std::uint32_t>(
                    1 + (((std::abs(i) * 7) + (std::abs(j) * 3) + k + 2) % 5));
                const auto snr = static_cast<float>(
                    20.0 - (0.3 * std::abs(i)) - (0.9 * std::abs(j)) -
                    (1.5 * std::abs(k)) + (0.5 * jitter(id + 1)));
                out.push_back(make(id, f, fdot, fddot, width, snr));
                ++id;
            }
        }
    }
    for (int n = 0; n < 1000; ++n) {
        const double f    = 140.0 + (5.0 * jitter((id * 3) + 1));
        const double fdot = (jitter((id * 3) + 2) - 0.5) * 2e-6;
        const auto width =
            static_cast<std::uint32_t>(1 + (static_cast<int>(id * 11) % 8));
        const auto snr = static_cast<float>(7.0 + (2.0 * jitter((id * 3) + 3)));
        out.push_back(make(id, f, fdot, 0.0, width, snr));
        ++id;
    }
    return out;
}

// Canonical labels: clusters numbered by first appearance in input order.
std::vector<std::uint64_t> canonical(std::vector<std::uint64_t> labels) {
    std::vector<std::uint64_t> map(labels.size(),
                                   std::numeric_limits<std::uint64_t>::max());
    std::uint64_t next = 0;
    for (auto& l : labels) {
        if (map[l] == std::numeric_limits<std::uint64_t>::max()) {
            map[l] = next++;
        }
        l = map[l];
    }
    return labels;
}

// Independent transitive closure over every pair, through the public
// distance function.
std::vector<std::uint64_t> pairwise_closure(const std::vector<Candidate>& in,
                                            const CandidateOptions& options) {
    std::vector<std::uint64_t> parent(in.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](std::uint64_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x         = parent[x];
        }
        return x;
    };
    for (std::size_t a = 0; a < in.size(); ++a) {
        for (std::size_t b = a + 1; b < in.size(); ++b) {
            const double duty =
                std::min(in[a].duty_cycle(), in[b].duty_cycle());
            const double span =
                loki::search::phase_track_distance(in[a], in[b], kTobs) * duty;
            if (span <= std::max(options.phase_radius, options.radius * duty)) {
                parent[find(b)] = find(a);
            }
        }
    }
    std::vector<std::uint64_t> labels(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        labels[i] = find(i);
    }
    return labels;
}

// The documented detection order: decreasing S/N, then ascending
// derivatives, narrower duty, more bins, lower id (docs/candidates.md).
bool brighter(const Candidate& a, const Candidate& b) {
    if (a.snr != b.snr) {
        return a.snr > b.snr;
    }
    if (a.derivatives != b.derivatives) {
        return a.derivatives < b.derivatives;
    }
    if (a.duty_cycle() != b.duty_cycle()) {
        return a.duty_cycle() < b.duty_cycle();
    }
    if (a.nbins != b.nbins) {
        return a.nbins > b.nbins;
    }
    return a.id < b.id;
}

// Anchored grouping without the frequency window: each trial, in detection
// order, joins the lowest-numbered existing cluster whose representative is
// within the link limit, else starts a cluster. The window in the library
// is a superset of the matching clusters, so the assignments must agree.
std::vector<std::uint64_t> anchored_reference(const std::vector<Candidate>& in,
                                              const CandidateOptions& options) {
    std::vector<std::size_t> order(in.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order,
                      [&](auto a, auto b) { return brighter(in[a], in[b]); });
    std::vector<std::size_t> representatives;
    std::vector<std::uint64_t> labels(in.size());
    for (const auto idx : order) {
        const auto& c = in[idx];
        auto match    = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t id = 0; id < representatives.size(); ++id) {
            const auto& r     = in[representatives[id]];
            const double duty = std::min(r.duty_cycle(), c.duty_cycle());
            const double span =
                loki::search::phase_track_distance(r, c, kTobs) * duty;
            if (span <= std::max(options.phase_radius, options.radius * duty)) {
                match = id;
                break;
            }
        }
        if (match == std::numeric_limits<std::uint64_t>::max()) {
            match = representatives.size();
            representatives.push_back(idx);
        }
        labels[idx] = match;
    }
    return labels;
}

// Every stored phase extent is the largest span from the representative to
// a member. The public distance is span / min(duty); undoing the division
// costs an ulp, hence the tolerance.
void check_phase_extents(const std::vector<Candidate>& in,
                         const loki::search::CandidateClustering& out) {
    std::vector<double> extent(out.clusters.size(), 0.0);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const auto& rep   = out.clusters[out.membership[i]].representative;
        const double duty = std::min(rep.duty_cycle(), in[i].duty_cycle());
        extent[out.membership[i]] = std::max(
            extent[out.membership[i]],
            loki::search::phase_track_distance(rep, in[i], kTobs) * duty);
    }
    for (std::size_t k = 0; k < out.clusters.size(); ++k) {
        CAPTURE(k);
        REQUIRE_THAT(out.clusters[k].phase_extent,
                     Catch::Matchers::WithinRel(extent[k], 1e-9) ||
                         Catch::Matchers::WithinAbs(extent[k], 1e-15));
    }
}
} // namespace

TEST_CASE("Clustering a bright source with thousands of trials",
          "[candidates][stress][slow]") {
    const auto input = bright_source_detections();
    REQUIRE(input.size() == 4965);
    CandidateOptions options;
    options.memory_gb = 0.25;

    const auto start = std::chrono::steady_clock::now();
    const auto out   = loki::search::cluster_candidates(input, kTobs, options);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    CAPTURE(seconds, out.clusters.size());
    // The reference implementation (per-pair recursive root finding with
    // heap allocation) took 1.4 s here on an M-series laptop; the bounded one
    // is expected well under 0.1 s. Loose enough for a loaded CI runner.
    CHECK(seconds < 1.0);

    // The partition equals an independent closure over every pair.
    REQUIRE(canonical(out.membership) ==
            canonical(pairwise_closure(input, options)));
    std::uint64_t members = 0;
    for (const auto& c : out.clusters) {
        REQUIRE(c.member_count >= 1);
        members += c.member_count;
    }
    REQUIRE(members == input.size());
    // Representatives are the brightest member of their cluster.
    for (std::size_t i = 0; i < input.size(); ++i) {
        REQUIRE(input[i].snr <=
                out.clusters[out.membership[i]].representative.snr);
    }
    check_phase_extents(input, out);

    // Anchored grouping takes the same bounded path for its link test.
    options.grouping  = "anchored";
    const auto anchor = loki::search::cluster_candidates(input, kTobs, options);
    CAPTURE(anchor.clusters.size());
    REQUIRE(anchor.membership == anchored_reference(input, options));
    check_phase_extents(input, anchor);
}
