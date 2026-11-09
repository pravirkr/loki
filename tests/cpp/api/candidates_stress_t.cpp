// A bright source seen in thousands of trials: the regime where the
// frequency-window prefilter of the clusterer does nothing and every pair
// must be tested. Guards both the result (against an independent pairwise
// closure and a fingerprint recorded from the reference implementation) and
// the running time (a return to per-pair root-finding would show here).
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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

std::uint64_t fnv1a(std::uint64_t h, std::uint64_t v) {
    for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xffU;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Every link decision and every stored span, in order.
std::uint64_t fingerprint(const loki::search::CandidateClustering& out) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const auto m : out.membership) {
        h = fnv1a(h, m);
    }
    for (const auto& c : out.clusters) {
        h = fnv1a(h, c.representative.id);
        h = fnv1a(h, c.member_count);
        h = fnv1a(h, std::bit_cast<std::uint64_t>(c.phase_extent));
    }
    return h;
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
    for (const auto& c : out.clusters) {
        REQUIRE(c.member_count >= 1);
    }
    // Representatives are the brightest member of their cluster.
    for (std::size_t i = 0; i < input.size(); ++i) {
        REQUIRE(input[i].snr <=
                out.clusters[out.membership[i]].representative.snr);
    }

    // Bit-exact against the reference implementation (every membership,
    // representative, count and stored phase extent).
    const auto fp = fingerprint(out);
    CAPTURE(fp);
    CHECK(fp == 0x3949e458c8b41812ULL);

    SECTION("anchored grouping") {
        options.grouping = "anchored";
        const auto anchor =
            loki::search::cluster_candidates(input, kTobs, options);
        const auto fpa = fingerprint(anchor);
        CAPTURE(fpa, anchor.clusters.size());
        CHECK(fpa == 0x3d6a9ac603f6b912ULL);
    }
}
