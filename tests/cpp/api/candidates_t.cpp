#include "loki/search/candidates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {
loki::search::Candidate trial(std::uint64_t id,
                              double frequency,
                              float snr,
                              std::uint32_t width = 1,
                              std::uint32_t nbins = 10) {
    loki::search::Candidate c;
    c.id             = id;
    c.derivatives[0] = frequency;
    c.parameters[0]  = frequency;
    c.width          = width;
    c.nbins          = nbins;
    c.snr            = snr;
    return c;
}
} // namespace

TEST_CASE("Candidate clustering anchors on the strongest trial",
          "[candidates]") {
    const std::vector input{trial(0, 10, 12), trial(1, 10.09, 11),
                            trial(2, 10.18, 10)};
    loki::search::CandidateOptions options;
    options.grouping     = "anchored";
    options.phase_radius = 0; // Exercise the legacy pulse-width-only rule.
    const auto out       = loki::search::cluster_candidates(input, 1, options);
    REQUIRE(out.clusters.size() == 2);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 1});
    REQUIRE(out.clusters[0].member_count == 2);
    REQUIRE(out.clusters[0].representative.id == 0);

    auto reversed = input;
    std::ranges::reverse(reversed);
    const auto other = loki::search::cluster_candidates(reversed, 1, options);
    REQUIRE(other.clusters[0].representative.id == 0);
    REQUIRE(other.membership == std::vector<std::uint64_t>{1, 0, 0});
}

TEST_CASE("Candidate widths are physical duty cycles and ties are stable",
          "[candidates]") {
    const std::vector input{trial(7, 10, 12, 2, 20), trial(3, 10, 12, 1, 10),
                            trial(9, 10, 11, 3, 10)};
    const auto out = loki::search::cluster_candidates(input, 10);
    REQUIRE(out.clusters.size() == 1);
    REQUIRE(out.clusters[0].representative.id ==
            7); // equivalent width, finer bins
    REQUIRE(out.clusters[0].member_count == 3);
    loki::search::CandidateOptions options;
    options.cluster = false;
    REQUIRE(
        loki::search::cluster_candidates(input, 10, options).clusters.size() ==
        3);
}

TEST_CASE("Phase track distance includes interior extrema", "[candidates]") {
    const auto a = trial(0, 100, 12);
    auto b       = a;
    b.derivatives[0] += 1;
    b.derivatives[2] = -6;
    // Over [-1,1], residual x-x^3 is zero at endpoints and midpoint.
    const double distance = loki::search::phase_track_distance(a, b, 2);
    REQUIRE(distance ==
            Catch::Approx(40 / (3 * std::sqrt(3.0))).epsilon(1e-12));
    const std::array input{a, b};
    REQUIRE(loki::search::cluster_candidates(input, 2).clusters.size() == 2);
    b                = a;
    b.derivatives[1] = 1;
    REQUIRE(loki::search::phase_track_distance(a, b, 2) == Catch::Approx(5));
}

TEST_CASE("Harmonics are separately flagged filtered and capped",
          "[candidates]") {
    std::vector input{trial(0, 10, 20), trial(1, 20, 14.1421356F),
                      trial(2, 31.123, 9)};
    loki::search::CandidateOptions options;
    options.flag_harmonics = true;
    auto out = loki::search::cluster_candidates(input, 100, options);
    REQUIRE(out.clusters[1].parent_cluster_id == 0);
    REQUIRE(out.clusters[1].harmonic_numerator == 2);
    REQUIRE(out.clusters[1].harmonic_denominator == 1);
    REQUIRE(out.clusters[1].selected);
    options.remove_harmonics = true;
    options.max_candidates   = 1;
    out = loki::search::cluster_candidates(input, 100, options);
    REQUIRE(out.clusters[0].selected);
    REQUIRE_FALSE(out.clusters[1].selected);
    REQUIRE_FALSE(out.clusters[2].selected);
    input[1].derivatives[1] = 0.1;
    out = loki::search::cluster_candidates(input, 100, options);
    REQUIRE(out.clusters[1].parent_cluster_id == -1);
}

TEST_CASE("Candidate processing validates input and bounds representatives",
          "[candidates]") {
    loki::search::CandidateOptions options;
    options.remove_harmonics = true;
    REQUIRE_THROWS(options.validate());
    auto bad  = trial(0, 10, 12);
    bad.width = 0;
    REQUIRE_THROWS(loki::search::cluster_candidates(std::array{bad}, 1));
    bad = trial(0, 10, std::numeric_limits<float>::quiet_NaN());
    REQUIRE_THROWS(loki::search::cluster_candidates(std::array{bad}, 1));
    options           = {};
    options.memory_gb = 0.0001;
    options.cluster   = false;
    std::vector<loki::search::Candidate> many;
    for (std::uint64_t i = 0; i < 100; ++i) {
        many.push_back(trial(i, 10, 12));
    }
    REQUIRE_THROWS(loki::search::cluster_candidates(many, 1, options));
    REQUIRE(loki::search::cluster_candidates({}, 1).clusters.empty());
}

TEST_CASE("EP winning-width trials use the same candidate interface",
          "[candidates]") {
    const std::array ffa{trial(0, 10, 10, 1, 64), trial(1, 10, 15, 2, 64),
                         trial(2, 10, 12, 3, 64)};
    const std::array ep{ffa[1]};
    const auto all_widths = loki::search::cluster_candidates(ffa, 100);
    const auto best_width = loki::search::cluster_candidates(ep, 100);
    REQUIRE(all_widths.clusters[0].representative.id ==
            best_width.clusters[0].representative.id);
    REQUIRE(all_widths.clusters[0].representative.width ==
            best_width.clusters[0].representative.width);
    REQUIRE(all_widths.clusters[0].representative.snr ==
            best_width.clusters[0].representative.snr);
}

TEST_CASE(
    "Phase-cycle suppression removes narrow-width duplicates without chaining",
    "[candidates]") {
    const std::vector input{trial(0, 10, 20, 1, 100),
                            trial(1, 10.0375, 12, 2, 100),
                            trial(2, 10.075, 11, 1, 100)};
    loki::search::CandidateOptions options;
    options.grouping = "anchored";
    auto out         = loki::search::cluster_candidates(input, 1, options);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 1});
    REQUIRE(out.clusters[0].representative.snr == 20);
    REQUIRE(out.clusters[0].representative.width == 1);
    REQUIRE(out.clusters[0].member_count == 2);
    options.phase_radius = 0;
    REQUIRE(
        loki::search::cluster_candidates(input, 1, options).clusters.size() ==
        3);
    options.phase_radius = 0.5;
    options.cluster      = false;
    REQUIRE(
        loki::search::cluster_candidates(input, 1, options).clusters.size() ==
        3);
}

TEST_CASE("Suppression separates acceleration tracks at identical frequencies",
          "[candidates]") {
    auto first              = trial(0, 10, 20, 1, 100);
    auto neighbor           = first;
    neighbor.id             = 1;
    neighbor.snr            = 12;
    neighbor.derivatives[1] = 0.25; // phase span 1/32 cycle over T=1
    auto distinct           = neighbor;
    distinct.id             = 2;
    distinct.snr            = 11;
    distinct.derivatives[1] = 2; // phase span 1/4 cycle
    const std::array input{first, neighbor, distinct};
    const auto out = loki::search::cluster_candidates(input, 1);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 1});
    REQUIRE(out.clusters[1].representative.id == 2);
    loki::search::CandidateOptions options;
    options.phase_radius = -1;
    REQUIRE_THROWS(options.validate());
    options.phase_radius = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(options.validate());
}

TEST_CASE(
    "Transitive components resolve late bridges and retain the brightest trial",
    "[candidates]") {
    // A and C arrive first; the weaker B connects them only afterwards.
    const std::vector input{trial(0, 10, 20), trial(1, 10.18, 19),
                            trial(2, 10.09, 10)};
    const auto out = loki::search::cluster_candidates(input, 1);
    REQUIRE(out.clusters.size() == 1);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 0});
    REQUIRE(out.clusters[0].member_count == 3);
    REQUIRE(out.clusters[0].representative.id == 0);
    REQUIRE(out.clusters[0].representative.snr == 20);
    REQUIRE(out.clusters[0].phase_extent == Catch::Approx(.18));
    auto reversed = input;
    std::ranges::reverse(reversed);
    const auto other = loki::search::cluster_candidates(reversed, 1);
    REQUIRE(other.membership == std::vector<std::uint64_t>{0, 0, 0});
    REQUIRE(other.clusters[0].representative.id == 0);
    loki::search::CandidateOptions options;
    options.grouping = "anchored";
    REQUIRE(
        loki::search::cluster_candidates(input, 1, options).clusters.size() ==
        2);
}

TEST_CASE(
    "Weak wider trials preserve transitive edges at duplicate physical tracks",
    "[candidates]") {
    const std::array input{
        trial(0, 10, 20, 1, 100), trial(1, 10.15, 19, 1, 100),
        trial(2, 10, 10, 20, 100), trial(3, 10.15, 9, 20, 100)};
    const auto out = loki::search::cluster_candidates(input, 1);
    REQUIRE(out.clusters.size() == 1);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 0, 0});
    REQUIRE(out.clusters[0].representative.width == 1);
    REQUIRE(out.clusters[0].member_count == 4);
    REQUIRE(out.clusters[0].phase_extent == Catch::Approx(.15));
    const std::array best_width_only{input[0], input[1]};
    REQUIRE(
        loki::search::cluster_candidates(best_width_only, 1).clusters.size() ==
        2);
}

TEST_CASE("Transitive graph index is bounded by the host memory budget",
          "[candidates]") {
    loki::search::CandidateOptions options;
    options.memory_gb = .0001;
    std::vector<loki::search::Candidate> input;
    for (std::uint64_t i = 0; i < 100; ++i) {
        input.push_back(trial(i, 10 + static_cast<double>(i), 20));
    }
    REQUIRE_THROWS(loki::search::cluster_candidates(input, 1, options));
    // Duplicate widths at one track do not require 100 independent graph nodes.
    for (auto& c : input) {
        c.derivatives[0] = 10;
    }
    const auto out = loki::search::cluster_candidates(input, 1, options);
    REQUIRE(out.clusters.size() == 1);
    REQUIRE(out.clusters[0].member_count == 100);
}

TEST_CASE("Transitive compression matches the full per-width phase graph",
          "[candidates]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> offset(-.12, .12);
    std::vector<loki::search::Candidate> input;
    for (std::uint64_t i = 0; i < 16; ++i) {
        auto c = trial(3 * i, 10 + offset(rng), static_cast<float>(100 - 3 * i),
                       1, 100);
        c.derivatives[1] = offset(rng);
        c.derivatives[2] = offset(rng);
        for (const std::uint32_t width : {1U, 5U, 20U}) {
            c.id    = input.size();
            c.width = width;
            c.snr   = static_cast<float>(100 - input.size());
            input.push_back(c);
        }
    }
    const auto n = input.size();
    std::vector<std::vector<bool>> connected(n, std::vector<bool>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double duty =
                std::min(input[i].duty_cycle(), input[j].duty_cycle());
            connected[i][j] =
                loki::search::phase_track_distance(input[i], input[j], 2) <=
                std::max(1.0, .05 / duty);
        }
    }
    // Independent transitive closure over EVERY width, without compression.
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                connected[i][j] =
                    connected[i][j] || (connected[i][k] && connected[k][j]);
            }
        }
    }
    const auto out = loki::search::cluster_candidates(input, 2);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE((out.membership[i] == out.membership[j]) ==
                    connected[i][j]);
        }
        auto first      = std::ranges::find(connected[i], true);
        const auto best = static_cast<std::size_t>(
            std::distance(connected[i].begin(), first));
        REQUIRE(out.clusters[out.membership[i]].representative.id ==
                input[best].id);
    }
}

TEST_CASE("Final transitive membership removes gaps after component unions",
          "[candidates]") {
    const std::array input{trial(0, 10, 20), trial(1, 10.18, 19),
                           trial(2, 10.09, 10), trial(3, 30, 9)};
    const auto out = loki::search::cluster_candidates(input, 1);
    REQUIRE(out.membership == std::vector<std::uint64_t>{0, 0, 0, 1});
    REQUIRE(out.clusters.size() == 2);
    REQUIRE(out.clusters[1].representative.id == 3);
}
