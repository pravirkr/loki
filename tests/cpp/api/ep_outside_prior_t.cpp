// The EP parameter box applies at the pruning's reference segment, while
// leaves are reported at the observation midpoint: a jerk search reports
// accelerations up to a_max + j_max * T / 2, and flags those outside the box.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"

#include "temp_dir.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::EPFreqSweep;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

TEST_CASE("EP leaves are reported at the midpoint, up to a_max + j_max T / 2, "
          "and flagged outside the prior box",
          "[ep_freq_sweep][candidates]") {
    const TempDir outdir("loki-ep-outside-prior");
    constexpr SizeType kNsamps = 1U << 16U;
    constexpr double kTsamp    = 64e-6;
    // Unphysical, but over a 4.2 s observation a box this large spans
    // several acceleration cells and j_max * T / 2 is twice a_max.
    constexpr double kAccMax  = 2e5; // m/s^2
    constexpr double kJerkMax = 2e5; // m/s^3
    const double tobs         = static_cast<double>(kNsamps) * kTsamp;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -kJerkMax, .max = kJerkMax},
        ParamLimit{.min = -kAccMax, .max = kAccMax},
        ParamLimit{.min = 140.0, .max = 142.0},
    };
    const PulsarSearchConfig cfg(
        kNsamps, kTsamp, /*nbins=*/32, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024,
        /*nsegments=*/8, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/3);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double phase = 141.0 * static_cast<double>(i) * kTsamp;
        ts_e[i] = noise(rng) + (phase - std::floor(phase) < 0.1 ? 0.3F : 0.0F);
    }
    // Reference segments away from the midpoint.
    const std::vector<SizeType> ref_segs{1U, 6U};
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, std::nullopt,
                      std::nullopt, ref_segs);
    sweep.execute(ts_e, ts_v, outdir.path(), "spool");
    const auto spool = outdir / "spool_ep_results.h5";

    // Every final leaf obeys the relation, and some lie outside the box.
    const double reach  = kAccMax + (kJerkMax * tobs / 2);
    std::size_t outside = 0;
    std::size_t leaves  = 0;
    {
        const HighFive::File file(spool.string(), HighFive::File::ReadOnly);
        const auto chunks = file.getGroup("chunks");
        for (const auto& chunk : chunks.listObjectNames()) {
            const auto runs = chunks.getGroup(chunk).getGroup("runs");
            for (const auto& run : runs.listObjectNames()) {
                const auto ds   = runs.getGroup(run).getDataSet("param_sets");
                const auto dims = ds.getDimensions();
                std::vector<double> rows(dims[0] * dims[1] * dims[2]);
                ds.read_raw(rows.data());
                for (std::size_t i = 0; i < dims[0]; ++i) {
                    const double jerk = rows[i * dims[1] * dims[2]];
                    const double acc  = rows[(i * dims[1] + 1) * dims[2]];
                    CAPTURE(run, i, jerk, acc);
                    REQUIRE(std::abs(acc) <= reach * (1 + 1e-9));
                    REQUIRE(std::abs(jerk) <= kJerkMax * (1 + 1e-9));
                    outside += std::abs(acc) > kAccMax ? 1 : 0;
                    ++leaves;
                }
            }
        }
    }
    CAPTURE(leaves, outside);
    REQUIRE(leaves > 0);
    REQUIRE(outside > 0);

    // The product flags exactly the detections outside the box.
    auto options         = loki::search::CandidateOptions::ep_defaults();
    options.max_per_cell = 0;
    options.memory_gb    = 0.05;
    const auto product   = outdir / "product.h5";
    EPFreqSweep::publish_candidates(spool, product, options, {}, 0.0);
    const HighFive::File file(product.string(), HighFive::File::ReadOnly);
    const auto params = file.getDataSet("detections/param_sets")
                            .read<std::vector<std::vector<double>>>();
    const auto flags  = file.getDataSet("detections/outside_prior")
                            .read<std::vector<std::uint8_t>>();
    REQUIRE(params.size() == leaves);
    std::size_t flagged = 0;
    for (std::size_t i = 0; i < params.size(); ++i) {
        const bool out = std::abs(params[i][0]) > kJerkMax ||
                         std::abs(params[i][1]) > kAccMax;
        REQUIRE((flags[i] != 0) == out);
        flagged += flags[i];
    }
    REQUIRE(flagged == outside);

    // Every EP detection carries its cell half-widths; a polynomial search
    // never resolves an orbit.
    const auto has_errors = file.getDataSet("detections/has_parameter_errors")
                                .read<std::vector<std::uint8_t>>();
    const auto errors     = file.getDataSet("detections/param_errors")
                                .read<std::vector<std::vector<double>>>();
    const auto basis      = file.getDataSet("detections/leaf_basis")
                                .read<std::vector<std::uint8_t>>();
    const auto p_orb =
        file.getDataSet("detections/p_orb").read<std::vector<float>>();
    REQUIRE(errors.size() == params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
        CHECK(has_errors[i] == 1);
        CHECK(basis[i] == 0);
        CHECK(p_orb[i] == 0.0F);
        for (const double e : errors[i]) {
            CHECK(e > 0.0);
        }
    }
}
