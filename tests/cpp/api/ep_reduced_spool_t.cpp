// A candidate sweep reduces each run's final leaves as it writes them: the
// product equals reducing a full spool at publication.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"

#include "temp_dir.hpp"

using loki::Backend;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::EPFreqSweep;
using loki::search::CandidateOptions;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

namespace {
template <typename T>
std::vector<T> column(const HighFive::File& file, const std::string& path) {
    return file.getDataSet(path).read<std::vector<T>>();
}

std::uint64_t count(const HighFive::File& file, const std::string& name) {
    return file.getAttribute(name).read<std::uint64_t>();
}

void require_same_product(const std::filesystem::path& a,
                          const std::filesystem::path& b) {
    const HighFive::File fa(a.string(), HighFive::File::ReadOnly);
    const HighFive::File fb(b.string(), HighFive::File::ReadOnly);
    for (const auto* name : {"n_detections", "n_cell_dropped", "n_dominated",
                             "n_clusters", "n_candidates"}) {
        CAPTURE(name);
        REQUIRE(count(fa, name) == count(fb, name));
    }
    for (const auto* table : {"detections", "clusters", "candidates"}) {
        CAPTURE(table);
        const auto t = std::string(table) + "/";
        REQUIRE(column<float>(fa, t + "snr") == column<float>(fb, t + "snr"));
        REQUIRE(column<double>(fa, t + "f0") == column<double>(fb, t + "f0"));
        REQUIRE(column<double>(fa, t + "f1") == column<double>(fb, t + "f1"));
        REQUIRE(column<std::uint64_t>(fa, t + "id") ==
                column<std::uint64_t>(fb, t + "id"));
        REQUIRE(column<std::uint64_t>(fa, t + "cluster_id") ==
                column<std::uint64_t>(fb, t + "cluster_id"));
        REQUIRE(column<std::uint64_t>(fa, t + "member_count") ==
                column<std::uint64_t>(fb, t + "member_count"));
        REQUIRE(column<std::uint32_t>(fa, t + "source_index") ==
                column<std::uint32_t>(fb, t + "source_index"));
    }
    const auto runs = column<std::string>(fa, "search/sources/run_path");
    REQUIRE(runs == column<std::string>(fb, "search/sources/run_path"));
    for (const auto& run : runs) {
        CAPTURE(run);
        const auto ra = fa.getGroup(run);
        const auto rb = fb.getGroup(run);
        for (const auto* name :
             {"n_final_leaves", "n_leaves_above_snr_min", "n_cell_dropped",
              "n_dominated", "n_leaves_retained"}) {
            CAPTURE(name);
            REQUIRE(ra.getAttribute(name).read<std::uint64_t>() ==
                    rb.getAttribute(name).read<std::uint64_t>());
        }
        REQUIRE(ra.getAttribute("max_cell_dropped_snr").read<float>() ==
                rb.getAttribute("max_cell_dropped_snr").read<float>());
        REQUIRE(column<std::uint32_t>(fa, run + "/leaf_index") ==
                column<std::uint32_t>(fb, run + "/leaf_index"));
        REQUIRE(column<float>(fa, run + "/scores") ==
                column<float>(fb, run + "/scores"));
        REQUIRE(column<std::uint32_t>(fa, run + "/width") ==
                column<std::uint32_t>(fb, run + "/width"));
    }
}

/// Every run records where it was reduced (and with which options).
bool reduced_at(const std::filesystem::path& product, const std::string& at) {
    const HighFive::File file(product.string(), HighFive::File::ReadOnly);
    const auto runs = column<std::string>(file, "search/sources/run_path");
    for (const auto& run : runs) {
        const auto group = file.getGroup(run);
        if (!group.hasAttribute("reduced_max_per_cell") ||
            group.getAttribute("reduced_at").read<std::string>() != at) {
            return false;
        }
    }
    return !runs.empty();
}

/// @p exact: also compare with reducing a full spool of a second search,
/// which needs a bit-reproducible backend.
void check_reduced_sweep(const Exec& exec, bool exact) {
    const TempDir outdir("loki-ep-reduced-spool");
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 142.0},
    };
    const PulsarSearchConfig cfg(
        kNsamps, 64e-6, /*nbins=*/32, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024,
        /*nsegments=*/8, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/2);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    // Noise plus a 141 Hz pulse (duty 0.1): many leaves pass snr_min.
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double phase = 141.0 * static_cast<double>(i) * 64e-6;
        ts_e[i] = noise(rng) + (phase - std::floor(phase) < 0.1 ? 0.3F : 0.0F);
    }
    const std::vector<SizeType> ref_segs{2U, 4U};
    auto options         = CandidateOptions::ep_defaults();
    options.max_per_cell = 4; // small enough that every run drops leaves
    options.memory_gb    = 0.05;

    // A full spool, reduced at publication.
    EPFreqSweep diagnostics(cfg, false, 0.1F, "taylor", 0.1F, {}, std::nullopt,
                            std::nullopt, ref_segs, exec);
    diagnostics.execute(ts_e, ts_v, outdir.path(), "full");
    const auto full      = outdir / "full_ep_results.h5";
    const auto from_full = outdir / "from_full.h5";
    EPFreqSweep::publish_candidates(full, from_full, options);
    // A product reduced at publication records its options too.
    REQUIRE(reduced_at(from_full, "publication"));
    auto keep_all         = options;
    keep_all.max_per_cell = 0;
    REQUIRE_THROWS_AS(
        EPFreqSweep::publish_candidates(from_full, outdir / "w.h5", keep_all),
        std::invalid_argument);

    // The same search, reducing each run as it writes it.
    EPFreqSweep sweep(cfg, options, false, 0.1F, "taylor", 0.1F, {},
                      std::nullopt, std::nullopt, ref_segs, exec);
    sweep.execute(ts_e, ts_v, outdir.path(), "reduced");
    const auto reduced = outdir / "reduced_ep_results.h5";
    {
        const HighFive::File product(reduced.string(),
                                     HighFive::File::ReadOnly);
        REQUIRE(count(product, "n_cell_dropped") > 0);
        REQUIRE(count(product, "n_detections") > 0);
    }
    REQUIRE(reduced_at(reduced, "search"));
    if (exact) {
        require_same_product(from_full, reduced);
    }

    // A reduced product republishes to the same product with the same
    // options: the reduction is idempotent.
    const auto again = outdir / "again.h5";
    EPFreqSweep::publish_candidates(reduced, again, options);
    require_same_product(reduced, again);
    REQUIRE(reduced_at(again, "search"));

    // Other options cannot be honoured: the dropped leaves are gone.
    auto full_options         = options;
    full_options.max_per_cell = 0;
    REQUIRE_THROWS_WITH(
        EPFreqSweep::publish_candidates(reduced, outdir / "x.h5", full_options),
        Catch::Matchers::ContainsSubstring("max_per_cell = 0 to keep"));
    auto wider         = options;
    wider.max_per_cell = 8;
    REQUIRE_THROWS_AS(
        EPFreqSweep::publish_candidates(reduced, outdir / "y.h5", wider),
        std::invalid_argument);
    REQUIRE_THROWS_AS(EPFreqSweep::publish_candidates(reduced, outdir / "z.h5",
                                                      options, {}, 6.0),
                      std::invalid_argument);
}
} // namespace

TEST_CASE("A CPU sweep that reduces runs as it writes them publishes the same "
          "product as reducing a full spool",
          "[ep_freq_sweep][candidates]") {
    check_reduced_sweep(Exec{}, true);
}

TEST_CASE("A CUDA sweep reduces runs as it writes them",
          "[ep_freq_sweep][candidates][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    check_reduced_sweep(Exec::cuda(0), false);
}
