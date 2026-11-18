#include "lib/search/slice_scratch.hpp"

#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/search/candidates.hpp"

#include "lib/search/candidate_processing.hpp"
#include "lib/search/cands.hpp"

#include "temp_dir.hpp"

using loki::SizeType;
using loki::search::PruneResultWriter;
using loki::search::PruneStatsCollection;
using loki::search::detail::LeafReductionSpec;
using loki::search::detail::SliceScratch;

namespace {

constexpr SizeType kNparams = 2; // [accel, freq]
constexpr SizeType kStride  = (kNparams + 2) * 2;

/// Final leaves of a run: more rows than one batch of the scratch reader
/// (2^16), with many rows per frequency cell so a reduction drops some.
struct Leaves {
    std::vector<double> leaves;
    std::vector<float> scores;
    std::vector<float> scores_ep;
    std::vector<std::uint32_t> widths;
};

Leaves make_leaves(SizeType n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> freq(100.0, 100.5);
    std::uniform_real_distribution<double> accel(-5.0, 5.0);
    std::uniform_real_distribution<float> snr(3.0F, 12.0F);
    std::uniform_int_distribution<std::uint32_t> width(1, 8);
    Leaves out;
    for (SizeType i = 0; i < n; ++i) {
        const double row[kStride] = {accel(rng), 0.1, freq(rng), 1e-4,
                                     0.0,        0.0, 0.0,       0.0};
        out.leaves.insert(out.leaves.end(), row, row + kStride);
        out.scores.push_back(snr(rng));
        out.scores_ep.push_back(snr(rng));
        out.widths.push_back(width(rng));
    }
    return out;
}

template <typename T>
std::vector<T> read(const HighFive::Group& run, const std::string& name) {
    std::vector<T> v;
    run.getDataSet(name).read(v);
    return v;
}

/// The run groups of @p a and @p b hold the same rows and reduction.
void require_same_run(const std::filesystem::path& a,
                      const std::filesystem::path& b) {
    const HighFive::File fa(a.string(), HighFive::File::ReadOnly);
    const HighFive::File fb(b.string(), HighFive::File::ReadOnly);
    const auto ra = fa.getGroup("runs/000_00");
    const auto rb = fb.getGroup("runs/000_00");
    std::vector<std::vector<std::vector<double>>> pa;
    std::vector<std::vector<std::vector<double>>> pb;
    ra.getDataSet("param_sets").read(pa);
    rb.getDataSet("param_sets").read(pb);
    CHECK(pa == pb);
    CHECK(read<float>(ra, "scores") == read<float>(rb, "scores"));
    CHECK(read<float>(ra, "scores_ep") == read<float>(rb, "scores_ep"));
    CHECK(read<std::uint32_t>(ra, "width") == read<std::uint32_t>(rb, "width"));
    CHECK(ra.exist("leaf_index") == rb.exist("leaf_index"));
    if (ra.exist("leaf_index")) {
        CHECK(read<std::uint32_t>(ra, "leaf_index") ==
              read<std::uint32_t>(rb, "leaf_index"));
        for (const auto* attr : {"n_final_leaves", "n_leaves_above_snr_min",
                                 "n_cell_dropped", "n_leaves_retained"}) {
            INFO(attr);
            CHECK(ra.getAttribute(attr).read<std::uint64_t>() ==
                  rb.getAttribute(attr).read<std::uint64_t>());
        }
    }
}

} // namespace

TEST_CASE("Slice scratch keeps rows in order and removes its files",
          "[seed_slices]") {
    const loki::test::TempDir dir("loki-slice-scratch");
    const auto stem = dir / "run.slices";
    const auto in   = make_leaves(1000, 1);
    {
        SliceScratch scratch(stem, kStride);
        // Two appends per column, as two slices give.
        const auto half = SizeType{400};
        scratch.append_leaves(std::span(in.leaves).first(half * kStride));
        scratch.append_leaves(std::span(in.leaves).subspan(half * kStride));
        scratch.append_scores(std::span(in.scores).first(half));
        scratch.append_scores(std::span(in.scores).subspan(half));
        scratch.append_scores_ep(in.scores_ep);
        scratch.append_widths(in.widths);
        REQUIRE(scratch.rows() == 1000);
        CHECK(scratch.has_widths());
        CHECK(scratch.bytes() == 1000 * ((kStride * 8) + 4 + 4 + 4));
        scratch.seal();
        std::vector<double> leaves(300 * kStride);
        std::vector<float> scores(300);
        std::vector<float> scores_ep(300);
        std::vector<std::uint32_t> widths(300);
        scratch.read(500, 300, leaves, scores, scores_ep, widths);
        CHECK(std::vector<double>(in.leaves.begin() + (500 * kStride),
                                  in.leaves.begin() + (800 * kStride)) ==
              leaves);
        CHECK(std::vector<float>(in.scores.begin() + 500,
                                 in.scores.begin() + 800) == scores);
        CHECK(std::vector<std::uint32_t>(in.widths.begin() + 500,
                                         in.widths.begin() + 800) == widths);
        CHECK_THROWS_AS(
            scratch.read(900, 200, leaves, scores, scores_ep, widths),
            std::logic_error);
        CHECK(std::filesystem::exists(stem.string() + ".leaves"));
    }
    CHECK_FALSE(std::filesystem::exists(stem.string() + ".leaves"));
    CHECK_FALSE(std::filesystem::exists(stem.string() + ".widths"));

    // Every stem is new, even for the same run and result file: a CUDA
    // sweep fills one chunk's scratch while it writes the last one's.
    const auto result = dir / "tmp_000_00_results.h5";
    CHECK(loki::search::detail::slice_scratch_stem(result, "000_00") !=
          loki::search::detail::slice_scratch_stem(result, "000_00"));
    CHECK(loki::search::detail::slice_scratch_stem(result, "000_00")
              .parent_path() == dir.path());

    SliceScratch uneven(dir / "uneven", kStride);
    uneven.append_scores(std::span(in.scores).first(3));
    CHECK_THROWS_AS(uneven.seal(), std::logic_error);
}

TEST_CASE("A run written from the slice scratch equals the in-memory write",
          "[seed_slices][hdf5]") {
    const loki::test::TempDir dir("loki-slice-scratch-writer");
    // Over two batches of the scratch reader, so batch edges are crossed.
    const auto in = make_leaves((SizeType{1} << 17U) + 12345, 2);
    const auto n  = in.scores.size();
    const std::vector<SizeType> snail{0, 1, 2, 3};
    const PruneStatsCollection stats;

    loki::search::CandidateOptions options;
    options.max_per_cell = 3;
    options.cell_bins    = 64;
    const LeafReductionSpec spec{
        .options = options, .snr_min = 6.0, .tobs = 100.0, .nbins = 16};

    for (const bool reduce : {false, true}) {
        INFO("reduce " << reduce);
        const auto tag    = std::string(reduce ? "r" : "a");
        const auto memory = dir / (tag + "_memory.h5");
        const auto disk   = dir / (tag + "_scratch.h5");
        const auto* red   = reduce ? &spec : nullptr;
        {
            PruneResultWriter writer(memory);
            writer.write_run_results(
                "000_00", snail,
                {std::span(const_cast<double*>(in.leaves.data()),
                           in.leaves.size()),
                 {}},
                {std::span(const_cast<float*>(in.scores.data()), n), {}},
                {std::span(const_cast<float*>(in.scores_ep.data()), n), {}},
                1.5, n, kNparams, stats, in.widths, red);
        }
        {
            SliceScratch scratch(dir / (tag + "_run"), kStride);
            scratch.append_leaves(in.leaves);
            scratch.append_scores(in.scores);
            scratch.append_scores_ep(in.scores_ep);
            scratch.append_widths(in.widths);
            PruneResultWriter writer(disk);
            writer.write_run_results("000_00", snail, scratch, 1.5, kNparams,
                                     stats, red);
        }
        require_same_run(memory, disk);
        if (reduce) {
            const HighFive::File f(disk.string(), HighFive::File::ReadOnly);
            const auto run = f.getGroup("runs/000_00");
            const auto kept =
                run.getAttribute("n_leaves_retained").read<std::uint64_t>();
            CHECK(kept > 0);
            CHECK(kept < run.getAttribute("n_leaves_above_snr_min")
                             .read<std::uint64_t>());
        }
    }

    // An empty run.
    SliceScratch empty(dir / "empty_run", kStride);
    PruneResultWriter writer(dir / "empty.h5");
    writer.write_run_results("000_00", snail, empty, 0.0, kNparams, stats,
                             &spec);
    const HighFive::File f((dir / "empty.h5").string(),
                           HighFive::File::ReadOnly);
    CHECK(f.getGroup("runs/000_00")
              .getDataSet("scores")
              .getDimensions()
              .front() == 0);
}
