#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/ffa_sweep_candidates.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::CandidateBuffer;
using loki::algorithms::FFAFreqSweep;
using loki::algorithms::flush_candidates;
using loki::algorithms::RegionDecode;
using loki::search::PulsarSearchConfig;

namespace {

// Two chunks with deliberately different width counts, mirroring the way
// generate_box_width_trials() grows the boxcar set with nbins.
std::vector<RegionDecode> make_decode_table() {
    RegionDecode wide;
    wide.param_limits  = {ParamLimit{.min = 1.0, .max = 2.0}};
    wide.param_counts  = {8};
    wide.param_strides = {1};
    wide.widths        = {1, 2, 3, 4, 6};
    wide.n_widths      = wide.widths.size();
    wide.ncoords       = 8;
    wide.nbins         = 32;
    wide.nsegments     = 1;

    RegionDecode narrow;
    narrow.param_limits  = {ParamLimit{.min = 2.0, .max = 4.0}};
    narrow.param_counts  = {5};
    narrow.param_strides = {1};
    narrow.widths        = {1, 2};
    narrow.n_widths      = narrow.widths.size();
    narrow.ncoords       = 5;
    narrow.nbins         = 8;
    narrow.nsegments     = 1;

    return {wide, narrow};
}

struct WrittenResults {
    std::vector<double> param_sets;
    std::vector<float> snr;
};

WrittenResults read_results(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    WrittenResults out;
    file.getDataSet("snr").read(out.snr);
    std::vector<std::vector<double>> rows;
    file.getDataSet("param_sets").read(rows);
    for (const auto& row : rows) {
        out.param_sets.insert(out.param_sets.end(), row.begin(), row.end());
    }
    return out;
}

// Run the same candidate stream through the buffer, flushing either only at
// the end or every time the buffer fills.
WrittenResults run_sweep(SizeType capacity,
                         std::span<const RegionDecode> table,
                         const std::filesystem::path& path) {
    CandidateBuffer buf(capacity);
    auto writer = loki::cands::FFAResultWriter(
        path, loki::cands::FFAResultWriter::Mode::kWrite);
    writer.write_metadata({"freq", "width"}, table[0].nbins, 0.2, 1.5);

    constexpr SizeType kNParams = 1;
    std::vector<double> scratch(64 * (kNParams + 1));

    for (SizeType region = 0; region < table.size(); ++region) {
        const auto n_scores = table[region].get_n_scores();
        for (SizeType s = 0; s < n_scores; ++s) {
            if (buf.is_full()) {
                flush_candidates(buf, table, writer, scratch, kNParams);
            }
            buf.push(static_cast<float>(s) + (100.0F * region),
                     static_cast<uint32_t>(s), static_cast<uint32_t>(region));
        }
    }
    flush_candidates(buf, table, writer, scratch, kNParams);
    return read_results(path);
}

// Mirrors FFAFreqSweepCUDA::copy_candidates_to_host: a chunk's compacted
// results arrive as one block and are appended in space()-sized slices, with
// a flush whenever the buffer fills.
WrittenResults run_sweep_bulk(SizeType capacity,
                              std::span<const RegionDecode> table,
                              const std::filesystem::path& path) {
    CandidateBuffer buf(capacity);
    auto writer = loki::cands::FFAResultWriter(
        path, loki::cands::FFAResultWriter::Mode::kWrite);
    writer.write_metadata({"freq", "width"}, table[0].nbins, 0.2, 1.5);

    constexpr SizeType kNParams = 1;
    std::vector<double> scratch(64 * (kNParams + 1));

    for (SizeType region = 0; region < table.size(); ++region) {
        const auto n_passing = table[region].get_n_scores();
        std::vector<float> scores(n_passing);
        std::vector<uint32_t> indices(n_passing);
        for (SizeType s = 0; s < n_passing; ++s) {
            scores[s]  = static_cast<float>(s) + (100.0F * region);
            indices[s] = static_cast<uint32_t>(s);
        }

        SizeType copied = 0;
        while (copied < n_passing) {
            if (buf.is_full()) {
                flush_candidates(buf, table, writer, scratch, kNParams);
            }
            const SizeType n = std::min(buf.get_space(), n_passing - copied);
            std::copy_n(scores.begin() + static_cast<loki::IndexType>(copied),
                        n, buf.get_scores_tail(n).begin());
            std::copy_n(indices.begin() + static_cast<loki::IndexType>(copied),
                        n, buf.get_indices_tail(n).begin());
            buf.commit(n, static_cast<uint32_t>(region));
            copied += n;
        }
    }
    flush_candidates(buf, table, writer, scratch, kNParams);
    return read_results(path);
}

} // namespace

TEST_CASE("CandidateBuffer tracks capacity and rejects overfilling",
          "[ffa_sweep_candidates]") {
    CandidateBuffer buf(3);
    REQUIRE(buf.get_capacity() == 3);
    REQUIRE(buf.get_size() == 0);
    REQUIRE(buf.get_space() == 3);
    REQUIRE_FALSE(buf.is_full());

    buf.push(1.0F, 10, 0);
    buf.push(2.0F, 20, 1);
    REQUIRE(buf.get_size() == 2);
    REQUIRE(buf.get_space() == 1);

    // Bulk API used by the CUDA path: fill the tail, then publish it.
    const std::vector<float> scores{3.0F};
    const std::vector<uint32_t> indices{30};
    std::ranges::copy(scores, buf.get_scores_tail(1).begin());
    std::ranges::copy(indices, buf.get_indices_tail(1).begin());
    buf.commit(1, 7);

    REQUIRE(buf.is_full());
    REQUIRE(buf.get_scores()[2] == 3.0F);
    REQUIRE(buf.get_local_indices()[2] == 30U);
    REQUIRE(buf.get_region_ids()[2] == 7U);

    // A bulk request beyond capacity must throw rather than corrupt memory.
    REQUIRE_THROWS(buf.get_scores_tail(1));

    buf.clear();
    REQUIRE(buf.get_size() == 0);
    REQUIRE(buf.get_space() == 3);
}

TEST_CASE("Flushing on overflow yields the same file as one final flush",
          "[ffa_sweep_candidates]") {
    const auto table       = make_decode_table();
    const auto tmp         = std::filesystem::temp_directory_path();
    const auto path_single = tmp / "loki_cands_single.h5";
    const auto path_many   = tmp / "loki_cands_many.h5";

    SizeType total = 0;
    for (const auto& dec : table) {
        total += dec.get_n_scores();
    }

    // Capacity large enough to hold everything: exactly one flush.
    const auto single = run_sweep(total, table, path_single);
    // Capacity of 3 forces many mid-sweep flushes, including one that splits
    // a chunk across two writes.
    const auto many = run_sweep(3, table, path_many);

    REQUIRE(single.snr.size() == total);
    REQUIRE(single.snr == many.snr);
    REQUIRE(single.param_sets == many.param_sets);

    std::filesystem::remove(path_single);
    std::filesystem::remove(path_many);
}

TEST_CASE("Bulk slice appends survive a capacity smaller than a chunk",
          "[ffa_sweep_candidates]") {
    const auto table     = make_decode_table();
    const auto tmp       = std::filesystem::temp_directory_path();
    const auto path_ref  = tmp / "loki_cands_bulk_ref.h5";
    const auto path_tiny = tmp / "loki_cands_bulk_tiny.h5";

    SizeType total = 0;
    for (const auto& dec : table) {
        total += dec.get_n_scores();
    }

    // Capacity 3 is smaller than either chunk, so every chunk is split across
    // several slices and flushes. This is the CUDA path's worst case.
    const auto reference = run_sweep(total, table, path_ref);
    const auto bulk      = run_sweep_bulk(3, table, path_tiny);

    REQUIRE(bulk.snr.size() == total);
    REQUIRE(bulk.snr == reference.snr);
    REQUIRE(bulk.param_sets == reference.param_sets);

    std::filesystem::remove(path_ref);
    std::filesystem::remove(path_tiny);
}

TEST_CASE("A real multi-chunk sweep is unaffected by accumulator capacity",
          "[ffa_sweep_candidates][.slow]") {
    // A wide frequency range so the planner produces several chunks with
    // differing nbins, and a low snr_min so a large fraction of the grid
    // survives and the accumulator overflows repeatedly.
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 6.4e-5;
    const std::vector<ParamLimit> limits{ParamLimit{.min = 50.0, .max = 500.0}};

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (auto& sample : ts_e) {
        sample = noise(rng);
    }
    // Inject a bright signal so at least one strong candidate exists.
    constexpr double kInjectedF0 = 123.0;
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double phase =
            std::fmod(static_cast<double>(i) * kTsamp * kInjectedF0, 1.0);
        if (phase < 0.05) {
            ts_e[i] += 8.0F;
        }
    }

    auto run = [&](SizeType max_passing_candidates,
                   const std::string& prefix) -> WrittenResults {
        const PulsarSearchConfig cfg(
            kNsamps, kTsamp, /*nbins=*/64, /*eta=*/1.0, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true,
            /*nthreads=*/1, /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0, /*nbins_max=*/256,
            /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/std::nullopt,
            /*bseg_ffa=*/std::nullopt, /*snr_min=*/1.0, max_passing_candidates);
        FFAFreqSweep sweep(cfg, /*show_progress=*/false);
        const auto outdir = std::filesystem::temp_directory_path();
        sweep.execute(ts_e, ts_v, outdir, prefix);
        return read_results(outdir / (prefix + "_ffa_results.h5"));
    };

    // Capacity far beyond what a run produces: a single flush at the end.
    const auto single = run(1U << 24U, "loki_sweep_single");
    // Capacity of 1000 forces hundreds of mid-chunk flushes.
    const auto many = run(1000, "loki_sweep_many");

    REQUIRE_FALSE(single.snr.empty());
    REQUIRE(single.snr == many.snr);
    REQUIRE(single.param_sets == many.param_sets);

    // The strongest candidate must decode to a frequency near the injection.
    const auto best = std::ranges::max_element(single.snr);
    const auto best_row =
        static_cast<SizeType>(std::distance(single.snr.begin(), best));
    const SizeType n_cols = single.param_sets.size() / single.snr.size();
    const double best_f0  = single.param_sets[best_row * n_cols];
    REQUIRE(std::abs(best_f0 - kInjectedF0) < 1.0);

    const auto tmp = std::filesystem::temp_directory_path();
    std::filesystem::remove(tmp / "loki_sweep_single_ffa_results.h5");
    std::filesystem::remove(tmp / "loki_sweep_many_ffa_results.h5");
}

TEST_CASE("Candidates decode against their own chunk's width set",
          "[ffa_sweep_candidates]") {
    const auto table = make_decode_table();
    const auto path =
        std::filesystem::temp_directory_path() / "loki_cands_decode.h5";

    SizeType total = 0;
    for (const auto& dec : table) {
        total += dec.get_n_scores();
    }
    const auto written = run_sweep(total, table, path);

    // Row layout is [freq, width]; chunk 0 has 5 widths and chunk 1 has 2, so
    // decoding both with a single width count would mislabel one of them.
    const auto& wide = table[0];
    for (SizeType s = 0; s < wide.get_n_scores(); ++s) {
        const auto expected_width =
            static_cast<double>(wide.widths[s % wide.n_widths]);
        REQUIRE(written.param_sets[(s * 2) + 1] == expected_width);
    }

    const auto& narrow = table[1];
    const auto offset  = wide.get_n_scores();
    for (SizeType s = 0; s < narrow.get_n_scores(); ++s) {
        const auto expected_width =
            static_cast<double>(narrow.widths[s % narrow.n_widths]);
        REQUIRE(written.param_sets[((offset + s) * 2) + 1] == expected_width);
    }

    std::filesystem::remove(path);
}
