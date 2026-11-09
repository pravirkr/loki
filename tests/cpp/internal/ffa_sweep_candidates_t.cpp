#include "lib/search/ffa_sweep_candidates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"

#include "lib/search/cands.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::FFAFreqSweep;
using loki::search::CandidateBuffer;
using loki::search::FFAResultMetadata;
using loki::search::flush_candidates;
using loki::search::PulsarSearchConfig;
using loki::search::RegionDecode;

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
    std::vector<std::uint16_t> width;
    std::vector<std::uint16_t> nbins;
};

WrittenResults read_results(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    WrittenResults out;
    file.getDataSet("snr").read(out.snr);
    file.getDataSet("width").read(out.width);
    file.getDataSet("nbins").read(out.nbins);
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
    auto writer = loki::search::FFAResultWriter(
        path, loki::search::FFAResultWriter::Mode::kWrite);
    FFAResultMetadata meta;
    meta.param_names = {"freq"};
    meta.nbins_min   = table[0].nbins;
    meta.nbins_max   = table[0].nbins;
    meta.ducy_max    = 0.2;
    meta.wtsp        = 1.5;
    writer.write_metadata(meta);

    constexpr SizeType kNParams = 1;
    std::vector<double> scratch(64 * kNParams);
    std::vector<std::uint16_t> width_scratch(64);
    std::vector<std::uint16_t> nbins_scratch(64);

    for (SizeType region = 0; region < table.size(); ++region) {
        const auto n_scores = table[region].get_n_scores();
        for (SizeType s = 0; s < n_scores; ++s) {
            if (buf.is_full()) {
                flush_candidates(buf, table, writer, scratch, width_scratch,
                                 nbins_scratch, kNParams);
            }
            buf.push(static_cast<float>(s) +
                         (100.0F * static_cast<float>(region)),
                     static_cast<uint32_t>(s), static_cast<uint32_t>(region));
        }
    }
    flush_candidates(buf, table, writer, scratch, width_scratch, nbins_scratch,
                     kNParams);
    writer.finalize();
    return read_results(path);
}

// Mirrors the CUDA frequency sweep's copy_candidates_to_host: a chunk's
// compacted results arrive as one block and are appended in space()-sized
// slices, with a flush whenever the buffer fills.
WrittenResults run_sweep_bulk(SizeType capacity,
                              std::span<const RegionDecode> table,
                              const std::filesystem::path& path) {
    CandidateBuffer buf(capacity);
    auto writer = loki::search::FFAResultWriter(
        path, loki::search::FFAResultWriter::Mode::kWrite);
    FFAResultMetadata meta;
    meta.param_names = {"freq"};
    meta.nbins_min   = table[0].nbins;
    meta.nbins_max   = table[0].nbins;
    meta.ducy_max    = 0.2;
    meta.wtsp        = 1.5;
    writer.write_metadata(meta);

    constexpr SizeType kNParams = 1;
    std::vector<double> scratch(64 * kNParams);
    std::vector<std::uint16_t> width_scratch(64);
    std::vector<std::uint16_t> nbins_scratch(64);

    for (SizeType region = 0; region < table.size(); ++region) {
        const auto n_passing = table[region].get_n_scores();
        std::vector<float> scores(n_passing);
        std::vector<uint32_t> indices(n_passing);
        for (SizeType s = 0; s < n_passing; ++s) {
            scores[s] =
                static_cast<float>(s) + (100.0F * static_cast<float>(region));
            indices[s] = static_cast<uint32_t>(s);
        }

        SizeType copied = 0;
        while (copied < n_passing) {
            if (buf.is_full()) {
                flush_candidates(buf, table, writer, scratch, width_scratch,
                                 nbins_scratch, kNParams);
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
    flush_candidates(buf, table, writer, scratch, width_scratch, nbins_scratch,
                     kNParams);
    writer.finalize();
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
    // survives and the accumulator overflows repeatedly. f_max stays below
    // 1 / (nbins_min * tsamp) = 244 Hz, the planner's floor of nbins_min
    // samples per period.
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 6.4e-5;
    const std::vector<ParamLimit> limits{ParamLimit{.min = 50.0, .max = 200.0}};

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

    const auto run = [&](SizeType max_passing_candidates,
                         const std::string& prefix) -> WrittenResults {
        const PulsarSearchConfig cfg(
            kNsamps, kTsamp, /*nbins=*/64, /*eta=*/1.0, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true,
            /*nthreads=*/1, /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0, /*nbins_max=*/256,
            /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/std::nullopt,
            /*nsegments=*/std::nullopt, /*snr_min=*/1.0,
            max_passing_candidates);
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
    const double best_f0 = single.param_sets[best_row];
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

    const auto& wide = table[0];
    for (SizeType s = 0; s < wide.get_n_scores(); ++s) {
        const auto expected_width = wide.widths[s % wide.n_widths];
        REQUIRE(written.width[s] == expected_width);
        REQUIRE(written.nbins[s] == wide.nbins);
    }

    const auto& narrow = table[1];
    const auto offset  = wide.get_n_scores();
    for (SizeType s = 0; s < narrow.get_n_scores(); ++s) {
        const auto expected_width = narrow.widths[s % narrow.n_widths];
        REQUIRE(written.width[offset + s] == expected_width);
        REQUIRE(written.nbins[offset + s] == narrow.nbins);
    }

    std::filesystem::remove(path);
}

TEST_CASE("Non-power-of-2 accelerated FFAFreqSweep shifts candidates to true "
          "observation midpoint",
          "[ffa_sweep_candidates][nsamps]") {
    // Choose kNsamps = 20000 and bseg_brute = 79 at dt = 1 ms so:
    // - true observation is 20.0 s (midpoint 10.0 s),
    // - the tree is 256 brute segments of ceil(20000 / 256) = 79 samples,
    //   20224 samples = 20.224 s (midpoint 10.112 s),
    // - epoch shift delta_t_epoch = -0.112 s, a shift of f0 by 3.8e-5 Hz.
    // Longer brute segments (157 or 625 samples) bias this accelerated
    // signal by ~2.5 frequency bins whatever the padding.
    constexpr SizeType kNsamps     = 20000;
    constexpr SizeType kBsegBrute  = 79;
    constexpr double kTsamp        = 1.0e-3;
    constexpr double kTobs         = static_cast<double>(kNsamps) * kTsamp;
    constexpr double kMidObs       = 0.5 * kTobs;
    constexpr double kSpeedOfLight = 2.99792458e8;

    constexpr double kTrueAcc  = 5000.0;
    constexpr double kTrueFreq = 20.12345;

    std::vector<float> ts_e(kNsamps, 0.0F);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double dt = (static_cast<double>(i) * kTsamp) - kMidObs;
        const double d_delay =
            dt * (1.0 - (0.5 * kTrueAcc * dt / kSpeedOfLight));
        double phase = std::fmod(kTrueFreq * d_delay, 1.0);
        if (phase < 0.0) {
            phase += 1.0;
        }
        const double diff = std::min(phase, 1.0 - phase);
        ts_e[i]           = static_cast<float>(
            10.0 * std::exp(-0.5 * (diff / 0.015) * (diff / 0.015)));
    }

    const std::vector<ParamLimit> limits{
        ParamLimit{.min = 4000.0, .max = 6000.0},
        ParamLimit{.min = 20.115, .max = 20.130},
    };
    const loki::search::FFASearchConfig cfg(
        kNsamps, kTsamp, /*nbins=*/32, /*eta=*/1.0, limits,
        /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/false,
        /*nthreads=*/2, /*max_process_memory_gb=*/4.0,
        /*octave_scale=*/2.0, /*nbins_max=*/128,
        /*nbins_min_lossy_bf=*/32, kBsegBrute, /*nsegments=*/std::nullopt,
        /*snr_min=*/8.0, /*max_passing_candidates=*/100000);

    FFAFreqSweep sweep(cfg, /*show_progress=*/false);
    const auto outdir        = std::filesystem::temp_directory_path();
    const std::string prefix = "loki_np2_accel_midpoint";
    sweep.execute(ts_e, ts_v, outdir, prefix);
    const auto h5_path = outdir / (prefix + "_ffa_results.h5");
    const auto res     = read_results(h5_path);
    std::filesystem::remove(h5_path);

    REQUIRE_FALSE(res.snr.empty());
    const auto best_it = std::ranges::max_element(res.snr);
    const auto best_row =
        static_cast<SizeType>(std::distance(res.snr.begin(), best_it));
    const double got_acc  = res.param_sets[(best_row * 2) + 0];
    const double got_freq = res.param_sets[(best_row * 2) + 1];

    // With the delta_t_epoch shift it recovers kTrueFreq within one
    // frequency bin (~9.5e-4 Hz).
    REQUIRE(cfg.get_bseg_brute() == 79U);
    REQUIRE(std::abs(got_acc - kTrueAcc) < 1000.0);
    REQUIRE(std::abs(got_freq - kTrueFreq) < 1.0e-3);
}
