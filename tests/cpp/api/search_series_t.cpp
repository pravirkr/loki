#include "loki/io/search_series.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/candidate_file.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"

#include "temp_dir.hpp"

using loki::Backend;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::io::PreprocessOptions;
using loki::io::SearchSeries;
using loki::pipelines::EPFreqSweep;
using loki::pipelines::FFAFreqSweep;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

namespace {

constexpr double kTsamp = 64e-6;

bool cuda_available() { return loki::is_available(Backend::kCUDA); }

/// The backends of this build.
std::vector<Exec> backends() {
    std::vector<Exec> out{Exec::cpu(2)};
    if (cuda_available()) {
        out.push_back(Exec::cuda(0));
    }
    return out;
}

/// White noise with a pulse train at @p f0 (duty cycle 5%).
std::vector<float> pulsar(SizeType n, double f0, float amp, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> x(n);
    for (SizeType i = 0; i < n; ++i) {
        x[i] = noise(rng);
        if (std::fmod(static_cast<double>(i) * kTsamp * f0, 1.0) < 0.05) {
            x[i] += amp;
        }
    }
    return x;
}

std::pair<std::vector<float>, std::vector<float>>
to_host(const SearchSeries& s) {
    std::vector<float> e(s.size());
    std::vector<float> v(s.size());
    s.copy_to_host(e, v);
    return {std::move(e), std::move(v)};
}

PreprocessOptions zap_options() {
    PreprocessOptions o;
    o.zap_periodic = true;
    o.birdies      = {{.freq = 300.0, .width = 2.0}};
    return o;
}

/// A row's identity that does not depend on the other rows: its grid cell.
using CellKey = std::tuple<std::uint32_t, std::uint32_t, std::vector<double>>;

/// The best S/N of each cell of @p rows.
std::map<CellKey, float>
best_by_cell(const std::vector<loki::io::CandidateRow>& rows) {
    std::map<CellKey, float> out;
    for (const auto& row : rows) {
        auto [it, fresh] =
            out.try_emplace({row.nbins, row.width, row.params}, row.snr);
        if (!fresh) {
            it->second = std::max(it->second, row.snr);
        }
    }
    return out;
}

/// The rows of @p a and @p b agree. On the CPU they are equal bit for bit.
/// The CUDA sweeps jitter run to run (float atomics, docs/performance.md),
/// which can add or drop a row near the threshold and so renumber the rows
/// after it. There every cell of @p a with S/N above @p snr_floor is in @p b
/// with its S/N within 1e-5.
void require_same_rows(const std::filesystem::path& a,
                       const std::filesystem::path& b,
                       bool exact,
                       float snr_floor) {
    constexpr std::size_t kTop = 1U << 16U;
    const auto sa              = loki::io::read_candidate_file_summary(a, kTop);
    const auto sb              = loki::io::read_candidate_file_summary(b, kTop);
    REQUIRE_FALSE(sa.top.empty());
    if (exact) {
        REQUIRE(sa.n_rows == sb.n_rows);
        REQUIRE(sa.top.size() == sb.top.size());
        for (std::size_t i = 0; i < sa.top.size(); ++i) {
            INFO("row " << i);
            CHECK(sa.top[i].id == sb.top[i].id);
            CHECK(sa.top[i].snr == sb.top[i].snr);
            CHECK(sa.top[i].params == sb.top[i].params);
            CHECK(sa.top[i].width == sb.top[i].width);
            CHECK(sa.top[i].nbins == sb.top[i].nbins);
            CHECK(sa.top[i].leaf == sb.top[i].leaf);
        }
        return;
    }
    const auto cells_a  = best_by_cell(sa.top);
    const auto cells_b  = best_by_cell(sb.top);
    std::size_t checked = 0;
    for (const auto& [key, snr] : cells_a) {
        if (snr < snr_floor) {
            continue;
        }
        INFO("nbins " << std::get<0>(key) << " width " << std::get<1>(key)
                      << " snr " << snr);
        const auto it = cells_b.find(key);
        REQUIRE(it != cells_b.end());
        CHECK(std::abs(it->second - snr) <= 1e-5F * snr);
        ++checked;
    }
    CHECK(checked > 0);
}

} // namespace

TEST_CASE("SearchSeries holds the folding inputs on its backend",
          "[io][search_series]") {
    const auto e = pulsar(4096, 50.0, 0.0F, 1);
    std::vector<float> v(e.size());
    for (SizeType i = 0; i < v.size(); ++i) {
        v[i] = 1.0F + (0.5F * static_cast<float>(i % 7));
    }
    for (const auto& exec : backends()) {
        INFO(loki::to_string(exec.backend));
        const auto s = SearchSeries::from_host(e, v, kTsamp, exec);
        CHECK_FALSE(s.empty());
        CHECK(s.size() == e.size());
        CHECK(s.tsamp() == kTsamp);
        CHECK(s.exec().backend == exec.backend);
        const auto [he, hv] = to_host(s);
        CHECK(he == e);
        CHECK(hv == v);

        std::vector<float> small(e.size() - 1);
        std::vector<float> full(e.size());
        CHECK_THROWS_AS(s.copy_to_host(small, full), std::invalid_argument);
        CHECK_THROWS_AS(SearchSeries::from_host(e, small, kTsamp, exec),
                        std::invalid_argument);
        CHECK_THROWS_AS(SearchSeries::from_host({}, {}, kTsamp, exec),
                        std::invalid_argument);
        CHECK_THROWS_AS(SearchSeries::from_host(e, v, 0.0, exec),
                        std::invalid_argument);

        SearchSeries moved = SearchSeries::from_host(e, v, kTsamp, exec);
        SearchSeries target(std::move(moved));
        CHECK(target.size() == e.size());
    }
    const SearchSeries none;
    CHECK(none.empty());
    CHECK(none.size() == 0);
    CHECK_THROWS_AS(none.exec(), std::logic_error);
}

TEST_CASE("SearchSeries::preprocess matches io::preprocess",
          "[io][search_series][preprocess]") {
    constexpr SizeType kN = 1U << 18U;
    const auto raw        = pulsar(kN, 37.0, 0.3F, 2);
    PreprocessOptions zscore;
    zscore.method = loki::io::PreprocessMethod::kZScore;
    const std::vector<std::pair<std::string, PreprocessOptions>> cases{
        {"robust", PreprocessOptions{}},
        {"robust + zap", zap_options()},
        {"zscore", zscore},
    };
    for (const auto& exec : backends()) {
        for (const auto& [name, options] : cases) {
            INFO(loki::to_string(exec.backend) << " " << name);
            std::vector<float> e(kN);
            std::vector<float> v(kN);
            const auto ref =
                loki::io::preprocess(raw, kTsamp, e, v, options, exec);
            const auto [s, rep] =
                SearchSeries::preprocess(raw, kTsamp, options, exec);
            REQUIRE(s.size() == kN);
            CHECK(s.exec().backend == exec.backend);
            CHECK(rep.n_masked == ref.n_masked);
            CHECK(rep.n_clipped == ref.n_clipped);
            CHECK(rep.n_zapped == ref.n_zapped);
            CHECK(rep.longest_masked_run == ref.longest_masked_run);
            // The same kernels on the same backend: bit for bit.
            const auto [he, hv] = to_host(s);
            CHECK(he == e);
            CHECK(hv == v);
        }
        CHECK_THROWS_AS(SearchSeries::preprocess({}, kTsamp, {}, exec),
                        std::invalid_argument);
        CHECK_THROWS_AS(SearchSeries::preprocess(raw, -1.0, {}, exec),
                        std::invalid_argument);
    }
}

TEST_CASE("FFAFreqSweep searches a SearchSeries as it searches spans",
          "[ffa_sweep][search_series]") {
    constexpr SizeType kN = 1U << 15U;
    const std::vector<ParamLimit> limits{ParamLimit{.min = 50.0, .max = 200.0}};
    const PulsarSearchConfig cfg(
        kN, kTsamp, /*nbins=*/64, /*eta=*/1.0, limits, /*ducy_max=*/0.2,
        /*wtsp=*/1.5, /*use_fourier=*/true, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/2.0,
        /*nbins_max=*/256, /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/std::nullopt, /*nsegments=*/std::nullopt,
        /*snr_min=*/3.0, /*max_passing_candidates=*/1U << 24U);
    const auto raw = pulsar(kN, 123.0, 1.0F, 3);
    const TempDir outdir("loki-search-series-ffa");
    for (const auto& exec : backends()) {
        const std::string tag(loki::to_string(exec.backend));
        INFO(tag);
        const auto [s, rep] = SearchSeries::preprocess(raw, kTsamp, {}, exec);
        const auto [e, v]   = to_host(s);

        FFAFreqSweep sweep(cfg, /*show_progress=*/false, exec);
        sweep.execute(e, v, outdir.path(), tag + "_spans");
        sweep.execute(s, outdir.path(), tag + "_series");
        require_same_rows(outdir / (tag + "_spans_ffa_results.h5"),
                          outdir / (tag + "_series_ffa_results.h5"),
                          exec.backend == Backend::kCPU, /*snr_floor=*/3.5F);
    }
    if (cuda_available()) {
        // A series on another backend is refused.
        const auto e = pulsar(kN, 123.0, 1.0F, 3);
        const std::vector<float> v(kN, 1.0F);
        const auto host_series = SearchSeries::from_host(e, v, kTsamp);
        FFAFreqSweep gpu(cfg, false, Exec::cuda(0));
        CHECK_THROWS_AS(gpu.execute(host_series, outdir.path(), "mismatch"),
                        std::invalid_argument);
    }
    FFAFreqSweep cpu(cfg, false, Exec::cpu(2));
    CHECK_THROWS_AS(cpu.execute(SearchSeries{}, outdir.path(), "empty"),
                    std::invalid_argument);
}

TEST_CASE("EPFreqSweep searches a SearchSeries as it searches spans",
          "[ep_freq_sweep][search_series]") {
    constexpr SizeType kN = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 142.0},
    };
    const PulsarSearchConfig cfg(
        kN, kTsamp, /*nbins=*/32, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024,
        /*nsegments=*/8, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/2);
    const auto raw = pulsar(kN, 141.0, 0.6F, 4);
    const std::vector<SizeType> ref_segs{4U};
    const TempDir outdir("loki-search-series-ep");
    for (const auto& exec : backends()) {
        const std::string tag(loki::to_string(exec.backend));
        INFO(tag);
        const auto [s, rep] = SearchSeries::preprocess(raw, kTsamp, {}, exec);
        const auto [e, v]   = to_host(s);

        EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {},
                          outdir / (tag + "_plan.h5"), std::nullopt, ref_segs,
                          exec);
        sweep.execute(e, v, outdir.path(), tag + "_spans");
        sweep.execute(s, outdir.path(), tag + "_series");
        require_same_rows(outdir / (tag + "_spans_ep_results.h5"),
                          outdir / (tag + "_series_ep_results.h5"),
                          exec.backend == Backend::kCPU, /*snr_floor=*/5.5F);
    }
    if (cuda_available()) {
        const std::vector<float> v(kN, 1.0F);
        const auto host_series = SearchSeries::from_host(raw, v, kTsamp);
        EPFreqSweep gpu(cfg, false, 0.1F, "taylor", 0.1F, {},
                        outdir / "mismatch_plan.h5", std::nullopt, ref_segs,
                        Exec::cuda(0));
        CHECK_THROWS_AS(gpu.execute(host_series, outdir.path(), "mismatch"),
                        std::invalid_argument);
    }
}
