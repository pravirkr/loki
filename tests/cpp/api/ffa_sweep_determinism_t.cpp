#include <cmath>
#include <filesystem>
#include <format>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::FFAFreqSweep;
using loki::search::PulsarSearchConfig;

namespace {

std::vector<float> read_snr(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    std::vector<float> snr;
    if (file.exist("snr")) {
        file.getDataSet("snr").read(snr);
    }
    return snr;
}

std::vector<double> read_param_sets_flat(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    std::vector<std::vector<double>> rows;
    file.getDataSet("param_sets").read(rows);
    std::vector<double> flat;
    for (const auto& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

} // namespace

TEST_CASE("CPU FFA sweep is invariant to OpenMP thread count",
          "[ffa_sweep][.slow]") {
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 6.4e-5;
    const std::vector<ParamLimit> limits{{50.0, 200.0}};

    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (auto& sample : ts_e) {
        sample = noise(rng);
    }

    const auto run = [&](int nthreads) {
        const PulsarSearchConfig cfg(
            kNsamps, kTsamp, /*nbins=*/64, /*eta=*/1.0, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true, nthreads,
            /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
            /*nbins_max=*/128, /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/std::nullopt, /*bseg_ffa=*/std::nullopt,
            /*snr_min=*/3.0, /*max_passing_candidates=*/1U << 20U);
        FFAFreqSweep sweep(cfg, false);
        const auto outdir = std::filesystem::temp_directory_path();
        const auto prefix = std::format("loki_det_{}", nthreads);
        sweep.execute(ts_e, ts_v, outdir, prefix);
        const auto path   = outdir / (prefix + "_ffa_results.h5");
        const auto snr    = read_snr(path);
        const auto params = read_param_sets_flat(path);
        std::filesystem::remove(path);
        return std::pair{snr, params};
    };

    const auto one = run(1);
    const auto two = run(2);
    REQUIRE(one.first == two.first);
    REQUIRE(one.second == two.second);
}
