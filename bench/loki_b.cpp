#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <omp.h>

#include "loki/detection/score.hpp"

namespace loki::detection {

class ScoresFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        nwidths = 15;
        nsamps  = state.range(0);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    template <typename T>
    std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
        std::vector<T> vec(size);
        std::uniform_real_distribution<T> dis(0.0, 1.0);
        std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
        return vec;
    }

    size_t nwidths{};
    size_t nsamps{};
};

class ScoresFixture2D : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        nwidths   = 15;
        nprofiles = state.range(0);
        nsamps    = 4096;
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    template <typename T>
    std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
        std::vector<T> vec(size);
        std::uniform_real_distribution<T> dis(0.0, 1.0);
        std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
        return vec;
    }

    size_t nwidths{};
    size_t nprofiles{};
    size_t nsamps{};
};

BENCHMARK_DEFINE_F(ScoresFixture, BM_loki_snr_1d)(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    const auto arr = generate_vector<float>(nsamps, gen);
    std::vector<size_t> widths(nwidths);
    std::iota(widths.begin(), widths.end(), 1);
    std::vector<float> out(widths.size());
    for (auto _ : state) {
        detection::snr_1d(std::span(arr), std::span(widths), std::span(out),
                          1.0F);
    }
}

BENCHMARK_DEFINE_F(ScoresFixture2D,
                   BM_loki_snr_2d_seq)(benchmark::State& state) {
    omp_set_num_threads(1);
    std::random_device rd;
    std::mt19937 gen(rd());
    const auto arr = generate_vector<float>(nprofiles * nsamps, gen);
    std::vector<size_t> widths(nwidths);
    std::iota(widths.begin(), widths.end(), 1);
    std::vector<float> out(nprofiles * widths.size());
    for (auto _ : state) {
        detection::snr_2d(std::span(arr), nprofiles, std::span(widths),
                          std::span(out), 1.0F);
    }
}

BENCHMARK_DEFINE_F(ScoresFixture2D,
                   BM_loki_snr_2d_par)(benchmark::State& state) {
    omp_set_num_threads(8);
    std::random_device rd;
    std::mt19937 gen(rd());
    const auto arr = generate_vector<float>(nprofiles * nsamps, gen);
    std::vector<size_t> widths(nwidths);
    std::iota(widths.begin(), widths.end(), 1);
    std::vector<float> out(nprofiles * widths.size());
    for (auto _ : state) {
        detection::snr_2d(std::span(arr), nprofiles, std::span(widths),
                          std::span(out), 1.0F);
    }
}

constexpr size_t kMinNsamps = 1 << 12;
constexpr size_t kMaxNsamps = 1 << 17;

BENCHMARK_REGISTER_F(ScoresFixture, BM_loki_snr_1d)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);

BENCHMARK_REGISTER_F(ScoresFixture2D, BM_loki_snr_2d_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);

BENCHMARK_REGISTER_F(ScoresFixture2D, BM_loki_snr_2d_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);

// BENCHMARK_MAIN();

} // namespace loki::detection