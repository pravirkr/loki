#include <cstdint>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

#include "lib/detail/math.hpp"

namespace {

using loki::SizeType;
using loki::io::PreprocessMethod;
using loki::io::PreprocessOptions;

constexpr double kTsamp = 64e-6;

std::vector<float> make_noise(SizeType n) {
    loki::math::PCG32 rng(2025);
    std::vector<float> x(n);
    for (auto& v : x) {
        v = static_cast<float>(rng() >> 8U) * (1.0F / 16777216.0F);
    }
    return x;
}

// Args: log2 n, nthreads, method (0 = robust, 1 = zscore), zap, window (s).
void bm_preprocess(benchmark::State& state) {
    const SizeType n    = SizeType{1} << static_cast<SizeType>(state.range(0));
    const auto nthreads = static_cast<int>(state.range(1));
    PreprocessOptions o;
    o.method        = state.range(2) == 0 ? PreprocessMethod::kRobust
                                          : PreprocessMethod::kZScore;
    o.zap_periodic  = state.range(3) != 0;
    o.filter_window = static_cast<double>(state.range(4));
    const auto raw  = make_noise(n);
    std::vector<float> ts_e(n);
    std::vector<float> ts_v(n);
    for (auto _ : state) {
        const auto rep = loki::io::preprocess(raw, kTsamp, ts_e, ts_v, o,
                                              loki::Exec::cpu(nthreads));
        benchmark::DoNotOptimize(rep.norm);
        benchmark::DoNotOptimize(ts_e.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

} // namespace

BENCHMARK(bm_preprocess)
    ->ArgNames({"log2n", "threads", "zscore", "zap", "window_s"})
    ->ArgsProduct({{25}, {1, 8}, {0}, {0, 1}, {1, 20}})
    ->Args({25, 8, 1, 0, 1})
    ->Args({25, 8, 1, 0, 20})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
