#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include "loki/common/types.hpp"
#include "detail/math.hpp"

namespace {

using loki::SizeType;
using loki::math::FilterMethod;

std::vector<float> make_noise(SizeType n) {
    loki::math::PCG32 rng(2024);
    std::vector<float> x(n);
    for (auto& v : x) {
        v = static_cast<float>(rng() >> 8U) * (1.0F / 16777216.0F);
    }
    return x;
}

void set_items(benchmark::State& state, SizeType n) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

// Args: n, window, use_heap (0 = sorted array, 1 = double heap).
void bm_running_median(benchmark::State& state) {
    const auto n = static_cast<SizeType>(state.range(0));
    const auto w = static_cast<SizeType>(state.range(1));
    const auto nthreads = static_cast<int>(state.range(3));
    loki::math::detail::tuning().heap_window =
        state.range(2) != 0 ? 1 : (SizeType{1} << 40U);
    const auto x = make_noise(n);
    std::vector<float> out(n);
    for (auto _ : state) {
        loki::math::running_filter(x, out, w, FilterMethod::kMedian, nthreads);
        benchmark::DoNotOptimize(out.data());
    }
    set_items(state, n);
}

// Args: n, window.
void bm_running_mean(benchmark::State& state) {
    const auto n = static_cast<SizeType>(state.range(0));
    const auto w = static_cast<SizeType>(state.range(1));
    const auto nthreads = static_cast<int>(state.range(2));
    const auto x = make_noise(n);
    std::vector<float> out(n);
    for (auto _ : state) {
        loki::math::running_filter(x, out, w, FilterMethod::kMean, nthreads);
        benchmark::DoNotOptimize(out.data());
    }
    set_items(state, n);
}

// Args: n, window, fast (0 = exact, 1 = fast). In-place baseline removal.
void bm_subtract_median(benchmark::State& state) {
    const auto n = static_cast<SizeType>(state.range(0));
    const auto w = static_cast<SizeType>(state.range(1));
    const bool fast = state.range(2) != 0;
    const auto nthreads = static_cast<int>(state.range(3));
    const auto src  = make_noise(n);
    auto x          = src;
    for (auto _ : state) {
        state.PauseTiming();
        x = src;
        state.ResumeTiming();
        loki::math::subtract_running_filter(x, w, FilterMethod::kMedian, fast, 101,
                                           nthreads);
        benchmark::DoNotOptimize(x.data());
    }
    set_items(state, n);
}

// Args: n, scale method (loki::ScaleMethod), use_radix.
void bm_zscore(benchmark::State& state) {
    const auto n      = static_cast<SizeType>(state.range(0));
    const auto scale  = static_cast<loki::ScaleMethod>(state.range(1));
    const auto nthreads = static_cast<int>(state.range(3));
    loki::math::detail::tuning().radix_select_size =
        state.range(2) != 0 ? 1 : (SizeType{1} << 40U);
    const auto src = make_noise(n);
    auto x         = src;
    for (auto _ : state) {
        state.PauseTiming();
        x = src;
        state.ResumeTiming();
        benchmark::DoNotOptimize(
            loki::math::zscore(x, loki::LocMethod::kMedian, scale, nthreads));
    }
    set_items(state, n);
}

} // namespace

BENCHMARK(bm_running_median)
    ->ArgNames({"n", "w", "heap", "threads"})
    ->ArgsProduct({{1 << 20, 1 << 23}, {5, 11, 21, 51, 101, 501, 1001, 4001, 15625}, {0, 1}, {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(bm_running_mean)
    ->ArgNames({"n", "w", "threads"})
    ->ArgsProduct({{1 << 20, 1 << 23}, {101, 15625}, {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(bm_subtract_median)
    ->ArgNames({"n", "w", "fast", "threads"})
    ->ArgsProduct({{1 << 23}, {15625}, {0, 1}, {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(bm_zscore)
    ->ArgNames({"n", "scale", "radix", "threads"})
    ->ArgsProduct({{1 << 20, 1 << 23},
                   {static_cast<int>(loki::ScaleMethod::kStd),
                    static_cast<int>(loki::ScaleMethod::kIqr),
                    static_cast<int>(loki::ScaleMethod::kMad)},
                   {0, 1},
                   {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
