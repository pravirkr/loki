#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/score_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

using loki::Backend;
using loki::SizeType;

namespace {

struct Hits {
    std::vector<uint32_t> indices;
    std::vector<float> scores;
};

// Device hits sorted by index (the atomic slot order is not reproducible).
Hits collect(const thrust::device_vector<float>& scores_d,
             const thrust::device_vector<uint32_t>& indices_d,
             SizeType n) {
    std::vector<float> scores(n);
    std::vector<uint32_t> indices(n);
    loki::cuda_utils::check_cuda_call(
        cudaMemcpy(scores.data(), thrust::raw_pointer_cast(scores_d.data()),
                   n * sizeof(float), cudaMemcpyDeviceToHost),
        "test D2H failed");
    loki::cuda_utils::check_cuda_call(
        cudaMemcpy(indices.data(), thrust::raw_pointer_cast(indices_d.data()),
                   n * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "test D2H failed");
    std::vector<SizeType> order(n);
    std::iota(order.begin(), order.end(), SizeType{0});
    std::ranges::sort(order, {}, [&](SizeType i) { return indices[i]; });
    Hits out;
    for (const auto i : order) {
        out.indices.push_back(indices[i]);
        out.scores.push_back(scores[i]);
    }
    return out;
}

bool bit_equal(const Hits& a, const Hits& b) {
    return a.indices == b.indices && a.scores.size() == b.scores.size() &&
           std::memcmp(a.scores.data(), b.scores.data(),
                       a.scores.size() * sizeof(float)) == 0;
}

// Random coordinates of one merge level into a level of `nsrc` coords.
struct LevelCoords {
    thrust::device_vector<uint32_t> i_tail;
    thrust::device_vector<uint32_t> i_head;
    thrust::device_vector<float> s_tail;
    thrust::device_vector<float> s_head;

    LevelCoords(SizeType n,
                SizeType nsrc,
                SizeType nbins,
                bool freq_only,
                std::mt19937& rng) {
        std::uniform_int_distribution<uint32_t> pick(
            0, static_cast<uint32_t>(nsrc - 1));
        std::uniform_real_distribution<float> unif(0.0F, 1.0F);
        std::vector<uint32_t> it(n);
        std::vector<uint32_t> ih(n);
        std::vector<float> st(n);
        std::vector<float> sh(n);
        for (SizeType c = 0; c < n; ++c) {
            it[c] = pick(rng);
            ih[c] = freq_only ? it[c] : pick(rng);
            st[c] = freq_only ? 0.0F : unif(rng) * static_cast<float>(nbins);
            sh[c] = unif(rng) * static_cast<float>(nbins);
        }
        i_tail.assign(it.begin(), it.end());
        i_head.assign(ih.begin(), ih.end());
        s_tail.assign(st.begin(), st.end());
        s_head.assign(sh.begin(), sh.end());
    }
    loki::coord::FFACoordDPtrs accel() {
        return {.i_tail     = thrust::raw_pointer_cast(i_tail.data()),
                .shift_tail = thrust::raw_pointer_cast(s_tail.data()),
                .i_head     = thrust::raw_pointer_cast(i_head.data()),
                .shift_head = thrust::raw_pointer_cast(s_head.data()),
                .size       = i_tail.size()};
    }
    loki::coord::FFACoordFreqDPtrs freq() {
        return {.idx   = thrust::raw_pointer_cast(i_tail.data()),
                .shift = thrust::raw_pointer_cast(s_head.data()),
                .size  = i_tail.size()};
    }
};

// A random source level (2^depth segments, positive variances with a few
// zeros) and random coordinates for the levels above it. Compares the
// unfused levels plus score_and_filter_cuda_d against the fused kernel.
void check_final_score(SizeType nbins,
                       SizeType ncoords_src,
                       SizeType ncoords_cur,
                       std::vector<uint32_t> widths,
                       float fold_scale,
                       bool freq_only,
                       std::uint32_t seed,
                       int depth = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> unif(0.0F, 1.0F);
    const SizeType stride = 2 * nbins;
    const SizeType nseg   = SizeType{1} << static_cast<unsigned>(depth);
    std::vector<float> source(nseg * ncoords_src * stride);
    for (SizeType p = 0; p < nseg * ncoords_src; ++p) {
        for (SizeType i = 0; i < nbins; ++i) {
            source[(p * stride) + i] = normal(rng) + 0.5F;
            // About 1% of the variances are zero (fold_norm_bin -> 0).
            source[(p * stride) + nbins + i] =
                unif(rng) < 0.01F ? 0.0F : 0.5F + unif(rng);
        }
    }
    // depth 2: an intermediate level of ncoords_mid coordinates.
    const SizeType ncoords_mid =
        depth == 2 ? (ncoords_src + ncoords_cur) / 2 : ncoords_src;
    LevelCoords mid(ncoords_mid, ncoords_src, nbins, freq_only, rng);
    LevelCoords fin(ncoords_cur, ncoords_mid, nbins, freq_only, rng);

    const thrust::device_vector<float> source_d(source.begin(), source.end());
    const thrust::device_vector<uint32_t> widths_d(widths.begin(),
                                                   widths.end());
    const SizeType nscores  = ncoords_cur * widths.size();
    const float threshold   = 0.0F;
    const float* source_ptr = thrust::raw_pointer_cast(source_d.data());
    cudaStream_t stream     = nullptr;
    loki::memory::DeviceCounter counter;

    // Unfused: write the levels, then score the final one.
    const auto merge = [&](const float* in, float* out, LevelCoords& c,
                           SizeType ncur, SizeType nprev, SizeType nsegs) {
        if (freq_only) {
            loki::core::ffa_iter_freq_cuda(in, out, c.freq(), ncur, nprev,
                                           nsegs, nbins, stream);
        } else {
            loki::core::ffa_iter_cuda(in, out, c.accel(), ncur, nprev, nsegs,
                                      nbins, stream);
        }
    };
    thrust::device_vector<float> mid_d(2 * ncoords_mid * stride);
    const float* parent_ptr = source_ptr;
    if (depth == 2) {
        merge(source_ptr, thrust::raw_pointer_cast(mid_d.data()), mid,
              ncoords_mid, ncoords_src, 2);
        parent_ptr = thrust::raw_pointer_cast(mid_d.data());
    }
    thrust::device_vector<float> fold_d(ncoords_cur * stride);
    merge(parent_ptr, thrust::raw_pointer_cast(fold_d.data()), fin, ncoords_cur,
          ncoords_mid, 1);
    thrust::device_vector<float> ref_scores_d(nscores);
    thrust::device_vector<uint32_t> ref_indices_d(nscores);
    const SizeType n_ref = loki::detection::score_and_filter_cuda_d(
        loki::cuda_utils::as_span(fold_d), loki::cuda_utils::as_span(widths_d),
        loki::cuda_utils::as_span(ref_scores_d),
        loki::cuda_utils::as_span(ref_indices_d), threshold, ncoords_cur, nbins,
        stream, counter, fold_scale);

    // Fused.
    thrust::device_vector<float> scores_d(nscores);
    thrust::device_vector<uint32_t> indices_d(nscores);
    counter.reset(stream);
    const loki::core::FFAFinalScoreArgs args{
        .widths     = thrust::raw_pointer_cast(widths_d.data()),
        .nwidths    = static_cast<uint32_t>(widths.size()),
        .threshold  = threshold,
        .fold_scale = fold_scale,
        .scores     = thrust::raw_pointer_cast(scores_d.data()),
        .indices    = thrust::raw_pointer_cast(indices_d.data()),
        .counter    = counter.data()};
    if (freq_only) {
        loki::core::ffa_freq_final_score_cuda(
            source_ptr, fin.freq(), mid.freq(), ncoords_cur, ncoords_mid,
            ncoords_src, depth, nbins, args, stream);
    } else {
        loki::core::ffa_final_score_cuda(source_ptr, fin.accel(), mid.accel(),
                                         ncoords_cur, ncoords_mid, ncoords_src,
                                         depth, nbins, args, stream);
    }
    const SizeType n_fused = counter.value_sync(stream);

    REQUIRE(n_ref > 0);
    REQUIRE(n_fused == n_ref);
    CHECK(bit_equal(collect(ref_scores_d, ref_indices_d, n_ref),
                    collect(scores_d, indices_d, n_fused)));
}

// Fourier variant: the unfused complex levels (the sweep's merge kernel), a
// cuFFT C2R (normalized) and score_and_filter_cuda_d, against the fused
// kernel (on-chip inverse FFT, direct or Bluestein, the 1/nbins deferred to
// fold_scale; depth 2 merges the level in between on chip too). Every
// (profile, width) passes, so the two score lists align by index; the
// phasors and the transforms differ in rounding, so compare with a
// tolerance.
void check_final_score_fourier(SizeType nbins,
                               SizeType ncoords_src,
                               SizeType ncoords_cur,
                               std::vector<uint32_t> widths,
                               bool freq_only,
                               std::uint32_t seed,
                               int depth = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const SizeType nbins_f = (nbins / 2) + 1;
    const SizeType stride  = 2 * nbins_f;
    const SizeType nseg    = SizeType{1} << static_cast<unsigned>(depth);
    // Source harmonics: e random; v the spectrum of a positive profile.
    std::vector<loki::ComplexTypeCUDA> source(nseg * ncoords_src * stride);
    for (SizeType p = 0; p < nseg * ncoords_src; ++p) {
        for (SizeType k = 0; k < nbins_f; ++k) {
            const float scale = 1.0F / std::sqrt(static_cast<float>(nbins));
            source[(p * stride) + k] = {normal(rng) * scale * 4.0F,
                                        normal(rng) * scale * 4.0F};
            source[(p * stride) + nbins_f + k] =
                k == 0 ? loki::ComplexTypeCUDA{static_cast<float>(nbins), 0.0F}
                       : loki::ComplexTypeCUDA{0.01F * normal(rng),
                                               0.01F * normal(rng)};
        }
    }
    const SizeType ncoords_mid =
        depth == 2 ? (ncoords_src + ncoords_cur) / 2 : ncoords_src;
    LevelCoords mid(ncoords_mid, ncoords_src, nbins, freq_only, rng);
    LevelCoords fin(ncoords_cur, ncoords_mid, nbins, freq_only, rng);
    const thrust::device_vector<loki::ComplexTypeCUDA> source_d(source.begin(),
                                                                source.end());
    const thrust::device_vector<uint32_t> widths_d(widths.begin(),
                                                   widths.end());
    const SizeType nscores = ncoords_cur * widths.size();
    const float threshold  = -1.0e30F;
    const auto* source_ptr = thrust::raw_pointer_cast(source_d.data());
    cudaStream_t stream    = nullptr;
    loki::memory::DeviceCounter counter;

    const auto merge = [&](const loki::ComplexTypeCUDA* in,
                           loki::ComplexTypeCUDA* out, LevelCoords& c,
                           SizeType ncur, SizeType nprev, SizeType nsegs) {
        if (freq_only) {
            loki::core::ffa_complex_iter_freq_sweep_cuda(
                in, out, c.freq(), ncur, nprev, nsegs, nbins_f, nbins, stream);
        } else {
            loki::core::ffa_complex_iter_sweep_cuda(
                in, out, c.accel(), ncur, nprev, nsegs, nbins_f, nbins, stream);
        }
    };
    thrust::device_vector<loki::ComplexTypeCUDA> mid_d(2 * ncoords_mid *
                                                       stride);
    const loki::ComplexTypeCUDA* parent_ptr = source_ptr;
    if (depth == 2) {
        merge(source_ptr, thrust::raw_pointer_cast(mid_d.data()), mid,
              ncoords_mid, ncoords_src, 2);
        parent_ptr = thrust::raw_pointer_cast(mid_d.data());
    }
    thrust::device_vector<loki::ComplexTypeCUDA> fold_c(ncoords_cur * stride);
    merge(parent_ptr, thrust::raw_pointer_cast(fold_c.data()), fin, ncoords_cur,
          ncoords_mid, 1);
    thrust::device_vector<float> fold_r(ncoords_cur * 2 * nbins);
    loki::math::irfft_batch_cuda(loki::cuda_utils::as_span(fold_c),
                                 loki::cuda_utils::as_span(fold_r),
                                 ncoords_cur * 2, nbins, stream, 0);
    thrust::device_vector<float> ref_scores_d(nscores);
    thrust::device_vector<uint32_t> ref_indices_d(nscores);
    const SizeType n_ref = loki::detection::score_and_filter_cuda_d(
        loki::cuda_utils::as_span(fold_r), loki::cuda_utils::as_span(widths_d),
        loki::cuda_utils::as_span(ref_scores_d),
        loki::cuda_utils::as_span(ref_indices_d), threshold, ncoords_cur, nbins,
        stream, counter, 1.0F);

    thrust::device_vector<float> scores_d(nscores);
    thrust::device_vector<uint32_t> indices_d(nscores);
    counter.reset(stream);
    thrust::device_vector<loki::ComplexTypeCUDA> bluestein_d;
    if (loki::core::ffa_fourier_needs_bluestein(nbins)) {
        const auto table = loki::core::ffa_fourier_bluestein_table(nbins);
        bluestein_d.assign(table.begin(), table.end());
    }
    loki::core::FFAFinalScoreArgs args{
        .widths     = thrust::raw_pointer_cast(widths_d.data()),
        .nwidths    = static_cast<uint32_t>(widths.size()),
        .threshold  = threshold,
        .fold_scale = 1.0F / static_cast<float>(nbins),
        .scores     = thrust::raw_pointer_cast(scores_d.data()),
        .indices    = thrust::raw_pointer_cast(indices_d.data()),
        .counter    = counter.data()};
    args.bluestein = thrust::raw_pointer_cast(bluestein_d.data());
    if (freq_only) {
        loki::core::ffa_freq_final_score_fourier_cuda(
            source_ptr, fin.freq(), mid.freq(), ncoords_cur, ncoords_mid,
            ncoords_src, depth, nbins, args, stream);
    } else {
        loki::core::ffa_final_score_fourier_cuda(
            source_ptr, fin.accel(), mid.accel(), ncoords_cur, ncoords_mid,
            ncoords_src, depth, nbins, args, stream);
    }
    const SizeType n_fused = counter.value_sync(stream);
    REQUIRE(n_ref == nscores);
    REQUIRE(n_fused == nscores);
    const auto ref   = collect(ref_scores_d, ref_indices_d, n_ref);
    const auto fused = collect(scores_d, indices_d, n_fused);
    REQUIRE(ref.indices == fused.indices);
    float max_diff = 0.0F;
    for (SizeType i = 0; i < nscores; ++i) {
        max_diff =
            std::max(max_diff, std::abs(ref.scores[i] - fused.scores[i]));
    }
    INFO("nbins " << nbins << " freq_only " << freq_only << " depth " << depth
                  << " max |dS/N| " << max_diff);
    CHECK(max_diff < 1.0e-5F);
}

std::vector<uint32_t> geometric_widths(SizeType nbins, SizeType count) {
    std::vector<uint32_t> w;
    double x = 1.0;
    while (w.size() < count) {
        const auto v = static_cast<uint32_t>(x);
        if (w.empty() || v > w.back()) {
            w.push_back(v);
        }
        x *= 1.3;
        if (v >= nbins / 2) {
            break;
        }
    }
    return w;
}

} // namespace

TEST_CASE("Fused final FFA level and score match the unfused path bit for bit",
          "[ffa][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const bool freq_only = GENERATE(false, true);
    SECTION("thread regime (nbins <= 64, >= 2^16 profiles)") {
        const SizeType nbins  = GENERATE(32, 64);
        const SizeType nwidth = GENERATE(5, 12, 20);
        const auto widths     = geometric_widths(nbins, nwidth);
        check_final_score(nbins, 30000, 70001, widths, 1.0F, freq_only, 1);
        check_final_score(nbins, 30000, 70001, widths,
                          1.0F / static_cast<float>(nbins), freq_only, 2);
    }
    SECTION("warp regime (larger nbins, or few profiles)") {
        const SizeType nbins = GENERATE(50, 64, 128, 256, 1024);
        const auto widths    = geometric_widths(nbins, 14);
        check_final_score(nbins, 500, 1201, widths, 1.0F, freq_only, 3);
        check_final_score(nbins, 500, 1201, widths,
                          1.0F / static_cast<float>(nbins), freq_only, 4);
    }
    SECTION("two levels merged on chip (depth 2)") {
        const SizeType nbins = GENERATE(32, 64, 128, 256);
        const auto widths    = geometric_widths(nbins, 9);
        check_final_score(nbins, 20000, 70001, widths, 1.0F, freq_only, 7, 2);
        check_final_score(nbins, 300, 1201, widths, 1.0F, freq_only, 8, 2);
    }
}
TEST_CASE("Fused Fourier final level and score match cuFFT C2R scoring",
          "[ffa][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    // Powers of two (direct C2R) and others (Bluestein, every M).
    const SizeType nbins =
        GENERATE(32, 64, 128, 256, 512, 1024, 7, 16, 25, 50, 63, 100, 129, 200,
                 333, 400, 800, 1000, 1023);
    const bool freq_only = GENERATE(false, true);
    const auto widths    = geometric_widths(nbins, 10);
    // Few profiles (warp regime) and, for nbins <= 64, >= 2^16 (thread);
    // the last level alone and the last two levels.
    const int depth = GENERATE(1, 2);
    check_final_score_fourier(nbins, 400, 1001, widths, freq_only, 5, depth);
    if (nbins <= 64) {
        check_final_score_fourier(nbins, 30000, 70001, widths, freq_only, 6,
                                  depth);
    }
}

TEST_CASE("Fused Fourier scoring matches cuFFT C2R for every nbins",
          "[ffa][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    for (SizeType nbins = 2; nbins <= 1024; ++nbins) {
        const auto widths =
            geometric_widths(nbins, std::min<SizeType>(nbins / 2, 6));
        check_final_score_fourier(nbins, 24, 61, widths, false,
                                  static_cast<std::uint32_t>(nbins), 1);
    }
}
#endif // LOKI_ENABLE_CUDA
