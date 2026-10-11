#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/ep_score_cuda.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/score_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

using loki::Backend;
using loki::ComplexTypeCUDA;
using loki::SizeType;

namespace {

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d) {
    const thrust::host_vector<T> h = d;
    return {h.begin(), h.end()};
}

template <typename T> bool bit_equal(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

std::vector<uint32_t> boxcar_widths(SizeType nbins, SizeType count = 12) {
    std::vector<uint32_t> w;
    double x = 1.0;
    while (w.size() < count && x < static_cast<double>(nbins)) {
        const auto v = static_cast<uint32_t>(x);
        if (w.empty() || v > w.back()) {
            w.push_back(v);
        }
        x *= count > 12 ? 1.08 : 1.4;
    }
    return w;
}

// Random pruning batch: parent folds in a circular tree, FFA segment folds,
// leaves pointing into both, some leaves invalid. FoldT float: time-domain
// rows (e, then v > 0); ComplexTypeCUDA: half spectra.
template <typename FoldT> struct Batch {
    static constexpr bool kComplex = std::is_same_v<FoldT, ComplexTypeCUDA>;
    SizeType nbins, nf, stride, n_leaves, capacity, physical_start;
    thrust::device_vector<FoldT> tree, ffa;
    thrust::device_vector<uint32_t> idx_tree, idx_ffa, widths;
    thrust::device_vector<float> shift;
    thrust::device_vector<uint8_t> valid;
    thrust::device_vector<double> leaves;

    Batch(SizeType nbins_,
          SizeType n,
          std::uint32_t seed,
          SizeType nwidths = 12)
        : nbins(nbins_),
          nf(kComplex ? (nbins_ / 2) + 1 : nbins_),
          stride(2 * nf),
          n_leaves(n),
          capacity(n + 5),
          physical_start(capacity - 17) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        const float scale = 4.0F / std::sqrt(static_cast<float>(nbins));
        // Fourier: e random; v the spectrum of a positive profile. The
        // merged spectra must stay Hermitian (real DC and Nyquist bins):
        // C2R ignores those imaginary parts, but cuFFT's result then
        // depends on the algorithm it picks. A real tree DC / Nyquist and a
        // zero FFA Nyquist keep them real through the phase shift.
        const SizeType nyq = (nbins % 2 == 0) ? nbins / 2 : nf;
        const auto folds   = [&](SizeType rows, bool is_ffa) {
            std::vector<FoldT> f(rows * stride);
            for (SizeType p = 0; p < rows; ++p) {
                for (SizeType k = 0; k < nf; ++k) {
                    if constexpr (kComplex) {
                        ComplexTypeCUDA e{normal(rng) * scale,
                                          normal(rng) * scale};
                        ComplexTypeCUDA v =
                            k == 0 ? ComplexTypeCUDA{static_cast<float>(nbins),
                                                     0.0F}
                                   : ComplexTypeCUDA{0.01F * normal(rng),
                                                     0.01F * normal(rng)};
                        if (k == 0 || k == nyq) {
                            e = {e.real(), 0.0F};
                            v = {v.real(), 0.0F};
                        }
                        if (is_ffa && k == nyq) {
                            e = {0.0F, 0.0F};
                            v = {0.0F, 0.0F};
                        }
                        f[(p * stride) + k]      = e;
                        f[(p * stride) + nf + k] = v;
                    } else {
                        f[(p * stride) + k] = normal(rng);
                        f[(p * stride) + nf + k] =
                            0.5F + (0.1F * std::abs(normal(rng)));
                    }
                }
            }
            return f;
        };
        const auto t = folds(capacity, false);
        const auto f = folds(37, true);
        tree.assign(t.begin(), t.end());
        ffa.assign(f.begin(), f.end());
        std::uniform_int_distribution<uint32_t> pick_tree(0, capacity - 1);
        std::uniform_int_distribution<uint32_t> pick_ffa(0, 36);
        std::uniform_real_distribution<float> pick_shift(
            0.0F, static_cast<float>(nbins));
        std::vector<uint32_t> it(n);
        std::vector<uint32_t> iff(n);
        std::vector<float> sh(n);
        std::vector<uint8_t> va(n);
        std::vector<double> lv(n * 10);
        for (SizeType i = 0; i < n; ++i) {
            it[i]  = pick_tree(rng);
            iff[i] = pick_ffa(rng);
            sh[i]  = pick_shift(rng);
            va[i]  = (i % 11) == 3 ? 0 : 1;
            for (SizeType j = 0; j < 10; ++j) {
                lv[(i * 10) + j] = static_cast<double>(normal(rng));
            }
        }
        idx_tree.assign(it.begin(), it.end());
        idx_ffa.assign(iff.begin(), iff.end());
        shift.assign(sh.begin(), sh.end());
        valid.assign(va.begin(), va.end());
        leaves.assign(lv.begin(), lv.end());
        const auto w = boxcar_widths(nbins, nwidths);
        widths.assign(w.begin(), w.end());
    }
};

// The unfused chain: shift-add, (cuFFT C2R,) thread/warp scorer.
template <typename FoldT> struct Reference {
    std::vector<FoldT> folds;
    std::vector<float> scores;
};

template <typename FoldT> Reference<FoldT> reference(const Batch<FoldT>& b) {
    using loki::cuda_utils::as_span;
    thrust::device_vector<FoldT> folds(b.n_leaves * b.stride);
    thrust::device_vector<float> scores(b.n_leaves);
    thrust::device_vector<uint8_t> filt(b.n_leaves);
    loki::memory::CUBScratchArena scratch(b.n_leaves, 1, nullptr);
    if constexpr (Batch<FoldT>::kComplex) {
        loki::core::shift_add_linear_complex_batch_cuda(
            thrust::raw_pointer_cast(b.tree.data()),
            thrust::raw_pointer_cast(b.idx_tree.data()),
            thrust::raw_pointer_cast(b.valid.data()),
            thrust::raw_pointer_cast(b.ffa.data()),
            thrust::raw_pointer_cast(b.idx_ffa.data()),
            thrust::raw_pointer_cast(b.shift.data()),
            thrust::raw_pointer_cast(folds.data()), b.nf, b.nbins, b.n_leaves,
            b.physical_start, b.capacity, nullptr);
        thrust::device_vector<ComplexTypeCUDA> work = folds;
        thrust::device_vector<float> real(b.n_leaves * 2 * b.nbins);
        loki::math::irfft_batch_cuda(as_span(work), as_span(real),
                                     b.n_leaves * 2, b.nbins, nullptr, 0);
        (void)loki::detection::score_and_filter_max_cuda_d(
            as_span(real), as_span(b.widths), as_span(scores), as_span(b.valid),
            as_span(filt), 0.0F, b.n_leaves, b.nbins, scratch, nullptr, 1.0F);
    } else {
        loki::core::shift_add_linear_batch_cuda(
            thrust::raw_pointer_cast(b.tree.data()),
            thrust::raw_pointer_cast(b.idx_tree.data()),
            thrust::raw_pointer_cast(b.valid.data()),
            thrust::raw_pointer_cast(b.ffa.data()),
            thrust::raw_pointer_cast(b.idx_ffa.data()),
            thrust::raw_pointer_cast(b.shift.data()),
            thrust::raw_pointer_cast(folds.data()), b.nbins, b.n_leaves,
            b.physical_start, b.capacity, nullptr);
        (void)loki::detection::score_and_filter_max_cuda_d(
            as_span(folds), as_span(b.widths), as_span(scores),
            as_span(b.valid), as_span(filt), 0.0F, b.n_leaves, b.nbins, scratch,
            nullptr);
    }
    return {to_host(folds), to_host(scores)};
}

template <typename FoldT> struct Fused {
    std::vector<FoldT> folds;
    std::vector<float> scores;
    std::vector<uint8_t> mask;
    uint32_t count{};
    float min{};
    float max{};
};

// The fused kernel, staged (folds_out) or direct (into a tree region that
// starts at write_head and wraps).
template <typename FoldT>
Fused<FoldT> run_fused(const Batch<FoldT>& b,
                       float threshold,
                       bool direct,
                       std::vector<double>* tree_leaves_out = nullptr,
                       std::vector<float>* tree_scores_out  = nullptr,
                       SizeType write_head                  = 0,
                       SizeType dst_capacity                = 0) {
    thrust::device_vector<FoldT> folds_out(b.n_leaves * b.stride);
    thrust::device_vector<float> scores(b.n_leaves);
    thrust::device_vector<uint8_t> mask(b.n_leaves);
    const SizeType words = 3 + loki::core::ep_shift_add_score_tile_words<FoldT>(
                                   b.nbins, b.n_leaves);
    thrust::device_vector<unsigned long long> state(words, 0ULL);
    thrust::device_vector<ComplexTypeCUDA> bluestein;
    if (Batch<FoldT>::kComplex &&
        loki::core::ffa_fourier_needs_bluestein(b.nbins)) {
        const auto table = loki::core::ffa_fourier_bluestein_table(b.nbins);
        bluestein.assign(table.begin(), table.end());
    }
    thrust::device_vector<double> dst_leaves(dst_capacity * 10, 0.0);
    thrust::device_vector<FoldT> dst_folds(dst_capacity * b.stride);
    thrust::device_vector<float> dst_scores(dst_capacity, 0.0F);
    auto* stats =
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(state.data()));
    loki::core::EPScoreArgs<FoldT> args{
        .folds_tree         = thrust::raw_pointer_cast(b.tree.data()),
        .indices_tree       = thrust::raw_pointer_cast(b.idx_tree.data()),
        .validation_mask    = thrust::raw_pointer_cast(b.valid.data()),
        .folds_ffa          = thrust::raw_pointer_cast(b.ffa.data()),
        .indices_ffa        = thrust::raw_pointer_cast(b.idx_ffa.data()),
        .phase_shift        = thrust::raw_pointer_cast(b.shift.data()),
        .folds_out          = thrust::raw_pointer_cast(folds_out.data()),
        .scores             = thrust::raw_pointer_cast(scores.data()),
        .filtered_mask      = thrust::raw_pointer_cast(mask.data()),
        .stats              = stats,
        .widths             = thrust::raw_pointer_cast(b.widths.data()),
        .nwidths            = static_cast<uint32_t>(b.widths.size()),
        .threshold          = threshold,
        .n_leaves           = static_cast<uint32_t>(b.n_leaves),
        .physical_start_idx = static_cast<uint32_t>(b.physical_start),
        .capacity           = static_cast<uint32_t>(b.capacity),
        .nbins              = static_cast<uint32_t>(b.nbins),
        .bluestein          = bluestein.empty()
                                  ? nullptr
                                  : thrust::raw_pointer_cast(bluestein.data()),
    };
    if (direct) {
        // The write region wraps at the tree's capacity, as the reads do.
        REQUIRE(dst_capacity == b.capacity);
        args.leaves_batch    = thrust::raw_pointer_cast(b.leaves.data());
        args.leaves_stride   = 10;
        args.tree_leaves     = thrust::raw_pointer_cast(dst_leaves.data());
        args.tree_folds      = thrust::raw_pointer_cast(dst_folds.data());
        args.tree_scores     = thrust::raw_pointer_cast(dst_scores.data());
        args.tree_write_head = static_cast<uint32_t>(write_head);
        args.tile_state      = thrust::raw_pointer_cast(state.data()) + 2;
    }
    loki::core::ep_shift_add_score_cuda<FoldT>(args, nullptr);
    loki::cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                      "fused kernel failed");
    std::vector<uint32_t> st(4);
    cudaMemcpy(st.data(), stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    Fused<FoldT> f{.folds  = direct ? to_host(dst_folds) : to_host(folds_out),
                   .scores = to_host(scores),
                   .mask   = to_host(mask),
                   .count  = st[0]};
    f.max = st[1] != 0
                ? std::bit_cast<float>(loki::core::ep_score_key_bits(st[1]))
                : std::numeric_limits<float>::lowest();
    f.min = st[1] != 0
                ? std::bit_cast<float>(loki::core::ep_score_key_bits(~st[2]))
                : std::numeric_limits<float>::max();
    if (tree_leaves_out != nullptr) {
        *tree_leaves_out = to_host(dst_leaves);
    }
    if (tree_scores_out != nullptr) {
        *tree_scores_out = to_host(dst_scores);
    }
    return f;
}

template <typename FoldT>
void check_batch(SizeType nbins,
                 SizeType n_leaves,
                 std::uint32_t seed,
                 SizeType nwidths = 12) {
    constexpr bool kComplex = Batch<FoldT>::kComplex;
    const Batch<FoldT> b(nbins, n_leaves, seed, nwidths);
    INFO("complex " << kComplex << " nbins " << nbins << " n_leaves "
                    << n_leaves << " nwidths " << b.widths.size());
    const auto ref   = reference(b);
    const auto valid = to_host(b.valid);

    // Threshold at the median of the valid reference scores.
    std::vector<float> vs;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (valid[i] != 0) {
            vs.push_back(ref.scores[i]);
        }
    }
    std::ranges::nth_element(vs, vs.begin() + static_cast<long>(vs.size() / 2));
    const float threshold = vs[vs.size() / 2];

    const auto staged = run_fused(b, threshold, false);
    float max_diff    = 0.0F;
    uint32_t npass    = 0;
    float mn          = std::numeric_limits<float>::max();
    float mx          = std::numeric_limits<float>::lowest();
    bool folds_exact  = true;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (valid[i] == 0) {
            REQUIRE(staged.mask[i] == 0);
            continue;
        }
        for (SizeType k = 0; k < b.stride; ++k) {
            folds_exact &= bit_equal(staged.folds[(i * b.stride) + k],
                                     ref.folds[(i * b.stride) + k]);
        }
        const float d = std::abs(staged.scores[i] - ref.scores[i]);
        max_diff      = std::max(max_diff, d);
        if (std::abs(ref.scores[i] - threshold) > 1.0e-4F) {
            CHECK((staged.mask[i] != 0) == (ref.scores[i] >= threshold));
        }
        CHECK((staged.mask[i] != 0) == (staged.scores[i] >= threshold));
        npass += staged.mask[i];
        mn = std::min(mn, staged.scores[i]);
        mx = std::max(mx, staged.scores[i]);
    }
    INFO("max |dS/N| " << max_diff);
    CHECK(folds_exact);
    if constexpr (kComplex) {
        CHECK(max_diff < 1.0e-5F);
    } else {
        // No transform: the same arithmetic as the unfused chain.
        CHECK(max_diff == 0.0F);
    }
    CHECK(staged.count == npass);
    CHECK(staged.min == mn);
    CHECK(staged.max == mx);

    // Direct: the same scores and mask; survivors in leaf order from
    // write_head (wrapping), with their leaves, folds and scores.
    const SizeType dst_capacity = b.capacity;
    const SizeType write_head   = dst_capacity - 3;
    std::vector<double> dst_leaves;
    std::vector<float> dst_scores;
    const auto dir = run_fused(b, threshold, true, &dst_leaves, &dst_scores,
                               write_head, dst_capacity);
    CHECK(dir.count == staged.count);
    CHECK(dir.mask == staged.mask);
    CHECK(std::memcmp(dir.scores.data(), staged.scores.data(),
                      n_leaves * sizeof(float)) == 0);
    const auto leaves = to_host(b.leaves);
    SizeType rank     = 0;
    bool direct_exact = true;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (staged.mask[i] == 0) {
            continue;
        }
        const SizeType slot = (write_head + rank++) % dst_capacity;
        for (SizeType j = 0; j < 10; ++j) {
            direct_exact &= dst_leaves[(slot * 10) + j] == leaves[(i * 10) + j];
        }
        direct_exact &= dst_scores[slot] == staged.scores[i];
        for (SizeType k = 0; k < b.stride; ++k) {
            direct_exact &= bit_equal(dir.folds[(slot * b.stride) + k],
                                      ref.folds[(i * b.stride) + k]);
        }
    }
    CHECK(direct_exact);
}

} // namespace

TEST_CASE("Fused EP shift-add + score matches the unfused chain",
          "[ep][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    for (SizeType nbins = 2; nbins <= 1024; ++nbins) {
        check_batch<ComplexTypeCUDA>(nbins, 53,
                                     static_cast<std::uint32_t>(nbins));
    }
    // The thread regime (nbins <= 64, at least 2^16 leaves) and many tiles.
    for (const SizeType nbins : {SizeType{32}, SizeType{50}, SizeType{64}}) {
        check_batch<ComplexTypeCUDA>(nbins, 70001, 7);
    }
    check_batch<ComplexTypeCUDA>(128, 20000, 8);
    // More widths than one register group (scored in groups of 32).
    check_batch<ComplexTypeCUDA>(256, 300, 10, 45);
    check_batch<ComplexTypeCUDA>(64, 70001, 11, 40);
    check_batch<ComplexTypeCUDA>(100, 300, 12, 70);
}

TEST_CASE("Fused EP row scoring (seeds, ascend) matches cuFFT + scorer",
          "[ep][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    using loki::cuda_utils::as_span;
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const auto check_rows = [](SizeType nbins, SizeType n_rows,
                               SizeType nwidths) {
        // The batch's tree rows are the rows to score.
        const Batch<ComplexTypeCUDA> b(nbins, n_rows, 3, nwidths);
        INFO("nbins " << nbins << " rows " << n_rows << " nwidths "
                      << b.widths.size());
        thrust::device_vector<ComplexTypeCUDA> work(
            b.tree.begin(),
            b.tree.begin() + static_cast<long>(n_rows * b.stride));
        thrust::device_vector<float> real(n_rows * 2 * nbins);
        loki::math::irfft_batch_cuda(as_span(work), as_span(real), n_rows * 2,
                                     nbins, nullptr, 0);
        thrust::device_vector<float> ref(n_rows);
        loki::detection::snr_boxcar_3d_max_cuda_d(
            as_span(real), as_span(b.widths), as_span(ref), n_rows, nbins,
            nullptr);
        thrust::device_vector<ComplexTypeCUDA> bluestein;
        if (loki::core::ffa_fourier_needs_bluestein(nbins)) {
            const auto table = loki::core::ffa_fourier_bluestein_table(nbins);
            bluestein.assign(table.begin(), table.end());
        }
        thrust::device_vector<float> got(n_rows);
        loki::core::ep_score_rows_fourier_cuda(
            thrust::raw_pointer_cast(b.tree.data()), n_rows, nbins,
            thrust::raw_pointer_cast(b.widths.data()), b.widths.size(),
            thrust::raw_pointer_cast(got.data()),
            bluestein.empty() ? nullptr
                              : thrust::raw_pointer_cast(bluestein.data()),
            nullptr);
        cudaDeviceSynchronize();
        const auto r   = to_host(ref);
        const auto g   = to_host(got);
        float max_diff = 0.0F;
        for (SizeType i = 0; i < n_rows; ++i) {
            max_diff = std::max(max_diff, std::abs(r[i] - g[i]));
        }
        INFO("max |dS/N| " << max_diff);
        CHECK(max_diff < 1.0e-5F);
    };
    for (SizeType nbins = 2; nbins <= 1024; nbins += (nbins < 70 ? 1 : 13)) {
        check_rows(nbins, 41, 12);
    }
    check_rows(64, 70001, 12); // thread regime
    check_rows(256, 500, 45);
}

TEST_CASE("Fused EP time-domain shift-add + score matches bit for bit",
          "[ep][cuda][score]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    for (SizeType nbins = 2; nbins <= 1100; nbins += (nbins < 130 ? 1 : 37)) {
        check_batch<float>(nbins, 53, static_cast<std::uint32_t>(nbins));
    }
    for (const SizeType nbins : {SizeType{32}, SizeType{50}, SizeType{64}}) {
        check_batch<float>(nbins, 70001, 7);
    }
    check_batch<float>(128, 20000, 8);
    check_batch<float>(256, 300, 10, 45);
    check_batch<float>(64, 70001, 11, 40);
    check_batch<float>(1400, 60, 12, 70);
}
#endif // LOKI_ENABLE_CUDA
