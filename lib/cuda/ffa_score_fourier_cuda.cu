#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/ffa_score_fourier_impl.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

namespace {

// In-place iterative radix-2 forward DFT (exp(-2 pi i j k / n)), n a power
// of two. Host only, for the Bluestein tables.
void fft_forward_pow2(std::vector<std::complex<double>>& x) {
    const SizeType n = x.size();
    for (SizeType i = 1, j = 0; i < n; ++i) {
        SizeType bit = n >> 1U;
        for (; (j & bit) != 0; bit >>= 1U) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
    for (SizeType len = 2; len <= n; len <<= 1U) {
        const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
        for (SizeType i = 0; i < n; i += len) {
            for (SizeType k = 0; k < len / 2; ++k) {
                const auto w = std::polar(1.0, ang * static_cast<double>(k));
                const auto u = x[i + k];
                const auto v = x[i + k + (len / 2)] * w;
                x[i + k]     = u + v;
                x[i + k + (len / 2)] = u - v;
            }
        }
    }
}

} // namespace

bool ffa_fourier_needs_bluestein(SizeType nbins) noexcept {
    return nbins < 32 || nbins > 1024 || !std::has_single_bit(nbins);
}

std::vector<ComplexTypeCUDA> ffa_fourier_bluestein_table(SizeType nbins) {
    // The fused scorers take nbins up to 1024, the batched GPU FFTs up to
    // math::kGpuFFTMaxN (M up to 4096).
    if (nbins < 2 || nbins > math::kGpuFFTMaxN) {
        throw std::invalid_argument(
            std::format("ffa_fourier_bluestein_table: nbins must be in [2, {}]",
                        math::kGpuFFTMaxN));
    }
    const SizeType n = nbins;
    const SizeType m = std::max<SizeType>(32, std::bit_ceil((2 * n) - 1));
    // c[j] = exp(i pi j^2 / n); j^2 is reduced mod 2n exactly first.
    std::vector<std::complex<double>> chirp(n);
    for (SizeType j = 0; j < n; ++j) {
        const SizeType r = (j * j) % (2 * n);
        chirp[j] = std::polar(1.0, std::numbers::pi * static_cast<double>(r) /
                                       static_cast<double>(n));
    }
    std::vector<std::complex<double>> kern(m, {0.0, 0.0});
    kern[0] = std::conj(chirp[0]);
    for (SizeType j = 1; j < n; ++j) {
        kern[j]     = std::conj(chirp[j]);
        kern[m - j] = std::conj(chirp[j]);
    }
    fft_forward_pow2(kern);
    std::vector<ComplexTypeCUDA> table;
    table.reserve(n + m);
    for (const auto& c : chirp) {
        table.emplace_back(static_cast<float>(c.real()),
                           static_cast<float>(c.imag()));
    }
    const double inv_m = 1.0 / static_cast<double>(m);
    for (const auto& k : kern) {
        table.emplace_back(static_cast<float>(k.real() * inv_m),
                           static_cast<float>(k.imag() * inv_m));
    }
    return table;
}

bool ffa_final_score_fourier_supported(SizeType nbins) noexcept {
    return nbins >= 2 && nbins <= 1024;
}

void ffa_final_score_fourier_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                  coord::FFACoordDPtrs coords_final,
                                  coord::FFACoordDPtrs coords_prev,
                                  SizeType ncoords_cur,
                                  SizeType ncoords_prev,
                                  SizeType ncoords_src,
                                  int depth,
                                  SizeType nbins,
                                  const FFAFinalScoreArgs& args,
                                  cudaStream_t stream) {
    (void)ncoords_prev;
    detail::launch_fourier(fold_in, detail::AccelParentsC{coords_final},
                           detail::AccelParentsC{coords_prev}, ncoords_cur,
                           ncoords_src, depth, nbins, args, stream);
}

} // namespace loki::core
