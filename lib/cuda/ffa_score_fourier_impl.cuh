#pragma once

/**
 * @file ffa_score_fourier_impl.cuh
 * @brief Templates of the fused Fourier final-level scorer: the final FFA
 * level(s) merged on chip, inverted with the portable block FFT
 * (ffa_portable_fft.cuh; direct for a power-of-two nbins in [32, 1024],
 * Bluestein otherwise) and scored. Included by one translation unit per
 * coordinate source (ffa_score_fourier_*_cuda.cu), so the instantiations
 * compile in parallel. See docs/fft.md. CUDA only.
 */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_portable_fft.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core::detail {

constexpr uint32_t align_up(uint32_t x, uint32_t a) {
    return ((x + a - 1) / a) * a;
}

// Profiles per block and the shared-memory row layout of a configuration.
// kPairs profiles are transformed per round, so kProfiles is a multiple.
// The rows are kept small (kMaxRowBytes) so that several blocks share an
// SM: the transforms are latency bound.
// Serial and Kadane (kKadane, the EP's Kadane scorer) keep one normalized row
// per profile.
template <uint32_t N, bool kSerial, uint32_t kPairs, bool kKadane = false>
struct RowLayout {
    // Serial: one normalized row per profile. Warp: e and v rows. An odd
    // pitch keeps the rows written by one warp on distinct banks.
    static constexpr uint32_t kPitch          = N + 1;
    static constexpr uint32_t kRowsPerProfile = (kSerial || kKadane) ? 1 : 2;
    static constexpr uint32_t kMaxRowBytes    = 12U * 1024U;
    static constexpr uint32_t kFit =
        kMaxRowBytes / (kRowsPerProfile * (N + 1) * 4U);
    static constexpr uint32_t kMaxWant = kKadane ? 32U : 16U;
    static constexpr uint32_t kWant =
        kSerial ? 32U : (kFit < 1U ? 1U : (kFit > kMaxWant ? kMaxWant : kFit));
    static constexpr uint32_t kProfiles =
        kWant < kPairs ? kPairs : (kWant / kPairs) * kPairs;
    static constexpr uint32_t kRowBytes =
        kProfiles * kRowsPerProfile * kPitch * 4U;
};

/**
 * Final FFA level (complex) + on-chip C2R + boxcar score. Each round runs
 * ffts_per_block C2R transforms: the e and v harmonics of ffts_per_block / 2
 * profiles. The merged harmonics go straight into the FFT registers, and the
 * real outputs into shared-memory rows that the time-domain scorers read.
 */
// Rows and shifts of one final profile. Depth 1: the final level from its
// two-segment parent level. Depth 2: the last two levels from the
// four-segment level below (row i is segment i), as two complex merge levels
// would form them. Rows are profile indices in the source level.
template <int kDepth> struct FourierRows {
    static constexpr int kRows   = 1 << kDepth;
    static constexpr int kShifts = (2 * kRows) - 2;
    uint32_t row[kRows];
    float shift[kShifts]; // s_t, s_h (final) then s_tt, s_th, s_ht, s_hh
};

// Source rows and shifts of final profile @p icoord (in traversal order).
template <int kDepth, class FinalParents, class PrevParents>
__device__ __forceinline__ FourierRows<kDepth>
load_fourier_rows(const FinalParents& parents,
                  const PrevParents& prev_parents,
                  uint32_t icoord,
                  uint32_t ncoords_src) {
    FourierRows<kDepth> r{};
    const TailHeadF th = parents.get(icoord);
    r.shift[0]         = th.shift_tail;
    r.shift[1]         = th.shift_head;
    if constexpr (kDepth == 1) {
        r.row[0] = th.i_tail;
        r.row[1] = ncoords_src + th.i_head;
    } else {
        const TailHeadF pt = prev_parents.get(th.i_tail);
        const TailHeadF ph = prev_parents.get(th.i_head);
        r.row[0]           = pt.i_tail;
        r.row[1]           = ncoords_src + pt.i_head;
        r.row[2]           = (2 * ncoords_src) + ph.i_tail;
        r.row[3]           = (3 * ncoords_src) + ph.i_head;
        r.shift[2]         = pt.shift_tail;
        r.shift[3]         = pt.shift_head;
        r.shift[4]         = ph.shift_tail;
        r.shift[5]         = ph.shift_head;
    }
    return r;
}

// --- Complex transform on chip: direct (power of two) or Bluestein ---------
//
// e and v are real, so one complex inverse DFT of X_e + i X_v gives both:
// e = Re z, v = Im z. The Hermitian halves are extended with the imaginary
// parts of the DC and Nyquist harmonics dropped, which is what C2R does.
// The inverse DFT is conj(FFT(conj(.))) with the forward block FFT.
//
// Direct (kChirp = false): M = N, a power of two.
//
// Bluestein (kChirp = true): any N. x[n] = sum_k X[k] exp(2 pi i k n / N)
// with k n = (k^2 + n^2 - (n - k)^2) / 2 is c[n] sum_k (X[k] c[k])
// conj(c[n - k]), c[j] = exp(i pi j^2 / N): a circular convolution of
// length M >= 2N - 1 (two M-point FFTs), with the table of
// ffa_fourier_bluestein_table (c, then the FFT of the wrapped conj(c) / M).

// Harmonic j of the full spectrum from its stored half (j < N).
__device__ __forceinline__ ComplexTypeCUDA
hermitian_at(const ComplexTypeCUDA* half, uint32_t j, uint32_t n) {
    const uint32_t nf = (n / 2) + 1;
    if (j < nf) {
        const ComplexTypeCUDA v = half[j];
        const bool real_bin     = j == 0 || 2 * j == n;
        return {v.real(), real_bin ? 0.0F : v.imag()};
    }
    const ComplexTypeCUDA v = half[n - j];
    return {v.real(), -v.imag()};
}

__device__ __forceinline__ ComplexTypeCUDA cmul(const ComplexTypeCUDA& a,
                                                const ComplexTypeCUDA& b) {
    return {fmaf(a.real(), b.real(), -a.imag() * b.imag()),
            fmaf(a.real(), b.imag(), a.imag() * b.real())};
}

template <class FinalParents,
          class PrevParents,
          class FFT,
          uint32_t M,
          bool kChirp,
          bool kSerial,
          uint32_t NW,
          int kDepth>
__global__
__launch_bounds__(FFT::max_threads_per_block) void kernel_final_score_c2c(
    const ComplexTypeCUDA* __restrict__ fold_in,
    FinalParents parents,
    PrevParents prev_parents,
    uint32_t ncoords_cur,
    uint32_t ncoords_src,
    uint32_t nbins,
    FFAFinalScoreArgs args) {
    using Rows                  = FourierRows<kDepth>;
    constexpr bool kTailShift   = FinalParents::kTailShift;
    constexpr uint32_t kNMax    = kChirp ? M / 2 : M;
    constexpr uint32_t kNFMax   = (kNMax / 2) + 1;
    constexpr uint32_t F        = FFT::ffts_per_block;
    constexpr uint32_t kEpt     = FFT::elements_per_thread;
    using Layout                = RowLayout<kNMax, kSerial, F>;
    using complex_type          = typename FFT::value_type;
    constexpr uint32_t kPB      = Layout::kProfiles;
    constexpr uint32_t kMaxIter = (kNFMax + FFT::stride - 1) / FFT::stride;
    static_assert(kPB % F == 0, "rounds must tile the profiles");
    static_assert(FFT::max_threads_per_block % kWarpSize == 0,
                  "the scorers need whole warps");
    static_assert(!kSerial || kPB <= FFT::max_threads_per_block,
                  "the serial scorer needs a thread per profile");

    extern __shared__ __align__(16) unsigned char smem_raw[]; // NOLINT
    __shared__ WidthConsts s_wc;
    __shared__ Rows s_src[kPB];
    auto* fft_smem = reinterpret_cast<complex_type*>(smem_raw);
    auto* s_half   = reinterpret_cast<ComplexTypeCUDA*>(
        smem_raw + align_up(FFT::shared_memory_size, 16U));
    auto* rows = reinterpret_cast<float*>(s_half + (F * 2 * kNFMax));

    const uint32_t n        = kChirp ? nbins : M;
    const uint32_t nf       = (n / 2) + 1;
    const uint32_t pitch    = n | 1U;
    const float inv_n       = 1.0F / static_cast<float>(n);
    const uint32_t tid      = (threadIdx.y * blockDim.x) + threadIdx.x;
    const uint32_t nthreads = blockDim.x * blockDim.y;
    const uint32_t p0       = args.profile_begin + (blockIdx.x * kPB);
    const uint32_t nprof    = min(kPB, ncoords_cur - p0);
    load_width_consts(args, static_cast<int>(n), s_wc, tid, nthreads);
    for (uint32_t i = tid; i < nprof; i += nthreads) {
        s_src[i] = load_fourier_rows<kDepth>(
            parents, prev_parents, tile_order(p0 + i, args), ncoords_src);
    }
    __syncthreads();

    const ComplexTypeCUDA* __restrict__ chirp = args.bluestein;
    const ComplexTypeCUDA* __restrict__ kern  = args.bluestein + n;
    const uint32_t fft_id                     = threadIdx.y;
    const uint32_t stride                     = 2 * nf;
    ComplexTypeCUDA* half_e                   = s_half + (fft_id * 2 * kNFMax);
    ComplexTypeCUDA* half_v                   = half_e + kNFMax;
    for (uint32_t r = 0; r < kPB; r += F) {
        const uint32_t q  = r + fft_id;
        const bool active = q < nprof;
        // Merge this profile's harmonics k = threadIdx.x + i * stride
        // into the half spectra (phasor recurrences as in the C2R
        // kernel).
        if (active) {
            const Rows& rs = s_src[q];
            Phasor ph[Rows::kShifts];
            Phasor st[Rows::kShifts];
#pragma unroll
            for (int j = 0; j < Rows::kShifts; ++j) {
                if (kTailShift || (j % 2) == 1) {
                    ph[j] = fourier_phasor(threadIdx.x, rs.shift[j], inv_n);
                    st[j] = fourier_phasor(FFT::stride, rs.shift[j], inv_n);
                } else {
                    ph[j] = Phasor{1.0F, 0.0F};
                    st[j] = Phasor{1.0F, 0.0F};
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < kMaxIter; ++i) {
                const uint32_t k = (i * FFT::stride) + threadIdx.x;
                if (k < nf) {
#pragma unroll
                    for (uint32_t comp = 0; comp < 2; ++comp) {
                        ComplexTypeCUDA src[Rows::kRows];
#pragma unroll
                        for (int j = 0; j < Rows::kRows; ++j) {
                            src[j] = fold_in[(static_cast<size_t>(rs.row[j]) *
                                              stride) +
                                             (comp * nf) + k];
                        }
                        ComplexTypeCUDA m;
                        if constexpr (kDepth == 1) {
                            m = merge_c<kTailShift>(src[0], src[1], ph[0],
                                                    ph[1]);
                        } else {
                            const auto pt  = merge_c<kTailShift>(src[0], src[1],
                                                                 ph[2], ph[3]);
                            const auto phd = merge_c<kTailShift>(src[2], src[3],
                                                                 ph[4], ph[5]);
                            m = merge_c<kTailShift>(pt, phd, ph[0], ph[1]);
                        }
                        (comp == 0 ? half_e : half_v)[k] = m;
                    }
                }
#pragma unroll
                for (int j = 0; j < Rows::kShifts; ++j) {
                    if (kTailShift || (j % 2) == 1) {
                        ph[j] = phasor_mul(ph[j], st[j]);
                    }
                }
            }
        }
        FFT::sync();

        complex_type a[FFT::storage_size];
#pragma unroll
        for (uint32_t i = 0; i < kEpt; ++i) {
            const uint32_t j = (i * FFT::stride) + threadIdx.x;
            ComplexTypeCUDA v{0.0F, 0.0F};
            if (active && j < n) {
                const ComplexTypeCUDA xe = hermitian_at(half_e, j, n);
                const ComplexTypeCUDA xv = hermitian_at(half_v, j, n);
                v = {xe.real() - xv.imag(), xe.imag() + xv.real()};
                if constexpr (kChirp) {
                    v = cmul(v, chirp[j]);
                } else {
                    v = {v.real(), -v.imag()};
                }
            }
            a[i] = complex_type{v.real(), v.imag()};
        }
        FFT().execute(a, fft_smem);
        if constexpr (kChirp) {
#pragma unroll
            for (uint32_t i = 0; i < kEpt; ++i) {
                const uint32_t j        = (i * FFT::stride) + threadIdx.x;
                const ComplexTypeCUDA y = cmul({a[i].x, a[i].y}, kern[j]);
                a[i]                    = complex_type{y.real(), -y.imag()};
            }
            FFT::sync();
            FFT().execute(a, fft_smem);
        }
        if (active) {
            const float s = args.fold_scale;
#pragma unroll
            for (uint32_t i = 0; i < kEpt; ++i) {
                const uint32_t j = (i * FFT::stride) + threadIdx.x;
                if (j < n) {
                    ComplexTypeCUDA z{a[i].x, -a[i].y};
                    if constexpr (kChirp) {
                        z = cmul(z, chirp[j]);
                    }
                    if constexpr (kSerial) {
                        rows[(q * pitch) + j] =
                            fold_norm_bin(z.real() * s, z.imag() * s);
                    } else {
                        rows[(((q * 2) + 0) * pitch) + j] = z.real();
                        rows[(((q * 2) + 1) * pitch) + j] = z.imag();
                    }
                }
            }
        }
        // The next round rewrites the half spectra and fft_smem.
        FFT::sync();
    }
    // The scorers read rows of every profile.
    __syncthreads();

    if constexpr (kSerial) {
        if (tid < nprof) {
            score_row_serial<NW>(
                rows + (tid * pitch), static_cast<int>(n), s_wc,
                tile_order(p0 + tid, args) * args.nwidths, args);
        }
    } else {
        const uint32_t warp_id = tid / kWarpSize;
        const auto lane_id     = static_cast<int>(tid % kWarpSize);
        const uint32_t nwarps  = nthreads / kWarpSize;
        for (uint32_t p = warp_id; p < nprof; p += nwarps) {
            float* e_row = rows + ((p * 2) * pitch);
            score_ev_warp<NW>(
                e_row, e_row + pitch, args.fold_scale, static_cast<int>(n),
                s_wc, tile_order(p0 + p, args) * args.nwidths, args, lane_id);
        }
    }
}

template <class FinalParents,
          class PrevParents,
          class FFT,
          uint32_t M,
          bool kChirp,
          bool kSerial,
          uint32_t NW,
          int kDepth>
void launch_c2c_cfg(const ComplexTypeCUDA* fold_in,
                    const FinalParents& parents,
                    const PrevParents& prev_parents,
                    SizeType ncoords_cur,
                    SizeType ncoords_src,
                    SizeType nbins,
                    const FFAFinalScoreArgs& args,
                    cudaStream_t stream) {
    constexpr uint32_t F      = FFT::ffts_per_block;
    constexpr uint32_t kNMax  = kChirp ? M / 2 : M;
    constexpr uint32_t kNFMax = (kNMax / 2) + 1;
    using Layout              = RowLayout<kNMax, kSerial, F>;
    const auto kernel = kernel_final_score_c2c<FinalParents, PrevParents, FFT,
                                               M, kChirp, kSerial, NW, kDepth>;
    const SizeType smem = align_up(FFT::shared_memory_size, 16U) +
                          (F * 2 * kNFMax * sizeof(ComplexTypeCUDA)) +
                          Layout::kRowBytes;
    cuda_utils::set_kernel_dynamic_smem(kernel, smem);
    const SizeType end    = std::min<SizeType>(ncoords_cur, args.profile_end);
    const SizeType nrange = end - args.profile_begin;
    const dim3 grid((nrange + Layout::kProfiles - 1) / Layout::kProfiles);
    kernel<<<grid, FFT::block_dim, smem, stream>>>(
        fold_in, parents, prev_parents, static_cast<uint32_t>(end),
        static_cast<uint32_t>(ncoords_src), static_cast<uint32_t>(nbins), args);
    cuda_utils::check_last_cuda_error("kernel_final_score_c2c launch failed");
}

template <class FinalParents,
          class PrevParents,
          class FFT,
          uint32_t M,
          bool kChirp,
          int kDepth>
void launch_c2c_m(const ComplexTypeCUDA* fold_in,
                  const FinalParents& parents,
                  const PrevParents& prev_parents,
                  SizeType ncoords_cur,
                  SizeType ncoords_src,
                  SizeType nbins,
                  const FFAFinalScoreArgs& args,
                  cudaStream_t stream) {
    constexpr SizeType kThreadRegimeMinProfiles = 1U << 16U;
    const auto launch                           = [&]<bool kSerial>() {
        if (args.nwidths <= 8) {
            launch_c2c_cfg<FinalParents, PrevParents, FFT, M, kChirp, kSerial,
                           8, kDepth>(fold_in, parents, prev_parents,
                                      ncoords_cur, ncoords_src, nbins, args,
                                      stream);
        } else if (args.nwidths <= 16) {
            launch_c2c_cfg<FinalParents, PrevParents, FFT, M, kChirp, kSerial,
                           16, kDepth>(fold_in, parents, prev_parents,
                                       ncoords_cur, ncoords_src, nbins, args,
                                       stream);
        } else {
            launch_c2c_cfg<FinalParents, PrevParents, FFT, M, kChirp, kSerial,
                           kMaxWidths, kDepth>(fold_in, parents, prev_parents,
                                               ncoords_cur, ncoords_src, nbins,
                                               args, stream);
        }
    };
    // Same regime choice as score_and_filter_cuda_d.
    constexpr uint32_t kNMax = kChirp ? M / 2 : M;
    if constexpr (kNMax <= 64) {
        if (nbins <= 64 && ncoords_cur >= kThreadRegimeMinProfiles) {
            launch.template operator()<true>();
            return;
        }
    }
    launch.template operator()<false>();
}

// Portable M-point transforms: 128 threads per block, few elements per
// thread (the merge and the scorers want threads; shared memory per thread
// bounds the occupancy).
constexpr uint32_t portable_ept(uint32_t m) {
    return m <= 128 ? 4 : (m <= 1024 ? 8 : 16);
}
template <uint32_t M>
using PortableC2C =
    PortableFFT<M, portable_ept(M), std::max(1U, 128U / (M / portable_ept(M)))>;

/**
 * Fused final level(s) + score through M-point complex transforms of the
 * portable block FFT: direct for a power-of-two nbins in
 * [32, 1024], Bluestein otherwise.
 */
template <class FinalParents, class PrevParents, int kDepth>
void launch_c2c(const ComplexTypeCUDA* fold_in,
                const FinalParents& parents,
                const PrevParents& prev_parents,
                SizeType ncoords_cur,
                SizeType ncoords_src,
                SizeType nbins,
                const FFAFinalScoreArgs& args,
                cudaStream_t stream) {
    const bool direct = !ffa_fourier_needs_bluestein(nbins);
    if (!direct) {
        error_check::check(args.bluestein != nullptr,
                           "ffa_final_score_fourier_cuda: nbins needs the "
                           "Bluestein table (args.bluestein)");
    }
    const SizeType m =
        direct ? nbins : std::max<SizeType>(32, std::bit_ceil((2 * nbins) - 1));
#define LOKI_C2C_M(M)                                                          \
    case M:                                                                    \
        if (direct) {                                                          \
            if constexpr (M <= 1024) {                                         \
                launch_c2c_m<FinalParents, PrevParents, PortableC2C<M>, M,     \
                             false, kDepth>(fold_in, parents, prev_parents,    \
                                            ncoords_cur, ncoords_src, nbins,   \
                                            args, stream);                     \
            }                                                                  \
        } else {                                                               \
            launch_c2c_m<FinalParents, PrevParents, PortableC2C<M>, M, true,   \
                         kDepth>(fold_in, parents, prev_parents, ncoords_cur,  \
                                 ncoords_src, nbins, args, stream);            \
        }                                                                      \
        break
    switch (m) {
        LOKI_C2C_M(32);
        LOKI_C2C_M(64);
        LOKI_C2C_M(128);
        LOKI_C2C_M(256);
        LOKI_C2C_M(512);
        LOKI_C2C_M(1024);
        LOKI_C2C_M(2048);
    default:
        throw std::invalid_argument(
            "ffa_final_score_fourier_cuda: unsupported nbins");
    }
#undef LOKI_C2C_M
}

/// Fused Fourier final level(s) + score; depth 1 or 2 (see
/// ffa_final_score_cuda for the level counts).
template <class FinalParents, class PrevParents>
void launch_fourier(const ComplexTypeCUDA* fold_in,
                    const FinalParents& parents,
                    const PrevParents& prev_parents,
                    SizeType ncoords_cur,
                    SizeType ncoords_src,
                    int depth,
                    SizeType nbins,
                    const FFAFinalScoreArgs& args,
                    cudaStream_t stream) {
    if (std::min<SizeType>(ncoords_cur, args.profile_end) <=
        args.profile_begin) {
        return;
    }
    error_check::check_less_equal(
        SizeType{args.nwidths}, SizeType{kMaxWidths},
        "ffa_final_score_fourier_cuda: too many boxcar widths");
    error_check::check_less_equal(
        ncoords_cur * args.nwidths, SizeType{UINT32_MAX},
        "ffa_final_score_fourier_cuda: score index exceeds 32 bits");
    error_check::check_less_equal(
        4 * ncoords_src, SizeType{UINT32_MAX},
        "ffa_final_score_fourier_cuda: source rows exceed 32 bits");
    if (depth == 1) {
        launch_c2c<FinalParents, PrevParents, 1>(fold_in, parents, prev_parents,
                                                 ncoords_cur, ncoords_src,
                                                 nbins, args, stream);
    } else if (depth == 2) {
        launch_c2c<FinalParents, PrevParents, 2>(fold_in, parents, prev_parents,
                                                 ncoords_cur, ncoords_src,
                                                 nbins, args, stream);
    } else {
        throw std::invalid_argument(
            "ffa_final_score_fourier_cuda: depth must be 1 or 2");
    }
}

} // namespace loki::core::detail
