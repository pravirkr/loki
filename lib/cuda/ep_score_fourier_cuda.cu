#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "lib/algorithms/ep_memory.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ep_fourier_device.cuh"
#include "lib/cuda/ep_score_cuda.cuh"
#include "lib/cuda/ep_score_device.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/cuda/ffa_score_fourier_impl.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

using detail::align_up;
using detail::cmul;
using detail::EPBlock;
using detail::fold_norm_bin;
using detail::hermitian_at;
using detail::kMaxWidths;
using detail::kWarpSize;
using detail::RowLayout;
using detail::WidthConstsView;

// Width constants (any count) after the kernel's other shared memory.
__host__ __device__ constexpr uint32_t width_consts_bytes(uint32_t nwidths) {
    return nwidths * 3U * 4U;
}

__device__ __forceinline__ WidthConstsView width_consts_at(unsigned char* p,
                                                           uint32_t nwidths) {
    auto* w = reinterpret_cast<int*>(p);
    auto* h = reinterpret_cast<float*>(w + nwidths);
    return {w, h, h + nwidths};
}

// Parent and FFA harmonics of one leaf held by a thread, loaded a round
// ahead so that their latency overlaps the previous round's transform.
struct LeafHarmonics {
    ComplexTypeCUDA tree_e;
    ComplexTypeCUDA tree_v;
    ComplexTypeCUDA ffa_e;
    ComplexTypeCUDA ffa_v;
};

// Without an FFA term (rows mode, @p ffa null) only the tree is loaded.
template <uint32_t kMaxIter, uint32_t kStride>
__device__ __forceinline__ void
load_leaf_harmonics(LeafHarmonics (&h)[kMaxIter],
                    const ComplexTypeCUDA* __restrict__ tree,
                    const ComplexTypeCUDA* __restrict__ ffa,
                    uint32_t nf) {
#pragma unroll
    for (uint32_t i = 0; i < kMaxIter; ++i) {
        const uint32_t k = (i * kStride) + threadIdx.x;
        if (k < nf) {
            if (ffa != nullptr) {
                h[i] = {tree[k], tree[nf + k], ffa[k], ffa[nf + k]};
            } else {
                h[i] = {tree[k], tree[nf + k], {}, {}};
            }
        }
    }
}

/**
 * One pruning batch, fused: each block takes kPB leaves. In rounds of F
 * leaves (one complex transform each: e + i v), the threads merge the
 * leaf's harmonics (ep_shift_add_harmonic, also stored to folds_out) into
 * shared half spectra, run the inverse DFT on chip (direct or Bluestein, as
 * kernel_final_score_c2c) and write the real profile rows. Then the rows are
 * scored with the thread or warp scorer of score_and_filter_max_cuda_d.
 */
template <class FFT, uint32_t M, bool kChirp, bool kSerial, uint32_t NW>
__global__
__launch_bounds__(FFT::max_threads_per_block) void kernel_ep_shift_add_score_c2c(
    EPFourierScoreArgs a) {
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
    __shared__ EPBlock<kPB> b;
    auto* fft_smem = reinterpret_cast<complex_type*>(smem_raw);
    auto* s_half   = reinterpret_cast<ComplexTypeCUDA*>(
        smem_raw + align_up(FFT::shared_memory_size, 16U));
    auto* rows           = reinterpret_cast<float*>(s_half + (F * 2 * kNFMax));
    WidthConstsView s_wc = width_consts_at(
        reinterpret_cast<unsigned char*>(rows) + Layout::kRowBytes, a.nwidths);

    const uint32_t n        = kChirp ? a.nbins : M;
    const uint32_t nf       = (n / 2) + 1;
    const uint32_t stride   = 2 * nf;
    const uint32_t pitch    = n | 1U;
    const uint32_t tid      = (threadIdx.y * blockDim.x) + threadIdx.x;
    const uint32_t nthreads = blockDim.x * blockDim.y;
    const bool direct       = a.tree_folds != nullptr;
    const bool rows_mode    = a.indices_tree == nullptr;
    uint32_t p0;
    uint32_t nprof;
    detail::ep_block_begin(b, a, kPB, tid, nthreads, p0, nprof);
    detail::load_width_consts(a.widths, a.nwidths, static_cast<int>(n), s_wc,
                              tid, nthreads);
    __syncthreads();

    const ComplexTypeCUDA* __restrict__ chirp = a.bluestein;
    const ComplexTypeCUDA* __restrict__ kern  = a.bluestein + n;
    const uint32_t fft_id                     = threadIdx.y;
    ComplexTypeCUDA* half_e                   = s_half + (fft_id * 2 * kNFMax);
    ComplexTypeCUDA* half_v                   = half_e + kNFMax;
    const auto load_round = [&](LeafHarmonics(&h)[kMaxIter], uint32_t r) {
        const uint32_t q = r + fft_id;
        if (q < nprof && b.valid[q] != 0) {
            load_leaf_harmonics<kMaxIter, FFT::stride>(
                h, a.folds_tree + (static_cast<size_t>(b.tree[q]) * stride),
                rows_mode
                    ? nullptr
                    : a.folds_ffa + (static_cast<size_t>(b.ffa[q]) * stride),
                nf);
        }
    };
    LeafHarmonics next[kMaxIter];
    load_round(next, 0);
    for (uint32_t r = 0; r < kPB; r += F) {
        const uint32_t q  = r + fft_id;
        const bool active = q < nprof && b.valid[q] != 0;
        LeafHarmonics cur[kMaxIter];
#pragma unroll
        for (uint32_t i = 0; i < kMaxIter; ++i) {
            cur[i] = next[i];
        }
        if (r + F < kPB) {
            load_round(next, r + F);
        }
        if (active) {
            ComplexTypeCUDA* __restrict__ out =
                a.folds_out + (static_cast<size_t>(p0 + q) * stride);
            const float shift = b.shift[q];
#pragma unroll
            for (uint32_t i = 0; i < kMaxIter; ++i) {
                const uint32_t k = (i * FFT::stride) + threadIdx.x;
                if (k < nf) {
                    ComplexTypeCUDA me = cur[i].tree_e;
                    ComplexTypeCUDA mv = cur[i].tree_v;
                    if (!rows_mode) {
                        detail::ep_shift_add_harmonic(
                            cur[i].tree_e, cur[i].tree_v, cur[i].ffa_e,
                            cur[i].ffa_v, k, shift, n, me, mv);
                    }
                    half_e[k] = me;
                    half_v[k] = mv;
                    if (!direct && !rows_mode) {
                        out[k]      = me;
                        out[nf + k] = mv;
                    }
                }
            }
        }
        FFT::sync();

        complex_type x[FFT::storage_size];
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
            x[i] = complex_type{v.real(), v.imag()};
        }
        FFT().execute(x, fft_smem);
        if constexpr (kChirp) {
#pragma unroll
            for (uint32_t i = 0; i < kEpt; ++i) {
                const uint32_t j        = (i * FFT::stride) + threadIdx.x;
                const ComplexTypeCUDA y = cmul({x[i].x, x[i].y}, kern[j]);
                x[i]                    = complex_type{y.real(), -y.imag()};
            }
            FFT::sync();
            FFT().execute(x, fft_smem);
        }
        if (active) {
            const float s = 1.0F / static_cast<float>(n);
#pragma unroll
            for (uint32_t i = 0; i < kEpt; ++i) {
                const uint32_t j = (i * FFT::stride) + threadIdx.x;
                if (j < n) {
                    ComplexTypeCUDA z{x[i].x, -x[i].y};
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

    uint32_t npass = 0;
    if constexpr (kSerial) {
        if (tid < nprof) {
            if (b.valid[tid] != 0) {
                float best = cuda::std::numeric_limits<float>::lowest();
                detail::score_row_serial_sink<NW>(
                    rows + (tid * pitch), static_cast<int>(n), s_wc, a.nwidths,
                    [&best](uint32_t /*iw*/, float snr) {
                        best = fmaxf(best, snr);
                    });
                detail::ep_block_record(b, a, tid, p0, best, npass);
            } else {
                a.filtered_mask[p0 + tid] = 0;
            }
        }
    } else {
        const uint32_t warp_id = tid / kWarpSize;
        const auto lane_id     = static_cast<int>(tid % kWarpSize);
        const uint32_t nwarps  = nthreads / kWarpSize;
        const float s          = 1.0F / static_cast<float>(n);
        for (uint32_t p = warp_id; p < nprof; p += nwarps) {
            if (b.valid[p] == 0) {
                if (lane_id == 0) {
                    a.filtered_mask[p0 + p] = 0; // not in rows mode
                }
                continue;
            }
            float best   = cuda::std::numeric_limits<float>::lowest();
            float* e_row = rows + ((p * 2) * pitch);
            detail::score_ev_warp_sink<NW>(
                e_row, e_row + pitch, s, static_cast<int>(n), s_wc, a.nwidths,
                [&best](uint32_t /*iw*/, float snr) {
                    best = fmaxf(best, snr);
                },
                lane_id);
            if (lane_id == 0) {
                detail::ep_block_record(b, a, p, p0, best, npass);
            }
        }
    }
    if (!detail::ep_block_finish(b, a, npass, p0, nprof, tid, nthreads)) {
        return;
    }

    // Direct mode: the survivors' folds. They are merged again (the rows
    // are hot in L2), with the same arithmetic, rather than kept on chip for
    // every profile.
    // No barrier inside: unrolled, all the loads are issued up front.
#pragma unroll
    for (uint32_t r = 0; r < kPB; r += F) {
        const uint32_t q = r + fft_id;
        if (q < nprof && b.flag[q] != 0) {
            LeafHarmonics h[kMaxIter];
            load_leaf_harmonics<kMaxIter, FFT::stride>(
                h, a.folds_tree + (static_cast<size_t>(b.tree[q]) * stride),
                a.folds_ffa + (static_cast<size_t>(b.ffa[q]) * stride), nf);
            ComplexTypeCUDA* __restrict__ out =
                a.tree_folds +
                (static_cast<size_t>(detail::ep_block_slot(b, a, q)) * stride);
            const float shift = b.shift[q];
#pragma unroll
            for (uint32_t i = 0; i < kMaxIter; ++i) {
                const uint32_t k = (i * FFT::stride) + threadIdx.x;
                if (k < nf) {
                    detail::ep_shift_add_harmonic(
                        h[i].tree_e, h[i].tree_v, h[i].ffa_e, h[i].ffa_v, k,
                        shift, n, out[k], out[nf + k]);
                }
            }
        }
    }
}

template <class FFT, uint32_t M, bool kChirp, bool kSerial, uint32_t NW>
void launch_cfg(const EPFourierScoreArgs& args, cudaStream_t stream) {
    constexpr uint32_t F      = FFT::ffts_per_block;
    constexpr uint32_t kNMax  = kChirp ? M / 2 : M;
    constexpr uint32_t kNFMax = (kNMax / 2) + 1;
    using Layout              = RowLayout<kNMax, kSerial, F>;
    const auto kernel =
        kernel_ep_shift_add_score_c2c<FFT, M, kChirp, kSerial, NW>;
    const SizeType smem = align_up(FFT::shared_memory_size, 16U) +
                          (F * 2 * kNFMax * sizeof(ComplexTypeCUDA)) +
                          Layout::kRowBytes + width_consts_bytes(args.nwidths);
    cuda_utils::set_kernel_dynamic_smem(kernel, smem);
    const dim3 grid((args.n_leaves + Layout::kProfiles - 1) /
                    Layout::kProfiles);
    kernel<<<grid, FFT::block_dim, smem, stream>>>(args);
    cuda_utils::check_last_cuda_error(
        "kernel_ep_shift_add_score_c2c launch failed");
}

template <class FFT, uint32_t M, bool kChirp>
void launch_m(const EPFourierScoreArgs& args,
              cudaStream_t stream,
              SizeType* profiles_per_block) {
    const auto launch = [&]<bool kSerial>() {
        if (profiles_per_block != nullptr) {
            constexpr uint32_t kNMax = kChirp ? M / 2 : M;
            *profiles_per_block =
                RowLayout<kNMax, kSerial, FFT::ffts_per_block>::kProfiles;
            return;
        }
        // More than kMaxWidths widths are scored in groups of kMaxWidths.
        if (args.nwidths <= 8) {
            launch_cfg<FFT, M, kChirp, kSerial, 8>(args, stream);
        } else if (args.nwidths <= 16) {
            launch_cfg<FFT, M, kChirp, kSerial, 16>(args, stream);
        } else {
            launch_cfg<FFT, M, kChirp, kSerial, kMaxWidths>(args, stream);
        }
    };
    // Same regime choice as score_and_filter_max_cuda_d.
    constexpr SizeType kThreadRegimeMinProfiles = 1U << 16U;
    constexpr uint32_t kNMax                    = kChirp ? M / 2 : M;
    if constexpr (kNMax <= 64) {
        if (args.nbins <= 64 && args.n_leaves >= kThreadRegimeMinProfiles) {
            launch.template operator()<true>();
            return;
        }
    }
    launch.template operator()<false>();
}

} // namespace

template <>
bool ep_shift_add_score_supported<ComplexTypeCUDA>(SizeType nbins,
                                                   SizeType nwidths) noexcept {
    return algorithms::detail::ep_cuda_fused_rows(nbins) && nwidths >= 1;
}

namespace {

// Launches the kernel, or with @p profiles_per_block only reports the
// profiles per block of the configuration it would launch.
void dispatch(const EPFourierScoreArgs& args,
              cudaStream_t stream,
              SizeType* profiles_per_block) {
    const SizeType nbins = args.nbins;
    error_check::check(
        ep_shift_add_score_supported<ComplexTypeCUDA>(nbins, args.nwidths),
        "ep_shift_add_score_fourier_cuda: unsupported nbins or "
        "number of widths");
    const bool direct = !ffa_fourier_needs_bluestein(nbins);
    if (!direct && profiles_per_block == nullptr) {
        error_check::check(args.bluestein != nullptr,
                           "ep_shift_add_score_fourier_cuda: nbins needs the "
                           "Bluestein table (args.bluestein)");
    }
    const SizeType m =
        direct ? nbins : std::max<SizeType>(32, std::bit_ceil((2 * nbins) - 1));
#define LOKI_EP_C2C_M(M)                                                       \
    case M:                                                                    \
        if (direct) {                                                          \
            if constexpr (M <= 1024) {                                         \
                launch_m<detail::PortableC2C<M>, M, false>(                    \
                    args, stream, profiles_per_block);                         \
            }                                                                  \
        } else {                                                               \
            launch_m<detail::PortableC2C<M>, M, true>(args, stream,            \
                                                      profiles_per_block);     \
        }                                                                      \
        break
    switch (m) {
        LOKI_EP_C2C_M(32);
        LOKI_EP_C2C_M(64);
        LOKI_EP_C2C_M(128);
        LOKI_EP_C2C_M(256);
        LOKI_EP_C2C_M(512);
        LOKI_EP_C2C_M(1024);
        LOKI_EP_C2C_M(2048);
    default:
        throw std::invalid_argument(
            "ep_shift_add_score_fourier_cuda: unsupported nbins");
    }
#undef LOKI_EP_C2C_M
}

} // namespace

template <>
SizeType ep_shift_add_score_tile_words<ComplexTypeCUDA>(SizeType nbins,
                                                        SizeType n_leaves) {
    EPFourierScoreArgs query{};
    query.nbins        = static_cast<uint32_t>(nbins);
    query.n_leaves     = static_cast<uint32_t>(n_leaves);
    query.nwidths      = 1;
    SizeType per_block = 1;
    dispatch(query, nullptr, &per_block);
    return ((n_leaves + per_block - 1) / per_block) + 1;
}

void ep_score_rows_fourier_cuda(const ComplexTypeCUDA* folds,
                                SizeType n_rows,
                                SizeType nbins,
                                const uint32_t* widths,
                                SizeType nwidths,
                                float* scores,
                                const ComplexTypeCUDA* bluestein,
                                cudaStream_t stream) {
    if (n_rows == 0) {
        return;
    }
    error_check::check_less_equal(n_rows, SizeType{UINT32_MAX},
                                  "ep_score_rows_fourier_cuda: too many rows");
    EPFourierScoreArgs args{};
    args.folds_tree = folds;
    args.scores     = scores;
    args.widths     = widths;
    args.nwidths    = static_cast<uint32_t>(nwidths);
    args.threshold  = cuda::std::numeric_limits<float>::lowest();
    args.n_leaves   = static_cast<uint32_t>(n_rows);
    args.nbins      = static_cast<uint32_t>(nbins);
    args.bluestein  = bluestein;
    dispatch(args, stream, nullptr);
}

template <>
void ep_shift_add_score_cuda<ComplexTypeCUDA>(const EPFourierScoreArgs& args,
                                              cudaStream_t stream) {
    if (args.n_leaves == 0) {
        return;
    }
    dispatch(args, stream, nullptr);
}

} // namespace loki::core
