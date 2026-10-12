#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/cuda/kernel_utils.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

using namespace detail; // NOLINT(google-build-using-namespace): device helpers

// One complex merge level of the frequency sweep. A thread owns one
// (coordinate, harmonic) and a run of segments, so its two phasors are
// computed once and reused for every segment (they do not depend on the
// segment). Harmonics are the fastest index, so loads and stores coalesce.
template <class Parents>
__global__ void
kernel_ffa_complex_iter_sweep(const ComplexTypeCUDA* __restrict__ fold_in,
                              ComplexTypeCUDA* __restrict__ fold_out,
                              Parents parents,
                              uint32_t ncoords_cur,
                              uint32_t ncoords_prev,
                              uint32_t nsegments,
                              uint32_t segs_per_thread,
                              uint32_t nbins_f,
                              float inv_nbins) {
    const size_t tid =
        (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const size_t per_group = static_cast<size_t>(ncoords_cur) * nbins_f;
    const size_t group     = tid / per_group;
    const size_t rem       = tid - (group * per_group);
    const auto icoord      = static_cast<uint32_t>(rem / nbins_f);
    const auto k =
        static_cast<uint32_t>(rem - (static_cast<size_t>(icoord) * nbins_f));
    const size_t seg_begin = group * segs_per_thread;
    if (seg_begin >= nsegments) {
        return;
    }
    const auto seg_end = static_cast<uint32_t>(
        min(static_cast<size_t>(nsegments), seg_begin + segs_per_thread));

    const TailHeadF th = parents.get(icoord);
    const Phasor ph    = fourier_phasor(k, th.shift_head, inv_nbins);
    Phasor pt{1.0F, 0.0F};
    if constexpr (Parents::kTailShift) {
        pt = fourier_phasor(k, th.shift_tail, inv_nbins);
    }
    const uint32_t stride = 2 * nbins_f;
    for (auto iseg = static_cast<uint32_t>(seg_begin); iseg < seg_end; ++iseg) {
        const ComplexTypeCUDA* tail =
            fold_in +
            profile_offset(2 * iseg, ncoords_prev, th.i_tail, stride) + k;
        const ComplexTypeCUDA* head =
            fold_in +
            profile_offset((2 * iseg) + 1, ncoords_prev, th.i_head, stride) + k;
        ComplexTypeCUDA* out =
            fold_out + profile_offset(iseg, ncoords_cur, icoord, stride) + k;
        out[0] = merge_c<Parents::kTailShift>(tail[0], head[0], pt, ph);
        out[nbins_f] =
            merge_c<Parents::kTailShift>(tail[nbins_f], head[nbins_f], pt, ph);
    }
}

template <class Parents>
void launch_complex_iter_sweep(const ComplexTypeCUDA* fold_in,
                               ComplexTypeCUDA* fold_out,
                               const Parents& parents,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               SizeType nsegments,
                               SizeType nbins_f,
                               SizeType nbins,
                               cudaStream_t stream) {
    const SizeType per_group = ncoords_cur * nbins_f;
    if (per_group == 0 || nsegments == 0) {
        return;
    }
    // Enough threads to fill the GPU; the rest of the work is a segment loop.
    constexpr SizeType kTargetThreads = SizeType{1} << 20U;
    const SizeType total              = per_group * nsegments;
    const SizeType segs_per_thread    = std::clamp<SizeType>(
        total / kTargetThreads, 1, std::min<SizeType>(nsegments, 64));
    const SizeType groups = (nsegments + segs_per_thread - 1) / segs_per_thread;
    constexpr SizeType kThreads = 256;
    const SizeType blocks = ((groups * per_group) + kThreads - 1) / kThreads;
    const dim3 grid(static_cast<uint32_t>(blocks));
    cuda_utils::check_kernel_launch_params(grid, dim3(kThreads));
    error_check::check_less_equal(
        ncoords_cur, SizeType{UINT32_MAX},
        "ffa_complex_iter_sweep_cuda: ncoords exceeds 32 bits");
    kernel_ffa_complex_iter_sweep<Parents><<<grid, kThreads, 0, stream>>>(
        fold_in, fold_out, parents, static_cast<uint32_t>(ncoords_cur),
        static_cast<uint32_t>(ncoords_prev), static_cast<uint32_t>(nsegments),
        static_cast<uint32_t>(segs_per_thread), static_cast<uint32_t>(nbins_f),
        1.0F / static_cast<float>(nbins));
    cuda_utils::check_last_cuda_error(
        "kernel_ffa_complex_iter_sweep launch failed");
}

// Phase-bin map of the time-domain brute fold, as ComputePhase
// (fold_cuda.cu) computes it.
__global__ void kernel_brute_fold_phase_map(const double* __restrict__ freqs,
                                            uint32_t nfreqs,
                                            uint32_t segment_len,
                                            uint32_t nbins,
                                            double tsamp,
                                            double t_ref,
                                            uint32_t* __restrict__ phase_map) {
    const size_t idx =
        (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(nfreqs) * segment_len) {
        return;
    }
    const auto ifreq = static_cast<uint32_t>(idx / segment_len);
    const auto isamp =
        static_cast<uint32_t>(idx - (static_cast<size_t>(ifreq) * segment_len));
    const double proper_time = (static_cast<double>(isamp) * tsamp) - t_ref;
    const float phase =
        utils::get_phase_idx_device(proper_time, freqs[ifreq], nbins, 0.0);
    const uint32_t shift = __float2uint_rz(phase + 0.5F);
    phase_map[idx]       = (shift == nbins) ? 0 : shift;
}

// Time-domain brute fold, one block per segment. The block's profiles for
// up to freqs_per_pass frequencies accumulate in shared memory (shared
// atomics), so each sample is read once per pass, and the profiles are
// written out contiguously: fold is [segment][freq][e, v][nbins].
__global__ void
kernel_brute_fold_segment(const float* __restrict__ ts_e,
                          const float* __restrict__ ts_v,
                          const uint32_t* __restrict__ phase_map,
                          float* __restrict__ fold,
                          uint32_t nfreqs,
                          uint32_t segment_len,
                          uint32_t nbins,
                          uint32_t freqs_per_pass) {
    extern __shared__ float s_prof[]; // NOLINT
    const size_t iseg     = blockIdx.x;
    const size_t seg_base = iseg * segment_len;
    const uint32_t stride = 2 * nbins;
    for (uint32_t f0 = 0; f0 < nfreqs; f0 += freqs_per_pass) {
        const uint32_t nf    = min(freqs_per_pass, nfreqs - f0);
        const uint32_t nelem = nf * stride;
        for (uint32_t j = threadIdx.x; j < nelem; j += blockDim.x) {
            s_prof[j] = 0.0F;
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < segment_len; i += blockDim.x) {
            const float e = ts_e[seg_base + i];
            const float v = ts_v[seg_base + i];
            for (uint32_t f = 0; f < nf; ++f) {
                const uint32_t bin =
                    phase_map[(static_cast<size_t>(f0 + f) * segment_len) + i];
                atomicAdd(&s_prof[(f * stride) + bin], e);
                atomicAdd(&s_prof[(f * stride) + nbins + bin], v);
            }
        }
        __syncthreads();
        float* out = fold + (((iseg * nfreqs) + f0) * stride);
        for (uint32_t j = threadIdx.x; j < nelem; j += blockDim.x) {
            out[j] = s_prof[j];
        }
        __syncthreads();
    }
}

struct FreqLvlC {
    const uint32_t* idx;
    const float* shift;
    uint32_t ncoords;
};

template <int K> struct FreqLvlsC {
    FreqLvlC level[K];
};

__device__ __forceinline__ uint32_t lower_bound_idx(const uint32_t* data,
                                                    uint32_t n,
                                                    uint32_t key) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1U);
        if (data[mid] < key) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// K frequency-only complex merge levels for one ancestor coordinate and a
// tile of 2^K segments, intermediates in shared memory (the complex
// counterpart of kernel_ffa_freq_fuse). Each child's phasor of harmonic k is
// computed once per level and reused for all of its segments.
template <int K>
__global__ void
kernel_ffa_complex_freq_fuse(const ComplexTypeCUDA* __restrict__ fold_in,
                             ComplexTypeCUDA* __restrict__ fold_out,
                             FreqLvlsC<K> levels,
                             uint32_t ncoords_in,
                             uint32_t ncoords_out,
                             uint32_t nbins_f,
                             float inv_nbins) {
    constexpr int kCap      = 1 << K;
    const uint32_t stride   = 2U * nbins_f;
    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const size_t block_id   = blockIdx.x;
    const auto ancestor     = static_cast<uint32_t>(block_id % ncoords_in);
    const auto tile         = static_cast<uint32_t>(block_id / ncoords_in);
    const uint32_t seg0     = tile << K;

    extern __shared__ __align__(16) unsigned char fuse_smem_raw[]; // NOLINT
    auto* fuse_smem = reinterpret_cast<ComplexTypeCUDA*>(fuse_smem_raw);
    ComplexTypeCUDA* bufs[2] = {fuse_smem, fuse_smem + (kCap * stride)};

    for (uint32_t i = tid; i < kCap * stride; i += nthreads) {
        const uint32_t prof = i / stride;
        const uint32_t elem = i - (prof * stride);
        bufs[0][i] =
            fold_in[profile_offset(seg0 + prof, ncoords_in, ancestor, stride) +
                    elem];
    }
    __shared__ uint32_t s_lo;
    __shared__ uint32_t s_nout;
    __shared__ float s_shift[kCap];
    __shared__ uint32_t s_local[kCap];

    uint32_t parent_lo = ancestor;
    uint32_t nparents  = 1;
    uint32_t nseg      = static_cast<uint32_t>(kCap);
    int cur_buf        = 0;
    for (int step = 0; step < K; ++step) {
        __syncthreads();
        if (tid == 0) {
            const FreqLvlC lvl = levels.level[step];
            const uint32_t lo =
                lower_bound_idx(lvl.idx, lvl.ncoords, parent_lo);
            const uint32_t hi =
                lower_bound_idx(lvl.idx, lvl.ncoords, parent_lo + nparents);
            s_lo   = lo;
            s_nout = hi - lo;
        }
        __syncthreads();
        const uint32_t lo   = s_lo;
        const uint32_t nout = s_nout;
        for (uint32_t child = tid; child < nout; child += nthreads) {
            s_shift[child] = levels.level[step].shift[lo + child];
            s_local[child] = levels.level[step].idx[lo + child] - parent_lo;
        }
        __syncthreads();

        const uint32_t nseg_out    = nseg >> 1U;
        ComplexTypeCUDA* nxt       = bufs[1 - cur_buf];
        const ComplexTypeCUDA* src = bufs[cur_buf];
        for (uint32_t item = tid; item < nout * nbins_f; item += nthreads) {
            const uint32_t child = item / nbins_f;
            const uint32_t k     = item - (child * nbins_f);
            const uint32_t local = s_local[child];
            const Phasor ph      = fourier_phasor(k, s_shift[child], inv_nbins);
            const Phasor one{1.0F, 0.0F};
            for (uint32_t seg = 0; seg < nseg_out; ++seg) {
                const ComplexTypeCUDA* tail =
                    src + ((((seg * 2U) * nparents) + local) * stride) + k;
                const ComplexTypeCUDA* head =
                    src + (((((seg * 2U) + 1U) * nparents) + local) * stride) +
                    k;
                ComplexTypeCUDA* dst =
                    nxt + (((seg * nout) + child) * stride) + k;
                dst[0] = merge_c<false>(tail[0], head[0], one, ph);
                dst[nbins_f] =
                    merge_c<false>(tail[nbins_f], head[nbins_f], one, ph);
            }
        }
        parent_lo = lo;
        nparents  = nout;
        nseg      = nseg_out;
        cur_buf ^= 1;
    }
    __syncthreads();
    const ComplexTypeCUDA* src = bufs[cur_buf];
    for (uint32_t i = tid; i < nparents * stride; i += nthreads) {
        const uint32_t child = i / stride;
        const uint32_t elem  = i - (child * stride);
        fold_out[profile_offset(tile, ncoords_out, parent_lo + child, stride) +
                 elem]       = src[i];
    }
}

template <int K> SizeType complex_freq_fuse_static_smem() {
    static const SizeType bytes = [] {
        cudaFuncAttributes attr{};
        cuda_utils::check_cuda_call(
            cudaFuncGetAttributes(&attr, kernel_ffa_complex_freq_fuse<K>),
            "cudaFuncGetAttributes kernel_ffa_complex_freq_fuse failed");
        return static_cast<SizeType>(attr.sharedSizeBytes);
    }();
    return bytes;
}

SizeType complex_freq_fuse_dynamic_smem(int k, SizeType nbins_f) {
    return 2ULL * (SizeType{1} << k) * 2ULL * nbins_f * sizeof(ComplexTypeCUDA);
}

template <int K>
void launch_complex_freq_fuse(const ComplexTypeCUDA* fold_in,
                              ComplexTypeCUDA* fold_out,
                              const uint32_t* const* level_idx,
                              const float* const* level_shift,
                              const uint32_t* level_ncoords,
                              uint32_t ncoords_in,
                              uint32_t ncoords_out,
                              uint32_t nsegments_in,
                              uint32_t nbins_f,
                              uint32_t nbins,
                              cudaStream_t stream) {
    FreqLvlsC<K> levels{};
    for (int step = 0; step < K; ++step) {
        levels.level[step] = FreqLvlC{.idx     = level_idx[step],
                                      .shift   = level_shift[step],
                                      .ncoords = level_ncoords[step]};
    }
    const size_t nblocks = static_cast<size_t>(nsegments_in >> K) * ncoords_in;
    const SizeType smem  = complex_freq_fuse_dynamic_smem(K, nbins_f);
    const SizeType total_smem = smem + complex_freq_fuse_static_smem<K>();
    error_check::check_less_equal(
        total_smem, cuda_utils::get_max_shared_memory_optin(),
        "launch_complex_freq_fuse: tile exceeds the shared memory per block");
    if (total_smem > cuda_utils::get_max_shared_memory()) {
        cuda_utils::check_cuda_call(
            cudaFuncSetAttribute(kernel_ffa_complex_freq_fuse<K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute kernel_ffa_complex_freq_fuse failed");
    }
    constexpr uint32_t kThreads = 256;
    cuda_utils::check_kernel_launch_params(dim3(static_cast<unsigned>(nblocks)),
                                           dim3(kThreads));
    kernel_ffa_complex_freq_fuse<K>
        <<<static_cast<unsigned>(nblocks), kThreads, smem, stream>>>(
            fold_in, fold_out, levels, ncoords_in, ncoords_out, nbins_f,
            1.0F / static_cast<float>(nbins));
    cuda_utils::check_last_cuda_error(
        "kernel_ffa_complex_freq_fuse launch failed");
}

} // namespace

bool ffa_complex_freq_fuse_fits_smem(int k, SizeType nbins_f) {
    SizeType static_bytes = 0;
    switch (k) {
    case 2:
        static_bytes = complex_freq_fuse_static_smem<2>();
        break;
    case 3:
        static_bytes = complex_freq_fuse_static_smem<3>();
        break;
    case 4:
        static_bytes = complex_freq_fuse_static_smem<4>();
        break;
    case 5:
        static_bytes = complex_freq_fuse_static_smem<5>();
        break;
    default:
        return false;
    }
    return complex_freq_fuse_dynamic_smem(k, nbins_f) + static_bytes <=
           cuda_utils::get_max_shared_memory_optin();
}

void ffa_complex_freq_fused_sweep_cuda(const ComplexTypeCUDA* fold_in,
                                       ComplexTypeCUDA* fold_out,
                                       const uint32_t* const* level_idx,
                                       const float* const* level_shift,
                                       const uint32_t* level_ncoords,
                                       uint32_t ncoords_in,
                                       uint32_t ncoords_out,
                                       uint32_t nsegments_in,
                                       uint32_t nbins_f,
                                       uint32_t nbins,
                                       int k,
                                       cudaStream_t stream) {
    switch (k) {
    case 2:
        launch_complex_freq_fuse<2>(fold_in, fold_out, level_idx, level_shift,
                                    level_ncoords, ncoords_in, ncoords_out,
                                    nsegments_in, nbins_f, nbins, stream);
        break;
    case 3:
        launch_complex_freq_fuse<3>(fold_in, fold_out, level_idx, level_shift,
                                    level_ncoords, ncoords_in, ncoords_out,
                                    nsegments_in, nbins_f, nbins, stream);
        break;
    case 4:
        launch_complex_freq_fuse<4>(fold_in, fold_out, level_idx, level_shift,
                                    level_ncoords, ncoords_in, ncoords_out,
                                    nsegments_in, nbins_f, nbins, stream);
        break;
    case 5:
        launch_complex_freq_fuse<5>(fold_in, fold_out, level_idx, level_shift,
                                    level_ncoords, ncoords_in, ncoords_out,
                                    nsegments_in, nbins_f, nbins, stream);
        break;
    default:
        throw std::invalid_argument(
            "ffa_complex_freq_fused_sweep_cuda: k must be 2 to 5");
    }
}

void brute_fold_phase_map_cuda(const double* freqs,
                               SizeType nfreqs,
                               SizeType segment_len,
                               SizeType nbins,
                               double tsamp,
                               double t_ref,
                               uint32_t* phase_map,
                               cudaStream_t stream) {
    const SizeType total = nfreqs * segment_len;
    if (total == 0) {
        return;
    }
    constexpr SizeType kThreads = 256;
    const dim3 grid((total + kThreads - 1) / kThreads);
    cuda_utils::check_kernel_launch_params(grid, dim3(kThreads));
    kernel_brute_fold_phase_map<<<grid, kThreads, 0, stream>>>(
        freqs, static_cast<uint32_t>(nfreqs),
        static_cast<uint32_t>(segment_len), static_cast<uint32_t>(nbins), tsamp,
        t_ref, phase_map);
    cuda_utils::check_last_cuda_error(
        "kernel_brute_fold_phase_map launch failed");
}

void brute_fold_segment_cuda(const float* ts_e,
                             const float* ts_v,
                             const uint32_t* phase_map,
                             float* fold,
                             SizeType nsegments,
                             SizeType nfreqs,
                             SizeType segment_len,
                             SizeType nbins,
                             cudaStream_t stream) {
    if (nsegments == 0 || nfreqs == 0) {
        return;
    }
    constexpr SizeType kSmemBudget = 32U * 1024U;
    const SizeType per_freq        = 2 * nbins * sizeof(float);
    const SizeType freqs_per_pass =
        std::clamp<SizeType>(kSmemBudget / per_freq, 1, nfreqs);
    const SizeType smem    = freqs_per_pass * per_freq;
    const SizeType threads = std::clamp<SizeType>(segment_len, 32, 256);
    const dim3 grid(static_cast<uint32_t>(nsegments));
    cuda_utils::check_kernel_launch_params(grid, dim3(threads), smem);
    kernel_brute_fold_segment<<<grid, static_cast<uint32_t>(threads), smem,
                                stream>>>(
        ts_e, ts_v, phase_map, fold, static_cast<uint32_t>(nfreqs),
        static_cast<uint32_t>(segment_len), static_cast<uint32_t>(nbins),
        static_cast<uint32_t>(freqs_per_pass));
    cuda_utils::check_last_cuda_error(
        "kernel_brute_fold_segment launch failed");
}

void ffa_complex_iter_sweep_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                 ComplexTypeCUDA* __restrict__ fold_out,
                                 coord::FFACoordDPtrs coords,
                                 SizeType ncoords_cur,
                                 SizeType ncoords_prev,
                                 SizeType nsegments,
                                 SizeType nbins_f,
                                 SizeType nbins,
                                 cudaStream_t stream) {
    launch_complex_iter_sweep(fold_in, fold_out, detail::AccelParentsC{coords},
                              ncoords_cur, ncoords_prev, nsegments, nbins_f,
                              nbins, stream);
}

void ffa_complex_iter_freq_sweep_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    ComplexTypeCUDA* __restrict__ fold_out,
    coord::FFACoordFreqDPtrs coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    cudaStream_t stream) {
    launch_complex_iter_sweep(fold_in, fold_out, detail::FreqParentsC{coords},
                              ncoords_cur, ncoords_prev, nsegments, nbins_f,
                              nbins, stream);
}

} // namespace loki::core
