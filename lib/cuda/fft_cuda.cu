#include "lib/cuda/fft_cuda.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_portable_fft.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::math {

namespace {

using core::detail::PortableFFT;

// Power-of-two lengths in [kDirectMin, kGpuFFTMaxN] are transformed
// directly; every other length by Bluestein's algorithm through an M-point
// transform, M = max(32, bit_ceil(2 n - 1)) (the layout of
// core::ffa_fourier_bluestein_table).
constexpr SizeType kDirectMin = 32;

bool needs_bluestein(SizeType n) {
    return n < kDirectMin || !std::has_single_bit(n);
}

SizeType transform_size(SizeType n) {
    return needs_bluestein(n)
               ? std::max<SizeType>(32, std::bit_ceil((2 * n) - 1))
               : n;
}

// Even rows of length n >= 4 are transformed as n / 2 complex values (real
// packing): half the transform length and no wasted imaginary part.
bool packed(SizeType n) { return n % 2 == 0 && n >= 4; }

/// Length of the complex transform of a real row of length @p n.
SizeType complex_length(SizeType n) { return packed(n) ? n / 2 : n; }

// M-point transforms: about 128 threads per block, as the fused scorers.
constexpr uint32_t batch_ept(uint32_t m) {
    return m <= 128 ? 4 : (m <= 1024 ? 8 : 16);
}
template <uint32_t M>
using BatchFFT =
    PortableFFT<M, batch_ept(M), std::max(1U, 128U / (M / batch_ept(M)))>;

__device__ __forceinline__ float2 cmulf(float2 a, float2 b) {
    return {fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x)};
}

/**
 * Inverse DFT of the full spectrum in @p a (element j = threadIdx.x + i *
 * stride holds X[j] for j < n, 0 beyond): a[i] becomes the unnormalized
 * x[j] = sum_k X[k] exp(2 pi i j k / n) for j < n. The inverse is
 * conj(FFT(conj(.))); Bluestein (kChirp) runs it as a circular convolution
 * with the chirp table (docs/fft.md).
 */
template <class FFT, bool kChirp>
__device__ __forceinline__ void inverse_dft(float2* a,
                                            float2* fft_smem,
                                            uint32_t n,
                                            const float2* __restrict__ chirp,
                                            const float2* __restrict__ kern) {
    constexpr uint32_t E = FFT::elements_per_thread;
#pragma unroll
    for (uint32_t i = 0; i < E; ++i) {
        const uint32_t j = (i * FFT::stride) + threadIdx.x;
        if constexpr (kChirp) {
            a[i] = j < n ? cmulf(a[i], chirp[j]) : float2{0.0F, 0.0F};
        } else {
            a[i] = {a[i].x, -a[i].y};
        }
    }
    FFT().execute(a, fft_smem);
    if constexpr (kChirp) {
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            const uint32_t j = (i * FFT::stride) + threadIdx.x;
            const float2 y   = cmulf(a[i], kern[j]);
            a[i]             = {y.x, -y.y};
        }
        FFT::sync();
        FFT().execute(a, fft_smem);
    }
#pragma unroll
    for (uint32_t i = 0; i < E; ++i) {
        const uint32_t j = (i * FFT::stride) + threadIdx.x;
        a[i]             = {a[i].x, -a[i].y};
        if constexpr (kChirp) {
            if (j < n) {
                a[i] = cmulf(a[i], chirp[j]);
            }
        }
    }
}

/// Forward DFT of the complex row in @p a (layout of inverse_dft):
/// conj(IDFT(conj(z))).
template <class FFT, bool kChirp>
__device__ __forceinline__ void forward_dft(float2* a,
                                            float2* fft_smem,
                                            uint32_t len,
                                            const float2* __restrict__ table) {
    if constexpr (kChirp) {
        constexpr uint32_t E = FFT::elements_per_thread;
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            a[i].y = -a[i].y;
        }
        inverse_dft<FFT, true>(a, fft_smem, len, table, table + len);
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            a[i].y = -a[i].y;
        }
    } else {
        FFT().execute(a, fft_smem);
    }
}

/// exp(-2 pi i k / n).
__device__ __forceinline__ float2 twiddle(uint32_t k, uint32_t n) {
    float2 w;
    sincospif(-2.0F * static_cast<float>(k) / static_cast<float>(n), &w.y,
              &w.x);
    return w;
}

/**
 * R2C of real rows: rows [blockIdx.x * F, + F), transform threadIdx.y; every
 * thread takes part in the transforms. kPacked (even n): the row is the
 * complex sequence z[m] = x[2m] + i x[2m+1] of length h = n / 2, and
 * X[k] = (Z[k] + conj Z[h-k]) / 2 + w^k (Z[k] - conj Z[h-k]) / (2i),
 * w = exp(-2 pi i / n). Otherwise the row is transformed as a complex
 * sequence of length n with zero imaginary parts.
 */
template <class FFT, bool kChirp, bool kPacked>
__global__ __launch_bounds__(FFT::max_threads_per_block) void kernel_rfft_rows(
    const float* __restrict__ in,
    float2* __restrict__ out,
    SizeType batch,
    uint32_t n,
    const float2* __restrict__ table) {
    constexpr uint32_t E = FFT::elements_per_thread;
    extern __shared__ __align__(16) unsigned char smem_raw[]; // NOLINT
    auto* fft_smem = reinterpret_cast<float2*>(smem_raw);
    const SizeType row =
        (SizeType{blockIdx.x} * FFT::ffts_per_block) + threadIdx.y;
    const bool active  = row < batch;
    const uint32_t nf  = (n / 2) + 1;
    const uint32_t len = kPacked ? n / 2 : n;
    const float* x     = in + (row * n);
    float2 a[E];
#pragma unroll
    for (uint32_t i = 0; i < E; ++i) {
        const uint32_t j = (i * FFT::stride) + threadIdx.x;
        float2 v{0.0F, 0.0F};
        if (active && j < len) {
            if constexpr (kPacked) {
                v = reinterpret_cast<const float2*>(x)[j];
            } else {
                v.x = x[j];
            }
        }
        a[i] = v;
    }
    forward_dft<FFT, kChirp>(a, fft_smem, len, table);
    float2* y = out + (row * nf);
    if constexpr (kPacked) {
        // Z[k] and Z[h - k] sit in different threads: through shared memory.
        float2* buf = fft_smem + (threadIdx.y * (FFT::stride * E));
        FFT::sync();
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            const uint32_t j = (i * FFT::stride) + threadIdx.x;
            if (j < len) {
                buf[j] = a[i];
            }
        }
        FFT::sync();
        if (active) {
#pragma unroll
            for (uint32_t i = 0; i < E; ++i) {
                const uint32_t k = (i * FFT::stride) + threadIdx.x;
                if (k < len) {
                    const float2 za = buf[k];
                    float2 zb       = buf[k == 0 ? 0 : len - k];
                    zb.y            = -zb.y;
                    const float2 e  = {0.5F * (za.x + zb.x),
                                       0.5F * (za.y + zb.y)};
                    const float2 o  = {0.5F * (za.y - zb.y),
                                       -0.5F * (za.x - zb.x)};
                    const float2 wo = cmulf(twiddle(k, n), o);
                    y[k]            = {e.x + wo.x, k == 0 ? 0.0F : e.y + wo.y};
                    if (k == 0) {
                        // Nyquist: X[h] = E[0] - O[0], both real.
                        y[len] = {e.x - o.x, 0.0F};
                    }
                }
            }
        }
    } else if (active) {
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            const uint32_t k = (i * FFT::stride) + threadIdx.x;
            if (k < nf) {
                const bool real_bin = k == 0 || 2 * k == n;
                y[k]                = {a[i].x, real_bin ? 0.0F : a[i].y};
            }
        }
    }
}

/**
 * C2R of half spectra, unnormalized times @p scale. kPacked (even n):
 * Z[k] = E + i O with E = X[k] + conj X[h-k] and O = (X[k] - conj X[h-k])
 * / w^k (twice the DFTs of the even and odd samples), then z = IDFT_h(Z)
 * gives x[2m] = Re z[m], x[2m+1] = Im z[m] times n, like an unnormalized
 * C2R. Otherwise the Hermitian extension of X is
 * inverted at length n. The imaginary parts of DC and Nyquist are dropped.
 */
template <class FFT, bool kChirp, bool kPacked>
__global__ __launch_bounds__(FFT::max_threads_per_block) void kernel_irfft_rows(
    const float2* __restrict__ in,
    float* __restrict__ out,
    SizeType batch,
    uint32_t n,
    const float2* __restrict__ table,
    float scale) {
    constexpr uint32_t E = FFT::elements_per_thread;
    extern __shared__ __align__(16) unsigned char smem_raw[]; // NOLINT
    auto* fft_smem = reinterpret_cast<float2*>(smem_raw);
    const SizeType row =
        (SizeType{blockIdx.x} * FFT::ffts_per_block) + threadIdx.y;
    const bool active  = row < batch;
    const uint32_t nf  = (n / 2) + 1;
    const uint32_t len = kPacked ? n / 2 : n;
    const float2* x    = in + (row * nf);
    // X[j] with the imaginary parts of DC and Nyquist dropped.
    const auto bin = [&](uint32_t j) {
        float2 v = x[j];
        v.y      = (j == 0 || 2 * j == n) ? 0.0F : v.y;
        return v;
    };
    float2 a[E];
#pragma unroll
    for (uint32_t i = 0; i < E; ++i) {
        const uint32_t j = (i * FFT::stride) + threadIdx.x;
        float2 v{0.0F, 0.0F};
        if (active && j < len) {
            if constexpr (kPacked) {
                const float2 xa = bin(j);
                float2 xb       = bin(len - j);
                xb.y            = -xb.y;
                const float2 e  = {xa.x + xb.x, xa.y + xb.y};
                float2 w        = twiddle(j, n);
                w.y             = -w.y; // 1 / w^j
                const float2 o  = cmulf({xa.x - xb.x, xa.y - xb.y}, w);
                v               = {e.x - o.y, e.y + o.x}; // e + i o
            } else if (j < nf) {
                v = bin(j);
            } else {
                v   = bin(n - j);
                v.y = -v.y;
            }
        }
        a[i] = v;
    }
    inverse_dft<FFT, kChirp>(a, fft_smem, len, table,
                             kChirp ? table + len : nullptr);
    if (active) {
        float* y = out + (row * n);
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            const uint32_t j = (i * FFT::stride) + threadIdx.x;
            if (j < len) {
                if constexpr (kPacked) {
                    reinterpret_cast<float2*>(y)[j] = {a[i].x * scale,
                                                       a[i].y * scale};
                } else {
                    y[j] = a[i].x * scale;
                }
            }
        }
    }
}

template <class FFT, class Kernel, class... Args>
void launch_rows(Kernel kernel,
                 SizeType batch,
                 cudaStream_t stream,
                 Args... args) {
    constexpr SizeType kSmem = FFT::shared_memory_size;
    const SizeType blocks =
        (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    error_check::check_less_equal(
        blocks, SizeType{std::numeric_limits<int32_t>::max()},
        "GpuFFTManager: batch too large for one launch");
    cuda_utils::set_kernel_dynamic_smem(kernel, kSmem);
    kernel<<<static_cast<unsigned int>(blocks), FFT::block_dim, kSmem,
             stream>>>(args...);
    cuda_utils::check_last_cuda_error("GpuFFTManager: kernel launch failed");
}

/// Calls @p f.template operator()<FFT, kChirp>() for the transform of
/// length @p n.
template <class F> void dispatch_size(SizeType n, F&& f) {
    const bool chirp = needs_bluestein(n);
    const SizeType m = transform_size(n);
#define LOKI_FFT_M(M)                                                          \
    case M:                                                                    \
        if (chirp) {                                                           \
            f.template operator()<BatchFFT<M>, true>();                        \
        } else {                                                               \
            if constexpr (M <= kGpuFFTMaxN) {                                  \
                f.template operator()<BatchFFT<M>, false>();                   \
            }                                                                  \
        }                                                                      \
        return
    switch (m) {
        LOKI_FFT_M(32);
        LOKI_FFT_M(64);
        LOKI_FFT_M(128);
        LOKI_FFT_M(256);
        LOKI_FFT_M(512);
        LOKI_FFT_M(1024);
        LOKI_FFT_M(2048);
        LOKI_FFT_M(4096);
    default:
        break;
    }
#undef LOKI_FFT_M
    throw std::invalid_argument(
        std::format("GpuFFTManager: unsupported length {}", n));
}

void check_extent(SizeType batch_size,
                  SizeType n_real,
                  SizeType in_size,
                  SizeType out_size,
                  SizeType in_row,
                  SizeType out_row) {
    error_check::check(gpu_fft_supported(n_real),
                       std::format("GpuFFTManager: length {} is not in [2, "
                                   "{}]",
                                   n_real, kGpuFFTMaxN));
    const bool fits =
        batch_size <= std::numeric_limits<SizeType>::max() / (2 * n_real);
    error_check::check(fits, "GpuFFTManager: batch_size overflows");
    error_check::check_greater_equal(in_size, batch_size * in_row,
                                     "GpuFFTManager: input too small");
    error_check::check_greater_equal(out_size, batch_size * out_row,
                                     "GpuFFTManager: output too small");
}

} // namespace

class GpuFFTManager::Impl {
public:
    explicit Impl(int device_id) : m_device_id(device_id) {}

    /// Device Bluestein table of the complex length @p n (nullptr for a
    /// direct one).
    const float2* table(SizeType n) {
        if (!needs_bluestein(n)) {
            return nullptr;
        }
        auto it = m_tables.find(n);
        if (it == m_tables.end()) {
            const auto host = core::ffa_fourier_bluestein_table(n);
            thrust::device_vector<ComplexTypeCUDA> dev(host.begin(),
                                                       host.end());
            it = m_tables.emplace(n, std::move(dev)).first;
        }
        return reinterpret_cast<const float2*>(
            thrust::raw_pointer_cast(it->second.data()));
    }

    void prepare(SizeType n) {
        error_check::check(gpu_fft_supported(n),
                           std::format("GpuFFTManager: length {} is not in "
                                       "[2, {}]",
                                       n, kGpuFFTMaxN));
        (void)table(complex_length(n));
        m_prepared.emplace(n, true);
    }

    int m_device_id;
    std::unordered_map<SizeType, thrust::device_vector<ComplexTypeCUDA>>
        m_tables;
    std::unordered_map<SizeType, bool> m_prepared;
};

GpuFFTManager::GpuFFTManager(int device_id)
    : m_impl(std::make_unique<Impl>(device_id)) {}
GpuFFTManager::~GpuFFTManager()                                   = default;
GpuFFTManager::GpuFFTManager(GpuFFTManager&&) noexcept            = default;
GpuFFTManager& GpuFFTManager::operator=(GpuFFTManager&&) noexcept = default;

void GpuFFTManager::prepare_plans(std::span<const SizeType> n_reals) {
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    for (const SizeType n : n_reals) {
        m_impl->prepare(n);
    }
}

void GpuFFTManager::rfft_batch(cuda::std::span<const float> real_input,
                               cuda::std::span<ComplexTypeCUDA> complex_output,
                               SizeType batch_size,
                               SizeType n_real,
                               cudaStream_t stream) {
    if (batch_size == 0) {
        return;
    }
    check_extent(batch_size, n_real, real_input.size(), complex_output.size(),
                 n_real, (n_real / 2) + 1);
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    const SizeType len  = complex_length(n_real);
    const float2* table = m_impl->table(len);
    auto* out           = reinterpret_cast<float2*>(complex_output.data());
    const auto run      = [&]<bool kPacked>() {
        dispatch_size(len, [&]<class FFT, bool kChirp>() {
            launch_rows<FFT>(kernel_rfft_rows<FFT, kChirp, kPacked>, batch_size,
                             stream, real_input.data(), out, batch_size,
                             static_cast<uint32_t>(n_real), table);
        });
    };
    if (packed(n_real)) {
        run.template operator()<true>();
    } else {
        run.template operator()<false>();
    }
}

void GpuFFTManager::irfft_batch(
    cuda::std::span<const ComplexTypeCUDA> complex_input,
    cuda::std::span<float> real_output,
    SizeType batch_size,
    SizeType n_real,
    cudaStream_t stream,
    bool normalize) {
    if (batch_size == 0) {
        return;
    }
    check_extent(batch_size, n_real, complex_input.size(), real_output.size(),
                 (n_real / 2) + 1, n_real);
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    const SizeType len  = complex_length(n_real);
    const float2* table = m_impl->table(len);
    const auto* in      = reinterpret_cast<const float2*>(complex_input.data());
    const float scale   = normalize ? 1.0F / static_cast<float>(n_real) : 1.0F;
    const auto run      = [&]<bool kPacked>() {
        dispatch_size(len, [&]<class FFT, bool kChirp>() {
            launch_rows<FFT>(kernel_irfft_rows<FFT, kChirp, kPacked>,
                             batch_size, stream, in, real_output.data(),
                             batch_size, static_cast<uint32_t>(n_real), table,
                             scale);
        });
    };
    if (packed(n_real)) {
        run.template operator()<true>();
    } else {
        run.template operator()<false>();
    }
}

bool GpuFFTManager::has_prepared(SizeType n_real) const noexcept {
    return m_impl->m_prepared.contains(n_real);
}

SizeType GpuFFTManager::n_cached_plans() const noexcept {
    return m_impl->m_prepared.size();
}

SizeType GpuFFTManager::table_bytes() const noexcept {
    SizeType bytes = 0;
    for (const auto& [n, t] : m_impl->m_tables) {
        bytes += t.size() * sizeof(ComplexTypeCUDA);
    }
    return bytes;
}

void rfft_batch_cuda(cuda::std::span<const float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream,
                     int device_id) {
    GpuFFTManager manager(device_id);
    manager.rfft_batch(real_input, complex_output, batch_size, n_real, stream);
    // The temporary table must outlive the transform.
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "rfft_batch_cuda: synchronize failed");
}

void irfft_batch_cuda(cuda::std::span<const ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream,
                      int device_id) {
    GpuFFTManager manager(device_id);
    manager.irfft_batch(complex_input, real_output, batch_size, n_real, stream);
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "irfft_batch_cuda: synchronize failed");
}

namespace detail {

std::unique_ptr<DeviceFFTPlans> make_fft_manager_gpu(int device_id) {
    return std::make_unique<GpuFFTPlans>(device_id);
}

} // namespace detail

} // namespace loki::math
