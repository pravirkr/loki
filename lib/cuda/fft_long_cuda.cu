#include "lib/cuda/fft_long_cuda.cuh"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <cuda_runtime.h>
#include <cufft.h>
#include <fmt/format.h>

#include "lib/cuda/cuda_utils.cuh"

namespace loki::math {

namespace {

constexpr std::string_view cufft_error_string(cufftResult error) noexcept {
    switch (error) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN: invalid plan handle";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU memory";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE: unsupported transform type";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE: invalid pointer or parameter";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR: driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED: failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED: the cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE: invalid transform size";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE: plan used on another GPU";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE: no work area set before execution";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED: feature not implemented";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED: operation not supported for these "
               "parameters";
    default:
        return "unknown cuFFT error";
    }
}

void check_cufft(cufftResult result, std::string_view what) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(fmt::format("LongRealFFT: {} failed: {} ({})",
                                             what, cufft_error_string(result),
                                             static_cast<int>(result)));
    }
}

/// A cuFFT handle with manual work-area allocation.
class Plan {
public:
    Plan() { check_cufft(cufftCreate(&m_handle), "cufftCreate"); }
    ~Plan() {
        if (m_valid) {
            cufftDestroy(m_handle);
        }
    }
    Plan(Plan&& o) noexcept
        : m_handle(o.m_handle),
          m_valid(std::exchange(o.m_valid, false)) {}
    Plan& operator=(Plan&&)      = delete;
    Plan(const Plan&)            = delete;
    Plan& operator=(const Plan&) = delete;

    [[nodiscard]] cufftHandle get() const noexcept { return m_handle; }

    /// Makes a single 1-D plan of @p type for length @p n; returns its work
    /// area in bytes.
    SizeType make(SizeType n, cufftType type) {
        check_cufft(cufftSetAutoAllocation(m_handle, 0),
                    "cufftSetAutoAllocation");
        auto len         = static_cast<long long>(n);
        std::size_t work = 0;
        check_cufft(cufftMakePlanMany64(m_handle, 1, &len, nullptr, 1, 0,
                                        nullptr, 1, 0, type, 1, &work),
                    fmt::format("cufftMakePlanMany64 (n = {})", n));
        return static_cast<SizeType>(work);
    }

    /// Work area a plan of @p type for length @p n would need, without
    /// making it.
    SizeType work_size(SizeType n, cufftType type) {
        auto len         = static_cast<long long>(n);
        std::size_t work = 0;
        check_cufft(cufftGetSizeMany64(m_handle, 1, &len, nullptr, 1, 0,
                                       nullptr, 1, 0, type, 1, &work),
                    fmt::format("cufftGetSizeMany64 (n = {})", n));
        return static_cast<SizeType>(work);
    }

private:
    cufftHandle m_handle{};
    bool m_valid{true};
};

__global__ void kernel_scale(float* __restrict__ x, SizeType n, float s) {
    const SizeType stride = static_cast<SizeType>(gridDim.x) * blockDim.x;
    for (SizeType i =
             (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        x[i] = __fmul_rn(x[i], s);
    }
}

} // namespace

class LongRealFFT::Impl {
public:
    Impl(SizeType n, int device_id, cudaStream_t stream)
        : m_n(n),
          m_device(device_id),
          m_stream(stream) {
        if (n < 2) {
            throw std::invalid_argument(
                fmt::format("LongRealFFT: length must be >= 2 (got {})", n));
        }
        // The handles, plans and work area all live on m_device; a setup
        // that throws releases what it made before the guard ends.
        const cuda_utils::CudaSetDeviceGuard guard(m_device);
        try {
            m_r2c.emplace();
            m_c2r.emplace();
            const SizeType w_r2c = m_r2c->make(n, CUFFT_R2C);
            const SizeType w_c2r = m_c2r->make(n, CUFFT_C2R);
            const SizeType work  = std::max(w_r2c, w_c2r);
            if (work > 0) {
                cuda_utils::check_cuda_call(
                    cudaMalloc(&m_work, work),
                    "LongRealFFT: work area allocation");
            }
            check_cufft(cufftSetWorkArea(m_r2c->get(), m_work),
                        "cufftSetWorkArea");
            check_cufft(cufftSetWorkArea(m_c2r->get(), m_work),
                        "cufftSetWorkArea");
            check_cufft(cufftSetStream(m_r2c->get(), m_stream),
                        "cufftSetStream");
            check_cufft(cufftSetStream(m_c2r->get(), m_stream),
                        "cufftSetStream");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() {
        const cuda_utils::CudaSetDeviceGuard guard(m_device);
        release();
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    [[nodiscard]] SizeType size() const noexcept { return m_n; }

    void rfft(const float* x_d, ComplexTypeCUDA* spec_d) {
        const cuda_utils::CudaSetDeviceGuard guard(m_device);
        // Out of place, R2C leaves its input unchanged.
        check_cufft(cufftExecR2C(m_r2c->get(),
                                 const_cast<cufftReal*>( // NOLINT
                                     reinterpret_cast<const cufftReal*>(x_d)),
                                 reinterpret_cast<cufftComplex*>(spec_d)),
                    "cufftExecR2C");
    }

    void irfft(ComplexTypeCUDA* spec_d, float* x_d) {
        const cuda_utils::CudaSetDeviceGuard guard(m_device);
        check_cufft(cufftExecC2R(m_c2r->get(),
                                 reinterpret_cast<cufftComplex*>(spec_d),
                                 reinterpret_cast<cufftReal*>(x_d)),
                    "cufftExecC2R");
        const float s = 1.0F / static_cast<float>(m_n);
        int dev       = 0;
        int sms       = 1;
        cuda_utils::check_cuda_call(cudaGetDevice(&dev));
        cuda_utils::check_cuda_call(
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        const SizeType blocks = std::clamp<SizeType>(
            (m_n + 255) / 256, 1, static_cast<SizeType>(32 * sms));
        kernel_scale<<<static_cast<unsigned int>(blocks), 256, 0, m_stream>>>(
            x_d, m_n, s);
        cuda_utils::check_last_cuda_error("LongRealFFT: scale");
    }

private:
    /// Destroys the plans, then the work area they use (on m_device).
    void release() noexcept {
        m_c2r.reset();
        m_r2c.reset();
        if (m_work != nullptr) {
            cudaFree(m_work);
            m_work = nullptr;
        }
    }

    SizeType m_n;
    int m_device;
    cudaStream_t m_stream;
    std::optional<Plan> m_r2c;
    std::optional<Plan> m_c2r;
    void* m_work{nullptr};
};

LongRealFFT::LongRealFFT(SizeType n, int device_id, cudaStream_t stream)
    : m_impl(std::make_unique<Impl>(n, device_id, stream)) {}

LongRealFFT::~LongRealFFT()                                 = default;
LongRealFFT::LongRealFFT(LongRealFFT&&) noexcept            = default;
LongRealFFT& LongRealFFT::operator=(LongRealFFT&&) noexcept = default;

SizeType LongRealFFT::workspace_bytes(SizeType n) {
    if (n < 2) {
        return 0;
    }
    Plan plan;
    return std::max(plan.work_size(n, CUFFT_R2C), plan.work_size(n, CUFFT_C2R));
}

SizeType LongRealFFT::size() const noexcept { return m_impl->size(); }

void LongRealFFT::rfft(const float* x_d, ComplexTypeCUDA* spec_d) {
    m_impl->rfft(x_d, spec_d);
}

void LongRealFFT::irfft(ComplexTypeCUDA* spec_d, float* x_d) {
    m_impl->irfft(spec_d, x_d);
}

} // namespace loki::math
