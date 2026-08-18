#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>

#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::math {

namespace {

constexpr SizeType kCUFFTWorkAreaBudgetBytes =
    256ULL * 1024ULL * 1024ULL; // 256 MB

// Normalization kernel for IRFFT output
__global__ void
normalize_kernel(float* __restrict__ data, int n_elements, float norm) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n_elements) {
        data[idx] *= norm;
    }
}

void check_batch_extent(SizeType batch_size,
                        SizeType length,
                        std::string_view what) {
    const bool fits =
        length == 0 ||
        batch_size <= (std::numeric_limits<SizeType>::max() / length);
    error_check::check(fits, std::format("CUFFTManager: {} * batch_size "
                                         "overflows SizeType",
                                         what));
}

int to_cufft_int(SizeType value, std::string_view what) {
    error_check::check_less_equal(
        value, static_cast<SizeType>(std::numeric_limits<int>::max()),
        std::format("CUFFTManager: {} exceeds cuFFT int limit", what));
    return static_cast<int>(value);
}

struct CUFFTPlanKey {
    int n_real{};
    int batch{};
    cufftType type{CUFFT_R2C};

    bool operator==(const CUFFTPlanKey& other) const {
        return n_real == other.n_real && batch == other.batch &&
               type == other.type;
    }
};

struct CUFFTPlanKeyHash {
    SizeType operator()(const CUFFTPlanKey& key) const {
        SizeType hash = std::hash<int>{}(key.n_real);
        hash ^= std::hash<int>{}(key.batch) + 0x9e3779b9U + (hash << 6U) +
                (hash >> 2U);
        hash ^= std::hash<int>{}(static_cast<int>(key.type)) + 0x9e3779b9U +
                (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

class CUFFTPlan {
public:
    CUFFTPlan() = default;
    CUFFTPlan(cufftHandle handle, SizeType work_size)
        : m_handle(handle),
          m_work_size(work_size) {}
    ~CUFFTPlan() {
        if (m_handle != 0) {
            cufftDestroy(m_handle);
        }
    }
    CUFFTPlan(const CUFFTPlan&)            = delete;
    CUFFTPlan& operator=(const CUFFTPlan&) = delete;
    CUFFTPlan(CUFFTPlan&& other) noexcept
        : m_handle(std::exchange(other.m_handle, 0)),
          m_work_size(std::exchange(other.m_work_size, 0)) {}
    CUFFTPlan& operator=(CUFFTPlan&& other) noexcept {
        if (this != &other) {
            if (m_handle != 0) {
                cufftDestroy(m_handle);
            }
            m_handle    = std::exchange(other.m_handle, 0);
            m_work_size = std::exchange(other.m_work_size, 0);
        }
        return *this;
    }

    [[nodiscard]] cufftHandle get() const noexcept { return m_handle; }
    [[nodiscard]] SizeType work_size() const noexcept { return m_work_size; }
    void set_work_size(SizeType work_size) noexcept { m_work_size = work_size; }

private:
    cufftHandle m_handle{0};
    SizeType m_work_size{0};
};

struct DeviceDeleter {
    void operator()(std::byte* ptr) const noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

bool estimate_work_size(int n_real,
                        int batch_size,
                        cufftType fft_type,
                        SizeType& work_size) {
    const int n_complex        = (n_real / 2) + 1;
    const int in_dist          = fft_type == CUFFT_R2C ? n_real : n_complex;
    const int out_dist         = fft_type == CUFFT_R2C ? n_complex : n_real;
    std::array<int, 1> n_arr   = {n_real};
    std::array<int, 1> inembed = {in_dist};
    std::array<int, 1> onembed = {out_dist};
    work_size                  = 0;
    const cufftResult status =
        cufftEstimateMany(1, n_arr, inembed, 1, in_dist, onembed, 1, out_dist,
                          fft_type, batch_size, &work_size);
    return status == CUFFT_SUCCESS;
}

CUFFTPlan make_cufft_plan(int n_real, int batch_size, cufftType fft_type) {
    const int n_complex        = (n_real / 2) + 1;
    const int in_dist          = fft_type == CUFFT_R2C ? n_real : n_complex;
    const int out_dist         = fft_type == CUFFT_R2C ? n_complex : n_real;
    std::array<int, 1> n_arr   = {n_real};
    std::array<int, 1> inembed = {in_dist};
    std::array<int, 1> onembed = {out_dist};

    cufftHandle handle = 0;
    cuda_utils::check_cufft_call(cufftCreate(&handle),
                                 "CUFFTManager: cufftCreate failed");
    CUFFTPlan owned(handle, 0);
    cuda_utils::check_cufft_call(cufftSetAutoAllocation(owned.get(), 0),
                                 "CUFFTManager: cufftSetAutoAllocation failed");
    SizeType work_size = 0;
    cuda_utils::check_cufft_call(
        cufftMakePlanMany(owned.get(),    // plan handle
                          1,              // rank
                          n_arr.data(),   // n
                          inembed.data(), // inembed
                          1,              // istride
                          in_dist,        // idist
                          onembed.data(), // onembed
                          1,              // ostride
                          out_dist,       // odist
                          fft_type,       // type
                          batch_size,     // batch size
                          &work_size),    // Get workspace size
        std::format("CUFFTManager: cufftMakePlanMany failed for n_real={}, "
                    "batch={}, type={}",
                    n_real, batch_size, static_cast<int>(fft_type)));
    owned.set_work_size(work_size);
    return owned;
}

} // namespace

class CUFFTManager::Impl {
public:
    explicit Impl(int device_id) : m_device_id(device_id) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
    }

    ~Impl() {
        try {
            cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
            m_plans.clear();
            m_work_area.reset();
            m_work_area_bytes = 0;
        } catch (...) {
            m_plans.clear();
            m_work_area.reset();
            m_work_area_bytes = 0;
        }
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    int m_device_id{0};
    std::unordered_set<SizeType> m_prepared;
    std::unordered_set<SizeType> m_exact;
    std::unordered_map<SizeType, SizeType> m_max_batch_by_n;
    std::unordered_map<CUFFTPlanKey, CUFFTPlan, CUFFTPlanKeyHash> m_plans;
    std::unique_ptr<std::byte, DeviceDeleter> m_work_area;
    SizeType m_work_area_bytes{0};

    [[nodiscard]] SizeType work_area_budget_bytes() const {
        const auto [free_mem_gb, total_mem_gb] =
            cuda_utils::get_cuda_memory_usage();
        const auto tenth_portion = static_cast<SizeType>(
            free_mem_gb * static_cast<double>(1ULL << 30U) / 10U);
        if (tenth_portion == 0) {
            return kCUFFTWorkAreaBudgetBytes;
        }
        return std::min(kCUFFTWorkAreaBudgetBytes, tenth_portion);
    }

    [[nodiscard]] SizeType clamp_max_batch(SizeType n_real,
                                           SizeType requested) const {
        SizeType batch = std::min(
            requested, static_cast<SizeType>(std::numeric_limits<int>::max()));
        error_check::check_greater(batch, SizeType{0},
                                   "CUFFTManager: max_batch must be positive");
        const SizeType budget   = work_area_budget_bytes();
        const int n_real_i      = to_cufft_int(n_real, "n_real");
        SizeType accepted_batch = 1;
        while (batch >= 1) {
            const int batch_i  = static_cast<int>(batch);
            SizeType r2c_bytes = 0;
            SizeType c2r_bytes = 0;
            const bool r2c_ok =
                estimate_work_size(n_real_i, batch_i, CUFFT_R2C, r2c_bytes);
            const bool c2r_ok =
                estimate_work_size(n_real_i, batch_i, CUFFT_C2R, c2r_bytes);
            if (r2c_ok && c2r_ok && std::max(r2c_bytes, c2r_bytes) <= budget) {
                accepted_batch = batch;
                break;
            }
            if (batch == 1) {
                break;
            }
            batch /= 2;
        }
        return accepted_batch;
    }

    void ensure_work_area(SizeType bytes) {
        if (bytes <= m_work_area_bytes) {
            return;
        }
        if (bytes == 0) {
            return;
        }
        // In-flight FFTs may still be using the old work area. The manager
        // forbids overlapping transforms, so a device sync is enough before
        // rebinding plans and freeing the previous buffer.
        cuda_utils::check_cuda_call(
            cudaDeviceSynchronize(),
            "CUFFTManager: cudaDeviceSynchronize failed before growing "
            "work area");
        void* new_ptr = nullptr;
        cuda_utils::check_cuda_call(
            cudaMalloc(&new_ptr, bytes),
            "CUFFTManager: cudaMalloc work area failed");
        std::unique_ptr<std::byte, DeviceDeleter> new_area(
            static_cast<std::byte*>(new_ptr));
        for (auto& plan_entry : m_plans) {
            if (plan_entry.second.work_size() > 0) {
                cuda_utils::check_cufft_call(
                    cufftSetWorkArea(plan_entry.second.get(), new_ptr),
                    "CUFFTManager: cufftSetWorkArea failed while growing "
                    "workspace");
            }
        }
        m_work_area       = std::move(new_area);
        m_work_area_bytes = bytes;
    }

    CUFFTPlan& get_or_create_plan(int n_real, int batch, cufftType type) {
        const CUFFTPlanKey key{.n_real = n_real, .batch = batch, .type = type};
        if (auto it = m_plans.find(key); it != m_plans.end()) {
            return it->second;
        }
        CUFFTPlan plan = make_cufft_plan(n_real, batch, type);
        ensure_work_area(plan.work_size());
        if (plan.work_size() > 0) {
            cuda_utils::check_cufft_call(
                cufftSetWorkArea(plan.get(), m_work_area.get()),
                "CUFFTManager: cufftSetWorkArea failed");
        }
        const auto [it, inserted] = m_plans.emplace(key, std::move(plan));
        if (!inserted) {
            throw error_check::DetailedException(
                "CUFFTManager: plan cache insert collided");
        }
        spdlog::debug("CUFFTManager: cached plan n_real={} batch={} type={} "
                      "(work={} bytes, total plans={})",
                      n_real, batch, static_cast<int>(type),
                      it->second.work_size(), m_plans.size());
        return it->second;
    }

    [[nodiscard]] SizeType max_batch_for(SizeType n_real) {
        if (const auto it = m_max_batch_by_n.find(n_real);
            it != m_max_batch_by_n.end()) {
            return it->second;
        }
        const SizeType clamped =
            clamp_max_batch(n_real, static_cast<SizeType>(kCUFFTBatchSizeMax));
        m_max_batch_by_n.emplace(n_real, clamped);
        return clamped;
    }

    void execute_rfft(float* __restrict__ real_ptr,
                      cufftComplex* __restrict__ complex_ptr,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream) {
        const SizeType n_complex = (n_real / 2) + 1;
        const SizeType max_batch = max_batch_for(n_real);
        const int n_real_i       = to_cufft_int(n_real, "n_real");
        SizeType offset          = 0;
        while (offset < batch_size) {
            const SizeType remaining   = batch_size - offset;
            const SizeType chunk_batch = std::min(max_batch, remaining);
            const int chunk_i = to_cufft_int(chunk_batch, "chunk_batch");
            CUFFTPlan& plan = get_or_create_plan(n_real_i, chunk_i, CUFFT_R2C);
            cuda_utils::check_cufft_call(cufftSetStream(plan.get(), stream),
                                         "CUFFTManager: cufftSetStream failed");
            float* chunk_real           = real_ptr + (offset * n_real);
            cufftComplex* chunk_complex = complex_ptr + (offset * n_complex);
            cuda_utils::check_cufft_call(
                cufftExecR2C(plan.get(), chunk_real, chunk_complex),
                "CUFFTManager: cufftExecR2C failed");
            offset += chunk_batch;
        }
    }

    void execute_irfft(cufftComplex* __restrict__ complex_ptr,
                       float* __restrict__ real_ptr,
                       SizeType batch_size,
                       SizeType n_real,
                       cudaStream_t stream) {
        const SizeType n_complex = (n_real / 2) + 1;
        const SizeType max_batch = max_batch_for(n_real);
        const int n_real_i       = to_cufft_int(n_real, "n_real");
        const float norm         = 1.0F / static_cast<float>(n_real);
        SizeType offset          = 0;

        while (offset < batch_size) {
            const SizeType remaining   = batch_size - offset;
            const SizeType chunk_batch = std::min(max_batch, remaining);
            const int chunk_i = to_cufft_int(chunk_batch, "chunk_batch");

            CUFFTPlan& plan = get_or_create_plan(n_real_i, chunk_i, CUFFT_C2R);
            cuda_utils::check_cufft_call(cufftSetStream(plan.get(), stream),
                                         "CUFFTManager: cufftSetStream failed");

            cufftComplex* chunk_complex = complex_ptr + (offset * n_complex);
            float* chunk_real           = real_ptr + (offset * n_real);
            cuda_utils::check_cufft_call(
                cufftExecC2R(plan.get(), chunk_complex, chunk_real),
                "CUFFTManager: cufftExecC2R failed");
            offset += chunk_batch;
        }

        // Launch a single normalization pass over the entire batch
        const SizeType total_elements = batch_size * n_real;
        error_check::check_less_equal(
            total_elements,
            static_cast<SizeType>(std::numeric_limits<int>::max()),
            "CUFFTManager: normalize element count exceeds int");
        constexpr int kThreadsPerBlock = 256;
        const auto blocks              = static_cast<int>(
            (total_elements + static_cast<SizeType>(kThreadsPerBlock) - 1) /
            static_cast<SizeType>(kThreadsPerBlock));
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        normalize_kernel<<<grid_dim, block_dim, 0, stream>>>(
            real_ptr, static_cast<int>(total_elements), norm);
        cuda_utils::check_last_cuda_error(
            "CUFFTManager: normalize_kernel launch failed");
    }
};

CUFFTManager::CUFFTManager(int device_id)
    : m_impl(std::make_unique<Impl>(device_id)) {}
CUFFTManager::~CUFFTManager()                                  = default;
CUFFTManager::CUFFTManager(CUFFTManager&&) noexcept            = default;
CUFFTManager& CUFFTManager::operator=(CUFFTManager&&) noexcept = default;

void CUFFTManager::prepare_plans(std::span<const SizeType> n_reals,
                                 SizeType max_batch) {
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    error_check::check_greater(max_batch, SizeType{0},
                               "CUFFTManager::prepare_plans: max_batch "
                               "must be positive");
    error_check::check_less_equal(
        max_batch, static_cast<SizeType>(std::numeric_limits<int>::max()),
        "CUFFTManager::prepare_plans: max_batch exceeds cuFFT int limit");

    std::vector<SizeType> unique_n(n_reals.begin(), n_reals.end());
    std::ranges::sort(unique_n);
    unique_n.erase(std::ranges::unique(unique_n).begin(), unique_n.end());

    SizeType n_new = 0;
    for (const auto n_real : unique_n) {
        if (n_real == 0) {
            continue;
        }
        to_cufft_int(n_real, "n_real");
        if (m_impl->m_exact.contains(n_real)) {
            throw error_check::DetailedException(std::format(
                "CUFFTManager: n_real={} already prepared with exact-batch "
                "cache",
                n_real));
        }
        const SizeType clamped = m_impl->clamp_max_batch(n_real, max_batch);
        const auto existing    = m_impl->m_max_batch_by_n.find(n_real);
        if (existing != m_impl->m_max_batch_by_n.end() &&
            m_impl->m_prepared.contains(n_real)) {
            if (clamped < existing->second) {
                throw error_check::DetailedException(std::format(
                    "CUFFTManager::prepare_plans: cannot shrink max_batch for "
                    "n_real={} from {} to {}",
                    n_real, existing->second, clamped));
            }
            if (clamped == existing->second) {
                continue;
            }
            const int n_real_i    = static_cast<int>(n_real);
            const int old_batch_i = static_cast<int>(existing->second);
            m_impl->m_plans.erase(CUFFTPlanKey{
                .n_real = n_real_i, .batch = old_batch_i, .type = CUFFT_R2C});
            m_impl->m_plans.erase(CUFFTPlanKey{
                .n_real = n_real_i, .batch = old_batch_i, .type = CUFFT_C2R});
        }

        const int n_real_i    = static_cast<int>(n_real);
        const int batch_i     = static_cast<int>(clamped);
        const SizeType before = m_impl->m_plans.size();
        m_impl->get_or_create_plan(n_real_i, batch_i, CUFFT_R2C);
        m_impl->get_or_create_plan(n_real_i, batch_i, CUFFT_C2R);
        n_new += m_impl->m_plans.size() - before;
        m_impl->m_max_batch_by_n[n_real] = clamped;
        m_impl->m_prepared.insert(n_real);
    }
    spdlog::info("CUFFTManager: cached {} plans for {} n_real values "
                 "(work area {} bytes)",
                 n_cached_plans(), m_impl->m_prepared.size(),
                 m_impl->m_work_area_bytes);
    spdlog::debug("CUFFTManager: created {} new plans this call", n_new);
}

void CUFFTManager::prepare_exact_plans(std::span<const SizeType> n_reals,
                                       SizeType max_batch) {
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    error_check::check_greater(max_batch, SizeType{0},
                               "CUFFTManager::prepare_exact_plans: max_batch "
                               "must be positive");
    error_check::check_less_equal(
        max_batch, static_cast<SizeType>(kCUFFTBatchSizeMax),
        "CUFFTManager::prepare_exact_plans: max_batch exceeds "
        "kCUFFTBatchSizeMax");

    std::vector<SizeType> unique_n(n_reals.begin(), n_reals.end());
    std::ranges::sort(unique_n);
    unique_n.erase(std::ranges::unique(unique_n).begin(), unique_n.end());

    SizeType n_registered = 0;
    for (const auto n_real : unique_n) {
        if (n_real == 0) {
            continue;
        }
        to_cufft_int(n_real, "n_real");
        if (m_impl->m_prepared.contains(n_real)) {
            throw error_check::DetailedException(std::format(
                "CUFFTManager: n_real={} already prepared with max-batch "
                "cache",
                n_real));
        }
        const SizeType clamped = m_impl->clamp_max_batch(n_real, max_batch);
        if (m_impl->m_exact.contains(n_real)) {
            const auto existing = m_impl->m_max_batch_by_n.find(n_real);
            if (existing != m_impl->m_max_batch_by_n.end() &&
                clamped < existing->second) {
                throw error_check::DetailedException(std::format(
                    "CUFFTManager::prepare_exact_plans: cannot shrink "
                    "max_batch for n_real={} from {} to {}",
                    n_real, existing->second, clamped));
            }
            if (existing != m_impl->m_max_batch_by_n.end() &&
                clamped == existing->second) {
                continue;
            }
            m_impl->m_max_batch_by_n[n_real] = clamped;
            ++n_registered;
            continue;
        }
        m_impl->m_max_batch_by_n[n_real] = clamped;
        m_impl->m_exact.insert(n_real);
        ++n_registered;
    }
    spdlog::info(
        "CUFFTManager: registered {} n_real values for exact-batch caching",
        n_registered);
}

void CUFFTManager::rfft_batch(cuda::std::span<float> real_input,
                              cuda::std::span<ComplexTypeCUDA> complex_output,
                              SizeType batch_size,
                              SizeType n_real,
                              cudaStream_t stream) {
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    error_check::check_greater(n_real, SizeType{0},
                               "CUFFTManager::rfft_batch: n_real must be "
                               "positive");
    if (batch_size == 0) {
        return;
    }
    to_cufft_int(n_real, "n_real");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch_extent(batch_size, n_real, "n_real");
    check_batch_extent(batch_size, n_complex, "n_complex");
    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "CUFFTManager::rfft_batch: real_input size does not match batch size");
    error_check::check_equal(
        complex_output.size(), batch_size * n_complex,
        "CUFFTManager::rfft_batch: complex_output size does not match batch "
        "size");
    if (m_impl->m_exact.contains(n_real)) {
        throw error_check::DetailedException(std::format(
            "CUFFTManager::rfft_batch: n_real={} uses exact-batch cache; "
            "use prepare_plans for RFFT",
            n_real));
    }

    auto* real_ptr    = real_input.data();
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_output.data());
    m_impl->execute_rfft(real_ptr, complex_ptr, batch_size, n_real, stream);
}

void CUFFTManager::irfft_batch(cuda::std::span<ComplexTypeCUDA> complex_input,
                               cuda::std::span<float> real_output,
                               SizeType batch_size,
                               SizeType n_real,
                               cudaStream_t stream) {
    cuda_utils::CudaSetDeviceGuard device_guard(m_impl->m_device_id);
    error_check::check_greater(n_real, SizeType{0},
                               "CUFFTManager::irfft_batch: n_real must be "
                               "positive");
    if (batch_size == 0) {
        return;
    }
    to_cufft_int(n_real, "n_real");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch_extent(batch_size, n_real, "n_real");
    check_batch_extent(batch_size, n_complex, "n_complex");
    error_check::check_equal(
        real_output.size(), batch_size * n_real,
        "CUFFTManager::irfft_batch: real_output size does not match batch "
        "size");
    error_check::check_equal(
        complex_input.size(), batch_size * n_complex,
        "CUFFTManager::irfft_batch: complex_input size does not match batch "
        "size");

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_input.data());
    auto* real_ptr    = real_output.data();
    m_impl->execute_irfft(complex_ptr, real_ptr, batch_size, n_real, stream);
}

bool CUFFTManager::has_prepared(SizeType n_real) const noexcept {
    return m_impl->m_prepared.contains(n_real) ||
           m_impl->m_exact.contains(n_real);
}

SizeType CUFFTManager::n_cached_plans() const noexcept {
    return m_impl->m_plans.size();
}

SizeType CUFFTManager::work_area_bytes() const noexcept {
    return m_impl->m_work_area_bytes;
}

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream,
                     int device_id) {
    CUFFTManager manager(device_id);
    manager.rfft_batch(real_input, complex_output, batch_size, n_real, stream);
}

void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream,
                      int device_id) {
    CUFFTManager manager(device_id);
    manager.irfft_batch(complex_input, real_output, batch_size, n_real, stream);
}

/*
// cuFFTDx descriptor
template <uint32_t N>
using C2R_FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>() +
                         cufftdx::Type<cufftdx::fft_type::c2r>() +
                         cufftdx::SM<CUFFTDX_SM>() + cufftdx::Block());



template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void irfft_c2r_kernel(const ComplexTypeCUDA* __restrict__ complex_input,
                          float* __restrict__ real_output,
                          const uint32_t* __restrict__ batch_counter,
                          uint32_t max_batch) {
    constexpr uint32_t N       = cufftdx::size_of<FFT>::value;
    constexpr uint32_t in_len  = FFT::input_length;  // N/2+1
    constexpr uint32_t out_len = FFT::output_length; // N
    constexpr uint32_t stride  = FFT::stride;

    const uint32_t local_fft  = threadIdx.y;
    const uint32_t global_fft = blockIdx.x * FFT::ffts_per_block + local_fft;
    // nfft = 2 * batch_counter
    const uint32_t nfft_required = min(*batch_counter * 2, max_batch);

    if (global_fft >= nfft_required) {
        return;
    }

    // Register storage
    ComplexTypeCUDA thread_data[FFT::storage_size];

    // Load complex spectrum
    const uint32_t base_in = global_fft * in_len;
    for (uint32_t i = 0; i < FFT::input_ept; ++i) {
        const uint32_t pos = threadIdx.x + stride * i;
        if (pos < in_len) {
            thread_data[i] = reinterpret_cast<const ComplexTypeCUDA*>(
                complex_input)[base_in + pos];
        }
    }

    // Shared memory
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    auto* shared_mem = reinterpret_cast<ComplexTypeCUDA*>(smem);

    static_assert(!FFT::requires_workspace,
                  "Workspace-required FFT not supported");

    // Execute IRFFT
    FFT().execute(thread_data, shared_mem);

    // Store real output with normalization
    const float norm        = 1.0f / static_cast<float>(N);
    const uint32_t base_out = global_fft * out_len;
    const float* out        = reinterpret_cast<const float*>(thread_data);

    for (uint32_t i = 0; i < FFT::output_ept; ++i) {
        const uint32_t pos = threadIdx.x + stride * i;
        if (pos < out_len) {
            real_output[base_out + pos] = out[i] * norm;
        }
    }
}

template <unsigned int N> struct IrfftLauncher {
    using FFT = C2R_FFT<N>;

    static void launch(const ComplexTypeCUDA* in,
                       float* out,
                       const uint32_t* counter,
                       uint32_t max_batch,
                       cudaStream_t stream) {
        const uint32_t blocks =
            (max_batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;

        if (blocks == 0)
            return;

        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(irfft_c2r_kernel<FFT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size);
            configured = true;
        }

        irfft_c2r_kernel<FFT>
            <<<blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                reinterpret_cast<const ComplexTypeCUDA*>(in), out, counter,
                max_batch);
        cuda_utils::check_last_cuda_error(
            "IrfftLauncher: irfft_c2r_kernel launch failed");
    }
};

} // namespace

// IrfftExecutorCUDADx implementation
IrfftExecutorCUDADx::IrfftExecutorCUDADx(int nbins, int max_leaves)
    : m_nbins(nbins),
      m_nbins_f(nbins / 2 + 1),
      m_max_batch(2 * max_leaves) {
    if (!is_supported(nbins)) {
        throw std::runtime_error(
            std::format("IrfftExecutorCUDADx: unsupported nbins={}", nbins));
    }
}

bool IrfftExecutorCUDADx::is_supported(int nbins) {

    switch (nbins) {
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
        return true;
    default:
        return false;
    }
}

void IrfftExecutorCUDADx::execute_async(
    cuda::std::span<const ComplexTypeCUDA> complex_input,
    cuda::std::span<float> real_output,
    const utils::DeviceCounter& batch_counter,
    cudaStream_t stream) const {
    const ComplexTypeCUDA* in = complex_input.data();
    float* out                = real_output.data();

    switch (m_nbins) {
    case 32:
        IrfftLauncher<32>::launch(in, out, batch_counter.data(), m_max_batch,
                                  stream);
        break;
    case 64:
        IrfftLauncher<64>::launch(in, out, batch_counter.data(), m_max_batch,
                                  stream);
        break;
    case 128:
        IrfftLauncher<128>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 256:
        IrfftLauncher<256>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 512:
        IrfftLauncher<512>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 1024:
        IrfftLauncher<1024>::launch(in, out, batch_counter.data(), m_max_batch,
                                    stream);
        break;
    default:
        throw std::runtime_error(
            std::format("IrfftExecutorCUDADx: unsupported nbins={}", m_nbins));
    }
}
*/

} // namespace loki::math