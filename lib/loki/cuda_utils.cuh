#pragma once

#include <cstdint>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <cub/version.cuh>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

// Check if we are on a modern CUB version (CCCL 3.0+)
#if CUB_VERSION >= 300000
#include <cuda/functional>
#include <cuda/std/numbers>
template <typename T> using CubMaxOp    = ::cuda::maximum<T>;
template <typename T> using CubMinOp    = ::cuda::minimum<T>;
template <typename T> using ThrustMaxOp = ::cuda::maximum<T>;
template <typename T> using ThrustMinOp = ::cuda::minimum<T>;
inline constexpr double kPI             = cuda::std::numbers::pi_v<double>;
#else
// Fall back to CUB operators for older CCCL
#include <thrust/functional.h>
template <typename T> using CubMaxOp    = cub::Max;
template <typename T> using CubMinOp    = cub::Min;
template <typename T> using ThrustMaxOp = thrust::maximum<T>;
template <typename T> using ThrustMinOp = thrust::minimum<T>;
inline constexpr double kPI             = 3.14159265358979323846; // NOLINT
#endif // CUB_VERSION >= 300000

namespace loki::cuda_utils {

namespace detail {

// Error code to string conversion
constexpr std::string_view cufft_error_string(cufftResult error) noexcept {
    switch (error) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS: The cuFFT operation was successful.";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN: cuFFT was passed an invalid plan handle.";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU "
               "memory.";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE: The cuFFT type provided is unsupported.";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE: User specified an invalid pointer or "
               "parameter";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR: Driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED: Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED: The cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE: User specified an invalid transform size";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA: Not currently in use";
#if CUFFT_VERSION < 12000
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST: Missing parameters in call";
#endif
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE: Execution of a plan was on different GPU "
               "than plan creation";
#if CUFFT_VERSION < 12000
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR: Internal plan database error";
#endif
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE: No workspace has been provided prior to "
               "plan execution";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED: Function does not implement "
               "functionality for parameters given.";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED: Operation is not supported for parameters "
               "given.";
#if CUFFT_VERSION >= 12000
    case CUFFT_MISSING_DEPENDENCY:
        return "CUFFT_MISSING_DEPENDENCY: cuFFT is unable to find a dependency";
    case CUFFT_NVRTC_FAILURE:
        return "CUFFT_NVRTC_FAILURE: An NVRTC failure was encountered during a "
               "cuFFT operation";
    case CUFFT_NVJITLINK_FAILURE:
        return "CUFFT_NVJITLINK_FAILURE: An nvJitLink failure was encountered "
               "during a cuFFT operation";
    case CUFFT_NVSHMEM_FAILURE:
        return "CUFFT_NVSHMEM_FAILURE: An NVSHMEM failure was encountered "
               "during a cuFFT operation";
#endif
    default:
        return "Unknown cuFFT error";
    }
}

constexpr std::string_view curand_error_string(curandStatus_t error) noexcept {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS: No errors.";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH: Header file and linked library "
               "version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED: Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED: Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR: Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE: Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE: Length requested is not a "
               "multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: GPU does not have "
               "double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE: Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE: Preexisting failure on "
               "library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED: Initialization of CUDA "
               "failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH: Architecture mismatch, GPU does "
               "not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR: Internal library error.";
    default:
        return "Unknown cuRAND error";
    }
}

/**
 * @brief Unified exception class for all CUDA-related library errors.
 */
class CudaException : public std::runtime_error {
public:
    // Error type identification
    enum class ErrorType : uint8_t { kCuda, kCufft, kCurand };

    // Constructor for cudaError_t
    explicit CudaException(
        cudaError_t code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_error(code, user_msg, loc)),
          m_error_code(code),
          m_error_type(ErrorType::kCuda),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    // Constructor for cufftResult
    explicit CudaException(
        cufftResult code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_error(code, user_msg, loc)),
          m_error_code(code),
          m_error_type(ErrorType::kCufft),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    // Constructor for curandStatus_t
    explicit CudaException(
        curandStatus_t code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_error(code, user_msg, loc)),
          m_error_code(code),
          m_error_type(ErrorType::kCurand),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    [[nodiscard]] ErrorType error_type() const noexcept { return m_error_type; }
    [[nodiscard]] const char* file() const noexcept { return m_file; }
    [[nodiscard]] uint32_t line() const noexcept { return m_line; }
    [[nodiscard]] const char* function() const noexcept { return m_func; }
    [[nodiscard]] std::string_view user_message() const noexcept {
        return m_user_msg;
    }
    template <typename T> [[nodiscard]] T code() const {
        return std::get<T>(m_error_code);
    }
    // Get error string for the specific error type
    [[nodiscard]] std::string error_string() const {
        return std::visit(
            [](auto&& error) -> std::string {
                using T = std::decay_t<decltype(error)>;
                if constexpr (std::is_same_v<T, cudaError_t>) {
                    return cudaGetErrorString(error);
                } else if constexpr (std::is_same_v<T, cufftResult>) {
                    return std::string(cufft_error_string(error));
                } else if constexpr (std::is_same_v<T, curandStatus_t>) {
                    return std::string(curand_error_string(error));
                }
            },
            m_error_code);
    }

private:
    std::variant<cudaError_t, cufftResult, curandStatus_t> m_error_code;
    ErrorType m_error_type;
    const char* m_file;
    uint32_t m_line;
    const char* m_func;
    std::string m_user_msg;

    // Generic error formatting using std::visit
    template <typename ErrorCode>
    static std::string format_error(ErrorCode code,
                                    std::string_view user_msg,
                                    const std::source_location& loc) {
        std::string library_name;
        std::string error_desc;

        if constexpr (std::is_same_v<ErrorCode, cudaError_t>) {
            library_name = "CUDA";
            error_desc   = cudaGetErrorString(code);
        } else if constexpr (std::is_same_v<ErrorCode, cufftResult>) {
            library_name = "cuFFT";
            error_desc   = cufft_error_string(code);
        } else if constexpr (std::is_same_v<ErrorCode, curandStatus_t>) {
            library_name = "cuRAND";
            error_desc   = curand_error_string(code);
        }

        auto base_msg = std::format("{} Error [{}]: {}", library_name,
                                    static_cast<int>(code), error_desc);

        return user_msg.empty()
                   ? std::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("{} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }
};

inline thread_local int tls_current_device = -1;

} // namespace detail

// Generic error checking function
inline void
check_cuda_call(cudaError_t result,
                std::string_view msg     = "",
                std::source_location loc = std::source_location::current()) {
    if (result != cudaSuccess) {
        throw detail::CudaException(result, msg, loc);
    }
}

inline void
check_cufft_call(cufftResult result,
                 std::string_view msg     = "",
                 std::source_location loc = std::source_location::current()) {
    if (result != CUFFT_SUCCESS) {
        throw detail::CudaException(result, msg, loc);
    }
}

inline void
check_curand_call(curandStatus_t result,
                  std::string_view msg     = "",
                  std::source_location loc = std::source_location::current()) {
    if (result != CURAND_STATUS_SUCCESS) {
        throw detail::CudaException(result, msg, loc);
    }
}

// Specialized checks
inline void check_last_cuda_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw detail::CudaException(error, msg, loc);
    }
}

/**
 * @brief Thread-local cached device properties for zero-overhead access.
 * Automatically invalidated on device switches.
 */
class CudaDeviceContext {
public:
    ~CudaDeviceContext()                                   = default;
    CudaDeviceContext(const CudaDeviceContext&)            = delete;
    CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;
    CudaDeviceContext(CudaDeviceContext&&)                 = delete;
    CudaDeviceContext& operator=(CudaDeviceContext&&)      = delete;

    // Get thread-local instance for current device
    [[nodiscard]] static const CudaDeviceContext& get() {
        assert(detail::tls_current_device >= 0 &&
               "CUDA device guard not active");
        thread_local CudaDeviceContext instance;
        instance.refresh_if_needed();
        return instance;
    }

    // Force refresh (useful after external device changes)
    static void force_refresh() {
        thread_local CudaDeviceContext instance;
        instance.refresh();
    }

    // Accessors - zero cost, just reads cached values
    [[nodiscard]] int get_device_id() const noexcept { return m_device_id; }
    [[nodiscard]] const cudaDeviceProp& get_properties() const noexcept {
        return m_props;
    }

    [[nodiscard]] SizeType get_max_shared_memory() const noexcept {
        return m_props.sharedMemPerBlock;
    }

    [[nodiscard]] int get_max_threads_per_block() const noexcept {
        return m_props.maxThreadsPerBlock;
    }

    [[nodiscard]] const int* get_max_threads_dim() const noexcept {
        return m_props.maxThreadsDim;
    }

    [[nodiscard]] const int* get_max_grid_size() const noexcept {
        return m_props.maxGridSize;
    }

    [[nodiscard]] std::string get_device_info() const {
        return std::format(
            "Device {}: {} (CC {}.{}), Mem: {} MiB, Shared Mem: {} bytes",
            m_device_id, m_props.name, m_props.major, m_props.minor,
            m_props.totalGlobalMem >> 20U, m_props.sharedMemPerBlock);
    }

    // Validation methods - compile away in release builds
#ifndef NDEBUG
    /**
     * @brief Checks kernel launch parameters against device limits before
     * launching. Throws std::runtime_error on failure.
     * @param grid Grid dimensions.
     * @param block Block dimensions.
     * @param location Source location information (automatically captured).
     * @throws std::runtime_error if dimensions exceed device limits.
     */
    void check_kernel_launch_params(dim3 grid,
                                    dim3 block,
                                    size_t shmem_size = 0,
                                    const std::source_location loc =
                                        std::source_location::current()) const {

        auto check_limit = [&](auto val, auto max, std::string_view dim) {
            if (val > static_cast<unsigned>(max)) {
                throw std::runtime_error(std::format(
                    "{} dimension {} exceeds device limit {} at {}:{}", dim,
                    val, max, loc.file_name(), loc.line()));
            }
        };

        check_limit(block.x, m_props.maxThreadsDim[0], "Block X");
        check_limit(block.y, m_props.maxThreadsDim[1], "Block Y");
        check_limit(block.z, m_props.maxThreadsDim[2], "Block Z");
        check_limit(block.x * block.y * block.z, m_props.maxThreadsPerBlock,
                    "Total threads");
        check_limit(grid.x, m_props.maxGridSize[0], "Grid X");
        check_limit(grid.y, m_props.maxGridSize[1], "Grid Y");
        check_limit(grid.z, m_props.maxGridSize[2], "Grid Z");
        check_limit(shmem_size, m_props.sharedMemPerBlock, "Shared memory");
    }
#else
    // No-op in release builds - completely optimized away
    constexpr void check_kernel_launch_params(
        dim3,
        dim3,
        size_t = 0,
        const std::source_location =
            std::source_location::current()) const noexcept {}
#endif

private:
    int m_device_id{-1};
    cudaDeviceProp m_props{};

    CudaDeviceContext() { refresh(); }

    void refresh() {
        check_cuda_call(cudaGetDevice(&m_device_id), "Failed to get device");
        check_cuda_call(cudaGetDeviceProperties(&m_props, m_device_id),
                        "Failed to get device properties");
    }

    void refresh_from_known_device(int device_id) {
        m_device_id = device_id;
        check_cuda_call(cudaGetDeviceProperties(&m_props, device_id),
                        "Failed to get device properties");
    }

    void refresh_if_needed() {
        const int current = detail::tls_current_device;
        if (current != m_device_id) {
            refresh_from_known_device(current);
        }
    }
};

// Cached device count (global, not per-thread)
[[nodiscard]] inline int get_device_count() {
    static int count = [] {
        int device_count;
        check_cuda_call(cudaGetDeviceCount(&device_count),
                        "Failed to get device count");
        return device_count;
    }();
    return count;
}

// Optimized set_device with validation
inline void set_device(int device_id) {
    const int device_count = get_device_count();
    if (device_id < 0 || device_id >= device_count) {
        throw detail::CudaException(
            cudaErrorInvalidDevice,
            std::format("Invalid device_id {} (must be 0..{})", device_id,
                        device_count - 1));
    }

    int& current = detail::tls_current_device;
    if (current == device_id) {
        return; // Already on correct device
    }

    check_cuda_call(cudaSetDevice(device_id),
                    std::format("cudaSetDevice({}) failed", device_id));

    current = device_id;
    // Force context refresh after device switch
    CudaDeviceContext::force_refresh();
}

/**
 * @brief RAII guard for setting and restoring the CUDA device.
 * Provides access to cached device context for the guarded device.
 */
class CudaSetDeviceGuard {
public:
    explicit CudaSetDeviceGuard(int device_id) {
        int& current  = detail::tls_current_device;
        m_prev_device = current;

        if (current != device_id) {
            set_device(device_id);
            m_device_changed = true;
        }
    }

    ~CudaSetDeviceGuard() {
        if (m_device_changed) {
            cudaSetDevice(m_prev_device);
        }
    }

    // Provide access to cached context for current device
    [[nodiscard]] static const CudaDeviceContext& context() {
        assert(detail::tls_current_device >= 0 &&
               "CUDA device guard not active");
        return CudaDeviceContext::get();
    }

    CudaSetDeviceGuard(const CudaSetDeviceGuard&)            = delete;
    CudaSetDeviceGuard& operator=(const CudaSetDeviceGuard&) = delete;
    CudaSetDeviceGuard(CudaSetDeviceGuard&&)                 = delete;
    CudaSetDeviceGuard& operator=(CudaSetDeviceGuard&&)      = delete;

private:
    int m_prev_device{-1};
    bool m_device_changed{false};
};

// Standalone utility functions that use cached context
[[nodiscard]] inline std::string get_device_info() {
    return CudaDeviceContext::get().get_device_info();
}

[[nodiscard]] inline SizeType get_max_shared_memory() {
    return CudaDeviceContext::get().get_max_shared_memory();
}

[[nodiscard]] inline std::pair<double, double> get_cuda_memory_usage() {
    SizeType free_mem, total_mem;
    check_cuda_call(cudaMemGetInfo(&free_mem, &total_mem),
                    "Failed to get CUDA memory usage");
    return {static_cast<double>(free_mem) / (1ULL << 30U),
            static_cast<double>(total_mem) / (1ULL << 30U)};
}

// Standalone check function - delegates to context
inline void check_kernel_launch_params(
    dim3 grid,
    dim3 block,
    size_t shmem_size              = 0,
    const std::source_location loc = std::source_location::current()) {
    CudaDeviceContext::get().check_kernel_launch_params(grid, block, shmem_size,
                                                        loc);
}

// Span helpers for Thrust device vectors
template <typename T>
[[nodiscard]] inline cuda::std::span<T>
as_span(thrust::device_vector<T>& v) noexcept {
    return {thrust::raw_pointer_cast(v.data()),
            static_cast<cuda::std::span<T>::size_type>(v.size())};
}

template <typename T>
[[nodiscard]] inline cuda::std::span<const T>
as_span(const thrust::device_vector<T>& v) noexcept {
    return {thrust::raw_pointer_cast(v.data()),
            static_cast<cuda::std::span<const T>::size_type>(v.size())};
}

// Overload for smaller given size
template <typename T>
[[nodiscard]] inline cuda::std::span<T> as_span(thrust::device_vector<T>& v,
                                                SizeType size) noexcept {
    return {thrust::raw_pointer_cast(v.data()),
            static_cast<cuda::std::span<T>::size_type>(size)};
}

template <typename T>
[[nodiscard]] inline cuda::std::span<const T>
as_span(const thrust::device_vector<T>& v, SizeType size) noexcept {
    return {thrust::raw_pointer_cast(v.data()),
            static_cast<cuda::std::span<const T>::size_type>(size)};
}

// Lifetime safety: forbid temporaries
template <typename T>
cuda::std::span<T> as_span(thrust::device_vector<T>&&) = delete;

template <typename T>
cuda::std::span<const T> as_span(const thrust::device_vector<T>&&) = delete;

} // namespace loki::cuda_utils