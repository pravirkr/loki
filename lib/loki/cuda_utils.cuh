#pragma once

#include <cstdint>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>

namespace loki::cuda_utils {

namespace {
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
        return "CUFFT_NVRTC_FAILURE: An NVRTC failure was encountered during a cuFFT operation";
    case CUFFT_NVJITLINK_FAILURE:
        return "CUFFT_NVJITLINK_FAILURE: An nvJitLink failure was encountered during a cuFFT operation";
    case CUFFT_NVSHMEM_FAILURE:
        return "CUFFT_NVSHMEM_FAILURE: An NVSHMEM failure was encountered during a cuFFT operation";
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
} // namespace

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

namespace {
// Generic error checking function
inline void
check_cuda_call(cudaError_t result,
                std::string_view msg     = "",
                std::source_location loc = std::source_location::current()) {
    if (result != cudaSuccess) {
        throw CudaException(result, msg, loc);
    }
}

inline void
check_cufft_call(cufftResult result,
                 std::string_view msg     = "",
                 std::source_location loc = std::source_location::current()) {
    if (result != CUFFT_SUCCESS) {
        throw CudaException(result, msg, loc);
    }
}

inline void
check_curand_call(curandStatus_t result,
                  std::string_view msg     = "",
                  std::source_location loc = std::source_location::current()) {
    if (result != CURAND_STATUS_SUCCESS) {
        throw CudaException(result, msg, loc);
    }
}

// Specialized checks
inline void check_last_cuda_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(error, msg, loc);
    }
}

inline void check_cuda_sync_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    check_cuda_call(cudaDeviceSynchronize(), "Synchronization failed", loc);
    check_last_cuda_error(msg, loc);
}

/**
 * @brief Checks kernel launch parameters against device limits before
 * launching. Throws std::runtime_error on failure.
 * @param grid Grid dimensions.
 * @param block Block dimensions.
 * @param location Source location information (automatically captured).
 * @throws std::runtime_error if dimensions exceed device limits.
 */
inline void check_kernel_launch_params(
    dim3 grid,
    dim3 block,
    size_t shmem_size              = 0,
    const std::source_location loc = std::source_location::current()) {
    int device;
    cudaDeviceProp props{};
    // Get current device, check for errors
    check_cuda_call(cudaGetDevice(&device), "Failed to get device", loc);
    check_cuda_call(cudaGetDeviceProperties(&props, device),
                    "Failed to get device properties", loc);

    auto check_limit = [&](auto val, auto max, std::string_view dim) {
        if (val > static_cast<unsigned>(max)) {
            throw std::runtime_error(
                std::format("{} dimension {} exceeds device limit {} at {}:{}",
                            dim, val, max, loc.file_name(), loc.line()));
        }
    };

    check_limit(block.x, props.maxThreadsDim[0], "Block X");
    check_limit(block.y, props.maxThreadsDim[1], "Block Y");
    check_limit(block.z, props.maxThreadsDim[2], "Block Z");
    check_limit(block.x * block.y * block.z, props.maxThreadsPerBlock,
                "Total threads");
    check_limit(grid.x, props.maxGridSize[0], "Grid X");
    check_limit(grid.y, props.maxGridSize[1], "Grid Y");
    check_limit(grid.z, props.maxGridSize[2], "Grid Z");
    check_limit(shmem_size, props.sharedMemPerBlock, "Shared memory");
}

[[nodiscard]] inline std::string get_device_info() noexcept {
    int device;
    cudaDeviceProp props{};
    if (auto err = cudaGetDevice(&device); err != cudaSuccess) {
        return std::format("Failed to get device: {}", cudaGetErrorString(err));
    }
    if (auto err = cudaGetDeviceProperties(&props, device);
        err != cudaSuccess) {
        return std::format("Failed to get properties: {}",
                           cudaGetErrorString(err));
    }

    return std::format("Device {}: {} (CC {}.{}), Mem: {} MiB", device,
                       props.name, props.major, props.minor,
                       props.totalGlobalMem >> 20U);
}

// Cached device count
[[nodiscard]] inline int get_device_count() {
    static int count = [] {
        int device_count;
        check_cuda_call(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount failed: Failed to get device count");
        return device_count;
    }();
    return count;
}

inline void set_device(int device_id) {
    const int device_count = get_device_count();
    if (device_id < 0 || device_id >= device_count) {
        throw CudaException(cudaErrorInvalidDevice,
                            std::format("Invalid device_id {} (0..{})",
                                        device_id, device_count - 1));
    }

    int current_device_id;
    check_cuda_call(cudaGetDevice(&current_device_id), "cudaGetDevice failed");

    if (current_device_id == device_id) {
        return;
    }

    check_cuda_call(cudaSetDevice(device_id),
                    std::format("cudaSetDevice({}) failed", device_id));
}

/**
 * @brief RAII guard for setting and restoring the CUDA device.
 */
class CudaSetDeviceGuard {
    int m_previous_device_id = -1;

public:
    explicit CudaSetDeviceGuard(int device_id) {
        check_cuda_call(cudaGetDevice(&m_previous_device_id),
                        "cudaGetDevice failed");
        if (m_previous_device_id != device_id) {
            set_device(device_id);
        }
    }

    ~CudaSetDeviceGuard() { cudaSetDevice(m_previous_device_id); }
    CudaSetDeviceGuard(const CudaSetDeviceGuard&)            = delete;
    CudaSetDeviceGuard& operator=(const CudaSetDeviceGuard&) = delete;
    CudaSetDeviceGuard(CudaSetDeviceGuard&&)                 = delete;
    CudaSetDeviceGuard& operator=(CudaSetDeviceGuard&&)      = delete;
};

} // namespace
} // namespace loki::cuda_utils