#pragma once

#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>
#include <cufft.h>

namespace loki::cuda_utils {

namespace {
// Error code to string conversion
constexpr std::string_view cufft_error_string(cufftResult error) noexcept {
    switch (error) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    default:
        return "Unknown cuFFT error";
    }
}
} // namespace

/**
 * @brief Custom exception class for CUDA errors.
 */
class CudaException : public std::runtime_error {
public:
    // Constructor for cudaError_t
    explicit CudaException(
        cudaError_t code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_cuda_error(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_cuda(true),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    // Constructor for cufftResult
    explicit CudaException(
        cufftResult code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_cufft_error(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_cuda(false),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    [[nodiscard]] constexpr int code() const noexcept { return m_code; }
    [[nodiscard]] constexpr bool is_cuda_error() const noexcept {
        return m_is_cuda;
    }
    [[nodiscard]] constexpr const char* file() const noexcept { return m_file; }
    [[nodiscard]] constexpr uint32_t line() const noexcept { return m_line; }
    [[nodiscard]] constexpr const char* function() const noexcept {
        return m_func;
    }
    [[nodiscard]] constexpr std::string_view user_message() const noexcept {
        return m_user_msg;
    }
    [[nodiscard]] std::string error_string() const {
        return m_is_cuda ? cudaGetErrorString(static_cast<cudaError_t>(m_code))
                         : std::string(cufft_error_string(
                               static_cast<cufftResult>(m_code)));
    }

private:
    int m_code;
    bool m_is_cuda;
    const char* m_file;
    uint32_t m_line;
    const char* m_func;
    std::string m_user_msg;

    // Helper to format the what() message for CUDA errors
    static std::string format_cuda_error(cudaError_t code,
                                         std::string_view user_msg,
                                         const std::source_location& loc) {
        auto base_msg =
            std::format("CUDA Error [{}]: {}", static_cast<int>(code),
                        cudaGetErrorString(code));
        return user_msg.empty()
                   ? std::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("{} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }

    // Helper to format the what() message for cuFFT errors
    static std::string format_cufft_error(cufftResult code,
                                          std::string_view user_msg,
                                          const std::source_location& loc) {
        auto base_msg =
            std::format("cuFFT Error [{}]: {}", static_cast<int>(code),
                        cufft_error_string(code));
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
check_cuda_call(cufftResult result,
                std::string_view msg     = "",
                std::source_location loc = std::source_location::current()) {
    if (result != CUFFT_SUCCESS) {
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
    size_t shmem_size = 0,
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

inline void set_device(int device_id) {
    int device_count;
    check_cuda_call(cudaGetDeviceCount(&device_count),
                    "Failed to get device count");
    if (device_id < 0 || device_id >= device_count) {
        throw CudaException(
            cudaErrorInvalidDevice,
            std::format("Invalid device_id: {}. Must be between 0 and {}",
                        device_id, device_count - 1));
    }
    check_cuda_call(cudaSetDevice(device_id),
                    std::format("Failed to set device {}", device_id));
}
} // namespace
} // namespace loki::cuda_utils