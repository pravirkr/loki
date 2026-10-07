#pragma once

/**
 * @file backend.hpp
 * @brief Execution backend selection and device-memory views for Loki.
 *
 * Every algorithm and workspace takes an Exec parameter. The same header
 * is installed for every build; available_backends() reports which backends
 * this build of the library contains. No GPU runtime header is included here.
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace loki {

/// @brief Execution backend of an algorithm instance.
enum class Backend : std::uint8_t { kCPU, kCUDA };

/// @brief Lower-case name: "cpu" or "cuda".
[[nodiscard]] std::string_view to_string(Backend backend) noexcept;

/// @brief Inverse of to_string().
/// @throws std::invalid_argument for an unknown name.
[[nodiscard]] Backend parse_backend(std::string_view name);

/// @brief Backends compiled into this build of the library ("cpu" first).
[[nodiscard]] std::vector<Backend> available_backends();

/// @brief Whether @p backend is compiled into this build.
[[nodiscard]] bool is_available(Backend backend);

/**
 * @brief Where and how an algorithm instance runs.
 *
 * `nthreads` is honored by Backend::kCPU (OpenMP workers) and `device` by the
 * GPU backends (device ordinal). The field that does not apply is ignored.
 * Constructing an algorithm with a backend this build does not contain
 * throws std::invalid_argument naming available_backends().
 */
struct Exec {
    Backend backend = Backend::kCPU;
    int nthreads    = 1;
    int device      = 0;

    [[nodiscard]] static constexpr Exec cpu(int nthreads = 1) noexcept {
        return {.backend = Backend::kCPU, .nthreads = nthreads, .device = 0};
    }
    [[nodiscard]] static constexpr Exec cuda(int device = 0) noexcept {
        return {.backend = Backend::kCUDA, .nthreads = 1, .device = device};
    }
};

/**
 * @brief Non-owning handle to a backend queue: `cudaStream_t` for CUDA.
 * Null means the backend's default stream. The caller keeps it alive.
 * Converts implicitly from the native handle, so a `cudaStream_t` can
 * be passed where a Stream is expected.
 */
struct Stream {
    void* native = nullptr;

    constexpr Stream() noexcept = default;
    constexpr Stream(void* handle) noexcept : native(handle) {} // NOLINT
};

/// @brief A device and ordinal. `id < 0` means "not stated".
struct Device {
    Backend backend = Backend::kCPU;
    int id          = -1;
};

/**
 * @brief Non-owning view of `size` elements of type T in device memory.
 *
 * Laid out like a DLPack tensor: `handle` is the allocation (the device
 * pointer on CUDA) and `byte_offset` the start of the view inside it.
 * There is deliberately no conversion from std::span: host and device memory
 * are told apart by which overload the caller writes, never by inspecting a
 * pointer.
 */
template <typename T> struct DeviceSpan {
    void* handle            = nullptr;
    std::size_t byte_offset = 0;
    std::size_t count       = 0;
    Device device{};

    constexpr DeviceSpan() noexcept = default;
    /// View of @p n elements starting at device pointer @p ptr.
    constexpr DeviceSpan(T* ptr, std::size_t n, Device dev = {}) noexcept
        : handle(const_cast<std::remove_const_t<T>*>(ptr)),
          count(n),
          device(dev) {}
    /// DeviceSpan<T> converts to DeviceSpan<const T>.
    template <typename U>
        requires(std::is_same_v<const U, T> && !std::is_same_v<U, T>)
    constexpr DeviceSpan(const DeviceSpan<U>& other) noexcept // NOLINT
        : handle(other.handle),
          byte_offset(other.byte_offset),
          count(other.count),
          device(other.device) {}

    /// Start of the view, for pointer-addressed backends (CUDA).
    [[nodiscard]] T* data() const noexcept {
        if (handle == nullptr) {
            return nullptr;
        }
        return reinterpret_cast<T*>(static_cast<std::byte*>(handle) +
                                    byte_offset);
    }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }
    [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
        return count * sizeof(T);
    }
    [[nodiscard]] constexpr bool empty() const noexcept { return count == 0; }

    /// @p n elements starting @p offset elements in. Keeps `handle` and
    /// advances `byte_offset`.
    [[nodiscard]] constexpr DeviceSpan subspan(std::size_t offset,
                                               std::size_t n) const noexcept {
        DeviceSpan out = *this;
        out.byte_offset += offset * sizeof(T);
        out.count = n;
        return out;
    }
};

} // namespace loki
