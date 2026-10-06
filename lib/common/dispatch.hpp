#pragma once

/**
 * @file dispatch.hpp
 * @brief Shared helpers for backend dispatch in the facade translation units.
 *
 * Every public class builds one engine per instance through a
 * make_<algo>_engine() function in its facade .cpp. That function is the only
 * place that tests LOKI_ENABLE_GPU. Everything else in this header is
 * backend-neutral and compiles in every build.
 */

#include <string_view>

#include "loki/common/backend.hpp"

namespace loki::detail {

#ifdef LOKI_ENABLE_GPU
/// The GPU backend this build contains. One GPU backend per build.
inline constexpr Backend kGPUBackend = Backend::kCUDA;
#endif

/// @throws std::invalid_argument: @p what got device memory on a backend
/// that only takes host memory.
[[noreturn]] void throw_no_device_memory(std::string_view what,
                                         Backend backend);

/// @throws std::invalid_argument: @p backend is not compiled into this build.
[[noreturn]] void throw_unavailable(std::string_view algorithm,
                                    Backend backend);

/// @throws std::invalid_argument: @p backend is compiled into this build, but
/// @p algorithm has no engine for it yet.
[[noreturn]] void throw_unimplemented(std::string_view algorithm,
                                      Backend backend);

/// Rejects a DeviceSpan whose stated device differs from the instance's.
/// A view with `id < 0` ("not stated") is accepted.
void check_device(const Device& view,
                  Backend backend,
                  int device,
                  std::string_view what);

/// Rejects a shared resource (workspace, FFT manager) that was built for a
/// different backend or device than the algorithm using it.
void check_same_exec(const Exec& resource,
                     const Exec& algorithm,
                     std::string_view what);

/// Classes built from a search config take their thread count from the
/// config. Logs a warning when @p exec carries a non-default thread count
/// that will be ignored.
void warn_ignored_nthreads(const Exec& exec, std::string_view algorithm);

/**
 * @brief GPU-owned state behind a public handle (workspace, FFT manager).
 *
 * The concrete types hold device containers and are defined in lib/cuda/.
 * Host code only owns and destroys them through this base.
 */
class DeviceStorage {
public:
    DeviceStorage()                                = default;
    virtual ~DeviceStorage()                       = default;
    DeviceStorage(const DeviceStorage&)            = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
    DeviceStorage(DeviceStorage&&)                 = delete;
    DeviceStorage& operator=(DeviceStorage&&)      = delete;

    [[nodiscard]] virtual float get_memory_usage_gib() const noexcept {
        return 0.0F;
    }
};

} // namespace loki::detail
