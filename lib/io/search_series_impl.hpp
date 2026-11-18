#pragma once

/**
 * @file search_series_impl.hpp
 * @brief Storage behind io::SearchSeries and its backend factories.
 * Internal.
 */

#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/io/search_series.hpp"

#include "lib/common/dispatch.hpp"

namespace loki::io {

namespace detail {

/// Device copy of ts_e and ts_v, defined in lib/cuda/search_series_cuda.cu.
class SeriesDeviceStorage : public loki::detail::DeviceStorage {
public:
    [[nodiscard]] virtual const float* ts_e() const noexcept = 0;
    [[nodiscard]] virtual const float* ts_v() const noexcept = 0;
    virtual void copy_to_host(std::span<float> ts_e,
                              std::span<float> ts_v) const   = 0;
};

} // namespace detail

// Public handle storage. Exactly one of `host_e`/`host_v` (CPU) or `device`
// (GPU) holds the series, matching `exec.backend`.
class SearchSeries::Impl {
public:
    Exec exec;
    double tsamp{0.0};
    SizeType n{0};
    std::vector<float> host_e;
    std::vector<float> host_v;
    std::unique_ptr<detail::SeriesDeviceStorage> device;
};

namespace detail {

/// GPU factories (lib/cuda/search_series_cuda.cu, preprocess_cuda.cu).
std::unique_ptr<SeriesDeviceStorage> make_series_storage_gpu(
    std::span<const float> ts_e, std::span<const float> ts_v, int device_id);

/// preprocess_gpu() that keeps ts_e and ts_v on the device.
std::pair<std::unique_ptr<SeriesDeviceStorage>, PreprocessReport>
preprocess_gpu_resident(std::span<const float> raw,
                        double tsamp,
                        const PreprocessOptions& options,
                        int device_id);

/// Host views of a CPU series. Throws if @p series is empty or not CPU.
std::pair<std::span<const float>, std::span<const float>>
host_series(const SearchSeries& series, std::string_view what);

/// Device views of a GPU series. Throws if @p series is empty or not GPU.
std::pair<DeviceSpan<const float>, DeviceSpan<const float>>
device_series(const SearchSeries& series, std::string_view what);

} // namespace detail

} // namespace loki::io
