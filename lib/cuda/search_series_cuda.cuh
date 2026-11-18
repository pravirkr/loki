#pragma once

/**
 * @file search_series_cuda.cuh
 * @brief Device storage of io::SearchSeries. Internal.
 */

#include <span>
#include <utility>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/io/search_series_impl.hpp"

namespace loki::io::detail {

/// ts_e and ts_v on one device. The buffers are moved in (from the
/// preprocessing engine) or uploaded.
class CudaSeriesStorage final : public SeriesDeviceStorage {
public:
    using Buffer = cuda_utils::DeviceVectorNoInit<float>;

    CudaSeriesStorage(Buffer ts_e, Buffer ts_v, int device_id)
        : m_e(std::move(ts_e)),
          m_v(std::move(ts_v)),
          m_device(device_id) {}

    ~CudaSeriesStorage() override {
        const cuda_utils::CudaSetDeviceGuard guard(m_device);
        m_e.clear();
        m_e.shrink_to_fit();
        m_v.clear();
        m_v.shrink_to_fit();
    }

    CudaSeriesStorage(const CudaSeriesStorage&)            = delete;
    CudaSeriesStorage& operator=(const CudaSeriesStorage&) = delete;
    CudaSeriesStorage(CudaSeriesStorage&&)                 = delete;
    CudaSeriesStorage& operator=(CudaSeriesStorage&&)      = delete;

    [[nodiscard]] const float* ts_e() const noexcept override {
        return thrust::raw_pointer_cast(m_e.data());
    }
    [[nodiscard]] const float* ts_v() const noexcept override {
        return thrust::raw_pointer_cast(m_v.data());
    }

    void copy_to_host(std::span<float> ts_e,
                      std::span<float> ts_v) const override;

    [[nodiscard]] float get_memory_usage_gib() const noexcept override {
        return static_cast<float>(
            static_cast<double>((m_e.size() + m_v.size()) * sizeof(float)) /
            (1024.0 * 1024.0 * 1024.0));
    }

private:
    Buffer m_e;
    Buffer m_v;
    int m_device;
};

} // namespace loki::io::detail
