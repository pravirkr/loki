#include "lib/cuda/search_series_cuda.cuh"

#include <memory>
#include <span>

#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/io/search_series_impl.hpp"

namespace loki::io::detail {

void CudaSeriesStorage::copy_to_host(std::span<float> ts_e,
                                     std::span<float> ts_v) const {
    const cuda_utils::CudaSetDeviceGuard guard(m_device);
    cuda_utils::check_cuda_call(cudaMemcpy(ts_e.data(), this->ts_e(),
                                           ts_e.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                "SearchSeries: download of ts_e failed");
    cuda_utils::check_cuda_call(cudaMemcpy(ts_v.data(), this->ts_v(),
                                           ts_v.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                "SearchSeries: download of ts_v failed");
}

std::unique_ptr<SeriesDeviceStorage> make_series_storage_gpu(
    std::span<const float> ts_e, std::span<const float> ts_v, int device_id) {
    const cuda_utils::CudaSetDeviceGuard guard(device_id);
    CudaSeriesStorage::Buffer e(ts_e.size());
    CudaSeriesStorage::Buffer v(ts_v.size());
    cuda_utils::check_cuda_call(
        cudaMemcpy(thrust::raw_pointer_cast(e.data()), ts_e.data(),
                   ts_e.size() * sizeof(float), cudaMemcpyHostToDevice),
        "SearchSeries: upload of ts_e failed");
    cuda_utils::check_cuda_call(
        cudaMemcpy(thrust::raw_pointer_cast(v.data()), ts_v.data(),
                   ts_v.size() * sizeof(float), cudaMemcpyHostToDevice),
        "SearchSeries: upload of ts_v failed");
    return std::make_unique<CudaSeriesStorage>(std::move(e), std::move(v),
                                               device_id);
}

} // namespace loki::io::detail
