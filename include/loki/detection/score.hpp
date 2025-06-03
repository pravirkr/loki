#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::detection {

class MatchedFilter {
public:
    MatchedFilter(std::span<const SizeType> widths_arr,
                  SizeType nprofiles,
                  SizeType nbins,
                  std::string_view shape = "boxcar");

    ~MatchedFilter();
    MatchedFilter(MatchedFilter&&) noexcept;
    MatchedFilter& operator=(MatchedFilter&&) noexcept;
    MatchedFilter(const MatchedFilter&)            = delete;
    MatchedFilter& operator=(const MatchedFilter&) = delete;

    std::vector<float> get_templates() const noexcept;
    SizeType get_ntemplates() const noexcept;
    SizeType get_nbins() const noexcept;
    void compute(std::span<const float> arr, std::span<float> out);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

std::vector<SizeType> generate_width_trials(SizeType nbins_max,
                                            float wtsp = 1.5F);

// Compute the S/N of single pulse proile
void snr_1d(std::span<const float> arr,
            std::span<const SizeType> widths,
            std::span<float> out,
            float stdnoise = 1.0F);

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr,
            SizeType nprofiles,
            std::span<const SizeType> widths,
            std::span<float> out,
            float stdnoise = 1.0F);

} // namespace loki::detection
