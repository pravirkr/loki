#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

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

// Generate boxcar width trials for matched filtering
std::vector<SizeType> generate_box_width_trials(SizeType fold_bins,
                                                double ducy_max = 0.2,
                                                double wtsp     = 1.5);

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

// Compute the S/N of a batch of folded profiles
void snr_boxcar_batch(xt::xtensor<float, 3>& folds,
                      std::span<const SizeType> widths,
                      std::span<float> out);

// Compute the S/N of a batch of ComplexType folded profiles
void snr_boxcar_batch_complex(xt::xtensor<ComplexType, 3>& folds,
                              std::span<const SizeType> widths,
                              std::span<float> out);

template <typename FoldType>
using ScoringFunction = std::function<void(
    xt::xtensor<FoldType, 3>&, std::span<const SizeType>, std::span<float>)>;

} // namespace loki::detection
