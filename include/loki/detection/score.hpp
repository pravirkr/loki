#pragma once

#include <functional>
#include <memory>
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

struct BoxcarWidthsCache {
    std::vector<SizeType> widths;
    SizeType wmax;
    SizeType ntemplates;
    std::vector<float> h_vals;
    std::vector<float> b_vals;
    std::vector<float> fold_norm_buffer;
    std::vector<float> psum_buffer;

    BoxcarWidthsCache(std::span<const SizeType> widths, SizeType nbins);
};

// Generate boxcar width trials for matched filtering
std::vector<SizeType> generate_box_width_trials(SizeType fold_bins,
                                                double ducy_max = 0.2,
                                                double wtsp     = 1.5);

// Compute the Boxcar S/N of a single pulse profile
void snr_boxcar_1d(std::span<const float> arr,
                   std::span<const SizeType> widths,
                   std::span<float> out,
                   float stdnoise = 1.0F);

// Compute the Boxcar S/N of a batch of single pulse profiles with common
// variance Useful for thresholding code
void snr_boxcar_2d_max(std::span<const float> arr,
                       SizeType nprofiles,
                       std::span<const SizeType> widths,
                       std::span<float> out,
                       float stdnoise = 1.0F,
                       int nthreads   = 1);

// Compute the Boxcar S/N (for each width) of a batch of E, V folded profiles
void snr_boxcar_3d(std::span<const float> arr,
                   SizeType nprofiles,
                   std::span<const SizeType> widths,
                   std::span<float> out,
                   int nthreads = 1);

// Compute the Boxcar S/N of a batch of E, V folded profiles
void snr_boxcar_3d_max(std::span<const float> arr,
                       SizeType nprofiles,
                       std::span<const SizeType> widths,
                       std::span<float> out,
                       int nthreads = 1);

// Compute the S/N of a batch of folded profiles
void snr_boxcar_batch(std::span<const float> batch_folds,
                      std::span<float> batch_scores,
                      SizeType n_batch,
                      BoxcarWidthsCache& cache);

template <typename FoldType>
using ScoringFunction = std::function<void(
    std::span<const FoldType>, std::span<float>, SizeType, BoxcarWidthsCache&)>;

#ifdef LOKI_ENABLE_CUDA

void snr_boxcar_2d_max_cuda(std::span<const float> arr,
                            SizeType nprofiles,
                            std::span<const SizeType> widths,
                            std::span<float> out,
                            float stdnoise = 1.0F,
                            int device_id  = 0);

void snr_boxcar_2d_max_cuda_d(cuda::std::span<const float> arr,
                              SizeType nprofiles,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> out,
                              float stdnoise = 1.0F,
                              int device_id  = 0);

void snr_boxcar_3d_cuda(std::span<const float> arr,
                        SizeType nprofiles,
                        std::span<const SizeType> widths,
                        std::span<float> out,
                        int device_id = 0);

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> arr,
                          SizeType nprofiles,
                          cuda::std::span<const SizeType> widths,
                          cuda::std::span<float> out,
                          int device_id = 0);

void snr_boxcar_3d_max_cuda(std::span<const float> arr,
                            SizeType nprofiles,
                            std::span<const SizeType> widths,
                            std::span<float> out,
                            int device_id = 0);

void snr_boxcar_3d_max_cuda_d(cuda::std::span<const float> arr,
                              SizeType nprofiles,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> out,
                              int device_id = 0);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::detection
