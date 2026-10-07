#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

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

    std::vector<float> get_templates() const;
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

// Compute the maximum Boxcar S/N of a single pulse profile with a cache
bool snr_boxcar_threshold_with_cache(std::span<const float> arr,
                                     SizeType nbins,
                                     BoxcarWidthsCache& cache,
                                     float threshold,
                                     float stdnoise = 1.0F) noexcept;

void snr_boxcar_2d(std::span<const float> folds,
                   std::span<const SizeType> widths,
                   std::span<float> scores,
                   SizeType nprofiles,
                   SizeType nbins,
                   float stdnoise = 1.0F,
                   Exec exec      = {});

void snr_boxcar_2d(DeviceSpan<const float> folds,
                   DeviceSpan<const uint32_t> widths,
                   DeviceSpan<float> scores,
                   SizeType nprofiles,
                   SizeType nbins,
                   float stdnoise = 1.0F,
                   Stream stream  = {});

// Compute the Boxcar S/N of a batch of single pulse profiles with common
// variance Useful for thresholding code
void snr_boxcar_2d_max(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       float stdnoise = 1.0F,
                       Exec exec      = {});

void snr_boxcar_2d_max(DeviceSpan<const float> folds,
                       DeviceSpan<const uint32_t> widths,
                       DeviceSpan<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       float stdnoise = 1.0F,
                       Stream stream  = {});

/// One boxcar S/N that passed a threshold. `score_index` is
/// `profile * nwidths + width_index`, matching the dense `snr_boxcar_3d`
/// layout.
struct SnrHit {
    uint32_t score_index{0};
    float snr{0.0F};
};

/// Append thresholded boxcar S/N hits for a packed tile of E/V profiles.
/// `psum` must hold `nbins + max(widths)` floats. Hits are appended in
/// profile-major, width-minor order.
void append_snr_boxcar_3d_hits(const float* folds,
                               SizeType profile_base,
                               SizeType nprofiles,
                               SizeType nbins,
                               std::span<const SizeType> widths,
                               float threshold,
                               std::span<float> psum,
                               std::vector<SnrHit>& hits);

// Compute the Boxcar S/N (for each width) of a batch of E, V folded profiles
void snr_boxcar_3d(std::span<const float> folds,
                   std::span<const SizeType> widths,
                   std::span<float> scores,
                   SizeType nprofiles,
                   SizeType nbins,
                   Exec exec = {});

void snr_boxcar_3d(DeviceSpan<const float> folds,
                   DeviceSpan<const uint32_t> widths,
                   DeviceSpan<float> scores,
                   SizeType nprofiles,
                   SizeType nbins,
                   Stream stream = {});

// Compute the Boxcar S/N of a batch of E, V folded profiles
void snr_boxcar_3d_max(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       Exec exec = {});

void snr_boxcar_3d_max(DeviceSpan<const float> folds,
                       DeviceSpan<const uint32_t> widths,
                       DeviceSpan<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       Stream stream = {});

// Compute the S/N of a batch of folded profiles
void snr_boxcar_3d_max_with_cache(std::span<const float> folds,
                                  std::span<float> scores,
                                  SizeType nprofiles,
                                  SizeType nbins,
                                  BoxcarWidthsCache& cache);

// Compute the S/N of a batch of folded profiles
SizeType score_and_filter_max_with_cache(std::span<const float> folds,
                                         std::span<float> scores,
                                         std::span<SizeType> indices_filtered,
                                         float threshold,
                                         SizeType nprofiles,
                                         SizeType nbins,
                                         BoxcarWidthsCache& cache);

} // namespace loki::detection
