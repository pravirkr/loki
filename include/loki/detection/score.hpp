#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

namespace loki::detection {

class MatchedFilter {
public:
    MatchedFilter(std::span<const SizeType> widths_arr,
                  SizeType nprofiles,
                  SizeType nbins,
                  std::string_view shape = "boxcar");

    std::vector<float> get_templates() const;
    SizeType get_ntemplates() const;
    SizeType get_nbins() const;
    void compute(std::span<const float> arr, std::span<float> out);

private:
    std::vector<SizeType> m_widths_arr;
    SizeType m_nprofiles;
    SizeType m_nbins;
    std::string_view m_shape;

    SizeType m_nbins_pow2;
    SizeType m_ntemplates;
    std::vector<float> m_templates;
    std::vector<float> m_arr_padded;
    std::vector<float> m_snr_arr;

    // FFTW plans
    utils::FFT2D m_fft2d;

    void initialise_templates();
    static void generate_boxcar_template(std::span<float>& arr, SizeType width);
    static void generate_gaussian_template(std::span<float>& arr,
                                           SizeType width);
    static void normalise(std::span<float>& arr);
    static SizeType get_nbins_pow2(SizeType nbins);
};

std::vector<SizeType> generate_width_trials(SizeType nbins_max,
                                            float spacing_factor = 1.5F);

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
