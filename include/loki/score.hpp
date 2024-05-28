#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <loki/fft.hpp>

class MatchedFilter {
public:
    MatchedFilter(const std::vector<size_t>& widths_arr, size_t nprofiles,
                  size_t nbins, std::string_view shape = "boxcar");

    std::vector<float> get_templates() const;
    std::size_t get_ntemplates() const;
    std::size_t get_nbins() const;
    void compute(std::span<const float> arr, std::span<float> out);

private:
    std::vector<size_t> m_widths_arr;
    std::size_t m_nbins;
    std::size_t m_nprofiles;
    std::string_view m_shape;

    std::size_t m_nbins_pow2;
    std::size_t m_ntemplates;
    std::vector<float> m_templates;
    std::vector<float> m_arr_padded;
    std::vector<float> m_snr_arr;

    // FFTW plans
    FFT2D m_fft2d;

    std::size_t get_nbins_pow2(std::size_t nbins);
    void initialise_templates();
    void generate_boxcar_template(std::span<float>& arr, size_t width);
    void generate_gaussian_template(std::span<float>& arr, size_t width);
    void normalise(std::span<float>& arr);
};

namespace loki {

// Compute the S/N of single pulse proile
void snr_1d(std::span<const float> arr, std::span<const size_t> widths,
            float stdnoise, std::span<float> out);

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr, size_t nprofiles,
            std::span<const size_t> widths, float stdnoise,
            std::span<float> out);

}  // namespace loki
