#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <loki/fft.hpp>

class MatchedFilter {
public:
  MatchedFilter(const std::vector<size_t> &widths_arr, size_t nprofiles,
                size_t nbins, std::string_view shape = "boxcar");

  std::vector<float> get_templates() const;
  std::size_t get_ntemplates() const;
  std::size_t get_nbins() const;
  void compute(std::span<const float> arr, std::span<float> out);

private:
  const std::vector<size_t> widths_arr;
  const std::size_t nbins;
  const std::size_t nprofiles;
  const std::string_view shape;

  std::size_t nbins_pow2;
  std::size_t ntemplates;
  std::vector<float> templates;
  std::vector<float> arr_padded;
  std::vector<float> snr_arr;

  // FFTW plans
  FFT_2D fft2d;

  std::size_t get_nbins_pow2(std::size_t nbins);
  void initialise_templates();
  void generate_boxcar_template(std::span<float> &arr, size_t width);
  void generate_gaussian_template(std::span<float> &arr, size_t width);
  void normalise(std::span<float> &arr);
};

namespace loki {

// Compute the S/N of single pulse proile
void snr_1d(std::span<const float> arr, std::span<const size_t> widths,
            float stdnoise, std::span<float> out);

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr, const size_t nprofiles,
            std::span<const size_t> widths, float stdnoise,
            std::span<float> out);

} // namespace loki
