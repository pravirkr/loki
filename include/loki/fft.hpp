#pragma once

#include <cstddef>
#include <fftw3.h>
#include <span>

class FFT_2D {
public:
  FFT_2D(size_t n1x, size_t n2x, size_t ny);

  ~FFT_2D();

  void circular_convolve(std::span<float> n1, std::span<float> n2,
                         std::span<float> out);

private:
  const size_t n1x;
  const size_t n2x;
  const size_t ny;

  size_t fft_size;
  fftwf_complex *n1_fft;
  fftwf_complex *n2_fft;
  fftwf_complex *n1n2_fft;
  fftwf_plan plan_forward;
  fftwf_plan plan_inverse;
};