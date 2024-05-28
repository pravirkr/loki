#pragma once

#include <cstddef>
#include <fftw3.h>
#include <span>

class FFT2D {
public:
    FFT2D(size_t n1x, size_t n2x, size_t ny);
    FFT2D(const FFT2D&)            = delete;
    FFT2D& operator=(const FFT2D&) = delete;
    FFT2D(FFT2D&&)                 = delete;
    FFT2D& operator=(FFT2D&&)      = delete;
    ~FFT2D();

    void circular_convolve(std::span<float> n1, std::span<float> n2,
                           std::span<float> out);

private:
    size_t m_n1x;
    size_t m_n2x;
    size_t m_ny;

    size_t m_fft_size;
    fftwf_complex* m_n1_fft;
    fftwf_complex* m_n2_fft;
    fftwf_complex* m_n1n2_fft;
    fftwf_plan m_plan_forward;
    fftwf_plan m_plan_inverse;
};