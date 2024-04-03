#pragma once

#include <cstddef>
#include <span>
#include <fftw3.h>

class FFT_2D {
public:
    FFT_2D(size_t n1x, size_t n2x, size_t ny) : n1x(n1x), n2x(n2x), ny(ny) {
        fft_size = ny / 2 + 1;
        n1_fft   = fftwf_alloc_complex(n1x * fft_size);
        n2_fft   = fftwf_alloc_complex(n2x * fft_size);
        n1n2_fft = fftwf_alloc_complex(n1x * n2x * fft_size);
        plan_forward
            = fftwf_plan_dft_r2c_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE);
        plan_inverse
            = fftwf_plan_dft_c2r_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE);
    };

    ~FFT_2D() {
        fftwf_free(n1_fft);
        fftwf_free(n2_fft);
        fftwf_free(n1n2_fft);
        fftwf_destroy_plan(plan_forward);
        fftwf_destroy_plan(plan_inverse);
    }

    void circular_convolve(std::span<float> n1, std::span<float> n2,
                           std::span<float> out) {
        // Forward FFT
        fftwf_execute_dft_r2c(plan_forward, n1.data(), n1_fft);
        fftwf_execute_dft_r2c(plan_forward, n2.data(), n2_fft);
        // Multiply the FFTs
        for (size_t i = 0; i < n1x * n2x * fft_size; ++i) {
            const size_t idx_n1
                = (i / (n2x * fft_size)) * fft_size + (i % fft_size);
            const size_t idx_n2
                = (i / fft_size) % n2x * fft_size + (i % fft_size);
            n1n2_fft[i][0] = n1_fft[idx_n1][0] * n2_fft[idx_n2][0]
                             - n1_fft[idx_n1][1] * n2_fft[idx_n2][1];
            n1n2_fft[i][1] = n1_fft[idx_n1][0] * n2_fft[idx_n2][1]
                             + n1_fft[idx_n1][1] * n2_fft[idx_n2][0];
        }
        // Inverse FFT
        fftwf_execute_dft_c2r(plan_inverse, n1n2_fft, out.data());
    }

private:
    const size_t n1x;
    const size_t n2x;
    const size_t ny;

    size_t fft_size;
    fftwf_complex* n1_fft;
    fftwf_complex* n2_fft;
    fftwf_complex* n1n2_fft;
    fftwf_plan plan_forward;
    fftwf_plan plan_inverse;
};