#include "fftw3.h"
#include <loki/fft.hpp>

FFTW::FFTW(size_t n, int sign)
    : m_norm(1.0F / static_cast<float>(n)),
      m_sign(sign) {}

FFTW::~FFTW() {
    if (m_plan != nullptr) {
        fftwf_destroy_plan(m_plan);
    }
}

void FFTW::fft(std::span<const std::complex<float>> in,
               std::span<std::complex<float>> out,
               bool normalize) {
    execute(in, out);
    if (normalize) {
        for (size_t i = 0; i < n; ++i) {
            out[i] *= m_norm;
        }
    }
}

void FFTW::setup(std::span<Complex> in, std::span<Complex> out) {
    plan(in, out);
}

FFT1D::FFT1D(size_t nx, int sign, std::span<Complex> in, std::span<Complex> out)
    : FFTW(2 * nx, sign),
      m_nx(nx) {
    setup(in, out);
}

void FFT1D::plan(std::span<Complex> in, std::span<Complex> out) {
    m_plan = fftwf_plan_dft_1d(
        static_cast<int>(m_nx), reinterpret_cast<fftwf_complex*>(in.data()),
        reinterpret_cast<fftwf_complex*>(out.data()), m_sign, FFTW_ESTIMATE);
}

void FFT1D::execute(std::span<Complex> in, std::span<Complex> out) {
    fftwf_execute_dft(m_plan, reinterpret_cast<fftwf_complex*>(in.data()),
                      reinterpret_cast<fftwf_complex*>(out.data()));
}

FFT1DR2C::FFT1DR2C(size_t nx, std::span<float> in, std::span<Complex> out)
    : FFTW(2 * (nx / 2 + 1), -1),
      m_nx(nx) {
    setup(in, out);
}

void FFT1DR2C::plan(std::span<float> in, std::span<Complex> out) {
    m_plan = fftwf_plan_dft_r2c_1d(static_cast<int>(m_nx), in.data(),
                                   reinterpret_cast<fftwf_complex*>(out.data()),
                                   FFTW_ESTIMATE);
}

void FFT1DR2C::execute(std::span<float> in, std::span<Complex> out) {
    fftwf_execute_dft_r2c(m_plan, in.data(),
                          reinterpret_cast<fftwf_complex*>(out.data()));
}

FFT1DC2R::FFT1DC2R(size_t nx, std::span<Complex> in, std::span<float> out)
    : FFTW(2 * (nx / 2 + 1), 1),
      m_nx(nx) {
    setup(in, out);
}

void FFT1DC2R::plan(std::span<Complex> in, std::span<float> out) {
    m_plan = fftwf_plan_dft_c2r_1d(static_cast<int>(m_nx),
                                   reinterpret_cast<fftwf_complex*>(in.data()),
                                   out.data(), FFTW_ESTIMATE);
}

void FFT1DC2R::execute(std::span<Complex> in, std::span<float> out) {
    fftwf_execute_dft_c2r(m_plan, reinterpret_cast<fftwf_complex*>(in.data()),
                          out.data());
}

FFT2D::FFT2D(size_t nx,
             size_t ny,
             int sign,
             std::span<Complex> in,
             std::span<Complex> out)
    : FFTW(2 * nx * ny, sign),
      m_nx(nx),
      m_ny(ny) {
    setup(in, out);
}

void FFT2D::plan(std::span<Complex> in, std::span<Complex> out) {
    m_plan = fftwf_plan_dft_2d(static_cast<int>(m_nx), static_cast<int>(m_ny),
                               reinterpret_cast<fftwf_complex*>(in.data()),
                               reinterpret_cast<fftwf_complex*>(out.data()),
                               m_sign, FFTW_ESTIMATE);
}

void FFT2D::execute(std::span<Complex> in, std::span<Complex> out) {
    fftwf_execute_dft(m_plan, reinterpret_cast<fftwf_complex*>(in.data()),
                      reinterpret_cast<fftwf_complex*>(out.data()));
}

FFT2DR2C::FFT2DR2C(size_t nx,
                   size_t ny,
                   std::span<float> in,
                   std::span<Complex> out)
    : FFTW(2 * nx * (ny / 2 + 1), -1),
      m_nx(nx),
      m_ny(ny) {
    setup(in, out);
}

void FFT2DR2C::plan(std::span<float> in, std::span<Complex> out) {
    m_plan = fftwf_plan_dft_r2c_2d(
        static_cast<int>(m_nx), static_cast<int>(m_ny), in.data(),
        reinterpret_cast<fftwf_complex*>(out.data()), FFTW_ESTIMATE);
}

void FFT2DR2C::execute(std::span<float> in, std::span<Complex> out) {
    fftwf_execute_dft_r2c(m_plan, in.data(),
                          reinterpret_cast<fftwf_complex*>(out.data()));
}

FFT2DC2R::FFT2DC2R(size_t nx,
                   size_t ny,
                   std::span<Complex> in,
                   std::span<float> out)
    : FFTW(2 * nx * (ny / 2 + 1), 1),
      m_nx(nx),
      m_ny(ny) {
    setup(in, out);
}

void FFT2DC2R::plan(std::span<Complex> in, std::span<float> out) {
    m_plan = fftwf_plan_dft_c2r_2d(
        static_cast<int>(m_nx), static_cast<int>(m_ny),
        reinterpret_cast<fftwf_complex*>(in.data()), out.data(), FFTW_ESTIMATE);
}

void FFT2DC2R::execute(std::span<Complex> in, std::span<float> out) {
    fftwf_execute_dft_c2r(m_plan, reinterpret_cast<fftwf_complex*>(in.data()),
                          out.data());
}

FFT2DC::FFT2DC(int n1x, int n2x, int ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_fft_size(ny / 2 + 1),
      m_n1_fft(fftwf_alloc_complex(n1x * m_fft_size)),
      m_n2_fft(fftwf_alloc_complex(n2x * m_fft_size)),
      m_n1n2_fft(fftwf_alloc_complex(n1x * n2x * m_fft_size)),
      m_plan_fw(
          fftwf_plan_dft_r2c_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE)),
      m_plan_bw(
          fftwf_plan_dft_c2r_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE)) {};

FFT2DC::~FFT2DC() {
    fftwf_free(m_n1_fft);
    fftwf_free(m_n2_fft);
    fftwf_free(m_n1n2_fft);
    fftwf_destroy_plan(m_plan_fw);
    fftwf_destroy_plan(m_plan_bw);
}

void FFT2DC::circular_convolve(std::span<float> n1,
                               std::span<float> n2,
                               std::span<float> out) {
    // Forward FFT
    fftwf_execute_dft_r2c(m_plan_fw, n1.data(), m_n1_fft);
    fftwf_execute_dft_r2c(m_plan_bw, n2.data(), m_n2_fft);
    // Multiply the FFTs
    for (size_t i = 0; i < m_n1x * m_n2x * m_fft_size; ++i) {
        const size_t idx_n1 =
            (i / (m_n2x * m_fft_size)) * m_fft_size + (i % m_fft_size);
        const size_t idx_n2 =
            (i / m_fft_size) % m_n2x * m_fft_size + (i % m_fft_size);
        m_n1n2_fft[i][0] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][0] -
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][1];
        m_n1n2_fft[i][1] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][1] +
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][0];
    }
    // Inverse FFT
    fftwf_execute_dft_c2r(m_plan_bw, m_n1n2_fft, out.data());
}
