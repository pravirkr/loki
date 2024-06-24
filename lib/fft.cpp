#include <loki/fft.hpp>

FFT2D::FFT2D(size_t n1x, size_t n2x, size_t ny)
    : m_n1x(n1x), m_n2x(n2x), m_ny(ny), m_fft_size(ny / 2 + 1),
      m_n1_fft(fftwf_alloc_complex(n1x * m_fft_size)),
      m_n2_fft(fftwf_alloc_complex(n2x * m_fft_size)),
      m_n1n2_fft(fftwf_alloc_complex(n1x * n2x * m_fft_size)),
      m_plan_forward(
          fftwf_plan_dft_r2c_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE)),
      m_plan_inverse(
          fftwf_plan_dft_c2r_2d(n1x, ny, nullptr, nullptr, FFTW_ESTIMATE)) {

      };

FFT2D::~FFT2D() {
  fftwf_free(m_n1_fft);
  fftwf_free(m_n2_fft);
  fftwf_free(m_n1n2_fft);
  fftwf_destroy_plan(m_plan_forward);
  fftwf_destroy_plan(m_plan_inverse);
}

void FFT2D::circular_convolve(std::span<float> n1, std::span<float> n2,
                              std::span<float> out) {
  // Forward FFT
  fftwf_execute_dft_r2c(m_plan_forward, n1.data(), m_n1_fft);
  fftwf_execute_dft_r2c(m_plan_forward, n2.data(), m_n2_fft);
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
  fftwf_execute_dft_c2r(m_plan_inverse, m_n1n2_fft, out.data());
}