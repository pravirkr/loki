#include "loki/utils/fft.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

namespace loki::utils {

FFT2D::FFT2D(size_t n1x, size_t n2x, size_t ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_fft_size(ny / 2 + 1),
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

void FFT2D::circular_convolve(std::span<float> n1,
                              std::span<float> n2,
                              std::span<float> out) {
    // Forward FFT
    fftwf_execute_dft_r2c(m_plan_forward, n1.data(), m_n1_fft);
    fftwf_execute_dft_r2c(m_plan_forward, n2.data(), m_n2_fft);
    // Multiply the FFTs
    for (size_t i = 0; i < m_n1x * m_n2x * m_fft_size; ++i) {
        const size_t idx_n1 =
            ((i / (m_n2x * m_fft_size)) * m_fft_size) + (i % m_fft_size);
        const size_t idx_n2 =
            ((i / m_fft_size) % m_n2x * m_fft_size) + (i % m_fft_size);
        m_n1n2_fft[i][0] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][0] -
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][1];
        m_n1n2_fft[i][1] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][1] +
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][0];
    }
    // Inverse FFT
    fftwf_execute_dft_c2r(m_plan_inverse, m_n1n2_fft, out.data());
}

void ensure_fftw_threading(int nthreads) {
    if (nthreads <= 0) {
        nthreads = omp_get_max_threads();
        spdlog::debug("configure_threading: Using max threads: {}", nthreads);
    }
    static std::once_flag init_flag;
    std::call_once(init_flag, [nthreads]() {
        if (fftwf_init_threads() == 0) {
            throw std::runtime_error("Failed to initialize FFTW threads");
        }
        fftwf_plan_with_nthreads(nthreads);
        spdlog::debug("FFTW threads initialized with {} threads", nthreads);
    });
}

void rfft_batch(std::span<float> real_input,
                std::span<ComplexType> complex_output,
                int batch_size,
                int n_real,
                int nthreads) {
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;
    auto* real_ptr      = real_input.data();
    auto* complex_ptr = reinterpret_cast<fftwf_complex*>(complex_output.data());

    fftwf_plan plan = fftwf_plan_many_dft_r2c(
        1,                     // rank
        &n_real,               // transform size
        batch_size,            // number of transforms
        real_ptr,              // input
        nullptr, 1, n_real,    // input layout: stride=1, dist=n_real
        complex_ptr,           // output
        nullptr, 1, n_complex, // output layout: stride=1, dist=n_complex
        FFTW_ESTIMATE          // fast planning
    );
    if (plan == nullptr) {
        throw std::runtime_error("Failed to create RFFT plan");
    }
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    spdlog::debug("RFFT batch completed: {} transforms of size {}", batch_size,
                  n_real);
}

void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads) {
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;
    auto* complex_ptr = reinterpret_cast<fftwf_complex*>(complex_input.data());
    auto* real_ptr    = real_output.data();

    fftwf_plan plan = fftwf_plan_many_dft_c2r(
        1,                     // rank
        &n_real,               // transform size
        batch_size,            // number of transforms
        complex_ptr,           // input
        nullptr, 1, n_complex, // input layout: stride=1, dist=n_complex
        real_ptr,              // output
        nullptr, 1, n_real,    // output layout: stride=1, dist=n_real
        FFTW_ESTIMATE          // fast planning
    );

    if (plan == nullptr) {
        throw std::runtime_error("Failed to create IRFFT plan");
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // FFTW C2R doesn't normalize - apply normalization
    const auto norm          = 1.0F / static_cast<float>(n_real);
    const int total_elements = batch_size * n_real;
    for (int i = 0; i < total_elements; ++i) {
        real_output[i] *= norm;
    }
    spdlog::debug("IRFFT batch completed: {} transforms of size {}", batch_size,
                  n_real);
}

} // namespace loki::utils