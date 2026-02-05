#include "loki/utils/fft.hpp"

#include <cstddef>
#include <format>
#include <mutex>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"

namespace loki::utils {

namespace {

// Global FFTW initialization (thread-safe)
std::once_flag g_fftw_init_flag; // NOLINT

void init_fftw_once(int nthreads) {
    std::call_once(g_fftw_init_flag, [nthreads]() {
        if (fftwf_init_threads() == 0) {
            throw std::runtime_error("Failed to initialize FFTW threads");
        }
        fftwf_plan_with_nthreads(nthreads);
        spdlog::debug("FFTW initialized with {} threads", nthreads);
    });
}

// RAII cleanup for all FFTW resources on program exit
struct FFTWGlobalCleanup {
    bool initialized{false};

    FFTWGlobalCleanup() = default;
    ~FFTWGlobalCleanup() {
        // Clean up RfftExecutor plans
        {
            std::lock_guard<std::mutex> lock(RfftExecutor::s_mutex);
            for (auto& [key, plan] : RfftExecutor::s_plan_cache) {
                fftwf_destroy_plan(plan);
            }
            RfftExecutor::s_plan_cache.clear();
            spdlog::debug("Cleaned up {} RfftExecutor plans",
                          RfftExecutor::s_plan_cache.size());
        }

        // Clean up IrfftExecutor plans
        {
            std::lock_guard<std::mutex> lock(IrfftExecutor::s_mutex);
            for (auto& [key, plan] : IrfftExecutor::s_plan_cache) {
                fftwf_destroy_plan(plan);
            }
            IrfftExecutor::s_plan_cache.clear();
            spdlog::debug("Cleaned up {} IrfftExecutor plans",
                          IrfftExecutor::s_plan_cache.size());
        }
        fftwf_cleanup_threads();
        spdlog::debug("FFTW global cleanup completed");
    }

    FFTWGlobalCleanup(const FFTWGlobalCleanup&)            = delete;
    FFTWGlobalCleanup& operator=(const FFTWGlobalCleanup&) = delete;
    FFTWGlobalCleanup(FFTWGlobalCleanup&&)                 = delete;
    FFTWGlobalCleanup& operator=(FFTWGlobalCleanup&&)      = delete;
};

// Static instance ensures cleanup on program exit
FFTWGlobalCleanup g_fftw_cleanup; // NOLINT
} // namespace

void ensure_fftw_threading(int nthreads) {
    if (nthreads <= 0) {
        nthreads = omp_get_max_threads();
        spdlog::debug("configure_threading: Using max threads: {}", nthreads);
    }
    init_fftw_once(nthreads);
}

// --- RfftExecutor Implementation ---

RfftExecutor::RfftExecutor(int n_real, int nthreads, int max_chunk_size)
    : m_n_real(n_real),
      m_n_complex((n_real / 2) + 1),
      m_nthreads(nthreads),
      m_max_chunk_size(max_chunk_size) {
    if (n_real <= 0) {
        throw std::invalid_argument(std::format(
            "RfftExecutor: n_real must be positive, got {}", n_real));
    }
    ensure_fftw_threading(m_nthreads);
    spdlog::debug("RfftExecutor created: n_real={}, max_chunk={}", m_n_real,
                  m_max_chunk_size);
}

void RfftExecutor::execute(std::span<const float> real_input,
                           std::span<ComplexType> complex_output,
                           int batch_size) {
    error_check::check_equal(real_input.size(),
                             static_cast<SizeType>(batch_size * m_n_real),
                             "RfftExecutor: real_input size mismatch");
    error_check::check_equal(complex_output.size(),
                             static_cast<SizeType>(batch_size * m_n_complex),
                             "RfftExecutor: complex_output size mismatch");

    // FFTW API is not const-correct, so cast away const as we
    // promising to preserve the input. (only works for 1D transforms)
    auto* in_ptr  = const_cast<float*>(real_input.data()); // NOLINT
    auto* out_ptr = reinterpret_cast<fftwf_complex*>(complex_output.data());

    // Handle large batches by chunking
    if (batch_size > m_max_chunk_size) {
        int offset          = 0;
        int chunks_executed = 0;

        // Get cached plan for chunk size
        const int chunk_size  = m_max_chunk_size;
        fftwf_plan chunk_plan = get_or_create_plan(chunk_size, in_ptr, out_ptr);

        while (offset < batch_size) {
            const int this_batch = std::min(chunk_size, batch_size - offset);
            if (this_batch == chunk_size) {
                // Use cached chunk plan
                fftwf_execute_dft_r2c(
                    chunk_plan,
                    in_ptr + static_cast<IndexType>(offset * m_n_real),
                    out_ptr + static_cast<IndexType>(offset * m_n_complex));
            } else {
                // Remainder: get or create plan (will be cached for reuse)
                fftwf_plan remainder_plan = get_or_create_plan(
                    this_batch,
                    in_ptr + static_cast<IndexType>(offset * m_n_real),
                    out_ptr + static_cast<IndexType>(offset * m_n_complex));
                fftwf_execute_dft_r2c(
                    remainder_plan,
                    in_ptr + static_cast<IndexType>(offset * m_n_real),
                    out_ptr + static_cast<IndexType>(offset * m_n_complex));
            }

            offset += this_batch;
            chunks_executed++;
        }

        spdlog::debug("RfftExecutor: Completed {} transforms in {} chunks",
                      batch_size, chunks_executed);
    } else {
        // Small batch: get or create plan (will be cached for reuse)
        fftwf_plan plan = get_or_create_plan(batch_size, in_ptr, out_ptr);
        fftwf_execute_dft_r2c(plan, in_ptr, out_ptr);

        spdlog::debug("RfftExecutor: Completed {} transforms", batch_size);
    }
}

fftwf_plan RfftExecutor::get_or_create_plan(int batch_size,
                                            float* in_ptr,
                                            fftwf_complex* out_ptr) {
    const PlanKey key{.n_real = m_n_real, .batch_size = batch_size};

    std::lock_guard<std::mutex> lock(s_mutex);

    auto it = s_plan_cache.find(key);
    if (it != s_plan_cache.end()) {
        spdlog::trace(
            "RfftExecutor: Reusing cached plan for n_real={}, batch={}",
            m_n_real, batch_size);
        return it->second;
    }

    // Create new plan
    fftwf_plan plan = fftwf_plan_many_dft_r2c(
        1,                       // rank
        &m_n_real,               // transform size
        batch_size,              // number of transforms
        in_ptr,                  // input (dummy)
        nullptr, 1, m_n_real,    // input layout: stride=1, dist=n_real
        out_ptr,                 // output (dummy)
        nullptr, 1, m_n_complex, // output layout: stride=1, dist=n_complex
        FFTW_ESTIMATE);          // fast planning

    if (plan == nullptr) {
        throw std::runtime_error(std::format(
            "RfftExecutor: Failed to create plan for n_real={}, batch_size={}",
            m_n_real, batch_size));
    }

    s_plan_cache[key] = plan;
    spdlog::debug(
        "RfftExecutor: Created and cached plan for n_real={}, batch={}",
        m_n_real, batch_size);

    return plan;
}

// --- IrfftExecutor Implementation ---

IrfftExecutor::IrfftExecutor(int n_real, int nthreads, int max_chunk_size)
    : m_n_real(n_real),
      m_n_complex((n_real / 2) + 1),
      m_nthreads(nthreads),
      m_max_chunk_size(max_chunk_size) {
    if (n_real <= 0) {
        throw std::invalid_argument(std::format(
            "IrfftExecutor: n_real must be positive, got {}", n_real));
    }
    ensure_fftw_threading(m_nthreads);
    spdlog::debug("IrfftExecutor created: n_real={}, max_chunk={}", m_n_real,
                  m_max_chunk_size);
}

void IrfftExecutor::execute(std::span<const ComplexType> complex_input,
                            std::span<float> real_output,
                            int batch_size) {
    error_check::check_equal(real_output.size(),
                             static_cast<SizeType>(batch_size * m_n_real),
                             "IrfftExecutor: real_output size mismatch");
    error_check::check_equal(complex_input.size(),
                             static_cast<SizeType>(batch_size * m_n_complex),
                             "IrfftExecutor: complex_input size mismatch");

    // FFTW API is not const-correct, so cast away const as we are
    // promising to preserve the input. (only works for 1D transforms)
    auto* in_ptr = reinterpret_cast<fftwf_complex*>(
        const_cast<ComplexType*>(complex_input.data())); // NOLINT
    auto* out_ptr = real_output.data();

    // Handle large batches by chunking
    if (batch_size > m_max_chunk_size) {
        int offset          = 0;
        int chunks_executed = 0;

        // Get cached plan for chunk size
        const int chunk_size  = m_max_chunk_size;
        fftwf_plan chunk_plan = get_or_create_plan(chunk_size, in_ptr, out_ptr);

        while (offset < batch_size) {
            const int this_batch = std::min(chunk_size, batch_size - offset);
            if (this_batch == chunk_size) {
                // Use cached chunk plan
                fftwf_execute_dft_c2r(
                    chunk_plan,
                    in_ptr + static_cast<IndexType>(offset * m_n_complex),
                    out_ptr + static_cast<IndexType>(offset * m_n_real));
            } else {
                // Remainder: get or create plan (will be cached for reuse)
                fftwf_plan remainder_plan = get_or_create_plan(
                    this_batch,
                    in_ptr + static_cast<IndexType>(offset * m_n_complex),
                    out_ptr + static_cast<IndexType>(offset * m_n_real));
                fftwf_execute_dft_c2r(
                    remainder_plan,
                    in_ptr + static_cast<IndexType>(offset * m_n_complex),
                    out_ptr + static_cast<IndexType>(offset * m_n_real));
            }

            offset += this_batch;
            chunks_executed++;
        }

        spdlog::debug("IrfftExecutor: Completed {} transforms in {} chunks",
                      batch_size, chunks_executed);
    } else {
        // Small batch: get or create plan (will be cached for reuse)
        fftwf_plan plan = get_or_create_plan(batch_size, in_ptr, out_ptr);
        fftwf_execute_dft_c2r(plan, in_ptr, out_ptr);

        spdlog::debug("IrfftExecutor: Completed {} transforms", batch_size);
    }

    // Normalize output (FFTW doesn't normalize C2R)
    const float norm         = 1.0F / static_cast<float>(m_n_real);
    const int total_elements = batch_size * m_n_real;

    for (int i = 0; i < total_elements; ++i) {
        real_output[i] *= norm;
    }
}

fftwf_plan IrfftExecutor::get_or_create_plan(int batch_size,
                                             fftwf_complex* in_ptr,
                                             float* out_ptr) {
    const PlanKey key{.n_real = m_n_real, .batch_size = batch_size};

    std::lock_guard<std::mutex> lock(s_mutex);

    auto it = s_plan_cache.find(key);
    if (it != s_plan_cache.end()) {
        spdlog::trace(
            "IrfftExecutor: Reusing cached plan for n_real={}, batch={}",
            m_n_real, batch_size);
        return it->second;
    }

    // Create new plan
    fftwf_plan plan = fftwf_plan_many_dft_c2r(
        1,                       // rank
        &m_n_real,               // transform size
        batch_size,              // number of transforms
        in_ptr,                  // input (dummy)
        nullptr, 1, m_n_complex, // input layout: stride=1, dist=n_complex
        out_ptr,                 // output (dummy)
        nullptr, 1, m_n_real,    // output layout: stride=1, dist=n_real
        FFTW_ESTIMATE            // fast planning
    );

    if (plan == nullptr) {
        throw std::runtime_error(std::format(
            "IrfftExecutor: Failed to create plan for n_real={}, batch_size={}",
            m_n_real, batch_size));
    }

    s_plan_cache[key] = plan;
    spdlog::debug(
        "IrfftExecutor: Created and cached plan for n_real={}, batch={}",
        m_n_real, batch_size);

    return plan;
}

// --- FFT2D Implementation ---
FFT2D::FFT2D(SizeType n1x, SizeType n2x, SizeType ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_fft_size((m_ny / 2) + 1),
      m_n1_fft(fftwf_alloc_complex(n1x * m_fft_size)),
      m_n2_fft(fftwf_alloc_complex(n2x * m_fft_size)),
      m_n1n2_fft(fftwf_alloc_complex(n1x * n2x * m_fft_size)),
      m_plan_forward(fftwf_plan_dft_r2c_2d(static_cast<int>(n1x),
                                           static_cast<int>(m_ny),
                                           nullptr,
                                           nullptr,
                                           FFTW_ESTIMATE)),
      m_plan_inverse(fftwf_plan_dft_c2r_2d(static_cast<int>(n1x),
                                           static_cast<int>(m_ny),
                                           nullptr,
                                           nullptr,
                                           FFTW_ESTIMATE)) {

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
        m_n1n2_fft[i][0] = (m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][0]) -
                           (m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][1]);
        m_n1n2_fft[i][1] = (m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][1]) +
                           (m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][0]);
    }
    // Inverse FFT
    fftwf_execute_dft_c2r(m_plan_inverse, m_n1n2_fft, out.data());
}

// --- Convenience functions (thin wrappers around executors) ---
void rfft_batch(std::span<const float> real_input,
                std::span<ComplexType> complex_output,
                int batch_size,
                int n_real,
                int nthreads) {
    RfftExecutor executor(n_real, nthreads);
    executor.execute(real_input, complex_output, batch_size);
}

void irfft_batch(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads) {
    IrfftExecutor executor(n_real, nthreads);
    executor.execute(complex_input, real_output, batch_size);
}

} // namespace loki::utils