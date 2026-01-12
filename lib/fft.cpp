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

constexpr int FFT_BATCH_SIZE_MAX = 16384; // safe, cache-friendly
struct PlanKey {
    int n_real;
    int chunk;

    bool operator<(const PlanKey& other) const {
        return std::tie(n_real, chunk) < std::tie(other.n_real, other.chunk);
    }
};

std::map<PlanKey, fftwf_plan> g_rfft_plan_cache;
std::map<PlanKey, fftwf_plan> g_irfft_plan_cache;
std::mutex g_plan_cache_mutex;

struct FFTWManager {
    std::unordered_map<int, fftwf_plan> plan_cache;
    std::mutex mutex;
    bool initialized{false};

    FFTWManager() = default;
    ~FFTWManager() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto const& [key, val] : plan_cache) {
            fftwf_destroy_plan(val);
        }
        if (initialized) {
            fftwf_cleanup_threads();
        }
    }

    FFTWManager(const FFTWManager&)            = delete;
    FFTWManager& operator=(const FFTWManager&) = delete;
    FFTWManager(FFTWManager&&)                 = delete;
    FFTWManager& operator=(FFTWManager&&)      = delete;
};

// static instance of FFTWManager
FFTWManager g_fftw_manager;
} // namespace

FFT2D::FFT2D(SizeType n1x, SizeType n2x, SizeType ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_fft_size((ny / 2) + 1),
      m_n1_fft(fftwf_alloc_complex(n1x * m_fft_size)),
      m_n2_fft(fftwf_alloc_complex(n2x * m_fft_size)),
      m_n1n2_fft(fftwf_alloc_complex(n1x * n2x * m_fft_size)),
      m_plan_forward(fftwf_plan_dft_r2c_2d(static_cast<int>(n1x),
                                           static_cast<int>(ny),
                                           nullptr,
                                           nullptr,
                                           FFTW_ESTIMATE)),
      m_plan_inverse(fftwf_plan_dft_c2r_2d(static_cast<int>(n1x),
                                           static_cast<int>(ny),
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
        m_n1n2_fft[i][0] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][0] -
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][1];
        m_n1n2_fft[i][1] = m_n1_fft[idx_n1][0] * m_n2_fft[idx_n2][1] +
                           m_n1_fft[idx_n1][1] * m_n2_fft[idx_n2][0];
    }
    // Inverse FFT
    fftwf_execute_dft_c2r(m_plan_inverse, m_n1n2_fft, out.data());
}

// --- IrfftExecutor ---

IrfftExecutor::IrfftExecutor(int n_real)
    : m_n_real(n_real),
      m_n_complex((n_real / 2) + 1) {
    std::lock_guard<std::mutex> lock(g_fftw_manager.mutex);
    if (!g_fftw_manager.initialized) {
        if (fftwf_init_threads() == 0) {
            throw std::runtime_error("Failed to initialize FFTW threads");
        }
        fftwf_plan_with_nthreads(1);
        g_fftw_manager.initialized = true;
    }
}

void IrfftExecutor::execute(std::span<const ComplexType> complex_input,
                            std::span<float> real_output,
                            int batch_size) {
    error_check::check_equal(
        real_output.size(), static_cast<SizeType>(batch_size * m_n_real),
        "IrfftExecutor: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), static_cast<SizeType>(batch_size * m_n_complex),
        "IrfftExecutor: complex_input size does not match batch size");

    auto* plan = get_plan(batch_size);

    // FFTW API is not const-correct, so cast away const as we are
    // promising to preserve the input. (only works for 1D transforms)
    auto* complex_ptr = reinterpret_cast<fftwf_complex*>(
        const_cast<ComplexType*>(complex_input.data())); // NOLINT
    auto* real_ptr = real_output.data();

    fftwf_execute_dft_c2r(plan, complex_ptr, real_ptr);

    // FFTW C2R doesn't normalize - apply normalization
    const auto norm          = 1.0F / static_cast<float>(m_n_real);
    const int total_elements = batch_size * m_n_real;
    for (int i = 0; i < total_elements; ++i) {
        real_output[i] *= norm;
    }

    spdlog::debug("IrfftExecutor: completed: {} transforms of size {}",
                  batch_size, m_n_real);
}

fftwf_plan IrfftExecutor::get_plan(int batch_size) {
    std::lock_guard<std::mutex> lock(s_mutex);
    auto it = s_plan_cache.find(batch_size);
    if (it != s_plan_cache.end()) {
        return it->second;
    }

    const unsigned flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
    fftwf_plan plan      = fftwf_plan_many_dft_c2r(
        1,                       // rank
        &m_n_real,               // transform size
        batch_size,              // number of transforms
        nullptr,                 // input (dummy)
        nullptr, 1, m_n_complex, // input layout: stride=1, dist=n_complex
        nullptr,                 // output (dummy)
        nullptr, 1, m_n_real,    // output layout: stride=1, dist=n_real
        flags                    // fast planning
    );

    if (plan == nullptr) {
        throw std::runtime_error(std::format(
            "Failed to create IRFFT plan for batch_size={}", batch_size));
    }

    s_plan_cache[batch_size] = plan;
    return plan;
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
    if (batch_size > FFT_BATCH_SIZE_MAX) {
        rfft_batch_chunked(real_input, complex_output, batch_size, n_real,
                           nthreads);
        return;
    }
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;

    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "RFFT batch: real_input size does not match batch size");
    error_check::check_equal(
        complex_output.size(), batch_size * n_complex,
        "RFFT batch: complex_output size does not match batch size");

    auto* real_ptr    = real_input.data();
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

void rfft_batch_chunked(std::span<float> real_input,
                        std::span<ComplexType> complex_output,
                        int batch_size,
                        int n_real,
                        int nthreads) {
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;
    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "RFFT batch: real_input size does not match batch size");
    error_check::check_equal(
        complex_output.size(), batch_size * n_complex,
        "RFFT batch: complex_output size does not match batch size");

    float* in_ptr = real_input.data();
    auto* out_ptr = reinterpret_cast<fftwf_complex*>(complex_output.data());
    // Determine chunk size - use smaller chunks for large batches
    const int chunk_size = std::min(FFT_BATCH_SIZE_MAX, batch_size);
    const PlanKey key{.n_real = n_real, .chunk = chunk_size};

    // Get or create cached plan (thread-safe)
    fftwf_plan plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_plan_cache_mutex);
        auto it = g_rfft_plan_cache.find(key);
        if (it != g_rfft_plan_cache.end()) {
            plan = it->second;
        } else {
            // Create plan with nullptr (FFTW will not use actual data for
            // ESTIMATE)
            plan =
                fftwf_plan_many_dft_r2c(1,           // rank
                                        &key.n_real, // transform size
                                        key.chunk,   // number of transforms
                                        nullptr, // input (dummy for planning)
                                        nullptr, 1, n_real, // input layout
                                        nullptr, // output (dummy for planning)
                                        nullptr, 1, n_complex, // output layout
                                        FFTW_ESTIMATE          // fast planning
                );

            if (plan == nullptr) {
                throw std::runtime_error(std::format(
                    "Failed to create RFFT plan for n_real={}, chunk={}",
                    n_real, chunk_size));
            }

            g_rfft_plan_cache[key] = plan;
            spdlog::debug(
                "RFFT: Created and cached plan for n_real={}, chunk={}", n_real,
                chunk_size);
        }
    }
    // Execute in chunks using the cached plan
    int offset          = 0;
    int chunks_executed = 0;
    while (offset < batch_size) {
        const int this_batch = std::min(chunk_size, batch_size - offset);

        // For the last chunk, if size differs, we need a different plan
        if (this_batch != chunk_size) {
            // Small remainder - create one-time plan
            fftwf_plan remainder_plan = fftwf_plan_many_dft_r2c(
                1, &n_real, this_batch, nullptr, nullptr, 1, n_real, nullptr,
                nullptr, 1, n_complex, FFTW_ESTIMATE);

            if (remainder_plan == nullptr) {
                throw std::runtime_error(
                    "Failed to create RFFT plan for remainder");
            }

            fftwf_execute_dft_r2c(remainder_plan, in_ptr + offset * n_real,
                                  out_ptr + offset * n_complex);
            fftwf_destroy_plan(remainder_plan);
        } else {
            fftwf_execute_dft_r2c(plan, in_ptr + offset * n_real,
                                  out_ptr + offset * n_complex);
        }

        offset += this_batch;
        chunks_executed++;
    }
    spdlog::debug("RFFT batch completed: {} transforms of size {} in {} chunks",
                  batch_size, n_real, chunks_executed);
}

void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads) {
    if (batch_size > FFT_BATCH_SIZE_MAX) {
        irfft_batch_chunked(complex_input, real_output, batch_size, n_real,
                            nthreads);
        return;
    }
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;

    error_check::check_equal(
        real_output.size(), batch_size * n_real,
        "IRFFT batch: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * n_complex,
        "IRFFT batch: complex_input size does not match batch size");

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

void irfft_batch_chunked(std::span<ComplexType> complex_input,
                         std::span<float> real_output,
                         int batch_size,
                         int n_real,
                         int nthreads) {
    ensure_fftw_threading(nthreads);
    const int n_complex = (n_real / 2) + 1;

    error_check::check_equal(
        real_output.size(), batch_size * n_real,
        "IRFFT batch: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * n_complex,
        "IRFFT batch: complex_input size does not match batch size");

    auto* in_ptr   = reinterpret_cast<fftwf_complex*>(complex_input.data());
    float* out_ptr = real_output.data();

    // Determine chunk size
    const int chunk_size = std::min(FFT_BATCH_SIZE_MAX, batch_size);
    const PlanKey key{n_real, chunk_size};

    // Get or create cached plan (thread-safe)
    fftwf_plan plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_plan_cache_mutex);
        auto it = g_irfft_plan_cache.find(key);
        if (it != g_irfft_plan_cache.end()) {
            plan = it->second;
        } else {
            plan =
                fftwf_plan_many_dft_c2r(1,           // rank
                                        &key.n_real, // transform size
                                        key.chunk,   // number of transforms
                                        nullptr,     // input (dummy)
                                        nullptr, 1, n_complex, // input layout
                                        nullptr,               // output (dummy)
                                        nullptr, 1, n_real,    // output layout
                                        FFTW_ESTIMATE          // fast planning
                );

            if (plan == nullptr) {
                throw std::runtime_error(std::format(
                    "Failed to create IRFFT plan for n_real={}, chunk={}",
                    n_real, chunk_size));
            }

            g_irfft_plan_cache[key] = plan;
            spdlog::debug(
                "IRFFT: Created and cached plan for n_real={}, chunk={}",
                n_real, chunk_size);
        }
    }

    // Execute in chunks
    int offset          = 0;
    int chunks_executed = 0;
    while (offset < batch_size) {
        const int this_batch = std::min(chunk_size, batch_size - offset);

        // For the last chunk, if size differs, create one-time plan
        if (this_batch != chunk_size) {
            fftwf_plan remainder_plan = fftwf_plan_many_dft_c2r(
                1, &n_real, this_batch, nullptr, nullptr, 1, n_complex, nullptr,
                nullptr, 1, n_real, FFTW_ESTIMATE);

            if (remainder_plan == nullptr) {
                throw std::runtime_error(
                    "Failed to create IRFFT plan for remainder");
            }

            fftwf_execute_dft_c2r(remainder_plan, in_ptr + offset * n_complex,
                                  out_ptr + offset * n_real);
            fftwf_destroy_plan(remainder_plan);
        } else {
            fftwf_execute_dft_c2r(plan, in_ptr + offset * n_complex,
                                  out_ptr + offset * n_real);
        }

        offset += this_batch;
        chunks_executed++;
    }

    // Normalize output (FFTW doesn't normalize inverse transforms)
    const float norm         = 1.0F / static_cast<float>(n_real);
    const int total_elements = batch_size * n_real;
    for (int i = 0; i < total_elements; ++i) {
        real_output[i] *= norm;
    }

    spdlog::debug(
        "IRFFT batch completed: {} transforms of size {} in {} chunks",
        batch_size, n_real, chunks_executed);
}

} // namespace loki::utils