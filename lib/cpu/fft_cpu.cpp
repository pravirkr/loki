/**
 * @file fft_cpu.cpp
 * @brief Host FFTs: the lane FFT for batched short rows, pocketfft for long
 * rows. The only library file that includes pocketfft.
 */

#include <algorithm>
#include <complex>
#include <format>
#include <limits>
#include <span>
#include <string_view>

#include <omp.h>
#include <pocketfft_hdronly.h>

#include "loki/common/types.hpp"

#include "lib/cpu/lane_fft.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/utils/fft_impl.hpp"

namespace loki::math {

namespace {

bool use_lanes(SizeType n_real) noexcept {
    return lane_fft::EVTransform::supports(n_real) && n_real <= kLaneFFTMaxSize;
}

void check_batch(SizeType input_size,
                 SizeType output_size,
                 SizeType batch_size,
                 SizeType n_in,
                 SizeType n_out,
                 std::string_view what) {
    error_check::check(
        n_in == 0 || batch_size <= std::numeric_limits<SizeType>::max() / n_in,
        std::format("{}: batch_size * row length overflows SizeType", what));
    error_check::check_equal(input_size, batch_size * n_in,
                             std::format("{}: input size does not match the "
                                         "batch size",
                                         what));
    error_check::check_equal(output_size, batch_size * n_out,
                             std::format("{}: output size does not match the "
                                         "batch size",
                                         what));
}

/// Calls fn(first_row, nrows) on contiguous row blocks, one per thread.
template <typename Fn>
void for_row_blocks(SizeType batch_size, int nthreads, Fn&& fn) {
    const auto nblocks = std::min<SizeType>(
        static_cast<SizeType>(std::max(nthreads, 1)), batch_size);
#pragma omp parallel for num_threads(static_cast<int>(nblocks))                \
    schedule(static) if (nblocks > 1)
    for (SizeType b = 0; b < nblocks; ++b) {
        const SizeType begin = (b * batch_size) / nblocks;
        const SizeType end   = ((b + 1) * batch_size) / nblocks;
        fn(begin, end - begin);
    }
}

void pocket_rfft(const float* in,
                 ComplexType* out,
                 SizeType nrows,
                 SizeType n_real) {
    const SizeType n_complex = (n_real / 2) + 1;
    const pocketfft::shape_t shape{nrows, n_real};
    const pocketfft::stride_t stride_in{
        static_cast<std::ptrdiff_t>(n_real * sizeof(float)), sizeof(float)};
    const pocketfft::stride_t stride_out{
        static_cast<std::ptrdiff_t>(n_complex * sizeof(ComplexType)),
        sizeof(ComplexType)};
    pocketfft::r2c(shape, stride_in, stride_out, /*axis=*/1, pocketfft::FORWARD,
                   in, out, 1.0F);
}

void pocket_irfft(const ComplexType* in,
                  float* out,
                  SizeType nrows,
                  SizeType n_real) {
    const SizeType n_complex = (n_real / 2) + 1;
    const pocketfft::shape_t shape{nrows, n_real};
    const pocketfft::stride_t stride_in{
        static_cast<std::ptrdiff_t>(n_complex * sizeof(ComplexType)),
        sizeof(ComplexType)};
    const pocketfft::stride_t stride_out{
        static_cast<std::ptrdiff_t>(n_real * sizeof(float)), sizeof(float)};
    pocketfft::c2r(shape, stride_in, stride_out, /*axis=*/1,
                   pocketfft::BACKWARD, in, out,
                   1.0F / static_cast<float>(n_real));
}

} // namespace

// --- CpuFFT ---

CpuFFT::CpuFFT(CpuFFT&& other) noexcept
    : m_registered(std::move(other.m_registered)),
      m_built(std::move(other.m_built)) {}

CpuFFT& CpuFFT::operator=(CpuFFT&& other) noexcept {
    m_registered = std::move(other.m_registered);
    m_built      = std::move(other.m_built);
    return *this;
}

void CpuFFT::prepare_plans(std::span<const SizeType> n_reals) {
    for (const SizeType n_real : n_reals) {
        error_check::check_greater(n_real, SizeType{0},
                                   "CpuFFT::prepare_plans: n_real must be "
                                   "positive");
        if (use_lanes(n_real)) {
            lane_fft::EVTransform::prepare(n_real);
        }
        const std::scoped_lock lock(m_mutex);
        m_registered.insert(n_real);
        m_built.insert(n_real);
    }
}

void CpuFFT::prepare_exact_plans(std::span<const SizeType> n_reals) {
    const std::scoped_lock lock(m_mutex);
    for (const SizeType n_real : n_reals) {
        error_check::check_greater(n_real, SizeType{0},
                                   "CpuFFT::prepare_exact_plans: n_real must "
                                   "be positive");
        m_registered.insert(n_real);
    }
}

bool CpuFFT::has_prepared(SizeType n_real) const {
    const std::scoped_lock lock(m_mutex);
    return m_registered.contains(n_real);
}

SizeType CpuFFT::n_cached_plans() const {
    const std::scoped_lock lock(m_mutex);
    return m_built.size();
}

void CpuFFT::note_built(SizeType n_real) {
    const std::scoped_lock lock(m_mutex);
    m_built.insert(n_real);
}

void CpuFFT::rfft_batch(std::span<const float> real_input,
                        std::span<ComplexType> complex_output,
                        SizeType batch_size,
                        SizeType n_real,
                        int nthreads) {
    error_check::check_greater(n_real, SizeType{0},
                               "CpuFFT::rfft_batch: n_real must be positive");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch(real_input.size(), complex_output.size(), batch_size, n_real,
                n_complex, "CpuFFT::rfft_batch");
    if (batch_size == 0) {
        return;
    }
    note_built(n_real);
    if (use_lanes(n_real)) {
        lane_fft::forward_batch(real_input.data(), complex_output.data(),
                                batch_size, n_real, nthreads);
        return;
    }
    for_row_blocks(batch_size, nthreads, [&](SizeType row, SizeType nrows) {
        pocket_rfft(real_input.data() + (row * n_real),
                    complex_output.data() + (row * n_complex), nrows, n_real);
    });
}

void CpuFFT::irfft_batch(std::span<const ComplexType> complex_input,
                         std::span<float> real_output,
                         SizeType batch_size,
                         SizeType n_real,
                         int nthreads) {
    error_check::check_greater(n_real, SizeType{0},
                               "CpuFFT::irfft_batch: n_real must be positive");
    const SizeType n_complex = (n_real / 2) + 1;
    check_batch(complex_input.size(), real_output.size(), batch_size, n_complex,
                n_real, "CpuFFT::irfft_batch");
    if (batch_size == 0) {
        return;
    }
    note_built(n_real);
    if (use_lanes(n_real)) {
        lane_fft::inverse_batch(complex_input.data(), real_output.data(),
                                batch_size, n_real, nthreads);
        return;
    }
    for_row_blocks(batch_size, nthreads, [&](SizeType row, SizeType nrows) {
        pocket_irfft(complex_input.data() + (row * n_complex),
                     real_output.data() + (row * n_real), nrows, n_real);
    });
}

void rfft_batch(std::span<const float> real_input,
                std::span<ComplexType> complex_output,
                SizeType batch_size,
                SizeType n_real,
                int nthreads) {
    CpuFFT().rfft_batch(real_input, complex_output, batch_size, n_real,
                        nthreads);
}

void irfft_batch(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 SizeType batch_size,
                 SizeType n_real,
                 int nthreads) {
    CpuFFT().irfft_batch(complex_input, real_output, batch_size, n_real,
                         nthreads);
}

// --- CircularConvolver ---

CircularConvolver::CircularConvolver(SizeType n1x, SizeType n2x, SizeType ny)
    : m_n1x(n1x),
      m_n2x(n2x),
      m_ny(ny),
      m_n1_fft(n1x * ((ny / 2) + 1)),
      m_n2_fft(n2x * ((ny / 2) + 1)),
      m_product(n1x * n2x * ((ny / 2) + 1)) {
    error_check::check_greater(ny, SizeType{0},
                               "CircularConvolver: ny must be positive");
}

void CircularConvolver::convolve(std::span<const float> n1,
                                 std::span<const float> n2,
                                 std::span<float> out) {
    error_check::check_equal(n1.size(), m_n1x * m_ny,
                             "CircularConvolver: n1 size mismatch");
    error_check::check_equal(n2.size(), m_n2x * m_ny,
                             "CircularConvolver: n2 size mismatch");
    error_check::check_equal(out.size(), m_n1x * m_n2x * m_ny,
                             "CircularConvolver: out size mismatch");
    const SizeType nf = (m_ny / 2) + 1;
    rfft_batch(n1, m_n1_fft, m_n1x, m_ny);
    rfft_batch(n2, m_n2_fft, m_n2x, m_ny);
    for (SizeType i = 0; i < m_n1x; ++i) {
        for (SizeType j = 0; j < m_n2x; ++j) {
            const ComplexType* a = m_n1_fft.data() + (i * nf);
            const ComplexType* b = m_n2_fft.data() + (j * nf);
            ComplexType* c       = m_product.data() + (((i * m_n2x) + j) * nf);
            for (SizeType k = 0; k < nf; ++k) {
                c[k] = a[k] * b[k];
            }
        }
    }
    irfft_batch(m_product, out, m_n1x * m_n2x, m_ny);
}

} // namespace loki::math
