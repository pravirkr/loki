#pragma once

/**
 * @file lane_fft.hpp
 * @brief Batched real FFTs of (e, v) profile pairs in lane layout. Internal.
 *
 * Up to kLanes profiles are transformed together, one profile per SIMD lane:
 * a profile's spectra (or samples) are interleaved across lanes as
 * `[point][kLanes]`, so every butterfly is a plain vertical SIMD operation.
 * The two real signals of a profile ride in one complex transform: z = e + iv.
 * The inverse leaves e in the real part and v in the imaginary part of its
 * result, already in the lane layout that `BoxcarScorer3D::score_lanes`
 * reads.
 */

#include <memory>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"

#include "lib/detail/utils.hpp"

namespace loki::lane_fft {

/// Profiles per transform call (one per lane).
inline constexpr SizeType kLanes = 16;

/**
 * @brief Forward and inverse real FFTs of kLanes (e, v) profiles at a time.
 *
 * Profiles hold an e and a v signal of n real samples, or of n/2 + 1 complex
 * bins: rows `[profile][e, v][n]` (samples) or `[profile][e, v][n/2 + 1]`
 * (spectra). The inverse applies the 1/n normalisation and, like a c2r
 * transform, ignores the imaginary parts of the DC and (even n) Nyquist bins.
 * One object per thread: it owns the scratch of its transforms.
 *
 * Any n >= 2: a mixed-radix Stockham FFT (radices 4, 2, 3, 5, 7, 11, 13), or
 * Bluestein's algorithm on such an FFT when n has a larger prime factor.
 */
class EVTransform {
public:
    explicit EVTransform(SizeType n);

    [[nodiscard]] static bool supports(SizeType n) noexcept;
    /// Build and cache the tables of size n ahead of the first transform.
    static void prepare(SizeType n);
    [[nodiscard]] SizeType size() const noexcept { return m_n; }

    /// Inverse transform of nprofiles <= kLanes spectra pairs. The samples
    /// are left in e_lanes() / v_lanes() as [n][kLanes]; lanes past
    /// nprofiles hold zeros.
    void inverse(const ComplexType* spectra, SizeType nprofiles) noexcept;
    [[nodiscard]] const float* e_lanes() const noexcept { return m_res_re; }
    [[nodiscard]] const float* v_lanes() const noexcept { return m_res_im; }

    /// Inverse transform of nrows <= 2 kLanes spectra rows `[nrows][n/2 + 1]`
    /// into sample rows `[nrows][n]`. Rows pair up as (e, v) = (2p, 2p + 1);
    /// an odd last row is paired with zeros.
    void inverse_rows(const ComplexType* spectra,
                      SizeType nrows,
                      float* samples) noexcept;

    /// Forward transform of nrows <= 2 kLanes sample rows `[nrows][n]` into
    /// spectra rows `[nrows][n/2 + 1]`, paired as in inverse_rows().
    void forward_rows(const float* samples,
                      SizeType nrows,
                      ComplexType* spectra) noexcept;

private:
    struct Plan; // immutable stages, twiddles and Bluestein tables; shared

    SizeType m_n;
    std::shared_ptr<const Plan> m_plan;
    // Four [max(n + 2, FFT size)][kLanes] buffers: the transform's ping-pong
    // buffers, which also hold the spectra in lane layout before packing /
    // after unpacking.
    utils::UninitVector<float> m_scratch;
    std::vector<float> m_zeros; // stands in for absent profiles
    float* m_z_re;
    float* m_z_im;
    float* m_y_re;
    float* m_y_im;
    const float* m_res_re{nullptr};
    const float* m_res_im{nullptr};

    [[nodiscard]] static std::shared_ptr<const Plan> get_plan(SizeType n);
    /// Transposes and packs nrows spectra rows, then inverts them.
    void inverse_impl(const ComplexType* spectra, SizeType nrows) noexcept;
    /// The n-point FFT of z (+i: inverse); m_res_* point at the result.
    template <bool Inverse> void transform() noexcept;
    /// The plan's FFT-size transform from buffer (re, im); returns the buffer
    /// pair holding the result, one of the four scratch buffers.
    template <bool Inverse>
    std::pair<float*, float*> run_stages(float* re, float* im) noexcept;
};

/// Inverse transforms of spectra rows `[nrows][n/2 + 1]` into sample rows
/// `[nrows][n]` (normalised), over nthreads. Rows pair up as in
/// EVTransform::inverse_rows(), so a row's rounding depends on its partner.
void inverse_batch(const ComplexType* spectra,
                   float* samples,
                   SizeType nrows,
                   SizeType n,
                   int nthreads);

/// Forward transforms of sample rows `[nrows][n]` into spectra rows
/// `[nrows][n/2 + 1]`, over nthreads, paired as inverse_batch().
void forward_batch(const float* samples,
                   ComplexType* spectra,
                   SizeType nrows,
                   SizeType n,
                   int nthreads);

} // namespace loki::lane_fft
