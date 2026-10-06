#pragma once

/**
 * @file boxcar_kernels.hpp
 * @brief CPU boxcar S/N kernels shared by lib/detection/score.cpp and
 * lib/cpu/score_cpu.cpp. Internal.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <omp.h>

#include "detail/utils.hpp"
#include "loki/common/types.hpp"

namespace loki::detection::detail {

/// Branchless e/sqrt(v) folded into the circular prefix sum. The first
/// `nbins` entries match `circular_prefix_sum` of the normalised profile, so
/// `diff_max` sees the same values as the two-pass form.
inline void ev_circular_prefix(const float* __restrict__ ts_e,
                               const float* __restrict__ ts_v,
                               float* __restrict__ psum,
                               SizeType nbins,
                               SizeType nsum) noexcept {
    float run = 0.0F;
    for (SizeType j = 0; j < nbins; ++j) {
        const float variance = ts_v[j];
        const float sample =
            (variance > 0.0F) ? (ts_e[j] / std::sqrt(variance)) : 0.0F;
        run += sample;
        psum[j] = run;
    }
    if (nsum <= nbins) {
        return;
    }
    const float last_sum          = psum[nbins - 1];
    const SizeType first_wrap_end = std::min(nsum, 2 * nbins);
    for (SizeType i = nbins; i < first_wrap_end; ++i) {
        psum[i] = psum[i - nbins] + last_sum;
    }
    if (nsum > 2 * nbins) {
        for (SizeType i = 2 * nbins; i < nsum; ++i) {
            const auto wrap_count   = i / nbins;
            const auto pos_in_cycle = i % nbins;
            psum[i] = psum[pos_in_cycle] +
                      (static_cast<float>(wrap_count) * last_sum);
        }
    }
}

template <bool Is3D, bool FindMax>
void snr_boxcar_impl(const float* __restrict__ folds,
                     SizeType nprofiles,
                     SizeType nbins,
                     const SizeType* __restrict__ widths,
                     SizeType nwidths,
                     float* __restrict__ scores,
                     float stdnoise = 1.0F, // stdnoise is only used for 2D
                     int nthreads   = 1) {
    nthreads            = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType wmax = *std::ranges::max_element(widths, widths + nwidths);

    // Precompute template parameters (h, b) for all widths
    std::vector<float> h_vals(nwidths);
    std::vector<float> b_vals(nwidths);
    for (SizeType iw = 0; iw < nwidths; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
    const float inv_stdnoise = Is3D ? 1.0F : (1.0F / stdnoise);

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(folds, widths, scores, nbins, nprofiles, nwidths, wmax, h_vals,     \
               b_vals, inv_stdnoise)
    {
        // Thread-local buffers
        std::vector<float> psum(nbins + wmax, 0.0F);

#pragma omp for
        for (SizeType i = 0; i < nprofiles; ++i) {
            if constexpr (Is3D) {
                const SizeType base_idx            = i * 2 * nbins;
                const float* __restrict__ ts_e_ptr = folds + base_idx;
                const float* __restrict__ ts_v_ptr = folds + base_idx + nbins;
                ev_circular_prefix(ts_e_ptr, ts_v_ptr, psum.data(), nbins,
                                   nbins + wmax);
            } else {
                const float* fold_ptr = folds + (i * nbins);
                utils::circular_prefix_sum(fold_ptr, psum.data(), nbins,
                                           nbins + wmax);
            }
            const float sum              = psum[nbins - 1];
            float* __restrict__ psum_ptr = psum.data();

            // Compute SNR for each width, find maximum
            float max_snr = std::numeric_limits<float>::lowest();
            for (SizeType iw = 0; iw < nwidths; ++iw) {
                const auto dmax =
                    utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
                const float snr_base =
                    ((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum);
                const float snr = snr_base * inv_stdnoise;
                if constexpr (FindMax) {
                    max_snr = std::max(max_snr, snr);
                } else {
                    scores[(i * nwidths) + iw] = snr;
                }
            }
            if constexpr (FindMax) {
                scores[i] = max_snr;
            }
        }
    }
}

} // namespace loki::detection::detail
