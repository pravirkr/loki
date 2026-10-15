#include <algorithm>
#include <span>

#include <omp.h>

#include "loki/common/types.hpp"

#include "lib/cpu/boxcar_kernels.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detection/score_engine.hpp"

namespace loki::detection::detail {

void snr_boxcar_2d_cpu(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       float stdnoise,
                       int nthreads) {
    const auto nwidths = widths.size();
    error_check::check_equal(folds.size(), (nprofiles * nbins),
                             "snr_boxcar_2d: folds size does not match");
    error_check::check_equal(
        scores.size(), nprofiles * nwidths,
        "snr_boxcar_2d: scores size does not match nprofiles * nwidths");
    snr_boxcar_impl<false, false>(folds.data(), nprofiles, nbins, widths.data(),
                                  nwidths, scores.data(), stdnoise, nthreads);
}

void snr_boxcar_2d_max_cpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           float stdnoise,
                           int nthreads) {
    const auto nwidths = widths.size();
    error_check::check_equal(folds.size(), (nprofiles * nbins),
                             "snr_boxcar_2d_max: arr size does not match");
    error_check::check_equal(
        scores.size(), nprofiles,
        "snr_boxcar_2d_max: scores size does not match nprofiles");
    snr_boxcar_impl<false, true>(folds.data(), nprofiles, nbins, widths.data(),
                                 nwidths, scores.data(), stdnoise, nthreads);
}

void snr_boxcar_3d_cpu(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       int nthreads) {
    const auto nwidths = widths.size();
    error_check::check_equal(folds.size(), (nprofiles * 2 * nbins),
                             "snr_boxcar_3d: folds size does not match");
    error_check::check_equal(
        scores.size(), nprofiles * nwidths,
        "snr_boxcar_3d: scores size does not match nprofiles * nwidths");
    nthreads                  = std::max(nthreads, 1);
    const float* folds_ptr    = folds.data();
    float* scores_ptr         = scores.data();
    const SizeType* width_ptr = widths.data();
#pragma omp parallel num_threads(nthreads) default(none)                       \
    firstprivate(folds_ptr, scores_ptr, width_ptr, nprofiles, nbins, nwidths)  \
    shared(nthreads)
    {
        // Contiguous blocks of whole profile groups per thread.
        const auto tid   = static_cast<SizeType>(omp_get_thread_num());
        const auto nthr  = static_cast<SizeType>(omp_get_num_threads());
        const auto begin = (nprofiles * tid) / nthr;
        const auto end   = (nprofiles * (tid + 1)) / nthr;
        BoxcarScorer3D scorer(nbins, width_ptr, nwidths);
        float* out = scores_ptr + (begin * nwidths);
        scorer.score(folds_ptr + (begin * 2 * nbins), end - begin,
                     [out, nwidths](SizeType p, SizeType iw, float snr) {
                         out[(p * nwidths) + iw] = snr;
                     });
    }
}

void snr_boxcar_3d_max_cpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           int nthreads) {
    const auto nwidths = widths.size();
    error_check::check_equal(folds.size(), (nprofiles * 2 * nbins),
                             "snr_boxcar_3d_max: folds size does not match");
    error_check::check_equal(
        scores.size(), nprofiles,
        "snr_boxcar_3d_max: scores size do not match nprofiles");
    snr_boxcar_impl<true, true>(folds.data(), nprofiles, nbins, widths.data(),
                                nwidths, scores.data(), 1.0F, nthreads);
}

} // namespace loki::detection::detail
