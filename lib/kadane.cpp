#include "loki/detection/kadane.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

#include <omp.h>
#include <xsimd/xsimd.hpp>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/simd_utils.hpp"

namespace loki::detection {
namespace {

template <SizeType NBiases, SizeType MaxNBINS>
SizeType score_and_filter_max_kadane_with_cache_impl(
    const float* __restrict__ arr,
    SizeType nprofiles,
    SizeType nbins,
    float* __restrict__ out,
    SizeType* __restrict__ indices_filtered,
    float threshold,
    BoxcarKadaneCache& cache) {
    using BatchType               = xsimd::batch<float>;
    constexpr SizeType kBatchSize = BatchType::size;
    constexpr SizeType kSimdAlign = xsimd::default_arch::alignment();

    const auto* __restrict__ biases = cache.biases.data();
    const SizeType vec_end          = nprofiles - (nprofiles % kBatchSize);

    // Temp buffers for the normalized and transposed data
    alignas(kSimdAlign) std::array<float, kBatchSize * MaxNBINS> temp_fn{};
    alignas(kSimdAlign) std::array<float, MaxNBINS * kBatchSize>
        transposed_fn{};
    alignas(kSimdAlign) std::array<float, kBatchSize> total_sums{};

    std::array<BatchType, NBiases> max_sum{};
    std::array<BatchType, NBiases> cur_max{};
    std::array<BatchType, NBiases> w_cur{};
    std::array<BatchType, NBiases> best_w{};
    std::array<BatchType, NBiases> min_sum{};
    std::array<BatchType, NBiases> cur_min{};
    std::array<BatchType, NBiases> wc_min{};
    std::array<BatchType, NBiases> best_wm{};

    const BatchType batch_nbins_f(static_cast<float>(nbins));
    const BatchType batch_zero(0.0F);
    const BatchType batch_one(1.0F);
    const BatchType batch_neg_inf(std::numeric_limits<float>::lowest());
    const BatchType batch_max_inf(std::numeric_limits<float>::max());

    SizeType nprofiles_passing = 0;

    for (SizeType i = 0; i < vec_end; i += kBatchSize) {
        const float* __restrict__ chunk_base  = arr + (i * 2 * nbins);
        float* __restrict__ temp_fn_ptr       = temp_fn.data();
        float* __restrict__ transposed_fn_ptr = transposed_fn.data();
        float* __restrict__ total_sums_ptr    = total_sums.data();

        // Step 1: Contiguous normalization and sum per profile in the batch
        for (std::size_t lane = 0; lane < kBatchSize; ++lane) {
            const float* __restrict__ ts_e_ptr =
                chunk_base + (lane * 2 * nbins);
            const float* __restrict__ ts_v_ptr = ts_e_ptr + nbins;
            float* __restrict__ dst            = temp_fn_ptr + (lane * nbins);
            float sum                          = 0.0F;

#pragma omp simd reduction(+ : sum)
            for (SizeType j = 0; j < nbins; ++j) {
                float val = ts_e_ptr[j] / std::sqrt(ts_v_ptr[j]);
                dst[j]    = val;
                sum += val;
            }
            total_sums_ptr[lane] = sum;
        }

        const BatchType mean_val =
            BatchType::load_aligned(total_sums_ptr) / batch_nbins_f;
        simd_utils::transpose<BatchType>(temp_fn_ptr, transposed_fn_ptr, nbins);

        // Step 2: init Kadane state for all biases
        UNROLL_VECTORIZE_N(NBiases)
        for (SizeType k = 0; k < NBiases; ++k) {
            max_sum[k] = batch_neg_inf;
            cur_max[k] = batch_zero;
            w_cur[k]   = batch_zero;
            best_w[k]  = batch_one;
            min_sum[k] = batch_max_inf;
            cur_min[k] = batch_zero;
            wc_min[k]  = batch_zero;
            best_wm[k] = batch_one;
        }

        // Step 3: single pass over bins, updating ALL biases
        for (SizeType j = 0; j < nbins; ++j) {
            const BatchType val_base =
                BatchType::load_aligned(&transposed_fn[j * kBatchSize]) -
                mean_val;

            UNROLL_VECTORIZE_N(NBiases)
            for (SizeType k = 0; k < NBiases; ++k) {
                BatchType val = val_base - BatchType(biases[k]);

                cur_max[k] += val;
                w_cur[k] += 1.0F;
                auto upd_max = (cur_max[k] > max_sum[k]);
                max_sum[k]   = xsimd::select(upd_max, cur_max[k], max_sum[k]);
                best_w[k]    = xsimd::select(upd_max, w_cur[k], best_w[k]);
                auto rst_max = (cur_max[k] < BatchType(0.0F));
                cur_max[k] =
                    xsimd::select(rst_max, BatchType(0.0F), cur_max[k]);
                w_cur[k] = xsimd::select(rst_max, BatchType(0.0F), w_cur[k]);

                cur_min[k] += val;
                wc_min[k] += 1.0F;
                auto upd_min = (cur_min[k] < min_sum[k]);
                min_sum[k]   = xsimd::select(upd_min, cur_min[k], min_sum[k]);
                best_wm[k]   = xsimd::select(upd_min, wc_min[k], best_wm[k]);
                auto rst_min = (cur_min[k] > BatchType(0.0F));
                cur_min[k] =
                    xsimd::select(rst_min, BatchType(0.0F), cur_min[k]);
                wc_min[k] = xsimd::select(rst_min, BatchType(0.0F), wc_min[k]);
            }
        }
        // Step 4: compute SNR from Kadane results
        BatchType max_snr = batch_neg_inf;

        UNROLL_VECTORIZE_N(NBiases)
        for (SizeType k = 0; k < NBiases; ++k) {
            const BatchType bias(biases[k]);
            BatchType best_biased_sum     = max_sum[k];
            BatchType best_width          = best_w[k];
            const BatchType wrapped_width = batch_nbins_f - best_wm[k];
            const BatchType wrapped_sum = (-batch_nbins_f * bias) - min_sum[k];
            const auto use_wrapped =
                (wrapped_width > batch_zero) & (wrapped_sum > best_biased_sum);
            best_biased_sum =
                xsimd::select(use_wrapped, wrapped_sum, best_biased_sum);
            best_width = xsimd::select(use_wrapped, wrapped_width, best_width);

            const BatchType unbiased_sum = best_biased_sum + best_width * bias;
            const BatchType denom = best_width * (batch_nbins_f - best_width);
            BatchType snr = unbiased_sum * xsimd::sqrt(batch_nbins_f / denom);

            auto valid =
                (best_width > batch_zero) & (best_width < batch_nbins_f);
            snr     = xsimd::select(valid, snr, batch_neg_inf);
            max_snr = xsimd::max(max_snr, snr);
        }

        max_snr.store_unaligned(out + i);

        // Check if the profile passes the threshold
        for (SizeType lane = 0; lane < kBatchSize; ++lane) {
            const SizeType profile_idx = i + lane;
            if (out[profile_idx] >= threshold) {
                indices_filtered[nprofiles_passing] = profile_idx;
                ++nprofiles_passing;
            }
        }
    }
    // Scalar tail
    auto* __restrict__ fold_norm = cache.fold_norm_buffer.data();
    for (SizeType i = vec_end; i < nprofiles; ++i) {
        const float* __restrict__ ts_e_ptr = arr + (i * 2 * nbins);
        const float* __restrict__ ts_v_ptr = ts_e_ptr + nbins;
        float total_sum                    = 0.0F;
        for (SizeType j = 0; j < nbins; ++j) {
            fold_norm[j] = ts_e_ptr[j] * (1.0F / std::sqrt(ts_v_ptr[j]));
            total_sum += fold_norm[j];
        }
        const float mean_val = total_sum / static_cast<float>(nbins);
        float max_snr        = std::numeric_limits<float>::lowest();
        for (SizeType k = 0; k < NBiases; ++k) {
            const float dc_offset = mean_val + biases[k];
            float max_sum         = std::numeric_limits<float>::lowest();
            float cur_max         = 0.0F;
            float min_sum         = std::numeric_limits<float>::max();
            float cur_min         = 0.0F;
            int w_cur = 0, best_w = 1, wc_min = 0, best_wm = 1;
            for (SizeType j = 0; j < nbins; ++j) {
                const float val = fold_norm[j] - dc_offset;
                // Track Max Contiguous Subarray
                cur_max += val;
                w_cur += 1;
                const bool update_max = (cur_max > max_sum);
                max_sum               = update_max ? cur_max : max_sum;
                best_w                = update_max ? w_cur : best_w;
                const bool reset_max  = (cur_max < 0.0F);
                cur_max               = reset_max ? 0.0F : cur_max;
                w_cur                 = reset_max ? 0 : w_cur;

                // Track Min Contiguous Subarray (for wrapped candidate)
                cur_min += val;
                wc_min += 1;
                const bool update_min = (cur_min < min_sum);
                min_sum               = update_min ? cur_min : min_sum;
                best_wm               = update_min ? wc_min : best_wm;
                const bool reset_min  = (cur_min > 0.0F);
                cur_min               = reset_min ? 0.0F : cur_min;
                wc_min                = reset_min ? 0 : wc_min;
            }
            // Candidate 1: Standard Non-wrapped
            float best_biased_sum = max_sum;
            auto best_width       = static_cast<SizeType>(best_w);
            // Candidate 2: Wrapped maximum
            const SizeType wrapped_width =
                nbins - static_cast<SizeType>(best_wm);
            const float wrapped_sum =
                (-static_cast<float>(nbins) * biases[k]) - min_sum;
            const bool use_wrapped =
                (wrapped_width > 0) && (wrapped_sum > best_biased_sum);
            best_biased_sum = use_wrapped ? wrapped_sum : best_biased_sum;
            best_width      = use_wrapped ? wrapped_width : best_width;

            if (best_width > 0 && best_width < nbins) {
                const float unbiased_sum =
                    best_biased_sum +
                    (static_cast<float>(best_width) * biases[k]);
                const float scale = std::sqrt(
                    static_cast<float>(nbins) /
                    static_cast<float>(best_width * (nbins - best_width)));
                max_snr = std::max(max_snr, unbiased_sum * scale);
            }
        }
        out[i] = max_snr;
        if (max_snr >= threshold) {
            indices_filtered[nprofiles_passing] = i;
            ++nprofiles_passing;
        }
    }
    return nprofiles_passing;
} // End score_and_filter_max_kadane_with_cache_impl definition

} // namespace

BoxcarKadaneCache::BoxcarKadaneCache(std::span<const float> biases,
                                     SizeType nbins)
    : biases(biases.begin(), biases.end()),
      n_biases(biases.size()),
      fold_norm_buffer(nbins) {}

SizeType
score_and_filter_max_kadane_with_cache(std::span<const float> folds,
                                       std::span<float> scores,
                                       std::span<SizeType> indices_filtered,
                                       float threshold,
                                       SizeType nprofiles,
                                       SizeType nbins,
                                       BoxcarKadaneCache& cache) {
    error_check::check_greater_equal(scores.size(), nprofiles,
                                     "score_and_filter_max_kadane_with_cache: "
                                     "scores should be at least nprofiles");
    error_check::check_greater_equal(indices_filtered.size(), nprofiles,
                                     "score_and_filter_max_kadane_with_cache: "
                                     "indices_filtered should be at least "
                                     "nprofiles");
    error_check::check_greater_equal(folds.size(), nprofiles * 2 * nbins,
                                     "score_and_filter_max_kadane_with_cache: "
                                     "folds should be at least nprofiles * "
                                     "2 * nbins");
    SizeType nprofiles_passing = 0;
    auto dispatch              = [&]<int B, SizeType N>() {
        nprofiles_passing = score_and_filter_max_kadane_with_cache_impl<B, N>(
            folds.data(), nprofiles, nbins, scores.data(),
            indices_filtered.data(), threshold, cache);
    };

    auto dispatch_bins = [&]<int B>() {
        if (nbins <= 32) {
            dispatch.template operator()<B, 32>();
        } else if (nbins <= 64) {
            dispatch.template operator()<B, 64>();
        } else if (nbins <= 128) {
            dispatch.template operator()<B, 128>();
        } else if (nbins <= 256) {
            dispatch.template operator()<B, 256>();
        } else if (nbins <= 512) {
            dispatch.template operator()<B, 512>();
        } else if (nbins <= 1024) {
            dispatch.template operator()<B, 1024>();
        } else {
            throw std::runtime_error("score_and_filter_max_kadane_with_cache: "
                                     "nbins exceeds compiled limit of 1024");
        }
    };

    switch (cache.n_biases) {
    case 2:
        dispatch_bins.template operator()<2>();
        break;
    case 3:
        dispatch_bins.template operator()<3>();
        break;
    case 4:
        dispatch_bins.template operator()<4>();
        break;
    default:
        throw std::runtime_error("Unsupported n_biases");
    }

    return nprofiles_passing;
}

} // namespace loki::detection