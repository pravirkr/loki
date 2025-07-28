#include "loki/detection/score.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>

#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/utils.hpp"
#include "loki/utils/fft.hpp"

namespace loki::detection {

namespace {
void normalise_l2(std::span<float> arr) {
    if (arr.empty()) {
        return;
    }
    const float mean = std::reduce(arr.begin(), arr.end(), 0.0F) /
                       static_cast<float>(arr.size());
    std::ranges::transform(arr, arr.begin(),
                           [mean](float val) { return val - mean; });
    // Compute norm (L2)
    const float norm = std::sqrt(
        std::inner_product(arr.begin(), arr.end(), arr.begin(), 0.0F));
    // Normalize in-place if norm is non-zero
    if (norm > 0.0F) {
        const float scale = 1.0F / norm;
        std::ranges::transform(arr, arr.begin(),
                               [scale](float val) { return val * scale; });
    }
}

void generate_boxcar_templates(std::span<float> templates,
                               std::span<const SizeType> widths,
                               SizeType nbins) {
    const auto ntemplates = widths.size();
    error_check::check_equal(templates.size(), ntemplates * nbins,
                             "generate_boxcar_templates: templates size does "
                             "not match");
    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        const auto width   = widths[iw];
        auto template_span = templates.subspan(iw * nbins, nbins);
        // Fill the first 'width' bins with 1.0, rest remain 0.0
        std::fill_n(template_span.begin(), std::min(width, nbins), 1.0F);
        normalise_l2(template_span);
    }
}

void generate_gaussian_templates(std::span<float> templates,
                                 std::span<const SizeType> widths,
                                 SizeType nbins) {
    const auto ntemplates = widths.size();
    error_check::check_equal(templates.size(), ntemplates * nbins,
                             "generate_gaussian_templates: templates size does "
                             "not match");
    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        const SizeType width = widths[iw];
        auto template_span   = templates.subspan(iw * nbins, nbins);
        const auto sigma =
            static_cast<float>(width) /
            (2.0F * std::sqrt(2.0F * std::numbers::ln2_v<float>));
        const auto xmax = static_cast<SizeType>(std::ceil(3.5F * sigma));
        const auto gaussian_width = (2 * xmax) + 1;

        if (nbins >= gaussian_width) {
            // Template fits entirely within nbins - center it
            const auto start = (nbins / 2) - xmax;
            for (SizeType i = 0; i < gaussian_width; ++i) {
                const auto x = static_cast<float>(static_cast<int>(i) -
                                                  static_cast<int>(xmax));
                template_span[start + i] =
                    std::exp(-x * x / (2.0F * sigma * sigma));
            }
        } else {
            // Template is larger than nbins - truncate it
            const auto start_offset = xmax - (nbins / 2);
            for (SizeType i = 0; i < nbins; ++i) {
                const auto x =
                    static_cast<float>(static_cast<int>(start_offset + i) -
                                       static_cast<int>(xmax));
                template_span[i] = std::exp(-x * x / (2.0F * sigma * sigma));
            }
        }
        normalise_l2(template_span);
    }
}

} // namespace

class MatchedFilter::Impl {
public:
    Impl(std::span<const SizeType> widths_arr,
         SizeType nprofiles,
         SizeType nbins,
         std::string_view shape)
        : m_widths_arr(widths_arr.begin(), widths_arr.end()),
          m_nprofiles(nprofiles),
          m_nbins(nbins),
          m_shape(shape),
          m_nbins_pow2(std::bit_ceil(m_nbins)),
          m_ntemplates(widths_arr.size()),
          m_fft2d(utils::FFT2D(m_nprofiles, m_ntemplates, m_nbins_pow2)) {
        // Allocate memory for the templates
        m_templates.resize(m_ntemplates * m_nbins_pow2, 0.0F);
        m_arr_padded.resize(nprofiles * m_nbins_pow2, 0.0F);
        m_snr_arr.resize(nprofiles * m_ntemplates * m_nbins_pow2, 0.0F);
        initialise_templates();
    }
    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    std::vector<float> get_templates() const { return m_templates; }
    SizeType get_ntemplates() const { return m_ntemplates; }
    SizeType get_nbins() const { return m_nbins_pow2; }
    void compute(std::span<const float> arr, std::span<float> out) {
        const SizeType arr_size = arr.size();
        if (arr_size != m_nprofiles * m_nbins) {
            throw std::invalid_argument("Input array size does not match");
        }
        const SizeType out_size = out.size();
        if (out_size != m_nprofiles * m_ntemplates) {
            throw std::invalid_argument("Output array size does not match");
        }
        if (arr_size != m_nbins_pow2) {
            std::ranges::copy(arr, m_arr_padded.begin());
        } else {
            m_arr_padded.assign(arr.begin(), arr.end());
        }
        m_fft2d.circular_convolve(std::span<float>(m_arr_padded),
                                  std::span<float>(m_templates),
                                  std::span<float>(m_snr_arr));
        // Find the maximum value for each profile and template and then scale
        for (SizeType i = 0; i < m_nprofiles; ++i) {
            for (SizeType j = 0; j < m_ntemplates; ++j) {
                const auto idx   = (i * m_ntemplates) + j;
                auto snr_subspan = std::span(m_snr_arr).subspan(
                    idx * m_nbins_pow2, m_nbins_pow2);
                out[idx] = *std::ranges::max_element(snr_subspan) /
                           static_cast<float>(m_nbins_pow2);
            }
        }
    }

private:
    std::vector<SizeType> m_widths_arr;
    SizeType m_nprofiles;
    SizeType m_nbins;
    std::string_view m_shape;

    SizeType m_nbins_pow2;
    SizeType m_ntemplates;
    std::vector<float> m_templates;
    std::vector<float> m_arr_padded;
    std::vector<float> m_snr_arr;

    // FFTW plans
    utils::FFT2D m_fft2d;

    void initialise_templates() {
        if (m_shape == "gaussian") {
            generate_gaussian_templates(m_templates, m_widths_arr,
                                        m_nbins_pow2);
        } else if (m_shape == "boxcar") {
            generate_boxcar_templates(m_templates, m_widths_arr, m_nbins_pow2);
        } else {
            throw std::invalid_argument(
                std::format("Invalid template shape: {}", m_shape));
        }
    }
}; // End MatchedFilter::Impl definition

MatchedFilter::MatchedFilter(std::span<const SizeType> widths_arr,
                             SizeType nprofiles,
                             SizeType nbins,
                             std::string_view shape)
    : m_impl(std::make_unique<Impl>(widths_arr, nprofiles, nbins, shape)) {}
MatchedFilter::~MatchedFilter()                              = default;
MatchedFilter::MatchedFilter(MatchedFilter&& other) noexcept = default;
MatchedFilter&
MatchedFilter::operator=(MatchedFilter&& other) noexcept = default;
std::vector<float> MatchedFilter::get_templates() const noexcept {
    return m_impl->get_templates();
}
SizeType MatchedFilter::get_ntemplates() const noexcept {
    return m_impl->get_ntemplates();
}
SizeType MatchedFilter::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
void MatchedFilter::compute(std::span<const float> arr, std::span<float> out) {
    m_impl->compute(arr, out);
}

// SnrBoxcarCache
BoxcarWidthsCache::BoxcarWidthsCache(std::span<const SizeType> widths,
                                     SizeType nbins)
    : widths(widths.begin(), widths.end()),
      wmax(*std::ranges::max_element(widths)),
      ntemplates(widths.size()),
      h_vals(widths.size()),
      b_vals(widths.size()),
      fold_norm_buffer(nbins),
      psum_buffer(nbins + wmax) {

    // Precompute h_vals and b_vals for all widths
    for (SizeType iw = 0; iw < widths.size(); ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
}
std::vector<SizeType>
generate_box_width_trials(SizeType fold_bins, double ducy_max, double wtsp) {
    const auto wmax = static_cast<SizeType>(
        std::max(1.0, ducy_max * static_cast<double>(fold_bins)));
    std::vector<SizeType> widths = {1};
    while (widths.back() < wmax) {
        const auto next_width = std::max(
            static_cast<SizeType>(widths.back() + 1),
            static_cast<SizeType>(wtsp * static_cast<double>(widths.back())));
        if (next_width > wmax) {
            break;
        }
        widths.push_back(next_width);
    }
    return widths;
}

void snr_boxcar_1d(std::span<const float> arr,
                   std::span<const SizeType> widths,
                   std::span<float> out,
                   float stdnoise) {
    const SizeType wmax       = *std::ranges::max_element(widths);
    const SizeType nbins      = arr.size();
    const SizeType ntemplates = widths.size();
    error_check::check_equal(out.size(), ntemplates,
                             "snr_boxcar_1d: out size does not match");
    std::vector<float> psum(nbins + wmax);
    utils::circular_prefix_sum(arr, std::span<float>(psum));
    const float sum              = psum[nbins - 1]; // sum of the input array
    float* __restrict__ psum_ptr = psum.data();

    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        // Height and baseline of a boxcar filter with width w bins
        // and zero mean and unit square sum
        const SizeType w = widths[iw];
        const float h    = std::sqrt(static_cast<float>(nbins - w) /
                                     static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);
        const float dmax = utils::diff_max(psum_ptr + w, psum_ptr, nbins);
        out[iw]          = ((h + b) * dmax - b * sum) / stdnoise;
    }
}

void snr_boxcar_2d_max(std::span<const float> arr,
                       const SizeType nprofiles,
                       std::span<const SizeType> widths,
                       std::span<float> out,
                       float stdnoise,
                       int nthreads) {
    nthreads                  = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType nbins      = arr.size() / nprofiles;
    const SizeType ntemplates = widths.size();
    const SizeType wmax       = *std::ranges::max_element(widths);
    error_check::check_equal(
        out.size(), nprofiles,
        "snr_boxcar_2d_max: out size does not match nprofiles");
    // Precompute template parameters (h, b) for all widths
    std::vector<float> h_vals(ntemplates);
    std::vector<float> b_vals(ntemplates);
    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(arr, widths, stdnoise, out, nbins, nprofiles, ntemplates, wmax,     \
               h_vals, b_vals)
    {
        // Thread-local buffer
        std::vector<float> psum(nbins + wmax, 0.0F);

#pragma omp for
        for (SizeType i = 0; i < nprofiles; ++i) {
            const auto fold = arr.subspan(i * nbins, nbins);
            utils::circular_prefix_sum(fold, std::span<float>(psum));
            const float sum              = psum[nbins - 1];
            float* __restrict__ psum_ptr = psum.data();

            // Compute SNR for each width, find maximum
            float max_snr = std::numeric_limits<float>::lowest();
            for (SizeType iw = 0; iw < ntemplates; ++iw) {
                const auto dmax =
                    utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
                const float snr =
                    (((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum)) /
                    stdnoise;
                max_snr = std::max(max_snr, snr);
            }
            out[i] = max_snr;
        }
    }
}

void snr_boxcar_3d(std::span<const float> arr,
                   SizeType nprofiles,
                   std::span<const SizeType> widths,
                   std::span<float> out,
                   int nthreads) {
    nthreads                  = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType nbins      = arr.size() / (2 * nprofiles);
    const SizeType ntemplates = widths.size();
    const SizeType wmax       = *std::ranges::max_element(widths);
    error_check::check_equal(
        out.size(), nprofiles * ntemplates,
        "snr_boxcar_3d: out size does not match nprofiles * ntemplates");
    // Precompute template parameters (h, b) for all widths
    std::vector<float> h_vals(ntemplates);
    std::vector<float> b_vals(ntemplates);
    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
#pragma omp parallel num_threads(nthreads) default(none) shared(               \
        arr, widths, out, nbins, nprofiles, ntemplates, wmax, h_vals, b_vals)
    {
        // Thread-local buffers
        std::vector<float> fold_norm(nbins, 0.0F);
        std::vector<float> psum(nbins + wmax, 0.0F);

#pragma omp for
        for (SizeType i = 0; i < nprofiles; ++i) {
            const SizeType base_idx            = i * 2 * nbins;
            const float* __restrict__ ts_e_ptr = arr.data() + base_idx;
            const float* __restrict__ ts_v_ptr = arr.data() + base_idx + nbins;
            float* __restrict__ fold_norm_ptr  = fold_norm.data();
            for (SizeType j = 0; j < nbins; ++j) {
                const float inv_sqrt_v = 1.0F / std::sqrt(ts_v_ptr[j]);
                fold_norm_ptr[j]       = ts_e_ptr[j] * inv_sqrt_v;
            }
            utils::circular_prefix_sum(std::span<const float>(fold_norm),
                                       std::span<float>(psum));
            const float sum              = psum[nbins - 1];
            float* __restrict__ psum_ptr = psum.data();

            for (SizeType iw = 0; iw < ntemplates; ++iw) {
                const auto dmax =
                    utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
                out[(i * ntemplates) + iw] =
                    ((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum);
            }
        }
    }
}

void snr_boxcar_3d_max(std::span<const float> arr,
                       SizeType nprofiles,
                       std::span<const SizeType> widths,
                       std::span<float> out,
                       int nthreads) {
    nthreads                  = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType nbins      = arr.size() / (2 * nprofiles);
    const SizeType ntemplates = widths.size();
    const SizeType wmax       = *std::ranges::max_element(widths);
    error_check::check_equal(
        out.size(), nprofiles,
        "snr_boxcar_3d_max: out size do not match nprofiles");
    // Precompute template parameters (h, b) for all widths
    std::vector<float> h_vals(ntemplates);
    std::vector<float> b_vals(ntemplates);
    for (SizeType iw = 0; iw < ntemplates; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
#pragma omp parallel num_threads(nthreads) default(none) shared(               \
        arr, widths, out, nbins, nprofiles, ntemplates, wmax, h_vals, b_vals)
    {
        // Thread-local buffers
        std::vector<float> fold_norm(nbins, 0.0F);
        std::vector<float> psum(nbins + wmax, 0.0F);

#pragma omp for
        for (SizeType i = 0; i < nprofiles; ++i) {
            const SizeType base_idx            = i * 2 * nbins;
            const float* __restrict__ ts_e_ptr = arr.data() + base_idx;
            const float* __restrict__ ts_v_ptr = arr.data() + base_idx + nbins;
            float* __restrict__ fold_norm_ptr  = fold_norm.data();
            for (SizeType j = 0; j < nbins; ++j) {
                const float inv_sqrt_v = 1.0F / std::sqrt(ts_v_ptr[j]);
                fold_norm_ptr[j]       = ts_e_ptr[j] * inv_sqrt_v;
            }
            utils::circular_prefix_sum(std::span<const float>(fold_norm),
                                       std::span<float>(psum));
            const float sum              = psum[nbins - 1];
            float* __restrict__ psum_ptr = psum.data();

            // Compute SNR for each width, find maximum
            float max_snr = std::numeric_limits<float>::lowest();
            for (SizeType iw = 0; iw < ntemplates; ++iw) {
                const auto dmax =
                    utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
                const float snr =
                    ((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum);
                max_snr = std::max(max_snr, snr);
            }
            out[i] = max_snr;
        }
    }
}

namespace {
void snr_boxcar_batch_kernel(const float* __restrict__ arr,
                             SizeType nprofiles,
                             SizeType nbins,
                             float* __restrict__ out,
                             BoxcarWidthsCache& cache) {
    // Use precomputed values from cache
    const auto* __restrict__ widths = cache.widths.data();
    const auto wmax                 = cache.wmax;
    const auto ntemplates           = cache.ntemplates;
    const auto* __restrict__ h_vals = cache.h_vals.data();
    const auto* __restrict__ b_vals = cache.b_vals.data();
    auto* __restrict__ fold_norm    = cache.fold_norm_buffer.data();
    auto* __restrict__ psum         = cache.psum_buffer.data();

    for (SizeType i = 0; i < nprofiles; ++i) {
        const SizeType base_idx            = i * 2 * nbins;
        const float* __restrict__ ts_e_ptr = arr + base_idx;
        const float* __restrict__ ts_v_ptr = arr + base_idx + nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            const float inv_sqrt_v = 1.0F / std::sqrt(ts_v_ptr[j]);
            fold_norm[j]           = ts_e_ptr[j] * inv_sqrt_v;
        }
        utils::circular_prefix_sum(std::span<const float>(fold_norm, nbins),
                                   std::span<float>(psum, nbins + wmax));
        const float sum              = psum[nbins - 1];
        float* __restrict__ psum_ptr = psum;

        // Compute SNR for each width, find maximum
        float max_snr = std::numeric_limits<float>::lowest();
        for (SizeType iw = 0; iw < ntemplates; ++iw) {
            const auto dmax =
                utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
            const float snr =
                ((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum);
            max_snr = std::max(max_snr, snr);
        }
        out[i] = max_snr;
    }
}
} // namespace

void snr_boxcar_batch(std::span<const float> batch_folds,
                      std::span<float> batch_scores,
                      SizeType n_batch,
                      BoxcarWidthsCache& cache) {
    error_check::check_equal(
        batch_scores.size(), n_batch,
        "snr_boxcar_batch: batch_scores size does not match n_batch");
    const auto nbins = batch_folds.size() / (2 * n_batch);
    snr_boxcar_batch_kernel(batch_folds.data(), n_batch, nbins,
                            batch_scores.data(), cache);
}

} // namespace loki::detection