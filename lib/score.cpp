#include "loki/detection/score.hpp"

#include <algorithm>
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
          m_nbins_pow2(utils::next_power_of_two(m_nbins)),
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

std::vector<SizeType> generate_width_trials(SizeType nbins_max, float wtsp) {
    std::vector<SizeType> widths = {1};
    while (widths.back() < nbins_max) {
        const auto next_width = std::max(
            static_cast<SizeType>(widths.back() + 1),
            static_cast<SizeType>(wtsp * static_cast<float>(widths.back())));
        if (next_width > nbins_max) {
            break;
        }
        widths.push_back(next_width);
    }
    return widths;
}

void snr_1d(std::span<const float> arr,
            std::span<const SizeType> widths,
            std::span<float> out,
            float stdnoise) {
    const SizeType wmax       = *std::ranges::max_element(widths);
    const SizeType nbins      = arr.size();
    const SizeType ntemplates = widths.size();
    if (out.size() != ntemplates) {
        throw std::invalid_argument(
            std::format("Output array size does not match (got {} != {})",
                        out.size(), ntemplates));
    }

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

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr,
            const SizeType nprofiles,
            std::span<const SizeType> widths,
            std::span<float> out,
            float stdnoise) {
    const SizeType nbins      = arr.size() / nprofiles;
    const SizeType ntemplates = widths.size();
    if (out.size() != nprofiles * ntemplates) {
        throw std::invalid_argument(
            std::format("Output array size does not match (got {} != {})",
                        out.size(), nprofiles * ntemplates));
    }
#pragma omp parallel for default(none)                                         \
    shared(arr, widths, stdnoise, out, nbins, nprofiles, ntemplates)
    for (SizeType i = 0; i < nprofiles; ++i) {
        snr_1d(arr.subspan(i * nbins, nbins), widths,
               out.subspan(i * ntemplates, ntemplates), stdnoise);
    }
}

} // namespace loki::detection