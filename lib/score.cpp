#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

#include <omp.h>

#include <loki/score.hpp>
#include <loki/utils.hpp>

MatchedFilter::MatchedFilter(const std::vector<size_t>& widths_arr,
                             size_t nprofiles,
                             size_t nbins,
                             std::string_view shape)
    : m_widths_arr(widths_arr),
      m_nbins(nbins),
      m_nprofiles(nprofiles),
      m_shape(shape),
      m_nbins_pow2(get_nbins_pow2(m_nbins)),
      m_ntemplates(widths_arr.size()),
      m_fft2d(FFT2D(m_nprofiles, m_ntemplates, m_nbins_pow2)) {
    // Allocate memory for the templates
    m_templates.resize(m_ntemplates * m_nbins_pow2, 0.0F);
    m_arr_padded.resize(nprofiles * m_nbins_pow2, 0.0F);
    m_snr_arr.resize(nprofiles * m_ntemplates * m_nbins_pow2, 0.0F);
    initialise_templates();
};

std::vector<float> MatchedFilter::get_templates() const { return m_templates; }
std::size_t MatchedFilter::get_ntemplates() const { return m_ntemplates; }
std::size_t MatchedFilter::get_nbins() const { return m_nbins_pow2; }
void MatchedFilter::compute(std::span<const float> arr, std::span<float> out) {
    const size_t arr_size = arr.size();
    if (arr_size != m_nprofiles * m_nbins) {
        throw std::invalid_argument("Input array size does not match");
    }
    const size_t out_size = out.size();
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
    for (size_t i = 0; i < m_nprofiles; ++i) {
        for (size_t j = 0; j < m_ntemplates; ++j) {
            const auto idx     = i * m_ntemplates + j;
            const auto snr_idx = m_snr_arr.begin() + idx * m_nbins_pow2;
            out[idx] = *std::max_element(snr_idx, snr_idx + m_nbins_pow2) /
                       m_nbins_pow2;
        }
    }
}

void MatchedFilter::initialise_templates() {
    if (m_shape == "gaussian") {
        for (size_t i = 0; i < m_ntemplates; ++i) {
            std::span<float> temp_arr(m_templates.data() + i * m_nbins_pow2,
                                      m_nbins_pow2);
            generate_gaussian_template(temp_arr, m_widths_arr[i]);
        }
    } else if (m_shape == "boxcar") {
        for (size_t i = 0; i < m_ntemplates; ++i) {
            std::span<float> temp_arr(m_templates.data() + i * m_nbins_pow2,
                                      m_nbins_pow2);
            generate_boxcar_template(temp_arr, m_widths_arr[i]);
        }
    } else {
        throw std::invalid_argument("Invalid shape");
    }
}

void MatchedFilter::normalise(std::span<float>& arr) {
    const float sum =
        std::inner_product(arr.begin(), arr.end(), arr.begin(), 0.0F);
    const float scale = 1.0F / std::sqrt(sum);
    std::transform(arr.begin(), arr.end(), arr.begin(),
                   [scale](float val) { return val * scale; });
}

std::size_t MatchedFilter::get_nbins_pow2(std::size_t nbins) {
    return 1 << static_cast<std::size_t>(std::ceil(std::log2(nbins)));
}

void MatchedFilter::generate_boxcar_template(std::span<float>& arr,
                                             size_t width) {
    const size_t temp_nbins = arr.size();
    const auto start        = temp_nbins / 2 - width / 2;
    const auto end          = start + width + (width % 2);
    std::fill(arr.begin(), arr.begin() + static_cast<int>(start), 0.0F);
    std::fill(arr.begin() + static_cast<int>(start),
              arr.begin() + static_cast<int>(end), 1.0F);
    std::fill(arr.begin() + static_cast<int>(end), arr.end(), 0.0F);
    normalise(arr);
}

void MatchedFilter::generate_gaussian_template(std::span<float>& arr,
                                               size_t width) {
    const size_t temp_nbins = arr.size();
    const float sigma =
        static_cast<float>(width) / (2.0F * std::sqrt(2.0F * std::log(2.0F)));
    const auto xmax = static_cast<size_t>(std::ceil(3.5 * sigma));

    const auto temp_start = temp_nbins / 2 - xmax;
    for (size_t i = 0; i < 2 * xmax + 1; ++i) {
        const auto x        = static_cast<float>(i - xmax);
        arr[temp_start + i] = std::exp(-x * x / (2.0F * sigma * sigma));
    }
    normalise(arr);
}

namespace loki {

// Compute the S/N of single pulse proile
void snr_1d(std::span<const float> arr,
            std::span<const size_t> widths,
            float stdnoise,
            std::span<float> out) {
    const size_t wmax       = *std::max_element(widths.begin(), widths.end());
    const size_t nbins      = arr.size();
    const size_t ntemplates = widths.size();
    if (out.size() != ntemplates) {
        throw std::invalid_argument("Output array size does not match");
    }

    std::vector<float> psum(nbins + wmax);
    circular_prefix_sum(arr, std::span<float>(psum));
    const float sum = psum[nbins - 1]; // sum of the input array
    const std::span<float> psum_span(psum);

    for (size_t iw = 0; iw < ntemplates; ++iw) {
        // Height and baseline of a boxcar filter with width w bins
        // and zero mean and unit square sum
        const size_t w = widths[iw];
        const float h  = std::sqrt(static_cast<float>(nbins - w) /
                                   static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);
        const float dmax =
            diff_max(psum_span.subspan(w, nbins), psum_span.subspan(0, nbins));
        out[iw] = ((h + b) * dmax - b * sum) / stdnoise;
    }
}

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr,
            const size_t nprofiles,
            std::span<const size_t> widths,
            float stdnoise,
            std::span<float> out) {
    const size_t nbins      = arr.size() / nprofiles;
    const size_t ntemplates = widths.size();
    if (out.size() != nprofiles * ntemplates) {
        throw std::invalid_argument("Output array size does not match");
    }
#pragma omp parallel for default(none)                                         \
    shared(arr, widths, stdnoise, out, nbins, nprofiles, ntemplates)
    for (size_t i = 0; i < nprofiles; ++i) {
        snr_1d(arr.subspan(i * nbins, nbins), widths, stdnoise,
               out.subspan(i * ntemplates, ntemplates));
    }
}

} // namespace loki
