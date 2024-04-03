#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "fft_utils.hpp"

class MatchedFilter {
public:
    MatchedFilter(const std::vector<size_t>& widths_arr, size_t nprofiles,
                  size_t nbins, std::string_view shape = "boxcar")
        : widths_arr(widths_arr), nbins(nbins), nprofiles(nprofiles),
          shape(shape),
          fft2d(nprofiles, widths_arr.size(), get_nbins_pow2(nbins)) {
        ntemplates = widths_arr.size();
        nbins_pow2 = get_nbins_pow2(nbins);
        // Allocate memory for the templates
        templates.resize(ntemplates * nbins_pow2, 0.0f);
        arr_padded.resize(nprofiles * nbins_pow2, 0.0f);
        snr_arr.resize(nprofiles * ntemplates * nbins_pow2, 0.0f);
        initialise_templates();
    };

    std::vector<float> get_templates() const { return templates; }
    std::size_t get_ntemplates() const { return ntemplates; }
    std::size_t get_nbins() const { return nbins_pow2; }
    void compute(std::span<const float> arr, std::span<float> out) {
        const size_t arr_size = arr.size();
        if (arr_size != nprofiles * nbins) {
            throw std::invalid_argument("Input array size does not match");
        }
        const size_t out_size = out.size();
        if (out_size != nprofiles * ntemplates) {
            throw std::invalid_argument("Output array size does not match");
        }
        if (arr_size != nbins_pow2) {
            std::ranges::copy(arr, arr_padded.begin());
        } else {
            arr_padded.assign(arr.begin(), arr.end());
        }
        fft2d.circular_convolve(std::span<float>(arr_padded),
                                std::span<float>(templates),
                                std::span<float>(snr_arr));
        // Find the maximum value for each profile and template and then scale
        for (size_t i = 0; i < nprofiles; ++i) {
            for (size_t j = 0; j < ntemplates; ++j) {
                const auto idx     = i * ntemplates + j;
                const auto snr_idx = snr_arr.begin() + idx * nbins_pow2;
                out[idx] = *std::max_element(snr_idx, snr_idx + nbins_pow2)
                           / nbins_pow2;
            }
        }
    }

private:
    const std::vector<size_t> widths_arr;
    const std::size_t nbins;
    const std::size_t nprofiles;
    const std::string_view shape;

    std::size_t nbins_pow2;
    std::size_t ntemplates;
    std::vector<float> templates;
    std::vector<float> arr_padded;
    std::vector<float> snr_arr;

    // FFTW plans
    FFT_2D fft2d;

    std::size_t get_nbins_pow2(std::size_t nbins) {
        return 1 << static_cast<std::size_t>(std::ceil(std::log2(nbins)));
    }

    void initialise_templates() {
        if (shape == "gaussian") {
            for (size_t i = 0; i < ntemplates; ++i) {
                std::span<float> temp_arr(templates.data() + i * nbins_pow2,
                                          nbins_pow2);
                generate_gaussian_template(temp_arr, widths_arr[i]);
            }
        } else if (shape == "boxcar") {
            for (size_t i = 0; i < ntemplates; ++i) {
                std::span<float> temp_arr(templates.data() + i * nbins_pow2,
                                          nbins_pow2);
                generate_boxcar_template(temp_arr, widths_arr[i]);
            }
        } else {
            throw std::invalid_argument("Invalid shape");
        }
    }

    void generate_boxcar_template(std::span<float>& arr, size_t width) {
        const size_t temp_nbins = arr.size();
        const auto temp_start   = arr.begin() + temp_nbins / 2 - width / 2;
        const auto temp_end     = temp_start + width + (width % 2);
        std::fill(arr.begin(), temp_start, 0.0f);
        std::fill(temp_start, temp_end, 1.0f);
        std::fill(temp_end, arr.end(), 0.0f);
        normalise(arr);
    }

    void generate_gaussian_template(std::span<float>& arr, size_t width) {
        const size_t temp_nbins = arr.size();
        const float sigma = width / (2.0f * std::sqrt(2.0f * std::log(2.0f)));
        const auto xmax   = static_cast<size_t>(std::ceil(3.5 * sigma));

        const auto temp_start = temp_nbins / 2 - xmax;
        for (size_t i = 0; i < 2 * xmax + 1; ++i) {
            const auto x        = i - xmax;
            arr[temp_start + i] = std::exp(-x * x / (2.0 * sigma * sigma));
        }
        normalise(arr);
    }

    void normalise(std::span<float>& arr) {
        const float norm = std::sqrt(
            std::inner_product(arr.begin(), arr.end(), arr.begin(), 0.0f));
        std::transform(arr.begin(), arr.end(), arr.begin(),
                       [norm](float val) { return val / norm; });
    }
};

namespace loki {

// out = x + scalar
void add_scalar(std::span<const float> x, const float scalar,
                std::span<float> out) {
    std::transform(x.begin(), x.end(), out.begin(),
                   [scalar](float xi) { return xi + scalar; });
}

// return max(x[i] - y[i])
float diff_max(std::span<const float> x, std::span<const float> y) {
    float max_diff = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < x.size(); ++i) {
        max_diff = std::max(max_diff, x[i] - y[i]);
    }
    return max_diff;
}

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out) {
    double acc         = 0;
    const size_t nbins = x.size();
    const size_t nsum  = out.size();
    const size_t jmax  = std::min(nbins, nsum);
    for (size_t j = 0; j < jmax; ++j) {
        acc += x[j];
        out[j] = static_cast<float>(acc);
    }
    if (nsum <= nbins) {
        return;
    }
    // Wrap around
    const size_t n_wraps = nsum / nbins;
    const size_t extra   = nsum % nbins;
    const float last     = out[jmax - 1];
    for (size_t i_wrap = 1; i_wrap < n_wraps; ++i_wrap) {
        add_scalar(out.subspan(0, nbins), i_wrap * last,
                   out.subspan(i_wrap * nbins, nbins));
    }
    add_scalar(out.subspan(0, extra), n_wraps * last,
               out.subspan(n_wraps * nbins, extra));
}

// Compute the S/N of single pulse proile
void snr_1d(std::span<const float> arr, std::span<const size_t> widths,
            float stdnoise, std::span<float> out) {
    const size_t wmax       = *std::max_element(widths.begin(), widths.end());
    const size_t nbins      = arr.size();
    const size_t ntemplates = widths.size();
    if (out.size() != ntemplates) {
        throw std::invalid_argument("Output array size does not match");
    }

    std::vector<float> psum(nbins + wmax);
    circular_prefix_sum(arr, std::span<float>(psum));
    const float sum = psum[nbins - 1];  // sum of the input array
    const std::span<float> psum_span(psum);

    for (size_t iw = 0; iw < ntemplates; ++iw) {
        // Height and baseline of a boxcar filter with width w bins
        // and zero mean and unit square sum
        const auto w     = static_cast<float>(widths[iw]);
        const float h    = std::sqrt((nbins - w) / (nbins * w));  // height = +h
        const float b    = w * h / (nbins - w);  // baseline = -b
        const float dmax = diff_max(psum_span.subspan(w, nbins),
                                    psum_span.subspan(0, nbins));
        const float snr  = ((h + b) * dmax - b * sum) / stdnoise;
        out[iw]          = snr;
    }
}

// Compute the S/N of array of single pulse profiles
void snr_2d(std::span<const float> arr, const size_t nprofiles,
            std::span<const size_t> widths, float stdnoise,
            std::span<float> out) {
    const size_t nbins      = arr.size() / nprofiles;
    const size_t ntemplates = widths.size();
    if (out.size() != nprofiles * ntemplates) {
        throw std::invalid_argument("Output array size does not match");
    }
#pragma omp parallel for
    for (size_t i = 0; i < nprofiles; ++i) {
        snr_1d(arr.subspan(i * nbins, nbins), widths, stdnoise,
               out.subspan(i * ntemplates, ntemplates));
    }
}

}  // namespace loki
