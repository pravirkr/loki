#include <algorithm>
#include <cstddef>
#include <span>

namespace loki {

void add_scalar(std::span<const float> x, const float scalar,
                std::span<float> out) {
  std::transform(x.begin(), x.end(), out.begin(),
                 [scalar](float xi) { return xi + scalar; });
}

float diff_max(std::span<const float> x, std::span<const float> y) {
  float max_diff = -std::numeric_limits<float>::max();
  for (size_t i = 0; i < x.size(); ++i) {
    max_diff = std::max(max_diff, x[i] - y[i]);
  }
  return max_diff;
}

void circular_prefix_sum(std::span<const float> x, std::span<float> out) {
  double acc = 0;
  const size_t nbins = x.size();
  const size_t nsum = out.size();
  const size_t jmax = std::min(nbins, nsum);
  for (size_t j = 0; j < jmax; ++j) {
    acc += x[j];
    out[j] = static_cast<float>(acc);
  }
  if (nsum <= nbins) {
    return;
  }
  // Wrap around
  const size_t n_wraps = nsum / nbins;
  const size_t extra = nsum % nbins;
  const float last = out[jmax - 1];
  for (size_t i_wrap = 1; i_wrap < n_wraps; ++i_wrap) {
    add_scalar(out.subspan(0, nbins), i_wrap * last,
               out.subspan(i_wrap * nbins, nbins));
  }
  add_scalar(out.subspan(0, extra), n_wraps * last,
             out.subspan(n_wraps * nbins, extra));
}

} // namespace loki