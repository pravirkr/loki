#pragma once

#include <span>

namespace loki {

// out = x + scalar
void add_scalar(std::span<const float> x, const float scalar,
                std::span<float> out);

// return max(x[i] - y[i])
float diff_max(std::span<const float> x, std::span<const float> y);

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

} // namespace loki