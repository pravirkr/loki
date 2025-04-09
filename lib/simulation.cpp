#include "loki/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace loki::simulation {
std::vector<float>
generate_folded_profile(SizeType nbins, float ducy, float center) {
    // Generate phase values from 0 to 1 (exclusive)
    std::vector<float> phase(nbins);
    std::iota(phase.begin(), phase.end(), 0);
    const float step = 1.0F / static_cast<float>(nbins);
    std::ranges::transform(phase, phase.begin(),
                           [step](float i) { return i * step; });

    // Calculate sigma
    float sigma = ducy / (2.0F * std::sqrt(2.0F * std::numbers::ln10_v<float>));

    // Calculate profile using wrapped phases
    std::vector<float> profile(nbins);
    std::ranges::transform(phase, profile.begin(), [sigma, center](float p) {
        float wrapped_phase = std::fmod(p - center + 0.5F, 1.0F) - 0.5F;
        return std::exp(-(wrapped_phase * wrapped_phase) /
                        (2.0F * sigma * sigma));
    });

    // Normalize by max value
    float max_val = std::ranges::max(profile);
    std::ranges::transform(profile, profile.begin(),
                           [max_val](float val) { return val / max_val; });

    // Calculate L2 norm and normalize
    const float sum_of_squares = std::inner_product(
        profile.begin(), profile.end(), profile.begin(), 0.0F);
    const float norm_factor = std::sqrt(sum_of_squares);
    std::ranges::transform(profile, profile.begin(), [norm_factor](float val) {
        return val / norm_factor;
    });

    return profile;
}
} // namespace loki::simulation
