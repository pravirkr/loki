#include "loki/simulation/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

#include "loki/exceptions.hpp"

namespace loki::simulation {
void generate_folded_profile(std::span<float> profile,
                             SizeType nbins,
                             float ducy,
                             float center) {
    error_check::check_greater_equal(
        profile.size(), nbins,
        "generate_folded_profile: profile size must be >= nbins");
    // Get subspan for the actual work area
    auto output = profile.subspan(0, nbins);
    // Generate phase values from 0 to 1 (exclusive)
    std::vector<float> phase(nbins);
    std::iota(phase.begin(), phase.end(), 0);
    const float step = 1.0F / static_cast<float>(nbins);
    std::ranges::transform(phase, phase.begin(),
                           [step](float i) { return i * step; });

    // Calculate sigma
    float sigma = ducy / (2.0F * std::sqrt(2.0F * std::numbers::ln10_v<float>));

    // Calculate profile using wrapped phases
    std::ranges::transform(phase, output.begin(), [sigma, center](float p) {
        float wrapped_phase = std::fmod(p - center + 0.5F, 1.0F) - 0.5F;
        return std::exp(-(wrapped_phase * wrapped_phase) /
                        (2.0F * sigma * sigma));
    });

    // Normalize by max value
    float max_val = std::ranges::max(output);
    std::ranges::transform(output, output.begin(),
                           [max_val](float val) { return val / max_val; });

    // Calculate L2 norm and normalize
    const float sum_of_squares =
        std::inner_product(output.begin(), output.end(), output.begin(), 0.0F);
    const float norm_factor = std::sqrt(sum_of_squares);
    std::ranges::transform(output, output.begin(), [norm_factor](float val) {
        return val / norm_factor;
    });
}

std::vector<float>
generate_folded_profile(SizeType nbins, float ducy, float center) {
    std::vector<float> profile(nbins);
    generate_folded_profile(profile, nbins, ducy, center);
    return profile;
}
} // namespace loki::simulation
