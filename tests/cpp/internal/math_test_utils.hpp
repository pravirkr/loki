#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "loki/common/types.hpp"

#include "lib/detail/math.hpp"

namespace loki::test {

/// Restores the math tuning knobs when it goes out of scope.
class TuningGuard {
public:
    TuningGuard() : m_saved(loki::math::detail::tuning()) {}
    ~TuningGuard() { loki::math::detail::tuning() = m_saved; }
    TuningGuard(const TuningGuard&)            = delete;
    TuningGuard& operator=(const TuningGuard&) = delete;
    TuningGuard(TuningGuard&&)                 = delete;
    TuningGuard& operator=(TuningGuard&&)      = delete;

    /// Median windows >= \p window use the heap algorithm.
    static void heap_from(SizeType window) {
        loki::math::detail::tuning().heap_window = window;
    }
    /// Order statistics on arrays >= \p n use radix select.
    static void radix_from(SizeType n) {
        loki::math::detail::tuning().radix_select_size = n;
    }

private:
    loki::math::detail::Tuning m_saved;
};

/// Uniform float in [0, 1).
inline float uniform01(loki::math::PCG32& rng) {
    return static_cast<float>(rng() >> 8U) * (1.0F / 16777216.0F);
}

enum class Pattern : std::uint8_t {
    kRandom,
    kSorted,
    kReversed,
    kConstant,
    kDuplicates,
    kSignedZeros,
    kDynamicRange,
    kSteps,
    kImpulses,
};

inline constexpr std::array<Pattern, 9> kAllPatterns = {
    Pattern::kRandom,       Pattern::kSorted,     Pattern::kReversed,
    Pattern::kConstant,     Pattern::kDuplicates, Pattern::kSignedZeros,
    Pattern::kDynamicRange, Pattern::kSteps,      Pattern::kImpulses,
};

// NOLINTNEXTLINE(modernize-use-string-view): callers concatenate the result
inline std::string pattern_name(Pattern p) {
    switch (p) {
    case Pattern::kRandom:
        return "random";
    case Pattern::kSorted:
        return "sorted";
    case Pattern::kReversed:
        return "reversed";
    case Pattern::kConstant:
        return "constant";
    case Pattern::kDuplicates:
        return "duplicates";
    case Pattern::kSignedZeros:
        return "signed_zeros";
    case Pattern::kDynamicRange:
        return "dynamic_range";
    case Pattern::kSteps:
        return "steps";
    case Pattern::kImpulses:
        return "impulses";
    }
    return "?";
}

/// Deterministic finite test series.
inline std::vector<float> make_series(Pattern p, SizeType n, uint64_t seed) {
    loki::math::PCG32 rng(seed);
    std::vector<float> x(n);
    for (SizeType i = 0; i < n; ++i) {
        const float u = uniform01(rng);
        switch (p) {
        case Pattern::kRandom:
            x[i] = (2.0F * u) - 1.0F;
            break;
        case Pattern::kSorted:
            x[i] = static_cast<float>(i) + u;
            break;
        case Pattern::kReversed:
            x[i] = static_cast<float>(n - i) + u;
            break;
        case Pattern::kConstant:
            x[i] = 3.5F;
            break;
        case Pattern::kDuplicates:
            x[i] = std::floor(u * 5.0F);
            break;
        case Pattern::kSignedZeros: {
            const auto k = static_cast<int>(u * 4.0F);
            // NOLINTNEXTLINE(readability-avoid-nested-conditional-operator)
            x[i] = k == 0 ? 0.0F : (k == 1 ? -0.0F : (k == 2 ? 1.0F : -1.0F));
            break;
        }
        case Pattern::kDynamicRange: {
            const float mag = std::pow(10.0F, (40.0F * u) - 20.0F);
            x[i]            = (rng() & 1U) != 0U ? mag : -mag;
            break;
        }
        case Pattern::kSteps:
            x[i] = ((i / 17) % 2 == 0 ? 0.0F : 5.0F) + (0.01F * u);
            break;
        case Pattern::kImpulses:
            x[i] = (i % 13 == 0) ? 100.0F : 0.01F * u;
            break;
        }
    }
    return x;
}

} // namespace loki::test
