#pragma once

#include <cstdint>

namespace loki::search {

/// The basis an EP leaf was last propagated in, stored in the leaf's basis
/// slot: the second value of its f0 row (`param_sets[..., -1, 1]` of a run's
/// final leaves). Seeds start as kPolynomial. In a circular-orbit search the
/// transform of every stage records the region its propagation used:
/// kCircularSnap when snap and acceleration resolved the orbit
/// (omega^2 = -snap / acceleration), kCircularCrackle when crackle and jerk
/// did (omega^2 = -crackle / jerk), kPolynomial for the Taylor channel.
/// Polynomial searches never change the slot, and loki reads it only for
/// circular-orbit (five-parameter) leaves: other leaf layouts, such as the
/// Chebyshev basis, may use that position for something else.
///
/// The codes are part of the output format: never renumber them. Values 3
/// and above are reserved for future bases (e.g. a physical orbit basis).
enum class LeafBasis : std::uint8_t {
    kPolynomial      = 0,
    kCircularSnap    = 1,
    kCircularCrackle = 2,
};

/// The leaf slot value of a basis code.
[[nodiscard]] constexpr double leaf_basis_value(LeafBasis basis) noexcept {
    return static_cast<double>(static_cast<std::uint8_t>(basis));
}

/// True when `code` is one of the codes above.
[[nodiscard]] constexpr bool is_known_leaf_basis(std::uint8_t code) noexcept {
    return code <= static_cast<std::uint8_t>(LeafBasis::kCircularCrackle);
}

} // namespace loki::search
