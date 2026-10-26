#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval    = 299792458.0;            // m/s
inline constexpr double kInvCval = 3.3356409519815204e-09; // s/m
// Cube root of the solar gravitational parameter μ⊙ = G M⊙. Units: m s^(-2/3).
inline constexpr double kGMsunOneThird = 5.10078726793173e6;
// To match the Python output for bitwise consistency
inline constexpr double kFloatEps = 1e-6; // half-up rounding
/// Bytes per "GB" in every memory limit, getter and log: 2^30 (binary, the
/// convention of Slurm's `--mem`, top and htop), so `max_process_memory_gb
/// = 16` is the same amount as `--mem=16G`. See docs/memory.md.
inline constexpr double kBytesPerGB = 1073741824.0;

namespace detail {
/// Hides @p bits from the optimiser. Under -ffast-math the compiler may
/// assume floats are finite and fold a bit test on a bit_cast float to a
/// constant; a volatile round trip keeps the test.
template <typename U> [[nodiscard]] constexpr U opaque_bits(U bits) noexcept {
    if (std::is_constant_evaluated()) {
        return bits;
    }
    volatile U v = bits;
    return v;
}
} // namespace detail

/**
 * @brief IEEE-754 finite check that remains valid with -ffast-math.
 *
 * std::isfinite is folded to true under -ffinite-math-only, and so is a
 * plain exponent-bit test on the float; test the hidden bits instead. Meant
 * for scalar validation; use all_finite() for arrays.
 */
[[nodiscard]] constexpr bool is_finite(float x) noexcept {
    const auto bits = detail::opaque_bits(std::bit_cast<uint32_t>(x));
    return (bits & 0x7F800000U) != 0x7F800000U;
}

[[nodiscard]] constexpr bool is_finite(double x) noexcept {
    const auto bits = detail::opaque_bits(std::bit_cast<uint64_t>(x));
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

/**
 * @brief IEEE-754 NaN check that remains valid with -ffast-math.
 */
[[nodiscard]] constexpr bool is_nan(float x) noexcept {
    return (detail::opaque_bits(std::bit_cast<uint32_t>(x)) & 0x7FFFFFFFU) >
           0x7F800000U;
}

[[nodiscard]] constexpr bool is_nan(double x) noexcept {
    return (detail::opaque_bits(std::bit_cast<uint64_t>(x)) &
            0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL;
}

/**
 * @brief True if every element of @p x is finite; valid with -ffast-math.
 *
 * Reads the elements as integers, so no float value exists for the
 * optimiser to assume finite. Branch-free and vectorisable.
 */
[[nodiscard]] inline bool all_finite(std::span<const float> x) noexcept {
    uint32_t bad = 0;
    for (const float& v : x) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        bad |= static_cast<uint32_t>((bits & 0x7F800000U) == 0x7F800000U);
    }
    return bad == 0;
}

/// @p bytes in GB of 2^30 bytes (kBytesPerGB).
constexpr float to_gib(SizeType bytes) noexcept {
    return static_cast<float>(static_cast<double>(bytes) / kBytesPerGB);
}

/// A count as a power of two for logs, with one decimal: 1048576 is
/// "2^20.0", 3000000 is "2^21.5". Zero is "0".
[[nodiscard]] std::string format_pow2(SizeType n);

/**
 * @brief Allocator for large scratch buffers: 64-byte aligned, and elements
 * of trivially copyable types are left uninitialised.
 *
 * `std::vector<T>::resize` value-initialises (zero-fills) every element; for
 * multi-GB fold buffers that every kernel overwrites before reading, that is
 * a wasted pass over memory plus a page fault per page. The 64-byte alignment
 * suits SIMD loads from the buffer start.
 */
template <typename T, std::size_t Align = 64> struct UninitAlignedAllocator {
    using value_type = T;
    template <typename U> struct rebind {
        using other = UninitAlignedAllocator<U, Align>;
    };

    UninitAlignedAllocator() noexcept = default;
    template <typename U>
    // NOLINTNEXTLINE(google-explicit-constructor): allocator rebind
    UninitAlignedAllocator(const UninitAlignedAllocator<U, Align>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(
            ::operator new(n * sizeof(T), std::align_val_t{Align}));
    }
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        ::operator delete(p, std::align_val_t{Align});
    }
    template <typename U> void construct(U* p) {
        if constexpr (!(std::is_trivially_copyable_v<U> &&
                        std::is_trivially_destructible_v<U>)) {
            ::new (static_cast<void*>(p)) U;
        }
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    template <typename U>
    bool operator==(const UninitAlignedAllocator<U, Align>&) const noexcept {
        return true;
    }
};

/// A vector whose `resize(n)` leaves trivially copyable elements
/// uninitialised (64-byte aligned storage).
template <typename T>
using UninitVector = std::vector<T, UninitAlignedAllocator<T>>;

/**
 * @brief Computes max_i(x[i] - y[i]) for two non-overlapping float arrays.
 *
 * @param x,y Non‑overlapping input arrays with at least @p size elements.
 * @param size Number of elements to process.
 * @return  The maximum of the pair‑wise differences.
 */
float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size) noexcept;

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(const float* __restrict__ x,
                         float* __restrict__ out,
                         SizeType nbins,
                         SizeType nsum) noexcept;

/**
 * @brief  Locate the index of the element **closest** to @p val in a sorted
 *         array.
 *
 * Breaks ties toward the *lower* index, i.e. when `val` lies exactly in the
 * middle of two equal‑distant neighbours the one with the smaller index is
 * returned.
 *
 * @param   arr_sorted  Monotonically non‑decreasing array.
 * @param   val         Search value (must be finite).
 * @param   rtol        Relative tolerance for floating-point comparison
 * (default: 1e-5).
 * @param   atol        Absolute tolerance for floating-point comparison
 *                      (default: 1e-8).
 * @return  Index of the nearest element.
 *
 * @throws  std::invalid_argument if the array is empty or @p val is NaN.
 *
 * @complex **O(log n)** comparisons via `std::ranges::lower_bound`.
 */
[[nodiscard]] SizeType
find_nearest_sorted_idx(std::span<const double> arr_sorted,
                        double val,
                        double rtol = 1e-5,
                        double atol = 1e-8);

/**
 * @brief  Two‑pointer **amortised O(1)** nearest‑index finder.
 *
 * Find the index of the closest value in a sorted array using a linear scan
 * starting from a hint index.
 *
 * In case of a tie, the index of the smaller value is returned.
 * Behavior is undefined if the array is not sorted in ascending order.
 * The hint_idx is updated to the returned index for use in subsequent calls.
 *
 * @param   arr_sorted  Monotonically non‑decreasing array.
 * @param   val         Search value (finite).
 * @param   hint_idx    [in/out]  Reference to an index used as a starting point
 * for the scan, updated to the returned index.
 * @param   rtol        Relative tolerance for floating-point comparison
 * (default: 1e-5).
 * @param   atol        Absolute tolerance for floating-point comparison
 *                      (default: 1e-8).
 * @return  Index of the nearest element (same tie rule as the binary version).
 *
 * @throws  std::invalid_argument if the array is empty, @p val is NaN.
 */
[[nodiscard]] SizeType
find_nearest_sorted_idx_scan(std::span<const double> arr_sorted,
                             double val,
                             SizeType& hint_idx,
                             double rtol = 1e-5,
                             double atol = 1e-8);

/**
 * @brief  Return @p num indices centred (as much as possible) around
 *         @p target_idx.
 *
 *
 * @param   indices      Sorted list of indices.
 * @param   target_idx   Index around which to centre the window.
 * @param   num          Number of neighbours to return (must be > 0).
 * @return  `std::vector` containing the selected neighbours.
 *
 * @throws  std::invalid_argument on empty @p indices or zero @p num.
 */
std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

/**
 * @brief Find the index of the element closest to a given value.
 *
 * Computes the index i such that |range[i] - value| is minimized.
 *
 * @param range Non-empty span of values.
 * @param value Target value.
 * @return Index of the closest element in the range.
 *
 */
SizeType find_nearest_index(std::span<const float> range, float value) noexcept;

/**
 * @brief Find the lower-bin index for a value in a sorted range.
 *
 *
 * @param range Sorted (ascending) span of values.
 * @param value Query value.
 * @return Index of the lower bin (i such that range[i] <= value < range[i+1])
 * @note Returns -1 if value < range[0] (intentional sentinel for underflow)
 * @note Returns range.size() - 1 if value >= range.back()
 *
 */
IndexType find_lower_bin_index(std::span<const float> range,
                               float value) noexcept;

/**
 * @brief  Generate `num_samples` evenly spaced points in [`start`, `stop`].
 *
 * @param   start        First value (included).
 * @param   stop         Last value (included if @p endpoint is true).
 * @param   num_samples  Number of points to generate.
 * @param   endpoint     When *true* the sequence **includes** @p stop.  When
 *                       *false* the sequence ends one step before @p stop.
 * @return  Vector of size @p num_samples (possibly empty).
 *
 * @note    By construction `(stop - start)` is evenly divisible by
 *          `(num_samples - 1)` when `endpoint == true`.
 */
std::vector<double> linspace(double start,
                             double stop,
                             SizeType num_samples = 50,
                             bool endpoint        = true) noexcept;

std::vector<SizeType>
determine_ref_segs(SizeType nsegments,
                   std::optional<SizeType> n_runs,
                   std::optional<std::vector<SizeType>> ref_segs);

std::vector<SizeType>
determine_ref_segs_pareto(SizeType nsegments,
                          std::optional<SizeType> n_runs,
                          std::optional<std::vector<SizeType>> ref_segs);

} // namespace loki::utils
