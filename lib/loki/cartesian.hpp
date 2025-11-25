#pragma once

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

/**
 * @brief Proxy for the current combination in the Cartesian product.
 * Provides read-only access to the combination elements.
 */
template <typename T, SizeType MaxDims = 5> class CartesianProductProxy {
public:
    using Container  = std::vector<T>;
    using ParamsSpan = std::span<const Container>;
    using Indices    = std::array<SizeType, MaxDims>;

    constexpr CartesianProductProxy(ParamsSpan params,
                                    const Indices& indices,
                                    SizeType dims) noexcept
        : m_params(params),
          m_indices(indices),
          m_dims(dims) {}

    // Read-only element access
    constexpr const T& operator[](SizeType i) const noexcept {
        return m_params[i][m_indices[i]];
    }
    constexpr SizeType size() const noexcept { return m_dims; }

    // Iterator for range-based for over the proxy
    class ConstIterator {
    public:
        using IteratorConcept  = std::forward_iterator_tag;
        using IteratorCategory = std::forward_iterator_tag;

        constexpr ConstIterator(const CartesianProductProxy* proxy,
                                SizeType pos) noexcept
            : m_proxy(proxy),
              m_pos(pos) {}

        constexpr const T& operator*() const noexcept {
            return (*m_proxy)[m_pos];
        }
        constexpr ConstIterator& operator++() noexcept {
            ++m_pos;
            return *this;
        }
        constexpr bool operator==(const ConstIterator& other) const noexcept {
            return m_proxy == other.m_proxy && m_pos == other.m_pos;
        }
        constexpr bool operator!=(const ConstIterator& other) const noexcept {
            return !(*this == other);
        }

    private:
        const CartesianProductProxy* m_proxy;
        SizeType m_pos;
    };

    constexpr ConstIterator begin() const noexcept { return {this, 0}; }
    constexpr ConstIterator end() const noexcept { return {this, m_dims}; }

private:
    ParamsSpan m_params;
    Indices m_indices;
    SizeType m_dims;
};

/**
 * @brief Iterator for the Cartesian product of float/double vectors.
 *
 * Thread Safety: Safe for concurrent reading, but iterators must not be
 * shared between threads.
 *
 * Note: The input vectors must outlive this view
 */
template <typename T, SizeType MaxDims = 5> class CartesianProductIterator {
public:
    using Container        = std::vector<T>;
    using ParamsSpan       = std::span<const Container>;
    using Indices          = std::array<SizeType, MaxDims>;
    using IteratorConcept  = std::forward_iterator_tag;
    using IteratorCategory = std::forward_iterator_tag;

    // End sentinel
    struct SentinelT {};

    // Begin iterator constructor.
    constexpr explicit CartesianProductIterator(ParamsSpan params)
        : m_params(params),
          m_dims(params.size()),
          m_done(false) {
        if (m_dims > MaxDims) {
            throw std::invalid_argument(
                std::format("Too many dimensions for CartesianProductIterator "
                            "(got {}, max {})",
                            m_dims, MaxDims));
        }
        m_indices.fill(0);
        // Mark as done if there are no dimensions or any vector is empty.
        if (m_dims == 0 || std::ranges::any_of(m_params, [](const auto& v) {
                return v.empty();
            })) {
            m_done = true;
        }
    }

    // Dereference returns a proxy for the current combination.
    constexpr CartesianProductProxy<T, MaxDims> operator*() const noexcept {
        return {m_params, m_indices, m_dims};
    }

    // Increment the iterator to the next combination (++ (odometer))
    constexpr CartesianProductIterator& operator++() noexcept {
        // Loop from the last dimension backwards.
        for (SizeType i = m_dims; i-- > 0;) {
            ++m_indices[i];
            if (m_indices[i] < m_params[i].size()) {
                ++m_flat_index;
                return *this;
            }
            // Reset the current index and continue to the previous dimension.
            m_indices[i] = 0;
        }
        // All combinations have been generated.
        m_done = true;
        ++m_flat_index; // becomes one-past-last; harmless if queried after end
        return *this;
    }

    constexpr SizeType flat_index() const noexcept { return m_flat_index; }

    // Iterator–iterator equality (same view & same state)
    constexpr bool
    operator==(const CartesianProductIterator& other) const noexcept {
        return m_params.data() == other.m_params.data() &&
               m_params.size() == other.m_params.size() &&
               m_done == other.m_done &&
               (m_done || m_indices == other.m_indices);
    }
    constexpr bool
    operator!=(const CartesianProductIterator& other) const noexcept {
        return !(*this == other);
    }

    // Iterator–sentinel equality
    friend constexpr bool operator==(const CartesianProductIterator& it,
                                     SentinelT /*unused*/) noexcept {
        return it.m_done;
    }
    friend constexpr bool
    operator==(SentinelT /*unused*/,
               const CartesianProductIterator& it) noexcept {
        return it.m_done;
    }
    friend constexpr bool operator!=(const CartesianProductIterator& it,
                                     SentinelT s) noexcept {
        return !(it == s);
    }
    friend constexpr bool
    operator!=(SentinelT s, const CartesianProductIterator& it) noexcept {
        return !(it == s);
    }

private:
    ParamsSpan m_params;
    Indices m_indices;
    SizeType m_dims;
    bool m_done{true};
    SizeType m_flat_index{};
};

/**
 * @brief
 *
 */
template <typename T, SizeType MaxDims = 5>
class CartesianProductView
    : public std::ranges::view_interface<CartesianProductView<T, MaxDims>> {
public:
    using Container  = std::vector<T>;
    using ParamsSpan = std::span<const Container>;
    using Iterator   = CartesianProductIterator<T, MaxDims>;
    using Sentinel   = typename Iterator::SentinelT;

    constexpr explicit CartesianProductView(ParamsSpan params) noexcept
        : m_params(params) {}

    constexpr Iterator begin() const noexcept { return Iterator(m_params); }
    constexpr Sentinel end() const noexcept { return {}; }

    constexpr bool empty() const noexcept {
        return m_params.empty() ||
               std::ranges::any_of(m_params,
                                   [](const auto& vec) { return vec.empty(); });
    }

    constexpr std::optional<SizeType> size() const noexcept {
        if (empty()) {
            return SizeType{0};
        }
        SizeType prod        = 1;
        constexpr auto kMaxv = std::numeric_limits<SizeType>::max();
        for (const auto& v : m_params) {
            const auto n = static_cast<SizeType>(v.size());
            if (n == 0) {
                return SizeType{0};
            }
            if (prod != 0 && n > 0 && prod > kMaxv / n) {
                // overflow — unknown in SizeType
                return std::nullopt;
            }
            prod *= n;
        }
        return prod;
    }

private:
    ParamsSpan m_params;
};

/**
 * @brief Helper function to create the Cartesian product view.
 * @tparam T      Floating-point type (float/double)
 * @tparam MaxDims Maximum number of dimensions supported (default: 5)
 */
template <typename T, SizeType MaxDims = 5>
constexpr auto
cartesian_product_view(const std::vector<std::vector<T>>& params) {
    return CartesianProductView<T, MaxDims>(
        std::span<const std::vector<T>>(params));
}

template <typename T, SizeType MaxDims = 5>
constexpr auto cartesian_product_view(std::span<const std::vector<T>> params) {
    return CartesianProductView<T, MaxDims>(params);
}

// Helper function for cartesian product with padded arrays
inline std::tuple<std::vector<double>, std::vector<SizeType>>
cartesian_prod_padded(std::span<const double> padded_arrays,
                      std::span<const SizeType> actual_counts,
                      SizeType n_batch,
                      SizeType nparams,
                      SizeType branch_max) {

    std::vector<SizeType> items_per_batch(n_batch);
    SizeType total_items = 0;

    // First pass: Calculate total items and items per batch
    for (SizeType i = 0; i < n_batch; ++i) {
        SizeType count_i = 1;
        for (SizeType j = 0; j < nparams; ++j) {
            count_i *= actual_counts[(i * nparams) + j];
        }
        items_per_batch[i] = count_i;
        total_items += count_i;
    }

    std::vector<double> cart_prod(total_items * nparams);
    std::vector<SizeType> origins(total_items);

    // Second pass: Generate combinations and fill arrays
    SizeType current_row_idx = 0;
    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType num_items_i = items_per_batch[i];

        // Fill origins for this batch
        std::ranges::fill(
            origins.begin() + static_cast<IndexType>(current_row_idx),
            origins.begin() +
                static_cast<IndexType>(current_row_idx + num_items_i),
            i);

        // Generate Cartesian product for batch 'i'
        std::vector<SizeType> indices(nparams, 0);

        for (SizeType item_row_idx = 0; item_row_idx < num_items_i;
             ++item_row_idx) {
            // Fill current combination
            const SizeType output_row = current_row_idx + item_row_idx;
            for (SizeType k = 0; k < nparams; ++k) {
                const SizeType padded_idx =
                    (i * nparams * branch_max) + (k * branch_max) + indices[k];
                cart_prod[(output_row * nparams) + k] =
                    padded_arrays[padded_idx];
            }

            // Odometer increment (except for last item)
            if (item_row_idx < num_items_i - 1) {
                for (SizeType k = nparams; k > 0; --k) {
                    const SizeType idx = k - 1;
                    const SizeType max_idx_k =
                        actual_counts[(i * nparams) + idx] - 1;
                    if (indices[idx] < max_idx_k) {
                        ++indices[idx];
                        break;
                    }
                    indices[idx] = 0;
                }
            }
        }
        current_row_idx += num_items_i;
    }

    return {std::move(cart_prod), std::move(origins)};
}
} // namespace loki::utils

// ------------------------------------------------------------------------
// Example usage:

// #include <iostream>
//
// int main() {
//     std::vector<std::vector<float>> params = {
//         {1.0f, 2.0f}, {10.0f, 20.0f, 30.0f}, {100.0f, 200.0f}};
//
//     for (const auto& combination : cartesian_product_view(params)) {
//         // Each 'combination' is a proxy providing access to the current
//         //tuple.
//         for (size_t i = 0; i < combination.size(); ++i) {
//             std::cout << combination[i] << " ";
//             // combo[i] is const T&
//         }
//         std::cout << "\n";
//     }
//     return 0;
// }
//  ------------------------------------------------------------------------
