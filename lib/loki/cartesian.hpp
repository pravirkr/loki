#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

// A proxy for the current combination in the Cartesian product.
// It provides read-only access to the combination elements.
class CartesianProductProxy {
public:
    // The constructor copies the current indices.
    constexpr CartesianProductProxy(std::span<const std::vector<float>> params,
                                    std::array<size_t, 5> indices,
                                    size_t dims) noexcept
        : m_params(params),
          m_indices(indices),
          m_dims(dims) {}

    // Access element at position i in the combination.
    constexpr float operator[](size_t i) const noexcept {
        return m_params[i][m_indices[i]];
    }

    // Returns the number of dimensions.
    constexpr size_t size() const noexcept { return m_dims; }

    // Optional: allows iterating over the elements in the combination.
    class ConstIterator {
    public:
        constexpr ConstIterator(const CartesianProductProxy* proxy,
                                size_t pos) noexcept
            : m_proxy(proxy),
              m_pos(pos) {}

        constexpr float operator*() const noexcept { return (*m_proxy)[m_pos]; }

        constexpr ConstIterator& operator++() noexcept {
            ++m_pos;
            return *this;
        }

        constexpr bool operator!=(const ConstIterator& other) const noexcept {
            return m_pos != other.m_pos;
        }

    private:
        const CartesianProductProxy* m_proxy;
        size_t m_pos;
    };

    constexpr ConstIterator begin() const noexcept { return {this, 0}; }
    constexpr ConstIterator end() const noexcept { return {this, m_dims}; }

private:
    std::span<const std::vector<float>> m_params;
    std::array<size_t, 5> m_indices;
    size_t m_dims;
};

/**
 * @brief A view over the Cartesian product of float vectors
 * 
 * Thread Safety: Safe for concurrent reading, but iterators must not be
 * shared between threads.
 * 
 * Note: The input vectors must outlive this view
 */
class CartesianProductIterator {
public:
    // Begin iterator constructor.
    constexpr explicit CartesianProductIterator(
        std::span<const std::vector<float>> params) noexcept
        : m_params(params),
          m_dims(params.size()),
          m_done(false) {
        m_indices.fill(0);
        // Mark as done if there are no dimensions or any vector is empty.
        if (m_dims == 0 || std::ranges::any_of(m_params, [](const auto& vec) {
                return vec.empty();
            })) {
            m_done = true;
        }
    }

    // End iterator constructor.
    constexpr CartesianProductIterator(
        std::span<const std::vector<float>> params, bool /*unused*/)
        : m_params(params),
          m_dims(params.size()),
          m_done(true) {
        m_indices.fill(0);
    }

    // Dereference returns a proxy for the current combination.
    constexpr CartesianProductProxy operator*() const noexcept {
        return {m_params, m_indices, m_dims};
    }

    // Increment the iterator to the next combination.
    constexpr CartesianProductIterator& operator++() noexcept {
        // Loop from the last dimension backwards.
        for (size_t i = m_dims; i-- > 0;) {
            ++m_indices[i];
            if (m_indices[i] < m_params[i].size()) {
                return *this;
            }
            // Reset the current index and continue to the previous dimension.
            m_indices[i] = 0;
        }
        // All combinations have been generated.
        m_done = true;
        return *this;
    }

    constexpr bool
    operator==(const CartesianProductIterator& other) const noexcept {
        return m_done == other.m_done;
    }
    constexpr bool
    operator!=(const CartesianProductIterator& other) const noexcept {
        return !(*this == other);
    }

private:
    std::span<const std::vector<float>> m_params;
    std::array<size_t, 5> m_indices{};
    size_t m_dims;
    bool m_done;
};

// A view that allows range-based iteration over the Cartesian product.
class CartesianProductView
    : public std::ranges::view_interface<CartesianProductView> {
public:
    explicit constexpr CartesianProductView(
        std::span<const std::vector<float>> params) noexcept
        : m_params(params) {}

    constexpr auto begin() const noexcept {
        return CartesianProductIterator(m_params);
    }

    constexpr auto end() const noexcept {
        return CartesianProductIterator(m_params, true);
    }

    constexpr bool empty() const noexcept {
        return m_params.empty() ||
               std::ranges::any_of(m_params,
                                   [](const auto& vec) { return vec.empty(); });
    }

private:
    std::span<const std::vector<float>> m_params;
};

// Helper function to create the Cartesian product view.
constexpr auto
cartesian_product_view(const std::vector<std::vector<float>>& params) {
    return CartesianProductView(std::span<const std::vector<float>>(params));
}

// ------------------------------------------------------------------------
// Example usage:

//#include <iostream>
//
//int main() {
//    std::vector<std::vector<float>> params = {
//        {1.0f, 2.0f}, {10.0f, 20.0f, 30.0f}, {100.0f, 200.0f}};
//
//    for (const auto& combination : cartesian_product_view(params)) {
//        // Each 'combination' is a proxy providing access to the current tuple.
//        for (size_t i = 0; i < combination.size(); ++i) {
//            std::cout << combination[i] << " ";
//        }
//        std::cout << "\n";
//    }
//    return 0;
//}
// ------------------------------------------------------------------------
