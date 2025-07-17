#include "loki/math.hpp"

namespace loki::math {

std::vector<float> generate_cheb_table(SizeType order_max, SizeType n_derivs) {
    const SizeType dim1       = n_derivs + 1;
    const SizeType dim2       = order_max + 1;
    const SizeType dim3       = order_max + 1;
    const SizeType total_size = dim1 * dim2 * dim3;

    std::vector<float> tab(total_size, 0.0F);

    // Helper lambda for 3D indexing: tab(i, j, k)
    auto idx3d = [=](SizeType i, SizeType j, SizeType k) -> SizeType {
        return (i * dim2 * dim3) + (j * dim3) + k;
    };

    // Base cases: T_0(x) = 1, T_1(x) = x
    tab[idx3d(0, 0, 0)] = 1.0F;
    tab[idx3d(0, 1, 1)] = 1.0F;

    // Generate Chebyshev polynomials using recurrence: T_{n+1}(x) = 2x T_n(x) -
    // T_{n-1}(x)
    for (SizeType jorder = 2; jorder <= order_max; ++jorder) {
        for (SizeType k = 0; k <= order_max; ++k) {
            const float prev2 = tab[idx3d(0, jorder - 2, k)];

            // Shift coefficients right (multiply by x) and apply recurrence
            const float rolled =
                (k > 0) ? tab[idx3d(0, jorder - 1, k - 1)] : 0.0F;
            tab[idx3d(0, jorder, k)] = 2.0F * rolled - prev2;
        }
    }

    // Generate derivatives
    std::vector<float> factor(order_max + 2);
    for (SizeType i = 0; i < order_max + 2; ++i) {
        factor[i] = static_cast<float>(i + 1);
    }

    for (SizeType ideriv = 1; ideriv <= n_derivs; ++ideriv) {
        for (SizeType jorder = 1; jorder <= order_max; ++jorder) {
            for (SizeType k = 0; k <= order_max; ++k) {
                // Shift coefficients left and multiply by factor
                const float prev_deriv =
                    (k < order_max) ? tab[idx3d(ideriv - 1, jorder, k + 1)]
                                    : 0.0F;
                tab[idx3d(ideriv, jorder, k)] = prev_deriv * factor[k];
            }
            // Ensure last coefficient is zero
            tab[idx3d(ideriv, jorder, order_max)] = 0.0F;
        }
    }

    return tab;
}

std::vector<float>
generalized_cheb_pols(SizeType poly_order, float t0, float scale) {
    // Get the base Chebyshev polynomials (no derivatives needed)
    const auto cheb_table = generate_cheb_table(poly_order, 0);
    const SizeType dim2   = poly_order + 1;
    const SizeType dim3   = poly_order + 1;

    // Extract the 2D slice [0, :, :] from the 3D table
    std::vector<float> cheb_pols(dim2 * dim3);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            const SizeType idx_3d =
                (0 * dim2 * dim3) + (i * dim3) + j; // [0, i, j]
            const SizeType idx_2d = (i * dim3) + j;
            cheb_pols[idx_2d]     = cheb_table[idx_3d];
        }
    }

    // Scale the polynomials: scale_factor = (1/scale)^k for k=0..poly_order
    std::vector<float> scale_factor(poly_order + 1);
    float scale_power     = 1.0F;
    const float inv_scale = 1.0F / scale;
    for (SizeType k = 0; k <= poly_order; ++k) {
        scale_factor[k] = scale_power;
        scale_power *= inv_scale;
    }

    // Apply scaling: cheb_pols * scale_factor (broadcast along columns)
    std::vector<float> scaled_pols(dim2 * dim3);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            const SizeType idx = (i * dim3) + j;
            scaled_pols[idx]   = cheb_pols[idx] * scale_factor[j];
        }
    }

    // Shift the origin to t0 using binomial expansion
    std::vector<float> shifted_pols(dim2 * dim3, 0.0F);
    const float neg_t0_over_scale = -t0 / scale;

    for (SizeType iorder = 0; iorder <= poly_order; ++iorder) {
        for (SizeType iterm = 0; iterm <= iorder; ++iterm) {
            const auto binom_coef =
                boost::math::binomial_coefficient<float>(iorder, iterm);
            const float power_term =
                std::pow(neg_t0_over_scale, static_cast<float>(iorder - iterm));

            const SizeType idx = (iorder * dim3) + iterm;
            shifted_pols[idx]  = binom_coef * power_term;
        }
    }

    // Matrix multiplication: result = scaled_pols * shifted_pols
    std::vector<float> result(dim2 * dim3, 0.0F);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            float sum = 0.0F;
            for (SizeType k = 0; k < dim3; ++k) {
                const SizeType scaled_idx  = (i * dim3) + k;
                const SizeType shifted_idx = (k * dim3) + j;
                sum += scaled_pols[scaled_idx] * shifted_pols[shifted_idx];
            }
            result[(i * dim3) + j] = sum;
        }
    }

    return result;
}

} // namespace loki::math