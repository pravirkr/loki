#include "loki/math.hpp"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace loki::math {

xt::xtensor<float, 3> generate_cheb_table(SizeType order_max,
                                          SizeType n_derivs) {
    xt::xtensor<float, 3> tab({n_derivs + 1, order_max + 1, order_max + 1},
                              0.0F);
    // Base cases: T_0(x) = 1, T_1(x) = x
    tab(0, 0, 0) = 1.0F;
    tab(0, 1, 1) = 1.0F;
    // Generate Chebyshev polynomials using recurrence: T_{n+1}(x) = 2x T_n(x) -
    // T_{n-1}(x)
    for (SizeType jorder = 2; jorder <= order_max; ++jorder) {
        auto prev  = xt::view(tab, 0, jorder - 1, xt::all());
        auto prev2 = xt::view(tab, 0, jorder - 2, xt::all());
        // Shift coefficients right (multiply by x) and apply recurrence
        auto rolled                         = xt::roll(prev, 1);
        xt::view(tab, 0, jorder, xt::all()) = 2.0F * rolled - prev2;
    }
    // Generate derivatives
    auto factor = xt::linspace<float>(1.0F, static_cast<float>(order_max + 2),
                                      order_max + 2);
    for (SizeType ideriv = 1; ideriv <= n_derivs; ++ideriv) {
        for (SizeType jorder = 1; jorder <= order_max; ++jorder) {
            auto prev_deriv = xt::view(tab, ideriv - 1, jorder, xt::all());
            // Shift coefficients left and multiply by factor
            auto rolled                              = xt::roll(prev_deriv, -1);
            xt::view(tab, ideriv, jorder, xt::all()) = rolled * factor;
            tab(ideriv, jorder, order_max) =
                0.0F; // Ensure last coefficient is zero
        }
    }
    return tab;
}

xt::xtensor<float, 2>
generalized_cheb_pols(SizeType poly_order, float t0, float scale) {
    // Get the base Chebyshev polynomials (no derivatives needed)
    xt::xtensor<float, 2> cheb_pols =
        xt::view(generate_cheb_table(poly_order, 0), 0, xt::all(), xt::all());

    // Scale the polynomials
    auto scale_factor = xt::pow(
        1.0F / scale, xt::arange<float>(static_cast<float>(poly_order + 1)));
    xt::xtensor<float, 2> scaled_pols =
        cheb_pols * xt::view(scale_factor, xt::newaxis(), xt::all());

    // Shift the origin to t0 using binomial expansion
    xt::xtensor<float, 2> shifted_pols({poly_order + 1, poly_order + 1}, 0.0F);
    for (SizeType iorder = 0; iorder <= poly_order; ++iorder) {
        for (SizeType iterm = 0; iterm <= iorder; ++iterm) {
            const auto binom_coef =
                boost::math::binomial_coefficient<float>(iorder, iterm);
            shifted_pols(iorder, iterm) =
                binom_coef *
                std::pow(-t0 / scale, static_cast<float>(iorder - iterm));
        }
    }

    // Matrix multiplication to combine scaled and shifted polynomials
    return xt::linalg::dot(scaled_pols, shifted_pols);
}

} // namespace loki::math