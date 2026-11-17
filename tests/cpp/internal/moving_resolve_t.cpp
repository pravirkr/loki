#include "moving_resolve.hpp"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "lib/core/chebyshev.hpp"
#include "lib/core/circular.hpp"
#include "lib/core/taylor.hpp"

using namespace moving_resolve;

namespace {

void check(const std::vector<Leaf>& cases, const Resolver& resolve) {
    for (const double t_cur : {kTInit, 18.3719}) {
        INFO("t_cur " << t_cur);
        const auto bad = failures(cases, resolve, t_cur);
        for (const auto& b : bad) {
            UNSCOPED_INFO(b);
        }
        CHECK(bad.empty());
    }
}

} // namespace

TEST_CASE("The moving-grid Taylor resolve picks the cell of the leaf's "
          "absolute frequency",
          "[resolve]") {
    for (const SizeType n_params : {2, 3, 4}) {
        INFO("n_params " << n_params);
        const auto lim = limits(n_params);
        check(taylor_leaves(n_params),
              [&](const std::vector<double>& leaves, SizeType n, double t_add,
                  double t_cur, std::vector<SizeType>& idx,
                  std::vector<float>& phase) {
                  loki::core::poly_taylor_resolve_batch(
                      leaves, idx, phase, lim, {t_add, kScale}, {t_cur, kScale},
                      {kTInit, kScale}, kNAccel, kNFreq, kNbins, n, n_params);
              });
    }
}

TEST_CASE("The moving-grid Chebyshev resolve picks the cell of the leaf's "
          "absolute frequency",
          "[resolve]") {
    for (const SizeType n_params : {2, 3, 4}) {
        INFO("n_params " << n_params);
        const auto lim = limits(n_params);
        check(chebyshev_leaves(n_params),
              [&](const std::vector<double>& leaves, SizeType n, double t_add,
                  double t_cur, std::vector<SizeType>& idx,
                  std::vector<float>& phase) {
                  loki::core::poly_chebyshev_resolve_batch(
                      leaves, idx, phase, lim, {t_add, kScale}, {t_cur, kScale},
                      {kTInit, kScale}, kNAccel, kNFreq, kNbins, n, n_params);
              });
    }
}

TEST_CASE("The moving-grid circular resolve picks the cell of the leaf's "
          "absolute frequency",
          "[resolve][circular]") {
    const auto lim = limits(5);
    check(circular_leaves(),
          [&](const std::vector<double>& leaves, SizeType n, double t_add,
              double t_cur, std::vector<SizeType>& idx,
              std::vector<float>& phase) {
              loki::core::circ_taylor_resolve_batch(
                  leaves, idx, phase, lim, {t_add, kScale}, {t_cur, kScale},
                  {kTInit, kScale}, kNAccel, kNFreq, kNbins, n, kSignif);
          });
}
