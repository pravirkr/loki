#pragma once

#include <string_view>

#include "loki/common/plans.hpp"

#include "lib/core/branching.hpp"

namespace loki::plans::detail {

/**
 * @brief The branching simulation of FFAPlanBase::get_branching_pattern
 * (reference segment 0, the FFA tile), reporting every stage's leaves per
 * seed of each frequency cell of the last FFA level to @p on_stage.
 *
 * @return The same forecast as `plan.forecast_branching(poly_basis)` with the
 * FFA tile: its mean, and the worst leaf of that tile.
 */
[[nodiscard]] BranchingForecast
simulate_branching_stages(const FFAPlanBase& plan,
                          std::string_view poly_basis,
                          const core::BranchingStageFn& on_stage);

} // namespace loki::plans::detail
