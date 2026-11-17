#pragma once

#include <functional>
#include <span>

#include "loki/common/types.hpp"

namespace loki::core {

/**
 * @brief Observer of a branching simulation, called once per pruning stage.
 *
 * @details `stage` is the pruning level minus one. `leaves_per_seed[i]` is
 * the number of leaves that one seed at frequency cell `i` of the simulated
 * grid has after that stage, before any pruning: the product of the cell's
 * branching factors so far (with the circular snap-occupancy factor). The
 * branching pattern's mean of a stage is the ratio of the cell sums of two
 * consecutive stages. The span is valid only during the call.
 */
using BranchingStageFn = std::function<void(
    SizeType stage, std::span<const double> leaves_per_seed)>;

} // namespace loki::core
