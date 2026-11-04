#pragma once

#include <string>
#include <vector>

namespace loki::simulation {

/// Every key of the simulation config schema as "table.key", for the test
/// that keeps the template in step.
[[nodiscard]] std::vector<std::string> simulate_config_keys();

} // namespace loki::simulation
