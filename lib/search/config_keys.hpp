#pragma once

#include <string>
#include <vector>

namespace loki::search {

/// Every key of the FFA config schema as "table.key" (and
/// "search.orbit.key"), for the test that keeps the template in step.
[[nodiscard]] std::vector<std::string> ffa_config_keys();

/// Every key of the EP config schema, as ffa_config_keys().
[[nodiscard]] std::vector<std::string> ep_config_keys();

} // namespace loki::search
