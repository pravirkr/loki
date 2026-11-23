#include "loki/common/build_info.hpp"

#include <string_view>

#include "lib/generated/git_commit.hpp"

namespace loki {

// The one user of the generated header, so that a new commit recompiles
// only this file.
std::string_view git_commit() noexcept { return LOKI_GIT_COMMIT; }

} // namespace loki
