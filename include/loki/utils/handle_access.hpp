#pragma once

/**
 * @file handle_access.hpp
 * @brief Forward declaration of the accessor the opaque public handles
 * befriend.
 */

namespace loki::detail {
/// Library-internal accessor to a handle's backend storage
/// (lib/common/dispatch.hpp).
struct HandleAccess;
} // namespace loki::detail
