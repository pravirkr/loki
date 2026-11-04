#include "loki/common/progress.hpp"

#include <atomic>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

namespace loki {

namespace {
std::atomic<ProgressMode>& mode_slot() noexcept {
    static std::atomic<ProgressMode> slot{ProgressMode::kAuto};
    return slot;
}
} // namespace

std::string_view to_string(ProgressMode mode) noexcept {
    switch (mode) {
    case ProgressMode::kAuto:
        return "auto";
    case ProgressMode::kRich:
        return "rich";
    case ProgressMode::kPlain:
        return "plain";
    case ProgressMode::kOff:
        return "off";
    }
    return "unknown";
}

ProgressMode parse_progress_mode(std::string_view name) {
    for (const auto m : {
             ProgressMode::kAuto,
             ProgressMode::kRich,
             ProgressMode::kPlain,
             ProgressMode::kOff,
         }) {
        if (name == to_string(m)) {
            return m;
        }
    }
    throw std::invalid_argument(
        fmt::format("Unknown progress mode '{}'. Expected 'auto', 'rich', "
                    "'plain' or 'off'",
                    name));
}

void set_progress_mode(ProgressMode mode) noexcept {
    mode_slot().store(mode, std::memory_order_relaxed);
}

ProgressMode get_progress_mode() noexcept {
    return mode_slot().load(std::memory_order_relaxed);
}

} // namespace loki
