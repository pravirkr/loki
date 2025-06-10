#pragma once

#include <format>
#include <source_location>
#include <stdexcept>
#include <string>

namespace loki::error_check {
class DetailedException : public std::runtime_error {
public:
    explicit DetailedException(
        std::string_view user_msg,
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(compose_message(user_msg, loc)) {}

private:
    static std::string compose_message(std::string_view msg,
                                       const std::source_location& loc) {
        return std::format("Error: {}\nIn {} ({}:{})", msg, loc.function_name(),
                           loc.file_name(), loc.line());
    }
};

inline void
check(bool condition,
      std::string_view msg,
      const std::source_location& loc = std::source_location::current()) {
    if (!condition) {
        throw DetailedException(msg, loc);
    }
}

inline void
check_equal(auto a,
            auto b,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    if (a != b) {
        std::string composed = msg.empty()
                                   ? std::format("Check failed: {} != {}", a, b)
                                   : std::format("{} ({} != {})", msg, a, b);
        throw DetailedException(composed, loc);
    }
}

inline void check_not_null(
    const void* ptr,
    std::string_view msg            = "Pointer must not be null",
    const std::source_location& loc = std::source_location::current()) {
    if (ptr == nullptr) {
        throw DetailedException(msg, loc);
    }
}

inline void
check_range(size_t index,
            size_t size,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    if (index >= size) {
        std::string composed =
            msg.empty()
                ? std::format("Index {} out of range [0, {})", index, size)
                : std::format("{} (index {} >= size {})", msg, index, size);
        throw DetailedException(composed, loc);
    }
}

} // namespace loki::error_check