#pragma once

#include <concepts>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

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
        return std::format("{}:{}:{}: error: {}\n  in function '{}'",
                           loc.file_name(), loc.line(), loc.column(), msg,
                           loc.function_name());
    }
};

// General check with custom comparator and message
template <typename T, typename U, typename Comparator>
inline void check_with_comparator(
    const T& a,
    const U& b,
    Comparator cmp,
    std::string_view op_str,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    if (!cmp(a, b)) {
        std::string composed =
            msg.empty() ? std::format("Check failed: {} {} {}", a, op_str, b)
                        : std::format("{} ({} {} {})", msg, a, op_str, b);
        throw DetailedException(composed, loc);
    }
}

// Specific relational checks
// Check if actual is equal to expected
template <typename T, typename U>
inline void
check_equal(const T& actual,
            const U& expected,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::equal_to<>(), "==", msg, loc);
}

// Check if actual is not equal to expected
template <typename T, typename U>
inline void check_not_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::not_equal_to<>(), "!=", msg,
                          loc);
}

// Check if actual is greater than expected
template <typename T, typename U>
inline void check_greater(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::greater<>(), ">", msg, loc);
}

// Check if actual is less than expected
template <typename T, typename U>
inline void
check_less(const T& actual,
           const U& expected,
           std::string_view msg            = "",
           const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::less<>(), "<", msg, loc);
}

// Check if actual is greater than or equal to expected
template <typename T, typename U>
inline void check_greater_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::greater_equal<>(), ">=", msg,
                          loc);
}

// Check if actual is less than or equal to expected
template <typename T, typename U>
inline void check_less_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::less_equal<>(), "<=", msg,
                          loc);
}

// Check a generic boolean condition
inline void
check(bool condition,
      std::string_view msg,
      const std::source_location& loc = std::source_location::current()) {
    if (!condition) {
        throw DetailedException(msg, loc);
    }
}

// Null pointer check
inline void check_not_null(
    const void* ptr,
    std::string_view msg            = "Pointer must not be null",
    const std::source_location& loc = std::source_location::current()) {
    if (ptr == nullptr) {
        throw DetailedException(msg, loc);
    }
}

// Check if index is within range [0, size)
template <std::integral Index, std::integral Size>
inline void
check_range(Index index,
            Size size,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    if (index < 0 || static_cast<std::make_unsigned_t<Index>>(index) >=
                         static_cast<std::make_unsigned_t<Size>>(size)) {
        std::string composed =
            msg.empty()
                ? std::format("Index {} out of range [0, {})", index, size)
                : std::format("{} (index {} >= size {})", msg, index, size);
        throw DetailedException(composed, loc);
    }
}

} // namespace loki::error_check