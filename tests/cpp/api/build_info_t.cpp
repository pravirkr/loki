#include "loki/common/build_info.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("git_commit names the source of the build", "[build_info]") {
    const std::string_view commit = loki::git_commit();
    REQUIRE_FALSE(commit.empty());
    // A `git describe` result or "unknown": one word.
    REQUIRE(std::ranges::none_of(commit, [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    }));
}
