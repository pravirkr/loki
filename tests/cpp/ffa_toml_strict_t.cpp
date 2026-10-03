#include <catch2/catch_test_macros.hpp>

#include "loki/search/configs.hpp"

TEST_CASE("FFATomlConfig rejects unknown keys", "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
unknown_key = true
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig rejects negative nbins", "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = -4
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig rejects use_boxcar_kadane at preflight",
          "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
use_boxcar_kadane = true
)";
    const auto cfg = loki::search::FFATomlConfig::from_string(toml);
    REQUIRE_THROWS_AS(cfg.to_search_config(), std::invalid_argument);
}
