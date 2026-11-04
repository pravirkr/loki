#include <algorithm>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/search/configs.hpp"
#include "loki/simulation/configs.hpp"

#include "lib/search/config_keys.hpp"
#include "lib/simulation/config_keys.hpp"

namespace {

/// The keys a template documents, set or commented out ("# key = ..."), as
/// "table.key" under the last "[table]" or "# [table]" header.
std::set<std::string> documented_keys(std::string_view text) {
    static const std::regex kHeader(R"(^\s*#?\s*\[([a-z_.]+)\]\s*(#.*)?$)");
    static const std::regex kKey(R"(^\s*#?\s*([a-z_0-9]+)\s*=)");
    std::set<std::string> keys;
    std::string table;
    std::size_t begin = 0;
    while (begin < text.size()) {
        auto end = text.find('\n', begin);
        end      = end == std::string_view::npos ? text.size() : end;
        const std::string line(text.substr(begin, end - begin));
        begin = end + 1;
        std::smatch match;
        if (std::regex_match(line, match, kHeader)) {
            table = match[1];
        } else if (!table.empty() && std::regex_search(line, match, kKey)) {
            keys.insert(table + "." + match[1].str());
        }
    }
    return keys;
}

/// Every schema key is documented in the template, and the template
/// documents no key the schema lacks. Aliases are read but not documented.
void check_in_step(std::vector<std::string> schema,
                   std::string_view text,
                   const std::set<std::string>& aliases) {
    const auto documented = documented_keys(text);
    for (const auto& key : schema) {
        if (!aliases.contains(key)) {
            CAPTURE(key);
            CHECK(documented.contains(key));
        }
    }
    std::ranges::sort(schema);
    for (const auto& key : documented) {
        CAPTURE(key);
        CHECK(std::ranges::binary_search(schema, key));
    }
}

} // namespace

TEST_CASE("The FFA template documents exactly the schema keys",
          "[config][ffa]") {
    check_in_step(loki::search::ffa_config_keys(),
                  loki::search::FFATomlConfig::default_toml_string(),
                  {"input.dt"});
}

TEST_CASE("The EP template documents exactly the schema keys", "[config][ep]") {
    check_in_step(loki::search::ep_config_keys(),
                  loki::search::EPTomlConfig::default_toml_string(),
                  {"input.dt"});
}

TEST_CASE("The simulation template documents exactly the schema keys",
          "[config][simulate]") {
    check_in_step(loki::simulation::simulate_config_keys(),
                  loki::simulation::SimulateTomlConfig::default_toml_string(),
                  {});
}
