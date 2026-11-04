#include "lib/detail/timing.hpp"

#include <thread>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("QuietScope marks only the calling thread, and nests", "[timing]") {
    REQUIRE_FALSE(loki::timing::quiet());
    {
        const loki::timing::QuietScope loud(false);
        REQUIRE_FALSE(loki::timing::quiet());
    }
    {
        const loki::timing::QuietScope outer(true);
        REQUIRE(loki::timing::quiet());
        {
            const loki::timing::QuietScope inner(true);
            REQUIRE(loki::timing::quiet());
        }
        REQUIRE(loki::timing::quiet());

        bool other_thread_quiet = true;
        std::thread([&] { other_thread_quiet = loki::timing::quiet(); }).join();
        REQUIRE_FALSE(other_thread_quiet);
    }
    REQUIRE_FALSE(loki::timing::quiet());
}
