#include <catch2/catch_test_macros.hpp>

#include "loki/common/backend.hpp"

TEST_CASE("Backend utilities", "[backend]") {
    SECTION("to_string and parse_backend") {
        REQUIRE(loki::to_string(loki::Backend::kCPU) == "cpu");
        REQUIRE(loki::to_string(loki::Backend::kCUDA) == "cuda");

        REQUIRE(loki::parse_backend("cpu") == loki::Backend::kCPU);
        REQUIRE(loki::parse_backend("cuda") == loki::Backend::kCUDA);
        REQUIRE_THROWS_AS(loki::parse_backend("invalid"),
                          std::invalid_argument);
    }

    SECTION("available_backends") {
        const auto backends = loki::available_backends();
        REQUIRE_FALSE(backends.empty());
        REQUIRE(backends.front() == loki::Backend::kCPU);
        REQUIRE(loki::is_available(loki::Backend::kCPU));
    }

    SECTION("Exec helpers") {
        const auto cpu_exec = loki::Exec::cpu(8);
        REQUIRE(cpu_exec.backend == loki::Backend::kCPU);
        REQUIRE(cpu_exec.nthreads == 8);
        REQUIRE(cpu_exec.device == 0);

        const auto cuda_exec = loki::Exec::cuda(2);
        REQUIRE(cuda_exec.backend == loki::Backend::kCUDA);
        REQUIRE(cuda_exec.nthreads == 1);
        REQUIRE(cuda_exec.device == 2);
    }

    SECTION("DeviceSpan operations") {
        int dummy = 42;
        loki::Device dev{.backend = loki::Backend::kCPU, .id = 0};
        loki::DeviceSpan<int> span(&dummy, 1, dev);

        REQUIRE_FALSE(span.empty());
        REQUIRE(span.size() == 1);
        REQUIRE(span.size_bytes() == sizeof(int));
        REQUIRE(span.data() == &dummy);

        loki::DeviceSpan<const int> const_span = span;
        REQUIRE(const_span.data() == &dummy);
        REQUIRE(const_span.size() == 1);
    }
}
