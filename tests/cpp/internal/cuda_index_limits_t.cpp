#include <catch2/catch_test_macros.hpp>

#include "lib/detail/index_limits.hpp"

using loki::index_limits::chunk_exceeds_cuda_index_limits;
using loki::index_limits::ChunkIndexUsage;

TEST_CASE("CUDA chunk index guard trips on oversized buffers",
          "[index_limits]") {
    ChunkIndexUsage ok{1000, 100, 64, 8, 32, 512};
    REQUIRE_FALSE(chunk_exceeds_cuda_index_limits(ok));

    ChunkIndexUsage huge_buffer{1ULL << 32U, 10, 64, 8, 10, 512};
    REQUIRE(chunk_exceeds_cuda_index_limits(huge_buffer));
}
