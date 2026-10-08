#include "lib/algorithms/ep_memory.hpp"

#include <cstddef>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"

#include "lib/utils/workspace_impl.hpp"

using Catch::Matchers::WithinRel;
using loki::ComplexType;
using loki::SizeType;
using loki::algorithms::detail::ep_shared_bytes;
using loki::algorithms::detail::ep_workspace_bytes;
using loki::algorithms::detail::kBytesPerGiB;
using loki::memory::EPWorkspaceCPU;
using loki::memory::FFAWorkspaceCPU;

// The planner's memory model must equal what EPFreqSweep really allocates:
// the plans it produces are only as good as these numbers.

TEMPLATE_TEST_CASE("EP memory model matches the per-thread EPWorkspaceCPU",
                   "[ep_memory]",
                   float,
                   ComplexType) {
    constexpr bool kIsComplex     = std::is_same_v<TestType, ComplexType>;
    constexpr SizeType kBatch     = 16;
    constexpr SizeType kBranchMax = 40;
    constexpr SizeType kMaxSugg   = 300;
    constexpr SizeType kNCoords   = 120;
    constexpr SizeType kNBins     = 16;
    constexpr SizeType kNSegments = 24;
    const SizeType nbins_ws       = kIsComplex ? (kNBins / 2) + 1 : kNBins;

    for (const SizeType nparams : {SizeType{2}, SizeType{3}, SizeType{4}}) {
        CAPTURE(nparams);
        const EPWorkspaceCPU<TestType> ws(kBatch, kBranchMax, kMaxSugg,
                                          kNCoords, nparams, nbins_ws,
                                          kNSegments);
        const auto actual =
            static_cast<double>(ws.get_memory_usage_gib()) * kBytesPerGiB;
        const auto model = static_cast<double>(
            ep_workspace_bytes<TestType>(nparams, kNBins, kNSegments, kNCoords,
                                         kMaxSugg, kBranchMax, kBatch));
        CHECK_THAT(actual, WithinRel(model, 1e-5));
    }
}

TEMPLATE_TEST_CASE("EP memory model matches the shared FFA buffers",
                   "[ep_memory]",
                   float,
                   ComplexType) {
    constexpr SizeType kBufferSize = 5000;
    constexpr SizeType kCoordSize  = 700;

    // nparams == 1 uses the frequency-only coordinates.
    for (const SizeType nparams : {SizeType{1}, SizeType{2}, SizeType{3}}) {
        CAPTURE(nparams);
        const FFAWorkspaceCPU<TestType> ffa(kBufferSize, kCoordSize, nparams);
        // EPFreqSweep also allocates an output fold of buffer_size elements.
        const SizeType actual =
            (ffa.fold_internal.size() * sizeof(TestType)) +
            (ffa.coords.size() * sizeof(loki::coord::FFACoord)) +
            (ffa.coords_freq.size() * sizeof(loki::coord::FFACoordFreq)) +
            (kBufferSize * sizeof(TestType));
        CHECK(ep_shared_bytes<TestType>(nparams, kBufferSize, kCoordSize) ==
              actual);
    }
}
