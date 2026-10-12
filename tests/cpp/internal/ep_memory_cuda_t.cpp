#ifdef LOKI_ENABLE_CUDA

#include "lib/algorithms/ep_memory_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/dynamic_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

using loki::Backend;
using loki::ComplexType;
using loki::ComplexTypeCUDA;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::detail::ep_cuda_cub_scratch_bytes;
using loki::algorithms::detail::ep_cuda_effective_max_sugg;
using loki::algorithms::detail::ep_cuda_ep_workspace_bytes;
using loki::algorithms::detail::ep_cuda_irfft_scratch_bytes;
using loki::algorithms::detail::ep_cuda_shared_bytes;
using loki::algorithms::detail::EPMemoryContext;
using loki::algorithms::detail::EPMemoryKind;
using loki::algorithms::detail::fold_bytes;
using loki::core::create_prune_dp_functs_cuda;
using loki::memory::EPWorkspaceCUDA;
using loki::memory::FFAWorkspaceCUDA;
using loki::search::PulsarSearchConfig;

// The CUDA policy of the planner's memory model must equal what EPFreqSweep
// allocates on the device. These tests build the real workspaces and compare.

namespace {

constexpr SizeType kBatch     = 64;
constexpr SizeType kBranchMax = 16;
constexpr SizeType kNsegments = 8;
constexpr SizeType kNbins     = 32;
constexpr SizeType kNcoords   = 500;

EPMemoryContext make_ctx(SizeType nparams) {
    EPMemoryContext ctx;
    ctx.nparams           = nparams;
    ctx.nsamps            = 1U << 16U;
    ctx.n_workers         = 1;
    ctx.batch_size        = kBatch;
    ctx.kind              = EPMemoryKind::kCuda;
    ctx.device            = 0;
    ctx.cub_scratch_bytes = &ep_cuda_cub_scratch_bytes;
    return ctx;
}

PulsarSearchConfig
make_cfg(SizeType nparams, bool kadane = false, SizeType nbins = kNbins) {
    std::vector<ParamLimit> limits(nparams,
                                   ParamLimit{.min = -1.0, .max = 1.0});
    limits.back() = ParamLimit{.min = 140.0, .max = 145.0};
    return {1U << 16U,
            64e-6,
            nbins,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/std::max<SizeType>(1024, nbins),
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/(1U << 16U) / 8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 20U,
            /*prune_poly_order=*/2,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0,
            /*m_p_min=*/1.4,
            /*propagator_significance=*/2.0,
            /*validation_significance=*/5.0,
            /*use_conservative_tile=*/false,
            /*use_boxcar_kadane=*/kadane};
}

} // namespace

TEMPLATE_TEST_CASE("CUDA EP memory model matches EPWorkspaceCUDA",
                   "[ep_memory][cuda]",
                   float,
                   ComplexType) {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    using FoldCUDA = std::conditional_t<std::is_same_v<TestType, float>, float,
                                        ComplexTypeCUDA>;
    const auto nparams = GENERATE(SizeType{1}, SizeType{2});
    const auto max_sugg =
        GENERATE(SizeType{1024}, SizeType{1} << 16U, SizeType{1} << 12U);
    CAPTURE(nparams, max_sugg);
    const auto ctx = make_ctx(nparams);

    // Allocation granularity of cudaMalloc: the free-memory delta is the
    // model plus at most this much per buffer.
    constexpr SizeType kSlack = SizeType{96} << 20U;

    // The engine passes the folded bin count: nbins / 2 + 1 for Fourier folds.
    constexpr SizeType kNbinsArg =
        std::is_same_v<TestType, ComplexType> ? (kNbins / 2) + 1 : kNbins;

    const auto free_before = loki::cuda_utils::get_cuda_memory_usage().first;
    EPWorkspaceCUDA<FoldCUDA> workspace(
        kBatch, kBranchMax,
        ep_cuda_effective_max_sugg(kBatch, kBranchMax, max_sugg), kNcoords,
        nparams, kNbinsArg, kNsegments);
    const auto free_after = loki::cuda_utils::get_cuda_memory_usage().first;

    const auto model = ep_cuda_ep_workspace_bytes<TestType>(
        ctx, kNbins, kNsegments, kNcoords, max_sugg, kBranchMax);
    // Exact: the model mirrors the members one by one.
    CHECK(workspace.get_memory_usage_bytes() == model);

    // The device agrees, up to the allocator's granularity.
    const auto delta = static_cast<SizeType>((free_before - free_after) *
                                             static_cast<double>(1ULL << 30U));
    CHECK(delta + kSlack >= model);
    CHECK(delta <= model + kSlack);
}

TEMPLATE_TEST_CASE("CUDA EP memory model matches FFAWorkspaceCUDA",
                   "[ep_memory][cuda]",
                   float,
                   ComplexType) {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    using FoldCUDA = std::conditional_t<std::is_same_v<TestType, float>, float,
                                        ComplexTypeCUDA>;
    const auto nparams         = GENERATE(SizeType{1}, SizeType{2});
    constexpr SizeType kBuffer = 40000;
    constexpr SizeType kCoords = 700;
    CAPTURE(nparams);

    FFAWorkspaceCUDA<FoldCUDA> workspace(kBuffer, kCoords, /*n_levels=*/6,
                                         nparams);
    // shared = workspace buffers + the output fold buffer.
    const auto output_fold_bytes = kBuffer * fold_bytes<TestType>();
    CHECK(workspace.get_buffers_bytes() + output_fold_bytes ==
          ep_cuda_shared_bytes<TestType>(nparams, kBuffer, kCoords));
}

TEST_CASE("CUDA EP memory model: world-tree capacity floor",
          "[ep_memory][cuda]") {
    // The world tree needs a capacity above the largest batch it receives.
    CHECK(ep_cuda_effective_max_sugg(64, 16, 100) == (64 * 16) + 1);
    CHECK(ep_cuda_effective_max_sugg(64, 16, 5000) == 5000);
}

TEST_CASE("CUDA EP memory model matches the prune irfft scratch",
          "[ep_memory][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto ctx = make_ctx(2);
    // n_coords_init is the product of the grid, which the model takes as
    // ncoords_ffa.
    const std::vector<SizeType> grid{20U, 25U};
    const std::vector<double> dparams{0.1, 0.1};
    REQUIRE(grid[0] * grid[1] == kNcoords);

    const auto bytes_of = [](const auto& functs) {
        return static_cast<SizeType>(std::llround(
            static_cast<double>(functs->get_irfft_scratch_memory_gib()) *
            static_cast<double>(1ULL << 30U)));
    };
    // Boxcar and Kadane (both fused), Fourier and float folds; and Fourier
    // folds above the on-chip limit, which keep the irfft scratch.
    for (const bool kadane : {false, true}) {
        for (const SizeType nbins : {kNbins, SizeType{2048}}) {
            CAPTURE(kadane, nbins);
            const auto fourier = create_prune_dp_functs_cuda<ComplexTypeCUDA>(
                "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
                make_cfg(2, kadane, nbins), kBatch, kBranchMax,
                /*device_id=*/0);
            CHECK(bytes_of(fourier) == ep_cuda_irfft_scratch_bytes<ComplexType>(
                                           ctx, nbins, kNcoords, kBranchMax));
            const auto time_domain = create_prune_dp_functs_cuda<float>(
                "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
                make_cfg(2, kadane, nbins), kBatch, kBranchMax,
                /*device_id=*/0);
            CHECK(bytes_of(time_domain) ==
                  ep_cuda_irfft_scratch_bytes<float>(ctx, nbins, kNcoords,
                                                     kBranchMax));
        }
    }
}

TEST_CASE("CUDA EP memory model: irfft scratch only for unfused scoring",
          "[ep_memory][cuda]") {
    auto ctx = make_ctx(2);
    // Every case holds the fused scorer's statistics and tile states
    // (8-byte words); a one-element stand-in replaces an unused scratch.
    const SizeType tiles    = (2 + (kBatch * kBranchMax) + 1) * 8;
    const SizeType stand_in = sizeof(ComplexType) + sizeof(float);
    CHECK(ep_cuda_irfft_scratch_bytes<float>(ctx, kNbins, kNcoords,
                                             kBranchMax) == stand_in + tiles);
    CHECK(ep_cuda_irfft_scratch_bytes<ComplexType>(
              ctx, kNbins, kNcoords, kBranchMax) == stand_in + tiles);
    // Fourier nbins above the on-chip limit (the boxcar and the Kadane scorer
    // alike): max(2 * batch * branch_max, 2 * ncoords) transforms of nbins.
    const auto full = [&](SizeType nbins) {
        const SizeType nfft = std::max(2 * kBatch * kBranchMax, 2 * kNcoords);
        return (nfft * ((nbins / 2) + 1) * sizeof(ComplexType)) +
               (nfft * nbins * sizeof(float)) + tiles;
    };
    CHECK(ep_cuda_irfft_scratch_bytes<ComplexType>(ctx, 2048, kNcoords,
                                                   kBranchMax) == full(2048));
}

TEST_CASE("CUDA prune functors take an external FFT manager without planning",
          "[ep_memory][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const std::vector<SizeType> grid{20U, 25U};
    const std::vector<double> dparams{0.1, 0.1};

    // The sweep prepares the plans once. The functor must reuse them and
    // build none of its own.
    // Fourier nbins above the on-chip limit: the batches still go through
    // the irfft.
    constexpr SizeType kBigNbins = 2048;
    loki::math::CUFFTManager shared(/*device_id=*/0);
    const std::vector<SizeType> n_reals{kBigNbins};
    shared.prepare_exact_plans(n_reals);
    const auto functs = create_prune_dp_functs_cuda<ComplexTypeCUDA>(
        "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
        make_cfg(2, /*kadane=*/true, kBigNbins), kBatch, kBranchMax,
        /*device_id=*/0, &shared);
    REQUIRE(functs != nullptr);
    CHECK(shared.has_prepared(kBigNbins));
    CHECK(shared.n_cached_plans() == 0);

    // A manager that does not prepare this nbins is refused.
    loki::math::CUFFTManager unprepared(/*device_id=*/0);
    CHECK_THROWS(create_prune_dp_functs_cuda<ComplexTypeCUDA>(
        "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
        make_cfg(2, /*kadane=*/true, kBigNbins), kBatch, kBranchMax,
        /*device_id=*/0, &unprepared));
    // The fused scoring (boxcar or Kadane) needs no plans at all.
    for (const bool kadane : {false, true}) {
        CHECK_NOTHROW(create_prune_dp_functs_cuda<ComplexTypeCUDA>(
            "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
            make_cfg(2, kadane), kBatch, kBranchMax, /*device_id=*/0,
            &unprepared));
    }
}

#endif // LOKI_ENABLE_CUDA
