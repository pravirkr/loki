#ifdef LOKI_ENABLE_CUDA

#include "lib/algorithms/ep_memory_cuda.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
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
#include "lib/cuda/prune_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/detection/thresholds_engine.hpp"
#include "lib/pipelines/ep_sweep_common.hpp"

using loki::Backend;
using loki::ComplexType;
using loki::ComplexTypeCUDA;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::detail::ep_cuda_cub_scratch_bytes;
using loki::algorithms::detail::ep_cuda_effective_max_sugg;
using loki::algorithms::detail::ep_cuda_ep_workspace_bytes;
using loki::algorithms::detail::ep_cuda_scorer_scratch_bytes;
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
    const auto nbins = GENERATE(kNbins, SizeType{1024});
    CAPTURE(nparams, max_sugg, nbins);
    const auto ctx = make_ctx(nparams);

    // Allocation granularity of cudaMalloc: the free-memory delta is the
    // model plus at most this much per buffer.
    constexpr SizeType kSlack = SizeType{96} << 20U;

    // The engine passes the folded bin count: nbins / 2 + 1 for Fourier folds.
    const SizeType nbins_arg =
        std::is_same_v<TestType, ComplexType> ? (nbins / 2) + 1 : nbins;

    const auto free_before = loki::cuda_utils::get_cuda_memory_usage().first;
    EPWorkspaceCUDA<FoldCUDA> workspace(
        kBatch, kBranchMax,
        ep_cuda_effective_max_sugg(kBatch, kBranchMax, max_sugg), kNcoords,
        nparams, nbins_arg, kNsegments);
    REQUIRE_NOTHROW(workspace.prune.validate(kBatch, kBranchMax, kNsegments));
    const auto free_after = loki::cuda_utils::get_cuda_memory_usage().first;

    const auto model = ep_cuda_ep_workspace_bytes<TestType>(
        ctx, nbins, kNsegments, kNcoords, max_sugg, kBranchMax);
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
            static_cast<double>(functs->get_scorer_scratch_memory_gib()) *
            static_cast<double>(1ULL << 30U)));
    };
    // Boxcar and Kadane (both fused), Fourier and float folds; and Fourier
    // the maximum supported fold width.
    for (const bool kadane : {false, true}) {
        for (const SizeType nbins : {kNbins, SizeType{1024}}) {
            CAPTURE(kadane, nbins);
            const auto fourier = create_prune_dp_functs_cuda<ComplexTypeCUDA>(
                "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
                make_cfg(2, kadane, nbins), kBatch, kBranchMax,
                /*device_id=*/0);
            CHECK(bytes_of(fourier) ==
                  ep_cuda_scorer_scratch_bytes(ctx, kBranchMax));
            const auto time_domain = create_prune_dp_functs_cuda<float>(
                "taylor", grid, dparams, kNsegments, /*tseg_ffa=*/1.0,
                make_cfg(2, kadane, nbins), kBatch, kBranchMax,
                /*device_id=*/0);
            CHECK(bytes_of(time_domain) ==
                  ep_cuda_scorer_scratch_bytes(ctx, kBranchMax));
        }
    }
}

TEST_CASE("CUDA EP memory model: only fused statistics scratch",
          "[ep_memory][cuda]") {
    const auto ctx       = make_ctx(2);
    const SizeType tiles = (2 + kBatch * kBranchMax + 1) * sizeof(uint64_t);
    CHECK(ep_cuda_scorer_scratch_bytes(ctx, kBranchMax) == tiles);
}

TEMPLATE_TEST_CASE("CUDA EP shared group seeds only each chunk's coordinates",
                   "[ep][cuda][workspace]",
                   float,
                   ComplexTypeCUDA) {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not built");
    }
    const loki::cuda_utils::CudaSetDeviceGuard guard(0);
    using HostFoldT   = loki::HostFoldType<TestType>;
    using DeviceFoldT = loki::DeviceFoldType<TestType>;
    const auto config = [](double f_max) {
        const std::vector<ParamLimit> limits{{.min = -10.0, .max = 10.0},
                                             {.min = 140.0, .max = f_max}};
        return PulsarSearchConfig(1U << 16U, 64e-6, 32, 1.0, limits, 0.3, 1.5,
                                  std::is_same_v<TestType, ComplexTypeCUDA>, 1,
                                  4.0, 2.0, 1024, 64, 1024, 8192);
    };
    const auto larger  = config(145.0);
    const auto smaller = config(141.0);
    const loki::plans::FFAPlan<HostFoldT> plan(larger);
    const loki::plans::FFAPlan<HostFoldT> small_plan(smaller);
    REQUIRE(small_plan.get_ncoords().back() < plan.get_ncoords().back());
    const auto ncoords   = plan.get_ncoords().back();
    const auto nsegments = plan.get_nsegments().back();
    const auto branch_max =
        std::max({SizeType{32}, plan.forecast_branching().max_children,
                  small_plan.forecast_branching().max_children});
    const auto capacity = ep_cuda_effective_max_sugg(kBatch, branch_max, 4096);
    const auto width    = std::is_same_v<TestType, ComplexTypeCUDA>
                              ? larger.get_nbins_f()
                              : larger.get_nbins();
    loki::cuda_utils::DeviceVectorNoInit<DeviceFoldT> arena(
        std::max(plan.get_buffer_size(), capacity * 2 * width));
    FFAWorkspaceCUDA<TestType> ffa(plan.get_buffer_size(),
                                   plan.get_coord_size(), plan.get_n_levels(),
                                   larger.get_nparams(), false);
    ffa.shared_fold_internal =
        loki::cuda_utils::as_span(arena).first(plan.get_buffer_size());
    const auto stream = cudaStreamPerThread;
    EPWorkspaceCUDA<TestType> ep(
        kBatch, branch_max, capacity, ncoords, larger.get_nparams(), width,
        nsegments, stream,
        loki::cuda_utils::as_span(arena).first(capacity * 2 * width));
    loki::math::CUFFTManager fft(0);
    thrust::device_vector<DeviceFoldT> folds(plan.get_buffer_size());
    std::vector<float> noise(larger.get_nsamps());
    std::mt19937 rng(17);
    std::normal_distribution<float> normal;
    for (auto& value : noise) {
        value = normal(rng);
    }
    thrust::device_vector<float> ts_e(noise.begin(), noise.end());
    thrust::device_vector<float> ts_v(noise.size(), 1.0F);
    std::vector<loki::pipelines::detail::EPResultPayload> results;
    loki::algorithms::EPCudaSharedPipeline<TestType> pipeline{
        .ep_workspace  = &ep,
        .ffa_workspace = &ffa,
        .fft_manager   = &fft,
        .fold_d        = loki::cuda_utils::as_span(folds),
        .ts_e_d        = loki::cuda_utils::as_span(ts_e),
        .ts_v_d        = loki::cuda_utils::as_span(ts_v),
        .stream        = stream,
        .ws_max_sugg   = capacity,
        .ws_branch_max = branch_max,
        .ws_ncoords    = ncoords,
        .ws_nsegments  = nsegments,
        .result_sink =
            [&results](loki::pipelines::detail::EPResultPayload result) {
                results.push_back(std::move(result));
            }};
    const auto outdir =
        std::filesystem::temp_directory_path() /
        (std::is_same_v<TestType, ComplexTypeCUDA> ? "loki_ep_group_fourier"
                                                   : "loki_ep_group_time");
    const std::vector<float> thresholds(nsegments - 1, 100.0F);
    for (const auto* cfg : {&larger, &smaller, &larger}) {
        loki::algorithms::EPMultiPassCudaCore<TestType> core(
            pipeline, *cfg, thresholds, 1, std::nullopt, {}, capacity, kBatch);
        core.execute_device(outdir, "chunk");
        const auto initial = results.back().stats.get_stats(0);
        REQUIRE(initial.has_value());
        const loki::plans::FFAPlan<HostFoldT> chunk_plan(*cfg);
        REQUIRE(initial->n_leaves == chunk_plan.get_ncoords().back());
    }
    REQUIRE(results.size() == 3);
    std::filesystem::remove_all(outdir);
}

TEST_CASE("CUDA threshold pools are allocated by run() only",
          "[ep_memory][thresholds][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const std::string mode =
        GENERATE(std::string{"legacy"}, std::string{"improved"});
    const std::vector<float> bp{2.0F, 2.0F, 1.5F, 2.0F};
    const std::vector<float> thresholds(bp.size(), 1.5F);
    const auto make = [&] {
        return loki::detection::detail::make_thresholds_gpu(
            bp, 0.1F, 1024, 32, 4, 0.05F, 10.0F, 60, 0.3F, 1.2F, 1.5F, 1, mode,
            16, 0, 11);
    };
    // The first launches load the kernels and reserve their local memory,
    // which the driver keeps. Measure after them.
    {
        auto warm_up = make();
        warm_up->run(6);
        (void)warm_up->evaluate(thresholds, 64, 21);
    }
    const auto used_bytes =
        [free_start = loki::cuda_utils::get_cuda_memory_usage().first] {
            return (free_start -
                    loki::cuda_utils::get_cuda_memory_usage().first) *
                   static_cast<double>(1ULL << 30U);
        };
    constexpr double kConstantsBytes = 2U << 20U;
    const auto same_states           = [](const auto& a, const auto& b) {
        REQUIRE(a.size() == b.size());
        for (SizeType i = 0; i < a.size(); ++i) {
            const auto& x = a[i];
            const auto& y = b[i];
            const std::array<float, 10> fx{
                x.success_h0,       x.success_h1,
                x.complexity,       x.complexity_cumul,
                x.success_h1_cumul, x.nbranches,
                x.threshold,        x.cost,
                x.threshold_prev,   x.success_h1_cumul_prev};
            const std::array<float, 10> fy{
                y.success_h0,       y.success_h1,
                y.complexity,       y.complexity_cumul,
                y.success_h1_cumul, y.nbranches,
                y.threshold,        y.cost,
                y.threshold_prev,   y.success_h1_cumul_prev};
            for (SizeType j = 0; j < fx.size(); ++j) {
                CHECK(std::bit_cast<uint32_t>(fx[j]) ==
                      std::bit_cast<uint32_t>(fy[j]));
            }
            CHECK(x.is_empty == y.is_empty);
        }
    };

    auto engine          = make();
    const auto run_bytes = static_cast<double>(engine->run_memory_bytes());
    REQUIRE(run_bytes > 8 * kConstantsBytes);
    // Construction and evaluation hold no simulation pools.
    const auto evaluated = engine->evaluate(thresholds, 64, 21);
    CHECK(evaluated.size() == bp.size());
    CHECK(used_bytes() <= kConstantsBytes);

    engine->run(6);
    CHECK(used_bytes() >= run_bytes);
    const auto first = engine->get_states();
    // A second run reuses the pools and repeats the first exactly.
    engine->run(6);
    same_states(first, engine->get_states());
    same_states(evaluated, engine->evaluate(thresholds, 64, 21));

    engine.reset();
    CHECK(used_bytes() <= kConstantsBytes);
}

#endif // LOKI_ENABLE_CUDA
