#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory_cuda.hpp"
#include "lib/algorithms/prune_engine.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/prune_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/pipelines/ep_sweep_common.hpp"
#include "lib/search/prune_mask.hpp"

#include "ep_seed_slices.hpp"
#include "temp_dir.hpp"

// The GPU counterpart of ep_seed_slices_t.cpp: K seed slices give the leaves
// and level stats of one prune, through the engine (result file) and the
// sweep's shared pipeline (result sink, with a pulsar mask). In the Fourier
// domain: the time-domain brute fold adds with float atomics, so its FFA
// output, and every score after it, jitters from one execute to the next.

using loki::Backend;
using loki::ComplexType;
using loki::ComplexTypeCUDA;
using loki::SizeType;
using loki::algorithms::ParamWindow;
using loki::algorithms::PruneRFIConfig;
using loki::test::RunResults;
using loki::test::TempDir;

namespace {

constexpr SizeType kBatch   = 64;
constexpr SizeType kMaxSugg = 1U << 16U;

std::vector<RunResults> run_engine(const loki::search::PulsarSearchConfig& cfg,
                                   const std::vector<float>& ts_e,
                                   const std::vector<float>& ts_v,
                                   SizeType n_slices,
                                   SizeType max_sugg = kMaxSugg,
                                   SizeType batch    = kBatch) {
    const TempDir outdir(fmt::format("loki-seed-slices-cuda-{}", n_slices));
    auto engine = loki::algorithms::detail::make_ep_gpu<ComplexType>(
        cfg, loki::test::seed_slice_thresholds(cfg), /*n_runs=*/2,
        /*ref_segs=*/std::nullopt, /*ascend_levels=*/{}, max_sugg, batch,
        "taylor", /*device_id=*/0);
    engine->set_seed_slices(loki::search::even_seed_slice_edges(
        n_slices, loki::plans::FFAPlan<ComplexType>(cfg)
                      .get_param_counts()
                      .back()
                      .back()));
    engine->execute(ts_e, ts_v, outdir.path(), "slices");
    return loki::test::read_run_results(outdir.path());
}

/// The sweep's shared buffers for one chunk, with a sink that keeps every
/// run.
template <typename FoldTypeCUDA> struct SharedChunk {
    using HostFoldT   = loki::HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = loki::DeviceFoldType<FoldTypeCUDA>;

    loki::search::PulsarSearchConfig cfg;
    loki::plans::FFAPlan<HostFoldT> plan;
    SizeType branch_max;
    SizeType capacity;
    SizeType width;
    loki::cuda_utils::DeviceVectorNoInit<DeviceFoldT> arena;
    loki::memory::FFAWorkspaceCUDA<FoldTypeCUDA> ffa;
    loki::memory::EPWorkspaceCUDA<FoldTypeCUDA> ep;
    loki::math::GpuFFTManager fft{0};
    thrust::device_vector<DeviceFoldT> folds;
    thrust::device_vector<float> ts_e;
    thrust::device_vector<float> ts_v;
    std::vector<loki::pipelines::detail::EPResultPayload> results;

    SharedChunk(loki::search::PulsarSearchConfig c,
                const std::vector<float>& e,
                const std::vector<float>& v)
        : cfg(std::move(c)),
          plan(cfg),
          branch_max(
              std::max(SizeType{32}, plan.forecast_branching().max_children)),
          capacity(loki::algorithms::detail::ep_cuda_effective_max_sugg(
              kBatch, branch_max, kMaxSugg)),
          width(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                    ? cfg.get_nbins_f()
                    : cfg.get_nbins()),
          arena(std::max(plan.get_buffer_size(), capacity * 2 * width)),
          ffa(plan.get_buffer_size(),
              plan.get_coord_size(),
              plan.get_n_levels(),
              cfg.get_nparams(),
              false),
          ep(kBatch,
             branch_max,
             capacity,
             plan.get_ncoords().back(),
             cfg.get_nparams(),
             width,
             plan.get_nsegments().back(),
             cudaStreamPerThread,
             loki::cuda_utils::as_span(arena).first(capacity * 2 * width)),
          folds(plan.get_buffer_size()),
          ts_e(e.begin(), e.end()),
          ts_v(v.begin(), v.end()) {
        ffa.shared_fold_internal =
            loki::cuda_utils::as_span(arena).first(plan.get_buffer_size());
        const auto tables =
            loki::algorithms::detail::ep_cuda_brute_table_sizes(plan);
        ffa.reserve_brute_tables(tables.nfreqs, tables.phase_map_entries);
    }

    std::vector<RunResults> run(SizeType n_slices, const PruneRFIConfig& rfi) {
        results.clear();
        const loki::algorithms::EPCudaSharedPipeline<FoldTypeCUDA> pipeline{
            .ep_workspace  = &ep,
            .ffa_workspace = &ffa,
            .fft_manager   = &fft,
            .fold_d        = loki::cuda_utils::as_span(folds),
            .ts_e_d        = loki::cuda_utils::as_span(ts_e),
            .ts_v_d        = loki::cuda_utils::as_span(ts_v),
            .stream        = cudaStreamPerThread,
            .ws_max_sugg   = capacity,
            .ws_branch_max = branch_max,
            .ws_ncoords    = plan.get_ncoords().back(),
            .ws_nsegments  = plan.get_nsegments().back(),
            .result_sink =
                [this](loki::pipelines::detail::EPResultPayload payload) {
                    results.push_back(std::move(payload));
                }};
        const TempDir outdir("loki-seed-slices-cuda-shared");
        loki::algorithms::EPMultiPassCudaCore<FoldTypeCUDA> core(
            pipeline, cfg, loki::test::seed_slice_thresholds(cfg),
            /*n_runs=*/2, std::nullopt, {}, kMaxSugg, kBatch, "taylor",
            /*device_id=*/0, rfi);
        core.set_seed_slices(loki::search::even_seed_slice_edges(
            n_slices, plan.get_param_counts().back().back()));
        core.execute_device(outdir.path(), "chunk");
        std::vector<RunResults> runs;
        for (const auto& payload : results) {
            runs.push_back(loki::test::to_run_results(payload));
        }
        std::ranges::sort(runs, {}, &RunResults::name);
        return runs;
    }
};

} // namespace

TEST_CASE("CUDA EP: seed slices give the leaves and stats of one prune",
          "[ep][seed_slices][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not built");
    }
    const loki::cuda_utils::CudaSetDeviceGuard guard(0);
    for (const SizeType nparams : {SizeType{2}, SizeType{5}}) {
        CAPTURE(nparams);
        const auto cfg = loki::test::seed_slice_config(nparams,
                                                       /*fourier=*/true,
                                                       /*nthreads=*/1);
        const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
        const auto whole        = run_engine(cfg, ts_e, ts_v, 1);
        REQUIRE(whole.size() == 2);
        REQUIRE(whole.front().n_leaves() > 0);
        CHECK(whole.front().total_overloads() == 0);
        // One prune repeats exactly, so slices can be compared exactly.
        loki::test::check_same_runs(whole, run_engine(cfg, ts_e, ts_v, 1));
        for (const SizeType k : {SizeType{3}, SizeType{1000}}) {
            CAPTURE(k);
            loki::test::check_same_runs(whole, run_engine(cfg, ts_e, ts_v, k));
        }
    }
}

TEST_CASE("CUDA EP: seed slices fit a tree smaller than the seeds",
          "[ep][seed_slices][cuda]") {
    // A planner that slices sizes the tree for the largest slice: with a
    // batch small enough that the tree's floor (one branched batch) is below
    // the seed count, the slices still give the leaves of one prune.
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not built");
    }
    const loki::cuda_utils::CudaSetDeviceGuard guard(0);
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/true,
                                                            /*nthreads=*/1);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    const loki::plans::FFAPlan<ComplexType> plan(cfg);
    const auto ncoords = plan.get_ncoords().back();
    const auto n_freq  = plan.get_param_counts().back().back();
    const auto branch_max =
        std::max<SizeType>(plan.forecast_branching().max_children, SizeType{1});
    constexpr SizeType kSmallBatch = 1;
    const auto max_sugg            = ncoords / 2;
    REQUIRE(loki::algorithms::detail::ep_cuda_effective_max_sugg(
                kSmallBatch, branch_max, max_sugg) < ncoords);
    const auto whole = run_engine(cfg, ts_e, ts_v, 1);
    const auto sliced =
        run_engine(cfg, ts_e, ts_v, n_freq, max_sugg, kSmallBatch);
    loki::test::check_same_runs(whole, sliced);
    for (const auto& run : sliced) {
        CHECK(run.total_overloads() == 0);
    }
}

TEST_CASE("CUDA EP sweep: seed slices are exact under a pulsar mask",
          "[ep][seed_slices][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not built");
    }
    const loki::cuda_utils::CudaSetDeviceGuard guard(0);
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/true,
                                                            /*nthreads=*/1);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    SharedChunk<ComplexTypeCUDA> chunk(cfg, ts_e, ts_v);
    const auto n_freq = chunk.plan.get_param_counts().back().back();
    PruneRFIConfig rfi;
    rfi.pulsar_mask  = {ParamWindow{.f_lo = 141.0, .f_hi = 142.5}};
    const auto whole = chunk.run(1, rfi);
    REQUIRE(whole.size() == 2);
    REQUIRE(whole.front().n_leaves() > 0);
    REQUIRE(whole.front().total_masked() > 0);
    CHECK(whole.front().total_overloads() == 0);
    loki::test::check_same_runs(whole, chunk.run(1, rfi));
    // One slice per cell: the cells under the window are fully masked
    // slices.
    for (const SizeType k : {SizeType{4}, n_freq}) {
        CAPTURE(k);
        loki::test::check_same_runs(whole, chunk.run(k, rfi));
    }

    // An early harvest of the pulsar from level 4: the mask its first
    // harvest widened carries across the slices of a run, so one slice per
    // cell harvests it as often as one prune does.
    PruneRFIConfig harvest;
    const auto n_levels = cfg.get_nsegments() - 1;
    harvest.harvest_scheme.assign(n_levels, 1.0e6F);
    for (SizeType level = 4; level <= n_levels; ++level) {
        harvest.harvest_scheme[level - 1] = 14.0F;
    }
    harvest.harvest_mask_ntiles = 2.0;
    const auto harvested        = chunk.run(1, harvest);
    REQUIRE(harvested.front().n_harvested_total > 0);
    const auto sliced = chunk.run(n_freq, harvest);
    REQUIRE(sliced.size() == harvested.size());
    for (SizeType r = 0; r < harvested.size(); ++r) {
        CHECK(sliced[r].n_harvested_total == harvested[r].n_harvested_total);
    }
}
#endif // LOKI_ENABLE_CUDA
