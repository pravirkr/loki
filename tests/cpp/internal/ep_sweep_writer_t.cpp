#include <filesystem>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/search/configs.hpp"

#include "lib/pipelines/ep_sweep_common.hpp"
#include "lib/search/cands.hpp"

namespace {

loki::search::PulsarSearchConfig writer_cfg() {
    return {1U << 16U,
            64e-6,
            32,
            1.0,
            std::vector<loki::ParamLimit>{{.min = 140.0, .max = 145.0}},
            0.3,
            1.5,
            false,
            1};
}

loki::pipelines::detail::EPResultPayload payload(std::string name,
                                                 loki::SizeType n) {
    loki::pipelines::detail::EPResultPayload result;
    result.run_name     = std::move(name);
    result.nparams      = 1;
    result.snail_scheme = {0, 1, 2};
    result.leaves.resize(n * 6, 142.0);
    result.scores.resize(n, 5.0F);
    result.scores_ep.resize(n, 4.0F);
    result.gflops = 2.0;
    return result;
}

} // namespace

TEST_CASE("EP sweep writer preserves the merged run layout", "[ep_writer]") {
    const auto dir =
        std::filesystem::temp_directory_path() / "loki_ep_writer_layout";
    std::filesystem::create_directories(dir);
    const auto cfg = writer_cfg();
    const std::vector<loki::algorithms::EPChunkConfig> chunks{
        {.cfg               = cfg,
         .threshold_scheme  = {1.0F},
         .branching_pattern = {2.0F},
         .nsegments         = 3}};
    const auto old_file = dir / "chunk_0000_pruning_nstages_3_results.h5";
    loki::search::PruneResultWriter old(old_file);
    old.write_metadata(cfg.get_param_names(), 3, chunks[0].max_sugg,
                       {chunks[0].threshold_scheme});
    loki::pipelines::detail::EPSweepWriter writer(dir / "direct.h5", chunks,
                                                  cfg, 0.1F, "taylor", 0.1F);
    for (loki::SizeType i = 0; i < 16; ++i) {
        auto run = payload(std::format("run_{:03d}", i), i == 0 ? 0 : 100);
        loki::search::PruneResultWriter append(
            old_file, loki::search::PruneResultWriter::Mode::kAppend);
        append.write_run_results(
            run.run_name, run.snail_scheme, {std::span<double>(run.leaves), {}},
            {std::span<float>(run.scores), {}},
            {std::span<float>(run.scores_ep), {}}, run.gflops,
            run.scores.size(), run.nparams, run.stats);
        writer.submit(std::move(run));
    }
    CHECK(writer.finish(1.0F) == 32.0);
    loki::pipelines::detail::merge_ep_sweep_results(
        dir, dir / "merged.h5", chunks, cfg, 0.1F, "taylor", 0.1F, 1.0F);
    const HighFive::File direct((dir / "direct.h5").string(),
                                HighFive::File::ReadOnly);
    const HighFive::File merged((dir / "merged.h5").string(),
                                HighFive::File::ReadOnly);
    // Root-group attributes: HDF5 1.10 cannot list them on a file handle.
    CHECK(direct.getGroup("/").listAttributeNames() ==
          merged.getGroup("/").listAttributeNames());
    for (loki::SizeType i = 0; i < 16; ++i) {
        const auto path = std::format("chunks/chunk_0000/runs/run_{:03d}", i);
        CHECK(direct.getGroup(path).listObjectNames() ==
              merged.getGroup(path).listObjectNames());
        CHECK(direct.getDataSet(path + "/param_sets").getDimensions() ==
              merged.getDataSet(path + "/param_sets").getDimensions());
        std::vector<float> a, b;
        direct.getDataSet(path + "/scores").read(a);
        merged.getDataSet(path + "/scores").read(b);
        CHECK(a == b);
    }
}

TEST_CASE("EP sweep writer propagates asynchronous write failures",
          "[ep_writer]") {
    const auto path =
        std::filesystem::temp_directory_path() / "loki_ep_writer_error.h5";
    const auto cfg = writer_cfg();
    const std::vector<loki::algorithms::EPChunkConfig> chunks{{.cfg = cfg}};
    std::filesystem::remove(path);
    loki::pipelines::detail::EPSweepWriter writer(path, chunks, cfg, 0.1F,
                                                  "taylor", 0.1F);
    writer.submit(payload("duplicate", 0));
    try {
        writer.submit(payload("duplicate", 0));
    } catch (const std::exception&) {
    }
    CHECK_THROWS(writer.finish(0.0F));
    // A failed sweep leaves only the partial file, never a complete-looking
    // one.
    CHECK_FALSE(std::filesystem::exists(path));
    CHECK_THROWS_AS(writer.finish(0.0F), std::logic_error);
    std::filesystem::remove(std::filesystem::path(path) += ".partial");
}
