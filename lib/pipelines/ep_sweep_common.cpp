#include "lib/pipelines/ep_sweep_common.hpp"

#include <cstddef>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hdf5.h>
#include <highfive/highfive.hpp>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"

namespace loki::pipelines::detail {

double
merge_ep_sweep_results(const std::filesystem::path& tmp_dir,
                       const std::filesystem::path& result_file,
                       const std::vector<algorithms::EPChunkConfig>& chunk_cfgs,
                       const search::PulsarSearchConfig& base_cfg,
                       float min_pd,
                       std::string_view poly_basis,
                       float ref_ducy,
                       float total_runtime) {
    HighFive::File main_h5(result_file.string(), HighFive::File::Overwrite);
    main_h5.createAttribute("ep_sweep_version", std::string("1.0.0-cpp"));
    main_h5.createAttribute("param_names", base_cfg.get_param_names());
    main_h5.createAttribute("nchunks", chunk_cfgs.size());
    main_h5.createAttribute("f_min", base_cfg.get_f_min());
    main_h5.createAttribute("f_max", base_cfg.get_f_max());
    main_h5.createAttribute("tobs", base_cfg.get_tobs());
    main_h5.createAttribute("tsamp", base_cfg.get_tsamp());
    main_h5.createAttribute("min_pd", min_pd);
    main_h5.createAttribute("poly_basis", std::string(poly_basis));
    main_h5.createAttribute("ref_ducy", ref_ducy);
    main_h5.createAttribute("total_runtime", total_runtime);

    auto main_chunks_group   = main_h5.createGroup("chunks");
    double accumulated_flops = 0.0;
    const SizeType nchunks   = chunk_cfgs.size();

    for (SizeType i = 0; i < nchunks; ++i) {
        const auto& chunk              = chunk_cfgs[i];
        const std::string chunk_prefix = std::format("chunk_{:04d}", i);
        auto chunk_group = main_chunks_group.createGroup(chunk_prefix);

        chunk_group.createAttribute("chunk_id", i);
        chunk_group.createAttribute("nominal_f_start", chunk.nominal_f_start);
        chunk_group.createAttribute("nominal_f_end", chunk.nominal_f_end);
        chunk_group.createAttribute("actual_f_start", chunk.actual_f_start);
        chunk_group.createAttribute("actual_f_end", chunk.actual_f_end);
        chunk_group.createAttribute("nbins", chunk.cfg.get_nbins());
        chunk_group.createAttribute("eta", chunk.cfg.get_eta());
        chunk_group.createAttribute("max_sugg", chunk.max_sugg);
        chunk_group.createAttribute("branch_max", chunk.branch_max);
        chunk_group.createAttribute("peak_complexity", chunk.peak_complexity);
        chunk_group.createAttribute("chunk_memory_gb", chunk.chunk_memory_gb);
        chunk_group.createAttribute("nsegments", chunk.nsegments);

        chunk_group.createDataSet("threshold_scheme", chunk.threshold_scheme);
        chunk_group.createDataSet("branching_pattern", chunk.branching_pattern);

        const auto chunk_result_file =
            tmp_dir / std::format("{}_pruning_nstages_{}_results.h5",
                                  chunk_prefix, chunk.nsegments);

        if (std::filesystem::exists(chunk_result_file)) {
            HighFive::File const chunk_h5(chunk_result_file.string(),
                                          HighFive::File::ReadOnly);
            if (chunk_h5.exist("runs")) {
                HighFive::Group const chunk_runs = chunk_h5.getGroup("runs");
                HighFive::Group const dst_runs =
                    chunk_group.createGroup("runs");
                for (const auto& run_name : chunk_runs.listObjectNames()) {
                    auto const run_grp = chunk_runs.getGroup(run_name);
                    if (run_grp.hasAttribute("total_pruning_gflops")) {
                        double run_gflops{};
                        run_grp.getAttribute("total_pruning_gflops")
                            .read(run_gflops);
                        accumulated_flops += run_gflops;
                    }
                    herr_t const status = H5Ocopy(
                        chunk_runs.getId(), run_name.c_str(), dst_runs.getId(),
                        run_name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
                    if (status < 0) {
                        throw std::runtime_error(std::format(
                            "EPFreqSweep: failed to copy run '{}' for chunk {}",
                            run_name, i));
                    }
                }
            }
        }
    }

    main_h5.createAttribute("total_pruning_gflops", accumulated_flops);
    return accumulated_flops;
}

void check_ep_group_budget(const algorithms::detail::EPChunkGroup& group,
                           double total_gb,
                           double limit_gb,
                           int n_workers) {
    if (total_gb > limit_gb) {
        throw std::runtime_error(std::format(
            "EPFreqSweep: chunks {}..{} need {:.2f} GB ({} workers), more "
            "than the limit {:.2f} GB. Re-plan (stale plan cache?).",
            group.begin, group.end - 1, total_gb, n_workers, limit_gb));
    }
}

void check_ep_workspace_vs_model(const algorithms::detail::EPChunkGroup& group,
                                 double actual_bytes,
                                 double model_bytes) {
    // get_memory_usage_gib() is a float: allow its rounding.
    constexpr double kRelTol = 1.0e-5;
    if (actual_bytes > model_bytes * (1.0 + kRelTol)) {
        throw std::logic_error(std::format(
            "EPFreqSweep: workspace for chunks {}..{} uses {:.0f} bytes, "
            "more than the memory model's {:.0f} (model drift)",
            group.begin, group.end - 1, actual_bytes, model_bytes));
    }
}

void check_ep_shared_vs_model(SizeType actual_bytes, SizeType model_bytes) {
    if (actual_bytes > model_bytes) {
        throw std::logic_error(std::format(
            "EPFreqSweep: shared FFA buffers use {} bytes, more than the "
            "memory model's {} (model drift)",
            actual_bytes, model_bytes));
    }
}

} // namespace loki::pipelines::detail
