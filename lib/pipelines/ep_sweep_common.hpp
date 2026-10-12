#pragma once

/**
 * @file ep_sweep_common.hpp
 * @brief Backend-neutral parts of the EPFreqSweep engines: the merge of the
 * per-chunk results and the runtime checks of a plan against the memory model
 * (docs/memory.md). Internal.
 */

#include <condition_variable>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/search/cands.hpp"

namespace loki::pipelines::detail {

/**
 * @brief Merges the per-chunk result files in @p tmp_dir into @p result_file.
 *
 * Chunk `i` is read from
 * `chunk_{i:04d}_pruning_nstages_{nsegments}_results.h5` and its "runs" group
 * is copied under `chunks/chunk_{i:04d}/runs`. The layout does not depend on
 * the backend that wrote the chunk files.
 * @return Total pruning GFLOPs of all runs.
 */
double
merge_ep_sweep_results(const std::filesystem::path& tmp_dir,
                       const std::filesystem::path& result_file,
                       const std::vector<algorithms::EPChunkConfig>& chunk_cfgs,
                       const search::PulsarSearchConfig& base_cfg,
                       float min_pd,
                       std::string_view poly_basis,
                       float ref_ducy,
                       float total_runtime);

/// Owning host snapshot; no spans may refer to a reusable device workspace.
struct EPResultPayload {
    SizeType chunk_id{0};
    std::string run_name;
    std::vector<SizeType> snail_scheme;
    std::vector<double> leaves;
    std::vector<float> scores;
    std::vector<float> scores_ep;
    double gflops{0.0};
    SizeType nparams{0};
    search::PruneStatsCollection stats;
};

/// One active write and at most one waiting payload. All final-file HDF5
/// operations run on the writer thread until finish() joins it.
/// Writes the sweep's results from one background thread. They go to
/// `<result_file>.partial`, which finish() renames to @p result_file, so an
/// aborted sweep never leaves a file that looks complete.
class EPSweepWriter {
public:
    EPSweepWriter(const std::filesystem::path& result_file,
                  const std::vector<algorithms::EPChunkConfig>& chunks,
                  const search::PulsarSearchConfig& cfg,
                  float min_pd,
                  std::string_view poly_basis,
                  float ref_ducy);
    ~EPSweepWriter();
    EPSweepWriter(const EPSweepWriter&)            = delete;
    EPSweepWriter& operator=(const EPSweepWriter&) = delete;
    void submit(EPResultPayload payload);
    double finish(float runtime);

private:
    void run();
    std::filesystem::path m_file;
    std::filesystem::path m_partial_file;
    SizeType m_nchunks;
    bool m_finished{false};
    std::mutex m_mutex;
    std::condition_variable m_ready;
    std::optional<EPResultPayload> m_pending;
    bool m_stopping{false};
    std::exception_ptr m_error;
    double m_gflops{0.0};
    std::thread m_thread;
};

/// Before allocating a chunk group: the model total must fit @p limit_gb.
/// The planner guarantees it, so this only fires on a stale cache or a bug.
/// @throws std::runtime_error
void check_ep_group_budget(const algorithms::detail::EPChunkGroup& group,
                           double total_gb,
                           double limit_gb,
                           int n_workers);

/// After allocating a group's workspace: it must not exceed the model.
/// @throws std::logic_error on a model drift.
void check_ep_workspace_vs_model(const algorithms::detail::EPChunkGroup& group,
                                 double actual_bytes,
                                 double model_bytes);

/// After allocating the shared FFA buffers: they must not exceed the model.
/// @throws std::logic_error on a model drift.
void check_ep_shared_vs_model(SizeType actual_bytes, SizeType model_bytes);

} // namespace loki::pipelines::detail
