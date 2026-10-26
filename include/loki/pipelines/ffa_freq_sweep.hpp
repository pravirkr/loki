#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/observation.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"

namespace loki::pipelines {

/**
 * @brief FFA search over a frequency range, split into regions that share one
 * workspace and FFT plan cache. Writes candidates to an HDF5 file.
 *
 * The CPU thread count comes from @p cfg; @p exec selects the backend and
 * device.
 */
class FFAFreqSweep {
public:
    explicit FFAFreqSweep(const search::FFASearchConfig& cfg,
                          bool show_progress = true,
                          Exec exec          = {});

    explicit FFAFreqSweep(const search::FFASearchConfig& cfg,
                          const search::CandidateOptions& candidates,
                          bool show_progress = true,
                          Exec exec          = {});

    ~FFAFreqSweep();
    FFAFreqSweep(FFAFreqSweep&&) noexcept;
    FFAFreqSweep& operator=(FFAFreqSweep&&) noexcept;
    FFAFreqSweep(const FFAFreqSweep&)            = delete;
    FFAFreqSweep& operator=(const FFAFreqSweep&) = delete;

    /// Runs the search, then groups and publishes the candidates as
    /// `<outdir>/<file_prefix>_ffa_results.h5`. If the search completes but
    /// the post-processing fails (candidate memory budget, disk, HDF5), the
    /// raw spool is kept as `<outdir>/<file_prefix>_ffa_spool.h5` and the
    /// error is rethrown; publish_candidates() resumes from it.
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir        = "./",
                 std::string_view file_prefix               = "test",
                 std::string_view config_toml               = {},
                 const io::ObservationMetadata& observation = {});

    /// Groups the detections of a kept @p spool and publishes the product at
    /// @p output, atomically, without searching again. Pass the same
    /// @p observation metadata as the original execute(): the spool does not
    /// store it.
    static void
    publish_candidates(const std::filesystem::path& spool,
                       const std::filesystem::path& output,
                       const search::CandidateOptions& candidates = {},
                       const io::ObservationMetadata& observation = {});

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::pipelines
