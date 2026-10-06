#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

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

    ~FFAFreqSweep();
    FFAFreqSweep(FFAFreqSweep&&) noexcept;
    FFAFreqSweep& operator=(FFAFreqSweep&&) noexcept;
    FFAFreqSweep(const FFAFreqSweep&)            = delete;
    FFAFreqSweep& operator=(const FFAFreqSweep&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test",
                 std::string_view config_toml        = {});

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms