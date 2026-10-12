#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

namespace loki::io {

/**
 * @brief Options for TimeSeries::read.
 *
 * When `preprocess` is true the payload goes through io::preprocess() with
 * `preprocessing` (robust dual-array statistics by default; set
 * `preprocessing.method = PreprocessMethod::kZScore` for the legacy
 * running-median detrend + z-score with `ts_v = 1`). When `preprocess` is
 * false, `get_ts_e()` is the payload converted to float32 and `get_ts_v()`
 * is 1.
 */
struct ReadOptions {
    bool preprocess{true};
    PreprocessOptions preprocessing;
    /// OpenMP threads used by preprocessing (values < 1 are treated as 1).
    int nthreads{1};
};

/**
 * @brief In-memory timeseries with intensity and per-sample variance.
 *
 * File bytes are read and written through psrio. That header stays inside the
 * implementation. `read` builds `ts_e` and `ts_v` here.
 *
 * Construction requires finite `ts_e`, finite `ts_v >= 0`, `ts_e == 0`
 * wherever `ts_v == 0` (masked samples) and at least one `ts_v > 0`. The fold
 * and score kernels rely on this and do not re-check it.
 */
class TimeSeries {
public:
    TimeSeries(std::vector<float> ts_e, std::vector<float> ts_v, double dt);
    ~TimeSeries();
    TimeSeries(const TimeSeries& other);
    TimeSeries& operator=(const TimeSeries& other);
    TimeSeries(TimeSeries&&) noexcept;
    TimeSeries& operator=(TimeSeries&&) noexcept;

    static TimeSeries read(const std::filesystem::path& path,
                           const ReadOptions& options = {});

    void write(const std::filesystem::path& path) const;

    /**
     * @brief Preprocess: `ts_e` is taken as the raw series, and `ts_e` /
     * `ts_v` are replaced with io::preprocess() output. The current `ts_v` is
     * ignored. If it throws, the series is unchanged. Needs two series of
     * scratch memory while it runs.
     */
    PreprocessReport preprocess(const PreprocessOptions& options = {},
                                Exec exec                        = {});

    [[nodiscard]] std::span<float> get_ts_e() noexcept;
    [[nodiscard]] std::span<const float> get_ts_e() const noexcept;
    [[nodiscard]] std::span<float> get_ts_v() noexcept;
    [[nodiscard]] std::span<const float> get_ts_v() const noexcept;

    [[nodiscard]] double get_dt() const noexcept;
    [[nodiscard]] SizeType get_nsamps() const noexcept;
    [[nodiscard]] double get_tobs() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::io
