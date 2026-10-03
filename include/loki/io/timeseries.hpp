#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::io {

/// Location estimate used when normalising a loaded timeseries.
enum class LocMethod : std::uint8_t { kMean, kMedian };

/// Scale estimate used when normalising a loaded timeseries.
enum class ScaleMethod : std::uint8_t { kStd, kIqr, kMad };

/**
 * @brief Options for TimeSeries::read.
 *
 * When `preprocess` is true the payload is baseline-subtracted with a sliding
 * median and z-scored. `get_ts_e()` is that series and `get_ts_v()` is 1.
 * When `preprocess` is false, `get_ts_e()` is the payload converted to float32
 * and `get_ts_v()` is still 1.
 *
 * Construction requires finite `ts_e` and strictly positive finite `ts_v` on
 * every sample. The FFA pipeline relies on this at ingress; scoring only
 * guards non-positive *fold-bin* variance when normalizing profiles.
 */
struct ReadOptions {
    bool preprocess{true};
    double filter_window{1.0};
    LocMethod loc{LocMethod::kMean};
    ScaleMethod scale{ScaleMethod::kIqr};
};

/**
 * @brief In-memory timeseries with intensity and per-sample variance.
 *
 * File bytes are read and written through psrio. That header stays inside the
 * implementation. `read` builds `ts_e` and `ts_v` here.
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
