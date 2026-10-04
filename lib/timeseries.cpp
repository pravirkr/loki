#include "loki/io/timeseries.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <psrio/psrio.hpp>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::io {
namespace {

void validate_ingress_arrays(std::span<const float> ts_e,
                             std::span<const float> ts_v) {
    for (SizeType i = 0; i < ts_e.size(); ++i) {
        if (!utils::is_finite(ts_e[i])) {
            throw std::invalid_argument(
                std::format("ts_e[{}] is not finite", i));
        }
        if (!utils::is_finite(ts_v[i]) || ts_v[i] <= 0.0F) {
            throw std::invalid_argument(std::format(
                "ts_v[{}] must be finite and positive (got {})", i, ts_v[i]));
        }
    }
}

[[nodiscard]] std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

enum class FileKind : std::uint8_t { kTim, kDat };

[[nodiscard]] FileKind kind_from_path(const std::filesystem::path& path) {
    const std::string ext = lower_ext(path);
    if (ext == ".tim") {
        return FileKind::kTim;
    }
    if (ext == ".dat") {
        return FileKind::kDat;
    }
    throw error_check::DetailedException(
        "timeseries path must end in .tim or .dat: " + path.string());
}

template <typename F> auto call_psrio(F&& fn) -> decltype(fn()) {
    try {
        return std::forward<F>(fn)();
    } catch (const psrio::Error& ex) {
        throw error_check::DetailedException(ex.what());
    }
}

[[nodiscard]] psrio::TimeSeries load_file(const std::filesystem::path& path) {
    switch (kind_from_path(path)) {
    case FileKind::kTim:
        return call_psrio([&] { return psrio::TimeSeries::from_tim(path); });
    case FileKind::kDat: {
        auto inf = path;
        inf.replace_extension(".inf");
        return call_psrio(
            [&] { return psrio::TimeSeries::from_dat(path, inf); });
    }
    }
    throw error_check::DetailedException("unhandled timeseries path");
}

/// Window length in samples for a filter window given in seconds.
[[nodiscard]] SizeType
window_in_samples(double window_sec, double tsamp, SizeType n) {
    const double bins = window_sec / tsamp;
    if (bins >= static_cast<double>(n)) {
        return n;
    }
    return bins >= 1.0 ? static_cast<SizeType>(std::llround(bins)) : 1;
}

/// Remove the running-median baseline and z-score, both in place.
void preprocess_samples(std::span<float> samples,
                        double tsamp,
                        const ReadOptions& options) {
    if (options.filter_window < 0.0) {
        throw error_check::DetailedException(
            "filter window must be non-negative");
    }
    if (samples.empty()) {
        return;
    }
    if (options.filter_window > 0.0) {
        const SizeType window =
            window_in_samples(options.filter_window, tsamp, samples.size());
        if (window > 1) {
            math::subtract_running_filter(
                samples, window, math::FilterMethod::kMedian,
                options.fast_median, options.fast_median_min_points,
                options.nthreads);
        }
    }
    math::zscore(samples, options.loc, options.scale, options.nthreads);
}

} // namespace

class TimeSeries::Impl {
public:
    std::vector<float> ts_e;
    std::vector<float> ts_v;
    psrio::Header header;
};

TimeSeries::TimeSeries(std::vector<float> ts_e,
                       std::vector<float> ts_v,
                       double dt)
    : m_impl(std::make_unique<Impl>()) {
    error_check::check(ts_e.size() == ts_v.size(),
                       "ts_e and ts_v must have the same length");
    error_check::check(!ts_e.empty(), "timeseries is empty");
    error_check::check(utils::is_finite(dt) && dt > 0.0, "dt must be positive");
    validate_ingress_arrays(ts_e, ts_v);
    m_impl->header.tsamp     = dt;
    m_impl->header.nsamples  = ts_e.size();
    m_impl->header.nbits     = 32;
    m_impl->header.nchans    = 1;
    m_impl->header.nifs      = 1;
    m_impl->header.data_type = "time series";
    m_impl->ts_e             = std::move(ts_e);
    m_impl->ts_v             = std::move(ts_v);
}

TimeSeries::~TimeSeries() = default;

TimeSeries::TimeSeries(const TimeSeries& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

TimeSeries& TimeSeries::operator=(const TimeSeries& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}

TimeSeries::TimeSeries(TimeSeries&&) noexcept = default;

TimeSeries& TimeSeries::operator=(TimeSeries&&) noexcept = default;

std::span<float> TimeSeries::get_ts_e() noexcept { return m_impl->ts_e; }

std::span<const float> TimeSeries::get_ts_e() const noexcept {
    return m_impl->ts_e;
}

std::span<float> TimeSeries::get_ts_v() noexcept { return m_impl->ts_v; }

std::span<const float> TimeSeries::get_ts_v() const noexcept {
    return m_impl->ts_v;
}

double TimeSeries::get_dt() const noexcept { return m_impl->header.tsamp; }

SizeType TimeSeries::get_nsamps() const noexcept { return m_impl->ts_e.size(); }

double TimeSeries::get_tobs() const noexcept {
    return static_cast<double>(get_nsamps()) * get_dt();
}

TimeSeries TimeSeries::read(const std::filesystem::path& path,
                            const ReadOptions& options) {
    const psrio::TimeSeries raw = load_file(path);
    psrio::Header header        = raw.header();
    std::vector<float> samples(raw.data().begin(), raw.data().end());
    if (options.preprocess) {
        preprocess_samples(samples, header.tsamp, options);
    }
    std::vector<float> variance(samples.size(), 1.0F);
    TimeSeries series(std::move(samples), std::move(variance), header.tsamp);
    const double tsamp             = header.tsamp;
    series.m_impl->header          = std::move(header);
    series.m_impl->header.tsamp    = tsamp;
    series.m_impl->header.nsamples = series.get_nsamps();
    return series;
}

void TimeSeries::write(const std::filesystem::path& path) const {
    error_check::check(utils::is_finite(m_impl->header.tsamp) &&
                           m_impl->header.tsamp > 0.0,
                       "tsamp must be positive");
    psrio::Header header = m_impl->header;
    header.nsamples      = m_impl->ts_e.size();
    header.nbits         = 32;
    if (header.raj != 0.0) {
        header.ra = psrio::astro::ra_to_string(header.raj);
    }
    if (header.dej != 0.0) {
        header.dec = psrio::astro::dec_to_string(header.dej);
    }
    psrio::TimeSeries series(
        std::vector<float>(m_impl->ts_e.begin(), m_impl->ts_e.end()), header);
    switch (kind_from_path(path)) {
    case FileKind::kTim:
        call_psrio([&] { return series.to_tim(path); });
        break;
    case FileKind::kDat: {
        const auto stem = path.parent_path() / path.stem();
        call_psrio([&] { return series.to_dat(stem); });
        break;
    }
    }
}

} // namespace loki::io
