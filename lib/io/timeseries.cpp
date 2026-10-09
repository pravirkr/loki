#include "loki/io/timeseries.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <psrio/psrio.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"

namespace loki::io {
namespace {

void validate_ingress_arrays(std::span<const float> ts_e,
                             std::span<const float> ts_v) {
    // Bulk finiteness first; the per-sample scan names the bad index.
    const bool finite = utils::all_finite(ts_e) && utils::all_finite(ts_v);
    bool any_weight   = false;
    for (SizeType i = 0; i < ts_e.size(); ++i) {
        if (!finite && !utils::is_finite(ts_e[i])) {
            throw std::invalid_argument(
                std::format("ts_e[{}] is not finite", i));
        }
        if ((!finite && !utils::is_finite(ts_v[i])) || ts_v[i] < 0.0F) {
            throw std::invalid_argument(std::format(
                "ts_v[{}] must be finite and >= 0 (got {})", i, ts_v[i]));
        }
        if (ts_v[i] == 0.0F && ts_e[i] != 0.0F) {
            throw std::invalid_argument(std::format(
                "ts_e[{}] must be 0 where ts_v is 0 (got {})", i, ts_e[i]));
        }
        any_weight = any_weight || ts_v[i] > 0.0F;
    }
    if (!any_weight) {
        throw std::invalid_argument("ts_v is zero on every sample");
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
    std::vector<float> variance(samples.size(), 1.0F);
    if (options.preprocess) {
        io::preprocess(samples, header.tsamp, samples, variance,
                       options.preprocessing,
                       Exec::cpu(std::max(options.nthreads, 1)));
    }
    TimeSeries series(std::move(samples), std::move(variance), header.tsamp);
    const double tsamp             = header.tsamp;
    series.m_impl->header          = std::move(header);
    series.m_impl->header.tsamp    = tsamp;
    series.m_impl->header.nsamples = series.get_nsamps();
    return series;
}

PreprocessReport TimeSeries::preprocess(const PreprocessOptions& options,
                                        Exec exec) {
    // Fresh outputs keep the series valid if preprocessing throws.
    std::vector<float> ts_e(m_impl->ts_e.size());
    std::vector<float> ts_v(m_impl->ts_v.size());
    auto report = io::preprocess(m_impl->ts_e, m_impl->header.tsamp, ts_e, ts_v,
                                 options, exec);
    m_impl->ts_e = std::move(ts_e);
    m_impl->ts_v = std::move(ts_v);
    return report;
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
