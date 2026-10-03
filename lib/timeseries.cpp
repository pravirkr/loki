#include "loki/io/timeseries.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <format>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <psrio/psrio.hpp>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
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
            throw std::invalid_argument(
                std::format("ts_v[{}] must be finite and positive (got {})", i,
                            ts_v[i]));
        }
    }
}

constexpr double kIqrScale = 1.349;
constexpr double kMadScale = 1.4826;

[[nodiscard]] bool is_finite_double(double value) noexcept {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
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

class WindowMedian {
public:
    void add(float value) {
        if (m_lo.empty() || value <= *m_lo.rbegin()) {
            m_lo.insert(value);
        } else {
            m_hi.insert(value);
        }
        rebalance();
    }

    void remove(float value) {
        const auto lower = m_lo.find(value);
        if (lower != m_lo.end()) {
            m_lo.erase(lower);
        } else {
            const auto upper = m_hi.find(value);
            if (upper == m_hi.end()) {
                throw error_check::DetailedException(
                    "running median lost a sample");
            }
            m_hi.erase(upper);
        }
        rebalance();
    }

    [[nodiscard]] double median() const {
        if (m_lo.empty()) {
            throw error_check::DetailedException(
                "running median window is empty");
        }
        if (m_lo.size() == m_hi.size()) {
            return 0.5 * (static_cast<double>(*m_lo.rbegin()) +
                          static_cast<double>(*m_hi.begin()));
        }
        return static_cast<double>(*m_lo.rbegin());
    }

private:
    void rebalance() {
        while (m_lo.size() > m_hi.size() + 1) {
            m_hi.insert(*m_lo.rbegin());
            m_lo.erase(std::prev(m_lo.end()));
        }
        while (m_hi.size() > m_lo.size()) {
            m_lo.insert(*m_hi.begin());
            m_hi.erase(m_hi.begin());
        }
    }

    std::multiset<float> m_lo;
    std::multiset<float> m_hi;
};

void subtract_running_median(std::span<float> samples,
                             double window_sec,
                             double tsamp) {
    if (window_sec < 0.0) {
        throw error_check::DetailedException(
            "filter window must be non-negative");
    }
    const SizeType n = samples.size();
    if (n == 0) {
        return;
    }
    SizeType window   = 1;
    const double bins = window_sec / tsamp;
    if (bins > static_cast<double>(n)) {
        window = n;
    } else if (bins >= 1.0) {
        window = static_cast<SizeType>(std::llround(bins));
    }
    const SizeType radius_left  = (window - 1) / 2;
    const SizeType radius_right = window - radius_left - 1;
    WindowMedian median;
    SizeType left  = 0;
    SizeType right = std::min(n - 1, radius_right);
    for (SizeType i = left; i <= right; ++i) {
        median.add(samples[i]);
    }
    std::vector<double> baseline(n);
    for (SizeType i = 0; i < n; ++i) {
        baseline[i] = median.median();
        if (i + 1 == n) {
            break;
        }
        const SizeType next_left =
            (i + 1 > radius_left) ? (i + 1 - radius_left) : 0;
        const SizeType next_right = std::min(n - 1, i + 1 + radius_right);
        while (left < next_left) {
            median.remove(samples[left]);
            ++left;
        }
        while (right < next_right) {
            ++right;
            median.add(samples[right]);
        }
    }
    for (SizeType i = 0; i < n; ++i) {
        samples[i] =
            static_cast<float>(static_cast<double>(samples[i]) - baseline[i]);
    }
}

[[nodiscard]] double percentile_sorted(const std::vector<float>& sorted,
                                       double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double index = (static_cast<double>(sorted.size()) - 1.0) * q;
    const auto low     = static_cast<SizeType>(std::floor(index));
    const auto high    = static_cast<SizeType>(std::ceil(index));
    const double frac  = index - static_cast<double>(low);
    return (static_cast<double>(sorted[low]) * (1.0 - frac)) +
           (static_cast<double>(sorted[high]) * frac);
}

[[nodiscard]] double location_of(std::span<const float> samples,
                                 LocMethod method) {
    if (method == LocMethod::kMean) {
        double sum = 0.0;
        for (float sample : samples) {
            sum += static_cast<double>(sample);
        }
        return sum / static_cast<double>(samples.size());
    }
    std::vector<float> sorted(samples.begin(), samples.end());
    std::ranges::sort(sorted);
    return percentile_sorted(sorted, 0.5);
}

[[nodiscard]] double scale_of(std::span<const float> samples,
                              ScaleMethod method) {
    if (method == ScaleMethod::kStd) {
        const double mean = location_of(samples, LocMethod::kMean);
        double accum      = 0.0;
        for (float sample : samples) {
            const double delta = static_cast<double>(sample) - mean;
            accum += delta * delta;
        }
        return std::sqrt(accum / static_cast<double>(samples.size()));
    }
    std::vector<float> sorted(samples.begin(), samples.end());
    std::ranges::sort(sorted);
    if (method == ScaleMethod::kIqr) {
        const double q75 = percentile_sorted(sorted, 0.75);
        const double q25 = percentile_sorted(sorted, 0.25);
        return (q75 - q25) / kIqrScale;
    }
    const double med = percentile_sorted(sorted, 0.5);
    std::vector<float> absdev(sorted.size());
    std::ranges::transform(sorted, absdev.begin(), [med](float sample) {
        return static_cast<float>(std::abs(static_cast<double>(sample) - med));
    });
    std::ranges::sort(absdev);
    return percentile_sorted(absdev, 0.5) * kMadScale;
}

void zscore(std::span<float> samples, LocMethod loc, ScaleMethod scale) {
    const double location    = location_of(samples, loc);
    const double scale_value = scale_of(samples, scale);
    for (float& sample : samples) {
        sample = static_cast<float>(static_cast<double>(sample) - location);
    }
    if (!(scale_value > 0.0) || !is_finite_double(scale_value)) {
        return;
    }
    for (float& sample : samples) {
        sample = static_cast<float>(static_cast<double>(sample) / scale_value);
    }
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
    error_check::check(is_finite_double(dt) && dt > 0.0, "dt must be positive");
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
        subtract_running_median(samples, options.filter_window, header.tsamp);
        zscore(samples, options.loc, options.scale);
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
    error_check::check(is_finite_double(m_impl->header.tsamp) &&
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
