#include "loki/io/sift.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"

#include "lib/detail/utils.hpp"

namespace loki::io {
namespace {

constexpr double kSecondsPerDay = 86400.0;

template <typename T>
std::optional<T> optional_attribute(const HighFive::Group& group,
                                    const char* name) {
    if (!group.hasAttribute(name)) {
        return std::nullopt;
    }
    return group.getAttribute(name).read<T>();
}

template <typename T>
std::vector<T> column(const HighFive::Group& table, const char* name) {
    std::vector<T> values;
    table.getDataSet(name).read(values);
    return values;
}

/// A candidate of one file with its file index, before grouping.
struct Record {
    SiftMember member;
};

/// Reads @p path's observation and its candidates at or above
/// @p snr_min into @p records. Returns the file and its segment key.
std::pair<SiftFile, std::string> read_file(const std::filesystem::path& path,
                                           SizeType index,
                                           double snr_min,
                                           std::vector<Record>& records) {
    std::optional<HighFive::File> opened;
    try {
        opened.emplace(path.string(), HighFive::File::ReadOnly);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            fmt::format("cannot read '{}': {}", path.string(), e.what()));
    }
    const auto& file = *opened;
    if (!file.hasAttribute("schema_version") || !file.exist("candidates")) {
        throw std::invalid_argument(fmt::format(
            "'{}' is not a published result file (a raw or spool file: "
            "publish it first with `loki publish`)",
            path.string()));
    }
    SiftFile out;
    out.path = path;
    std::string key;
    if (file.exist("observation")) {
        const auto obs    = file.getGroup("observation");
        out.dm            = optional_attribute<double>(obs, "dm");
        out.start_mjd     = optional_attribute<double>(obs, "start_mjd");
        out.reference_mjd = optional_attribute<double>(obs, "reference_mjd");
        out.barycentric   = optional_attribute<bool>(obs, "barycentric");
        out.tobs = optional_attribute<double>(obs, "tobs").value_or(0.0);
        if (out.dm && !utils::is_finite(*out.dm)) {
            out.dm.reset();
        }
        if (out.start_mjd && !utils::is_finite(*out.start_mjd)) {
            out.start_mjd.reset();
        }
        if (out.reference_mjd && !utils::is_finite(*out.reference_mjd)) {
            out.reference_mjd.reset();
        }
        if (!utils::is_finite(out.tobs) || out.tobs < 0.0) {
            out.tobs = 0.0;
        }
        if (out.start_mjd) {
            // Within a microsecond: the same first sample.
            key = fmt::format("mjd:{:.11f}", *out.start_mjd);
        } else {
            key = "input:" + optional_attribute<std::string>(obs, "input_path")
                                 .value_or(path.string());
        }
    } else {
        key = "file:" + path.string();
    }
    if (out.tobs <= 0 && file.hasAttribute("tobs")) {
        double root_tobs = 0.0;
        file.getAttribute("tobs").read(root_tobs);
        if (utils::is_finite(root_tobs) && root_tobs > 0.0) {
            out.tobs = root_tobs;
        }
    }
    const auto table = file.getGroup("candidates");
    const auto f0    = column<double>(table, "f0");
    const auto f1    = column<double>(table, "f1");
    const auto snr   = column<float>(table, "snr");
    const auto duty  = column<double>(table, "duty_cycle");
    const auto ids   = column<std::uint64_t>(table, "id");
    std::vector<std::uint8_t> selected;
    if (table.exist("selected")) {
        selected = column<std::uint8_t>(table, "selected");
    }
    if (f1.size() != f0.size() || snr.size() != f0.size() ||
        duty.size() != f0.size() || ids.size() != f0.size() ||
        (!selected.empty() && selected.size() != f0.size())) {
        throw std::runtime_error(fmt::format(
            "'{}': candidate columns differ in length", path.string()));
    }
    for (std::size_t i = 0; i < f0.size(); ++i) {
        if (!utils::is_finite(f0[i]) || !(f0[i] > 0.0) ||
            !utils::is_finite(f1[i]) || !utils::is_finite(snr[i]) ||
            !utils::is_finite(duty[i]) || snr[i] < snr_min ||
            (!selected.empty() && selected[i] == 0)) {
            continue;
        }
        records.push_back({.member = {.file       = index,
                                      .id         = ids[i],
                                      .freq       = f0[i],
                                      .fdot       = f1[i],
                                      .snr        = snr[i],
                                      .duty_cycle = duty[i]}});
    }
    return {out, key};
}

/// The only ratio p/q with p, q <= @p order_max within @p tolerance (a
/// fraction) of @p ratio, or {0, 0}.
std::pair<std::uint32_t, std::uint32_t>
unique_ratio(double ratio, double tolerance, std::uint32_t order_max) {
    std::pair<std::uint32_t, std::uint32_t> found{0, 0};
    for (std::uint32_t q = 1; q <= order_max; ++q) {
        const double pr = std::round(ratio * q);
        if (pr < 1 || pr > order_max) {
            continue;
        }
        const auto p = static_cast<std::uint32_t>(pr);
        if (std::gcd(p, q) != 1 ||
            std::abs((pr / q) - ratio) > tolerance * ratio) {
            continue;
        }
        if (found.first != 0) {
            return {0, 0};
        }
        found = {p, q};
    }
    return found;
}

} // namespace

SiftResult sift(std::span<const std::filesystem::path> files,
                const SiftOptions& options) {
    if (!utils::is_finite(options.tolerance) || !(options.tolerance > 0.0) ||
        options.tolerance >= 0.1) {
        throw std::invalid_argument("sift tolerance must be in (0, 0.1)");
    }
    if (!utils::is_finite(options.snr_min)) {
        throw std::invalid_argument("sift snr_min must be finite");
    }
    if (!utils::is_finite(options.harmonic_bins) ||
        options.harmonic_bins < 0.0) {
        throw std::invalid_argument("sift harmonic_bins must be >= 0");
    }
    SiftResult result;
    std::vector<Record> records;
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < files.size(); ++i) {
        auto [file, key] = read_file(files[i], i, options.snr_min, records);
        result.files.push_back(std::move(file));
        keys.push_back(std::move(key));
    }
    // Segments numbered by start MJD (then key), so the order is the
    // observing order.
    std::map<std::pair<double, std::string>, SizeType> segment_of;
    for (std::size_t i = 0; i < files.size(); ++i) {
        segment_of.emplace(
            std::pair{result.files[i].start_mjd.value_or(0.0), keys[i]}, 0);
    }
    SizeType next = 0;
    for (auto& [key, segment] : segment_of) {
        segment = next++;
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
        result.files[i].segment =
            segment_of.at({result.files[i].start_mjd.value_or(0.0), keys[i]});
    }
    result.n_segments   = next;
    result.n_candidates = records.size();

    // The DM range of all files, for the edge ratio.
    std::optional<double> dm_lo;
    std::optional<double> dm_hi;
    for (const auto& file : result.files) {
        if (file.dm) {
            dm_lo = std::min(dm_lo.value_or(*file.dm), *file.dm);
            dm_hi = std::max(dm_hi.value_or(*file.dm), *file.dm);
        }
    }

    // Friends-of-friends in frequency.
    std::ranges::sort(records, [](const Record& a, const Record& b) {
        return a.member.freq < b.member.freq;
    });
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i == 0 || records[i].member.freq - records[i - 1].member.freq >
                          options.tolerance * records[i].member.freq) {
            spans.emplace_back(i, i + 1);
        } else {
            spans.back().second = i + 1;
        }
    }

    for (const auto& [begin, end] : spans) {
        SiftGroup group;
        std::map<double, float> curve;
        std::map<SizeType, const SiftMember*> best_in_segment;
        for (std::size_t i = begin; i < end; ++i) {
            const auto& r = records[i];
            group.members.push_back(r.member);
            if (const auto dm = result.files[r.member.file].dm;
                dm.has_value()) {
                auto& best = curve[*dm];
                best       = std::max(best, r.member.snr);
            }
        }
        std::ranges::stable_sort(
            group.members,
            [](const auto& a, const auto& b) { return a.snr > b.snr; });
        for (const auto& m : group.members) {
            const auto segment = result.files[m.file].segment;
            best_in_segment.try_emplace(segment, &m);
        }
        if (best_in_segment.size() < options.min_segments) {
            continue;
        }
        for (const auto& [segment, m] : best_in_segment) {
            group.segments.push_back(segment);
        }
        group.dm_curve.assign(curve.begin(), curve.end());
        const float best_snr = group.members.front().snr;
        // One DM trial has no edge to compare with.
        if (dm_lo && *dm_hi > *dm_lo && !curve.empty() && best_snr > 0) {
            const auto at = [&](double dm) -> float {
                const auto it = curve.find(dm);
                return it == curve.end() ? 0.0F : it->second;
            };
            group.edge_ratio =
                static_cast<double>(std::max(at(*dm_lo), at(*dm_hi))) /
                best_snr;
        }
        // Taylor extrapolation between consecutive segments' best members.
        const SiftMember* previous = nullptr;
        for (const auto& [segment, m] : best_in_segment) {
            const auto& file = result.files[m->file];
            if (previous != nullptr) {
                const auto& before = result.files[previous->file];
                if (before.reference_mjd && file.reference_mjd &&
                    file.tobs > 0) {
                    const double dt =
                        (*file.reference_mjd - *before.reference_mjd) *
                        kSecondsPerDay;
                    const double bins = std::abs(m->freq - previous->freq -
                                                 (previous->fdot * dt)) *
                                        file.tobs;
                    group.extrapolation_bins =
                        std::max(group.extrapolation_bins.value_or(0.0), bins);
                }
            }
            previous = m;
        }
        result.groups.push_back(std::move(group));
    }
    std::ranges::stable_sort(result.groups, [](const auto& a, const auto& b) {
        return a.members.front().snr > b.members.front().snr;
    });

    // Harmonic relations: a weaker group to a stronger root group.
    if (options.harmonic_order_max > 0) {
        // Each group's strongest member per file (members by decreasing
        // S/N, so the first one seen).
        std::vector<std::map<SizeType, const SiftMember*>> best_in_file(
            result.groups.size());
        for (std::size_t g = 0; g < result.groups.size(); ++g) {
            for (const auto& m : result.groups[g].members) {
                best_in_file[g].try_emplace(m.file, &m);
            }
        }
        // The smallest |f_c - ratio f_p| * tobs over the files with members
        // of both groups, if any.
        const auto offset_bins = [&](std::size_t c, std::size_t p,
                                     double ratio) -> std::optional<double> {
            std::optional<double> best;
            for (const auto& [file, m] : best_in_file[c]) {
                const auto it = best_in_file[p].find(file);
                if (it == best_in_file[p].end()) {
                    continue;
                }
                const double bins =
                    std::abs(m->freq - (ratio * it->second->freq)) *
                    result.files[file].tobs;
                best = std::min(best.value_or(bins), bins);
            }
            return best;
        };
        for (std::size_t h = 0; h < result.groups.size(); ++h) {
            auto& child              = result.groups[h];
            const auto& hb           = child.members.front();
            const auto& child_dm     = result.files[hb.file].dm;
            std::uint64_t best_order = 0;
            for (std::size_t f = 0; f < h; ++f) {
                const auto& parent = result.groups[f];
                if (parent.harmonic_of >= 0) {
                    continue;
                }
                const auto& pb = parent.members.front();
                const auto [p, q] =
                    unique_ratio(hb.freq / pb.freq, options.tolerance,
                                 options.harmonic_order_max);
                if (p == 0) {
                    continue;
                }
                if (child_dm && !parent.dm_curve.empty() &&
                    (*child_dm < parent.dm_curve.front().first ||
                     *child_dm > parent.dm_curve.back().first)) {
                    continue;
                }
                const std::uint64_t order = static_cast<std::uint64_t>(p) * q;
                if (best_order != 0 && order >= best_order) {
                    continue;
                }
                const auto bins = offset_bins(
                    h, f, static_cast<double>(p) / static_cast<double>(q));
                if (!bins || *bins > options.harmonic_bins) {
                    continue;
                }
                best_order                 = order;
                child.harmonic_of          = static_cast<std::int64_t>(f);
                child.harmonic_numerator   = p;
                child.harmonic_denominator = q;
                child.harmonic_offset_bins = bins;
            }
        }
    }
    return result;
}

} // namespace loki::io
