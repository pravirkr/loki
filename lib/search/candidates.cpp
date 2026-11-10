#include "loki/search/candidates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <spdlog/spdlog.h>

#include "lib/detail/utils.hpp"
#include "lib/search/candidate_processing.hpp"

namespace loki::search {
namespace {
/// Degree of the phase-difference polynomial: one per frequency derivative.
constexpr std::size_t kPhaseDegree =
    std::tuple_size_v<decltype(Candidate::derivatives)>;

/// Dense polynomial of degree at most kPhaseDegree: c[0] + c[1] x + ...,
/// with n coefficients (n = 0 is the zero polynomial). Fixed storage: this
/// runs once per candidate pair, and a bright source brings thousands of
/// pairs per detection.
struct Polynomial {
    std::array<double, kPhaseDegree + 1> c{};
    std::size_t n{};
};

/// Points of [-1, 1] collected by roots(): critical points, endpoint zeros
/// and one bisection root per sign change, at most 3 * degree - 1.
struct Points {
    std::array<double, 3 * kPhaseDegree> x{};
    std::size_t n{};
    void push(double value) { x.at(n++) = value; }
    [[nodiscard]] double* begin() { return x.data(); }
    [[nodiscard]] double* end() { return x.data() + n; }
};

void positive(double value, const char* label) {
    if (!utils::is_finite(value) || value <= 0) {
        throw std::invalid_argument(label);
    }
}

double evaluate(const Polynomial& p, double x) {
    double y = 0;
    for (std::size_t i = p.n; i > 0; --i) {
        y = std::fma(y, x, p.c[i - 1]);
    }
    return y;
}

Polynomial derivative(const Polynomial& p) {
    Polynomial result;
    for (std::size_t i = 1; i < p.n; ++i) {
        result.c[i - 1] = p.c[i] * static_cast<double>(i);
    }
    result.n = p.n > 0 ? p.n - 1 : 0;
    return result;
}

// Isolate roots on monotone intervals bounded by roots of the derivative.
// Degree <= 6. Repeated roots are included at stationary points; redundant
// points are harmless when calculating the enclosing phase range.
Points roots(Polynomial p) {
    while (p.n > 0 && p.c[p.n - 1] == 0) {
        --p.n;
    }
    Points result;
    if (p.n <= 1) {
        return result;
    }
    if (p.n == 2) {
        const double x = -p.c[0] / p.c[1];
        if (x >= -1 && x <= 1) {
            result.push(x);
        }
        return result;
    }
    const auto critical = roots(derivative(p));
    Points bounds;
    bounds.push(-1);
    for (std::size_t i = 0; i < critical.n; ++i) {
        bounds.push(critical.x[i]);
    }
    bounds.push(1);
    std::ranges::sort(bounds);
    result = critical;
    for (std::size_t i = 1; i < bounds.n; ++i) {
        double lo        = bounds.x[i - 1];
        double hi        = bounds.x[i];
        const double flo = evaluate(p, lo);
        const double fhi = evaluate(p, hi);
        if (flo == 0) {
            result.push(lo);
        }
        if (fhi == 0) {
            result.push(hi);
        }
        if ((flo < 0 && fhi > 0) || (flo > 0 && fhi < 0)) {
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double mid = std::midpoint(lo, hi);
                const double fm  = evaluate(p, mid);
                if ((fm < 0) == (flo < 0)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            result.push(std::midpoint(lo, hi));
        }
    }
    std::ranges::sort(result);
    return result;
}

/// Frequency reach of a link of phase span @p span over @p tobs. The phase
/// difference on [-1, 1] has p'(0) = delta_f * Tobs / 2, and Bernstein's
/// inequality at the midpoint bounds |p'(0)| by degree * span / 2, so every
/// linkable track lies within |delta_f| <= degree * span / Tobs, even when
/// higher derivatives cancel the endpoint phase. 1 % slack covers rounding.
double link_reach(double span, double tobs) {
    return 1.01 * static_cast<double>(kPhaseDegree) * span / tobs;
}

/// Span (max - min) over [-1, 1] of the phase difference between the tracks
/// of @p a (scaled by @p scale_a) and @p b. With @p reject_above set, a pair
/// whose span provably exceeds it returns early with a value above
/// @p reject_above that is a lower bound on the span, not the span itself;
/// the caller's comparison against @p reject_above is unchanged either way.
double phase_span(const Candidate& a,
                  const Candidate& b,
                  double tobs,
                  double scale_a      = 1.0,
                  double reject_above = std::numeric_limits<double>::max()) {
    Polynomial p;
    double factor = 1;
    for (std::size_t k = 0; k < a.derivatives.size(); ++k) {
        factor *= (tobs / 2) / static_cast<double>(k + 1);
        p.c[k + 1] = ((scale_a * a.derivatives[k]) - b.derivatives[k]) * factor;
        if (!utils::is_finite(p.c[k + 1])) {
            throw std::invalid_argument("phase-track coefficients overflow");
        }
    }
    // Horner over leading zero coefficients returns the same value, so the
    // trimmed polynomial evaluates bit for bit as the full one.
    p.n = kPhaseDegree + 1;
    while (p.n > 0 && p.c[p.n - 1] == 0) {
        --p.n;
    }
    const double at_hi = evaluate(p, 1);
    const double at_lo = evaluate(p, -1);
    double low         = std::min(at_hi, at_lo);
    double high        = std::max(at_hi, at_lo);
    if (reject_above < std::numeric_limits<double>::max()) {
        // The span is at least the spread of the endpoint values, and, since
        // p(0) = 0 lies between the extrema, at least each endpoint's
        // magnitude. Rounding in the extremum evaluation can put the computed
        // span below the second bound by a few ulp at most; the factor
        // keeps this a strict lower bound of the computed span, so a pair
        // rejected here is one the full computation would reject. Pairs
        // inside the margin take the exact path.
        const double bound =
            std::max({high - low, std::abs(at_hi), std::abs(at_lo)}) *
            (1 - 1e-6);
        if (bound > reject_above) {
            return bound;
        }
    }
    const auto critical = roots(derivative(p));
    for (std::size_t i = 0; i < critical.n; ++i) {
        const double y = evaluate(p, critical.x[i]);
        low            = std::min(low, y);
        high           = std::max(high, y);
    }
    return high - low;
}
} // namespace

void CandidateOptions::validate() const {
    if (grouping != "transitive" && grouping != "anchored") {
        throw std::invalid_argument(
            "candidate grouping must be transitive or anchored");
    }
    positive(radius, "candidate radius must be finite and positive");
    if (!utils::is_finite(phase_radius) || phase_radius < 0) {
        throw std::invalid_argument(
            "candidate phase_radius must be finite and nonnegative");
    }
    if (!utils::is_finite(cell_log_width) || cell_log_width < 0) {
        throw std::invalid_argument(
            "candidate cell_log_width must be finite and nonnegative");
    }
    if (max_per_cell != 0 && cell_log_width == 0) {
        throw std::invalid_argument(
            "candidate max_per_cell requires a positive cell_log_width");
    }
    positive(memory_gb, "candidate memory_gb must be finite and positive");
    if (memory_gb < 0.0001 || memory_gb > 1.0e6) {
        throw std::invalid_argument(
            "candidate memory_gb must be in [0.0001, 1e6]");
    }
    if (remove_harmonics && !flag_harmonics) {
        throw std::invalid_argument("remove_harmonics requires flag_harmonics");
    }
    if (harmonic_denominator_max == 0 || harmonic_denominator_max > 10000) {
        throw std::invalid_argument(
            "harmonic denominator must be in [1, 10000]");
    }
    positive(harmonic_phase_distance,
             "harmonic phase distance must be positive");
    positive(harmonic_snr_distance, "harmonic S/N distance must be positive");
}

double Candidate::duty_cycle() const {
    return static_cast<double>(width) / static_cast<double>(nbins);
}

void Candidate::validate() const {
    if (has_parameter_errors &&
        !std::ranges::all_of(parameter_errors, [](double x) {
            return utils::is_finite(x) && x >= 0;
        })) {
        throw std::invalid_argument(
            "candidate uncertainties must be finite and nonnegative");
    }
    if (width == 0 || nbins == 0 || width > nbins || !utils::is_finite(snr) ||
        !std::ranges::all_of(derivatives,
                             [](double x) { return utils::is_finite(x); }) ||
        !std::ranges::all_of(parameters,
                             [](double x) { return utils::is_finite(x); }) ||
        derivatives[0] <= 0) {
        throw std::invalid_argument(
            "invalid candidate parameters, score or width");
    }
}

double
phase_track_distance(const Candidate& a, const Candidate& b, double tobs) {
    a.validate();
    b.validate();
    positive(tobs, "candidate tobs must be finite and positive");
    return phase_span(a, b, tobs) / std::min(a.duty_cycle(), b.duty_cycle());
}

namespace detail {
bool candidate_brighter(const Candidate& a, const Candidate& b) {
    if (a.snr != b.snr) {
        return a.snr > b.snr;
    }
    if (a.derivatives != b.derivatives) {
        return a.derivatives < b.derivatives;
    }
    if (a.duty_cycle() != b.duty_cycle()) {
        return a.duty_cycle() < b.duty_cycle();
    }
    if (a.nbins != b.nbins) {
        return a.nbins > b.nbins;
    }
    return a.id < b.id;
}

CellCap::CellCap(const CandidateOptions& options, std::size_t reserved_bytes)
    : m_max(options.max_per_cell),
      m_width(options.cell_log_width),
      m_memory_gb(options.memory_gb -
                  static_cast<double>(reserved_bytes) / utils::kBytesPerGB) {
    options.validate();
    if (m_max == 0) {
        throw std::invalid_argument("candidate cell cap needs max_per_cell");
    }
    if (m_memory_gb <= 0) {
        throw std::runtime_error(
            "candidate source index exhausts the host memory budget");
    }
}

void CellCap::add(const Candidate& c) {
    c.validate();
    const double cell = std::floor(std::log(c.derivatives[0]) / m_width);
    if (!(std::abs(cell) <
          static_cast<double>(std::numeric_limits<std::int64_t>::max()))) {
        throw std::invalid_argument(
            "candidate frequency cell exceeds the cell index range");
    }
    auto& kept = m_cells[static_cast<std::int64_t>(cell)];
    if (kept.size() < m_max) {
        // A kept trial is held by its cell, whose growth may double it, and
        // once more by take(); a cell adds its hash node and buckets.
        constexpr std::size_t kBytesPerTrial = 3 * sizeof(Candidate);
        constexpr std::size_t kBytesPerCell =
            sizeof(std::vector<Candidate>) + 128;
        const double required =
            static_cast<double>(m_kept + 1) * kBytesPerTrial +
            static_cast<double>(m_cells.size()) * kBytesPerCell;
        if (required > m_memory_gb * utils::kBytesPerGB / 2) {
            throw std::runtime_error(
                "candidate cell index exceeds host memory budget; increase "
                "candidate memory_gb or cell_log_width");
        }
        kept.push_back(c);
        std::ranges::push_heap(kept, candidate_brighter);
        ++m_kept;
        return;
    }
    ++m_dropped;
    if (candidate_brighter(c, kept.front())) {
        std::ranges::pop_heap(kept, candidate_brighter);
        kept.back() = c;
        std::ranges::push_heap(kept, candidate_brighter);
    }
}

std::vector<Candidate> CellCap::take() {
    std::vector<Candidate> out;
    out.reserve(m_kept);
    for (auto it = m_cells.begin(); it != m_cells.end();
         it      = m_cells.erase(it)) {
        out.insert(out.end(), it->second.begin(), it->second.end());
    }
    m_kept = 0;
    return out;
}

CandidateClusterer::CandidateClusterer(double tobs,
                                       const CandidateOptions& options,
                                       std::size_t reserved_bytes)
    : m_tobs(tobs),
      m_options(options),
      m_memory_gb(options.memory_gb -
                  static_cast<double>(reserved_bytes) / utils::kBytesPerGB) {
    positive(tobs, "candidate tobs must be finite and positive");
    options.validate();
    if (m_memory_gb <= 0) {
        throw std::runtime_error(
            "candidate source index exhausts the host memory budget");
    }
}

std::uint64_t CandidateClusterer::append(const Candidate& c) {
    c.validate();
    if (m_options.cluster && m_options.grouping == "transitive") {
        return append_transitive(c);
    }
    std::uint64_t match = std::numeric_limits<std::uint64_t>::max();
    if (m_options.cluster) {
        // min(duty_a,duty_b) <= duty_c, including the phase-cycle floor,
        // making this a conservative prefilter for every Taylor order.
        const double reach = link_reach(
            std::max(m_options.radius * c.duty_cycle(), m_options.phase_radius),
            m_tobs);
        const auto end = m_index.upper_bound(c.derivatives[0] + reach);
        for (auto it = m_index.lower_bound(c.derivatives[0] - reach); it != end;
             ++it) {
            if (it->second >= match) {
                continue;
            }
            const double limit = std::max(
                m_options.phase_radius,
                m_options.radius *
                    std::min(clusters[it->second].representative.duty_cycle(),
                             c.duty_cycle()));
            if (phase_span(clusters[it->second].representative, c, m_tobs, 1.0,
                           limit) <= limit) {
                match = it->second;
            }
        }
    }
    if (match != std::numeric_limits<std::uint64_t>::max()) {
        clusters[match].phase_extent =
            std::max(clusters[match].phase_extent,
                     phase_span(clusters[match].representative, c, m_tobs));
        ++clusters[match].member_count;
        return match;
    }
    // Includes old + new vector allocations during growth, map nodes, and
    // harmonic/serialization scratch. The remaining half is reserved for I/O.
    constexpr std::size_t kBytesPerCluster = 8 * sizeof(CandidateCluster) + 256;
    const double required = static_cast<double>(clusters.size() + 1) *
                            static_cast<double>(kBytesPerCluster);
    if (required > m_memory_gb * utils::kBytesPerGB / 2) {
        throw std::runtime_error("candidate representative index exceeds host "
                                 "memory budget; increase candidate memory_gb");
    }
    const auto id = static_cast<std::uint64_t>(clusters.size());
    clusters.push_back(CandidateCluster{.representative = c});
    if (m_options.cluster) {
        m_index.emplace(c.derivatives[0], id);
    }
    return id;
}

std::uint64_t CandidateClusterer::root(std::uint64_t node) {
    while (m_nodes[node].parent != node) {
        m_nodes[node].parent = m_nodes[m_nodes[node].parent].parent;
        node                 = m_nodes[node].parent;
    }
    return node;
}

std::uint64_t CandidateClusterer::append_transitive(const Candidate& c) {
    const auto found   = m_tracks.find(c.derivatives);
    std::uint64_t node = 0;
    if (found != m_tracks.end()) {
        node = found->second;
        ++m_nodes[root(node)].member_count;
        if (c.duty_cycle() <= m_nodes[node].duty) {
            return node;
        }
        // Identical tracks are a clique. Their widest trial covers the union
        // of their neighbor sets under the narrower-duty link rule. Expanding
        // this envelope must discover NEW edges even when the score is weaker.
        m_nodes[node].duty = c.duty_cycle();
    } else {
        // Both node/cluster vector reallocations, two tree indexes, union-find
        // remapping and serialization scratch are covered conservatively.
        constexpr std::size_t kBytesPerNode =
            8 * sizeof(Node) + 8 * sizeof(CandidateCluster) + 512;
        if (static_cast<double>(m_nodes.size() + 1) * kBytesPerNode >
            m_memory_gb * utils::kBytesPerGB / 2) {
            throw std::runtime_error(
                "candidate transitive track index exceeds host "
                "memory budget; increase candidate memory_gb");
        }
        node = static_cast<std::uint64_t>(m_nodes.size());
        m_nodes.push_back(
            Node{.representative = c, .duty = c.duty_cycle(), .parent = node});
        m_tracks.emplace(c.derivatives, node);
        m_index.emplace(c.derivatives[0], node);
    }
    // Every matching track is within this frequency interval (link_reach).
    const double reach = link_reach(
        std::max(m_options.phase_radius, m_options.radius * m_nodes[node].duty),
        m_tobs);
    const auto end = m_index.upper_bound(c.derivatives[0] + reach);
    for (auto it = m_index.lower_bound(c.derivatives[0] - reach); it != end;
         ++it) {
        const auto other = it->second;
        auto a           = root(node);
        auto b           = root(other);
        if (a == b) {
            continue;
        }
        const double limit =
            std::max(m_options.phase_radius,
                     m_options.radius *
                         std::min(m_nodes[node].duty, m_nodes[other].duty));
        if (phase_span(c, m_nodes[other].representative, m_tobs, 1.0, limit) >
            limit) {
            continue;
        }
        // Inputs are globally sorted. The earliest root is the strongest
        // ORIGINAL physical trial, including deterministic tie breakers.
        if (a > b) {
            std::swap(a, b);
        }
        m_nodes[b].parent = a;
        m_nodes[a].member_count += m_nodes[b].member_count;
    }
    return node;
}

void CandidateClusterer::finish_transitive() {
    m_remap.resize(m_nodes.size());
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        if (root(i) == i) {
            m_remap[i] = clusters.size();
            clusters.push_back(
                CandidateCluster{.representative = m_nodes[i].representative,
                                 .member_count   = m_nodes[i].member_count});
        }
    }
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        m_remap[i]    = m_remap[root(i)];
        auto& cluster = clusters[m_remap[i]];
        cluster.phase_extent =
            std::max(cluster.phase_extent,
                     phase_span(cluster.representative,
                                m_nodes[i].representative, m_tobs));
    }
}

std::uint64_t CandidateClusterer::cluster_id(std::uint64_t provisional) const {
    return m_options.cluster && m_options.grouping == "transitive"
               ? m_remap.at(provisional)
               : provisional;
}

void CandidateClusterer::finish() {
    if (m_options.cluster && m_options.grouping == "transitive") {
        finish_transitive();
    }
    if (m_options.flag_harmonics) {
        // Every pair of clusters, times every denominator: quadratic.
        const double pairs = 0.5 * static_cast<double>(clusters.size()) *
                             static_cast<double>(clusters.size());
        const double work =
            pairs * static_cast<double>(m_options.harmonic_denominator_max);
        if (work > 1e10) {
            spdlog::warn("Harmonic flagging compares {} clusters pairwise "
                         "with denominators up to {} (~{:.1e} steps); this "
                         "may take hours. Lower harmonic_denominator_max or "
                         "snr_min, or disable flag_harmonics.",
                         clusters.size(), m_options.harmonic_denominator_max,
                         work);
        }
        for (std::size_t h = 0; h < clusters.size(); ++h) {
            auto& child    = clusters[h];
            const auto& hc = child.representative;
            for (std::size_t f = 0; f < h; ++f) {
                const auto& parent = clusters[f];
                if (parent.parent_cluster_id >= 0) {
                    continue;
                }
                const auto& fc     = parent.representative;
                const double ratio = hc.derivatives[0] / fc.derivatives[0];
                std::uint32_t pn   = 0;
                std::uint32_t qn   = 1;
                double best        = std::numeric_limits<double>::max();
                for (std::uint32_t q = 1;
                     q <= m_options.harmonic_denominator_max; ++q) {
                    const double p = std::round(ratio * q);
                    if (p < 1 ||
                        p > std::numeric_limits<std::uint32_t>::max()) {
                        continue;
                    }
                    const double error = std::abs(p / q - ratio);
                    if (error < best) {
                        best = error;
                        pn   = static_cast<std::uint32_t>(p);
                        qn   = q;
                    }
                }
                if (pn == 0 || pn == qn) {
                    continue;
                }
                const auto divisor = std::gcd(pn, qn);
                pn /= divisor;
                qn /= divisor;
                const double expected =
                    fc.snr / std::sqrt(static_cast<double>(pn) * qn);
                if (std::abs(hc.snr - expected) >
                    m_options.harmonic_snr_distance) {
                    continue;
                }
                const double duty = hc.derivatives[0] >= fc.derivatives[0]
                                        ? hc.duty_cycle()
                                        : fc.duty_cycle();
                if (phase_span(fc, hc, m_tobs, static_cast<double>(pn) / qn) /
                        duty <=
                    m_options.harmonic_phase_distance) {
                    child.parent_cluster_id    = static_cast<std::int64_t>(f);
                    child.harmonic_numerator   = pn;
                    child.harmonic_denominator = qn;
                    break;
                }
            }
        }
    }
    std::uint64_t selected = 0;
    for (auto& cluster : clusters) {
        cluster.selected =
            !(m_options.remove_harmonics && cluster.parent_cluster_id >= 0);
        if (cluster.selected && m_options.max_candidates != 0 &&
            selected >= m_options.max_candidates) {
            cluster.selected = false;
        }
        selected += cluster.selected ? 1 : 0;
    }
}
} // namespace detail

CandidateClustering cluster_candidates(std::span<const Candidate> candidates,
                                       double tobs,
                                       const CandidateOptions& options) {
    detail::CandidateClusterer processor(tobs, options);
    // The convenience API's input and index are charged separately from the
    // representative budget reserved inside the processor.
    if (static_cast<double>(candidates.size()) * sizeof(std::size_t) >
        options.memory_gb * utils::kBytesPerGB / 4) {
        throw std::runtime_error("candidate input index exceeds memory budget");
    }
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    for (const auto& c : candidates) {
        c.validate();
    }
    std::ranges::sort(order, [&](auto a, auto b) {
        return detail::candidate_brighter(candidates[a], candidates[b]);
    });
    CandidateClustering result;
    result.membership.resize(candidates.size());
    for (auto i : order) {
        result.membership[i] = processor.append(candidates[i]);
    }
    processor.finish();
    for (auto& id : result.membership) {
        id = processor.cluster_id(id);
    }
    result.clusters = std::move(processor.clusters);
    return result;
}
} // namespace loki::search
