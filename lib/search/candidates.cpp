#include "loki/search/candidates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/common/interrupt.hpp"

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

/// Most points roots() collects for a polynomial of degree @p degree: the
/// points of its derivative, plus at most one root per interval between
/// them, plus the endpoints, R(d) = 2 R(d - 1) + 3 with R(1) = 1 (exact
/// duplicates are stored once). Rounding can add sign changes beyond a
/// polynomial's real roots, so the bound counts intervals, not roots.
constexpr std::size_t max_points(std::size_t degree) {
    return degree <= 1 ? 1 : (2 * max_points(degree - 1)) + 3;
}

/// Points of [-1, 1] collected by roots(): the derivative's points, zeros at
/// the bounds and one bisection root per sign change. They only bound the
/// polynomial's range, so an exact duplicate is stored once (a zero at a
/// bound is met from both sides, at every derivative level).
struct Points {
    // Only [0, n) is read: no zero fill, which would write 1 KB per
    // recursion level of roots() on the clustering hot path.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init): see above
    std::array<double, max_points(kPhaseDegree)> x;
    std::size_t n{};
    void push(double value) {
        if (std::find(x.begin(), x.begin() + n, value) == x.begin() + n) {
            x.at(n++) = value;
        }
    }
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
    std::copy_n(critical.x.begin(), critical.n, result.x.begin());
    result.n = critical.n;
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

/// Time samples of a phase-difference track for the dominance test.
constexpr std::size_t kDominanceSamples = 129;
/// Widest pulse, in duty cycle, the dominance test considers, and the step
/// between the widths it tries.
constexpr double kDominanceMaxDuty   = 0.3;
constexpr double kDominanceWidthStep = 1.5;

/// Fraction of the S/N of a pulse of duty @p w that a trial recovers when its
/// phase differs from the pulse's track by @p dphi (cycles, sampled uniformly
/// over the observation). A boxcar of width W >= w that holds a fraction q
/// of the samples recovers about q sqrt(w / W) of it; the best W wins.
double dominance_ratio(std::span<double> dphi, double w) {
    std::ranges::sort(dphi);
    const auto n = dphi.size();
    double best  = 0;
    double width = w;
    while (true) {
        std::size_t most = 0;
        std::size_t j    = 0;
        for (std::size_t i = 0; i < n; ++i) {
            while (j < n && dphi[j] <= dphi[i] + width) {
                ++j;
            }
            most = std::max(most, j - i);
        }
        best =
            std::max(best, static_cast<double>(most) / static_cast<double>(n) *
                               std::sqrt(w / width));
        if (width >= kDominanceMaxDuty) {
            break;
        }
        width = std::min(width * kDominanceWidthStep, kDominanceMaxDuty);
    }
    return best;
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

CandidateOptions CandidateOptions::ep_defaults() {
    CandidateOptions options;
    options.max_per_cell    = 32;
    options.cell_bins       = 64;
    options.dominance_kappa = 1.5;
    return options;
}

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
    if (!utils::is_finite(cell_bins) || cell_bins < 0) {
        throw std::invalid_argument(
            "candidate cell_bins must be finite and nonnegative");
    }
    if (!utils::is_finite(dominance_kappa) || dominance_kappa < 0) {
        throw std::invalid_argument(
            "candidate dominance_kappa must be finite and nonnegative");
    }
    if (max_per_cell != 0 && (cell_bins > 0) == (cell_log_width > 0)) {
        throw std::invalid_argument(
            "candidate max_per_cell requires exactly one positive cell width, "
            "cell_bins or cell_log_width");
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
    if (!utils::is_finite(harmonic_doppler_tolerance) ||
        harmonic_doppler_tolerance < 0 || harmonic_doppler_tolerance >= 0.1) {
        throw std::invalid_argument(
            "harmonic Doppler tolerance must be in [0, 0.1)");
    }
    if (!utils::is_finite(harmonic_doppler_min_snr) ||
        harmonic_doppler_min_snr < 0) {
        throw std::invalid_argument(
            "harmonic Doppler minimum S/N must be finite and non-negative");
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
    if (!is_known_leaf_basis(static_cast<std::uint8_t>(leaf_basis)) ||
        !utils::is_finite(p_orb) || p_orb < 0) {
        throw std::invalid_argument("invalid candidate leaf basis or p_orb");
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

CellCap::CellCap(const CandidateOptions& options,
                 double tobs,
                 std::size_t reserved_bytes)
    : m_max(options.max_per_cell),
      m_cap(options.max_per_cell),
      m_log(options.cell_log_width > 0),
      m_scale(m_log ? 1 / options.cell_log_width : tobs / options.cell_bins),
      m_memory_gb(options.memory_gb -
                  static_cast<double>(reserved_bytes) / utils::kBytesPerGB) {
    options.validate();
    if (m_max == 0) {
        throw std::invalid_argument("candidate cell cap needs max_per_cell");
    }
    if (!m_log && (!utils::is_finite(tobs) || tobs <= 0)) {
        throw std::invalid_argument(
            "candidate cell_bins needs a positive observation length");
    }
    if (m_memory_gb <= 0) {
        throw std::runtime_error(
            "candidate source index exhausts the host memory budget");
    }
}

void CellCap::add(const Candidate& c) {
    c.validate();
    const double cell = std::floor(
        (m_log ? std::log(c.derivatives[0]) : c.derivatives[0]) * m_scale);
    if (!(std::abs(cell) <
          static_cast<double>(std::numeric_limits<std::int64_t>::max()))) {
        throw std::invalid_argument(
            "candidate frequency cell exceeds the cell index range");
    }
    auto& kept = m_cells[{c.source_region, static_cast<std::int64_t>(cell)}];
    if (kept.size() < m_cap) {
        // A kept trial is held by its cell, whose growth may double it, and
        // once more by take(); a cell adds its hash node and buckets.
        constexpr std::size_t kBytesPerTrial = 3 * sizeof(Candidate);
        constexpr std::size_t kBytesPerCell =
            sizeof(std::vector<Candidate>) + 128;
        const auto required = [&] {
            const std::size_t extra = kept.size() < m_cap ? 1U : 0U;
            return (static_cast<double>(m_kept + extra) * kBytesPerTrial) +
                   (static_cast<double>(m_cells.size()) * kBytesPerCell);
        };
        // Over budget, every cell keeps fewer trials: halving the cap of all
        // cells keeps the same trials whatever the arrival order, since the
        // kept count only grows and the strongest m/2 of a cell's strongest
        // m are its strongest m/2.
        while (required() > m_memory_gb * utils::kBytesPerGB / 2) {
            if (m_cap == 1) {
                throw std::runtime_error(
                    "candidate cells exceed host memory budget with one trial "
                    "each; increase candidate memory_gb or the cell width");
            }
            m_cap /= 2;
            for (auto& [key, trials] : m_cells) {
                while (trials.size() > m_cap) {
                    std::ranges::pop_heap(trials, candidate_brighter);
                    m_max_dropped_snr =
                        std::max(m_max_dropped_snr, trials.back().snr);
                    trials.pop_back();
                    --m_kept;
                    ++m_dropped;
                }
            }
        }
        if (kept.size() >= m_cap) {
            add_to_full(kept, c);
            return;
        }
        kept.push_back(c);
        std::ranges::push_heap(kept, candidate_brighter);
        ++m_kept;
        return;
    }
    add_to_full(kept, c);
}

void CellCap::add_to_full(std::vector<Candidate>& kept, const Candidate& c) {
    ++m_dropped;
    if (candidate_brighter(c, kept.front())) {
        std::ranges::pop_heap(kept, candidate_brighter);
        m_max_dropped_snr = std::max(m_max_dropped_snr, kept.back().snr);
        kept.back()       = c;
        std::ranges::push_heap(kept, candidate_brighter);
    } else {
        m_max_dropped_snr = std::max(m_max_dropped_snr, c.snr);
    }
}

DominanceReducer::DominanceReducer(double tobs, const CandidateOptions& options)
    : m_tobs(tobs),
      m_options(options),
      m_bins(options.cell_bins > 0),
      m_reach(m_bins ? 2 * options.cell_bins / tobs
                     : 2 * options.cell_log_width) {
    options.validate();
    if (options.dominance_kappa <= 0 || options.max_per_cell == 0) {
        throw std::invalid_argument(
            "candidate dominance needs a kappa and the cell cap");
    }
    if (!utils::is_finite(tobs) || tobs <= 0) {
        throw std::invalid_argument(
            "candidate dominance needs a positive observation length");
    }
}

bool DominanceReducer::explains(const Candidate& strong,
                                const Candidate& weak) const {
    if (weak.snr > m_options.dominance_kappa * strong.snr) {
        return false; // the ratio never exceeds one
    }
    const double w     = std::min(strong.duty_cycle(), weak.duty_cycle());
    const double limit = std::max(m_options.phase_radius, m_options.radius * w);
    if (phase_span(weak, strong, m_tobs, 1.0, limit) <= limit) {
        return true;
    }
    Polynomial p;
    double factor = 1;
    for (std::size_t k = 0; k < weak.derivatives.size(); ++k) {
        factor *= (m_tobs / 2) / static_cast<double>(k + 1);
        p.c[k + 1] = (weak.derivatives[k] - strong.derivatives[k]) * factor;
    }
    p.n = kPhaseDegree + 1;
    std::array<double, kDominanceSamples> dphi{};
    for (std::size_t j = 0; j < dphi.size(); ++j) {
        dphi[j] = evaluate(p, -1 + (2.0 * static_cast<double>(j) /
                                    static_cast<double>(dphi.size() - 1)));
    }
    return weak.snr <=
           m_options.dominance_kappa * strong.snr * dominance_ratio(dphi, w);
}

std::uint64_t DominanceReducer::reduce(std::vector<Candidate>& trials) const {
    std::ranges::sort(trials, [](const Candidate& a, const Candidate& b) {
        if (a.source_region != b.source_region) {
            return a.source_region < b.source_region;
        }
        return candidate_brighter(a, b);
    });
    std::vector<Candidate> kept;
    kept.reserve(trials.size());
    // Kept trials of the current region by frequency.
    std::multimap<double, std::size_t> by_freq;
    std::uint64_t absorbed = 0;
    for (std::size_t i = 0; i < trials.size(); ++i) {
        if (i % 1024 == 0) {
            loki::check_interrupt();
        }
        const auto& c = trials[i];
        if (i > 0 && c.source_region != trials[i - 1].source_region) {
            by_freq.clear();
        }
        const double f     = c.derivatives[0];
        const double reach = m_bins ? m_reach : m_reach * f;
        bool explained     = false;
        for (auto it = by_freq.lower_bound(f - reach);
             it != by_freq.end() && it->first <= f + reach; ++it) {
            if (explains(kept[it->second], c)) {
                explained = true;
                break;
            }
        }
        if (explained) {
            ++absorbed;
            continue;
        }
        by_freq.emplace(f, kept.size());
        kept.push_back(c);
    }
    trials = std::move(kept);
    return absorbed;
}

Candidate ep_leaf_candidate(std::span<const double> leaf,
                            std::size_t nparams,
                            float score,
                            std::uint32_t width,
                            std::uint32_t nbins,
                            std::uint32_t run,
                            std::uint32_t index) {
    Candidate c;
    c.id            = (static_cast<std::uint64_t>(run) << 32U) | index;
    c.source_region = run;
    c.source_index  = index;
    c.width         = width;
    c.nbins         = nbins;
    c.snr           = score;
    // Each parameter's uncertainty is the width of the leaf's cell (its
    // grid step in that parameter).
    for (std::size_t k = 0; k < nparams; ++k) {
        c.parameters[k]       = leaf[2 * k];
        c.parameter_errors[k] = std::abs(leaf[(2 * k) + 1]);
    }
    c.has_parameter_errors = true;
    c.derivatives[0]       = c.parameters[nparams - 1];
    for (std::size_t k = 1; k < nparams; ++k) {
        c.derivatives[k] =
            -c.derivatives[0] * c.parameters[nparams - 1 - k] * utils::kInvCval;
    }
    // Only circular-orbit searches (five parameters, Taylor basis) write the
    // basis slot (after f0); other layouts may use that position otherwise.
    // Circular parameters are [crackle, snap, jerk, acceleration, frequency].
    if (nparams == 5) {
        const double basis_value = leaf[(2 * (nparams + 1)) + 1];
        if (!(basis_value >= 0.0 && basis_value <= 255.0) ||
            basis_value != std::floor(basis_value) ||
            !is_known_leaf_basis(static_cast<std::uint8_t>(basis_value))) {
            throw std::invalid_argument("unknown EP leaf basis code");
        }
        c.leaf_basis = static_cast<LeafBasis>(basis_value);
    }
    if (c.leaf_basis != LeafBasis::kPolynomial) {
        const bool from_snap  = c.leaf_basis == LeafBasis::kCircularSnap;
        const double num      = from_snap ? c.parameters[1] : c.parameters[0];
        const double den      = from_snap ? c.parameters[3] : c.parameters[2];
        const double omega_sq = den != 0.0 ? -num / den : 0.0;
        if (omega_sq > 0.0 && utils::is_finite(omega_sq)) {
            c.p_orb = static_cast<float>(2.0 * std::numbers::pi /
                                         std::sqrt(omega_sq));
        }
    }
    c.validate();
    return c;
}

RunLeafReduction reduce_run_leaves(
    const LeafReductionSpec& spec,
    std::size_t n_leaves,
    std::size_t nparams,
    const std::function<std::span<const double>(std::size_t)>& leaf,
    const std::function<float(std::size_t)>& score,
    std::span<const std::uint32_t> widths) {
    if (widths.size() != n_leaves) {
        throw std::invalid_argument("leaf reduction needs one width per leaf");
    }
    if (n_leaves > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "EP leaf index exceeds detection identity capacity");
    }
    CellCap cap(spec.options, spec.tobs);
    RunLeafReduction out;
    out.final_leaves = n_leaves;
    for (std::size_t i = 0; i < n_leaves; ++i) {
        if (i % 65536 == 0) {
            loki::check_interrupt();
        }
        const float s = score(i);
        if (!utils::is_finite(s)) {
            throw std::invalid_argument("non-finite final EP score");
        }
        if (s < spec.snr_min) {
            continue;
        }
        ++out.above_snr_min;
        cap.add(ep_leaf_candidate(leaf(i), nparams, s, widths[i], spec.nbins, 0,
                                  static_cast<std::uint32_t>(i)));
    }
    out.max_cell_dropped_snr = cap.max_dropped_snr();
    out.cell_dropped         = cap.dropped();
    out.max_per_cell_used    = cap.max_per_cell_used();
    if (out.max_per_cell_used < spec.options.max_per_cell) {
        spdlog::warn("EP run cells kept {} leaves each instead of {} to fit "
                     "candidate memory_gb",
                     out.max_per_cell_used, spec.options.max_per_cell);
    }
    auto kept = cap.take();
    if (spec.options.dominance_kappa > 0) {
        out.dominated = DominanceReducer(spec.tobs, spec.options).reduce(kept);
    }
    out.retained.reserve(kept.size());
    for (const auto& c : kept) {
        out.retained.push_back(c.source_index);
    }
    std::ranges::sort(out.retained);
    return out;
}

std::vector<Candidate> CellCap::take() {
    std::vector<Candidate> out;
    out.reserve(m_kept);
    for (auto it = m_cells.begin(); it != m_cells.end();
         it      = m_cells.erase(it)) {
        out.insert(out.end(), it->second.begin(), it->second.end());
    }
    m_kept            = 0;
    m_max_dropped_snr = 0;
    m_cap             = m_max;
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

void CandidateClusterer::flag_harmonic_families() {
    if (clusters.size() <= 1) {
        return;
    }
    const auto max_order     = m_options.harmonic_denominator_max;
    const double doppler_tol = m_options.harmonic_doppler_tolerance;
    const double n_grid =
        static_cast<double>(max_order) * static_cast<double>(max_order);
    const double est_pairs =
        std::min(n_grid, (0.65 * n_grid) + static_cast<double>(max_order));
    const std::size_t bytes_per_entry =
        sizeof(std::uint32_t) * 2 + sizeof(double) +
        (doppler_tol > 0.0 ? sizeof(double) : std::size_t{0});
    if (est_pairs * static_cast<double>(bytes_per_entry) >
        m_memory_gb * utils::kBytesPerGB / 2.0) {
        throw std::runtime_error(
            "harmonic ratio table exceeds host memory budget; lower "
            "harmonic_denominator_max or increase candidate memory_gb");
    }

    // The reduced ratios p/q, simplest (smallest p * q) first.
    struct Ratio {
        std::uint32_t p;
        std::uint32_t q;
        double value;
    };
    std::vector<Ratio> ratios;
    ratios.reserve(static_cast<std::size_t>(est_pairs));
    for (std::uint32_t p = 1; p <= max_order; ++p) {
        if (p % 64U == 0U) {
            loki::check_interrupt();
        }
        for (std::uint32_t q = 1; q <= max_order; ++q) {
            if (std::gcd(p, q) == 1) {
                ratios.push_back({p, q, static_cast<double>(p) / q});
            }
        }
    }
    std::ranges::sort(ratios, [](const Ratio& a, const Ratio& b) {
        const auto oa = static_cast<std::uint64_t>(a.p) * a.q;
        const auto ob = static_cast<std::uint64_t>(b.p) * b.q;
        return oa != ob ? oa < ob : a.value < b.value;
    });
    std::vector<double> values;
    if (doppler_tol > 0.0) {
        values.reserve(ratios.size());
        for (const auto& r : ratios) {
            values.push_back(r.value);
        }
        std::ranges::sort(values);
    }
    // The track test needs |f - r f_parent| * T / 2 <= limit * n^2 (Markov's
    // inequality: a degree-n polynomial with p(0) = 0 spans at least
    // |p'(0)| / n^2 on [-1, 1]), and limit <= harmonic_phase_distance (duty
    // cycles are at most 1).
    constexpr auto kDegree = static_cast<double>(
        std::tuple_size_v<decltype(Candidate{}.derivatives)>);
    const double track_window =
        2 * m_options.harmonic_phase_distance * kDegree * kDegree / m_tobs;
    // Roots so far by frequency (index into clusters).
    std::multimap<double, std::size_t> roots;
    for (std::size_t h = 0; h < clusters.size(); ++h) {
        loki::check_interrupt();
        auto& child              = clusters[h];
        const auto& hc           = child.representative;
        const double fh          = hc.derivatives[0];
        std::uint64_t best_order = 0;
        std::size_t best_parent  = 0;
        const Ratio* best_ratio  = nullptr;
        for (const auto& r : ratios) {
            const auto order = static_cast<std::uint64_t>(r.p) * r.q;
            if (best_ratio != nullptr && order > best_order) {
                break;
            }
            double lo_freq = (fh - track_window) / r.value;
            double hi_freq = (fh + track_window) / r.value;
            if (doppler_tol > 0.0) {
                const double target_c = fh / r.value;
                lo_freq = std::min(lo_freq, (target_c / (1.0 + doppler_tol)) *
                                                (1.0 - 1.0e-12));
                hi_freq = std::max(hi_freq, (target_c / (1.0 - doppler_tol)) *
                                                (1.0 + 1.0e-12));
            }
            const auto lo = roots.lower_bound(lo_freq);
            const auto hi = roots.upper_bound(hi_freq);
            for (auto it = lo; it != hi; ++it) {
                const std::size_t f = it->second;
                if (best_ratio != nullptr && f >= best_parent) {
                    continue;
                }
                const auto& fc = clusters[f].representative;
                // A harmonic is not stronger than its parent.
                if (hc.snr > fc.snr + m_options.harmonic_snr_distance) {
                    continue;
                }
                const double fc_freq  = fc.derivatives[0];
                const double target_f = r.value * fc_freq;
                const double df       = std::abs(fh - target_f);
                bool related          = false;
                if (doppler_tol > 0.0 &&
                    fc.snr >= m_options.harmonic_doppler_min_snr &&
                    df <= doppler_tol * target_f * (1.0 + 1.0e-12)) {
                    // Only an unambiguous ratio: with large orders some p/q
                    // lies within the window of almost any frequency.
                    const double ratio = fh / fc_freq;
                    const double v_lo =
                        (ratio / (1.0 + doppler_tol)) * (1.0 - 1.0e-12);
                    const double v_hi =
                        (ratio / (1.0 - doppler_tol)) * (1.0 + 1.0e-12);
                    std::uint32_t in_window = 0;
                    for (auto vit = std::ranges::lower_bound(values, v_lo);
                         vit != values.end() && *vit <= v_hi; ++vit) {
                        const double vf = (*vit) * fc_freq;
                        if (std::abs(fh - vf) <=
                            doppler_tol * vf * (1.0 + 1.0e-12)) {
                            ++in_window;
                            if (in_window > 1) {
                                break;
                            }
                        }
                    }
                    related = (in_window == 1);
                }
                if (!related) {
                    const double duty = fh >= fc.derivatives[0]
                                            ? hc.duty_cycle()
                                            : fc.duty_cycle();
                    const double limit =
                        m_options.harmonic_phase_distance * duty;
                    related =
                        df * (m_tobs / 2) <= limit * kDegree * kDegree &&
                        phase_span(fc, hc, m_tobs, r.value, limit) <= limit;
                }
                if (related) {
                    best_order  = order;
                    best_parent = f;
                    best_ratio  = &r;
                }
            }
        }
        if (best_ratio != nullptr) {
            child.parent_cluster_id    = static_cast<std::int64_t>(best_parent);
            child.harmonic_numerator   = best_ratio->p;
            child.harmonic_denominator = best_ratio->q;
        } else {
            roots.emplace(fh, h);
        }
    }
}

void CandidateClusterer::finish() {
    if (m_options.cluster && m_options.grouping == "transitive") {
        finish_transitive();
    }
    if (m_options.flag_harmonics) {
        flag_harmonic_families();
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
