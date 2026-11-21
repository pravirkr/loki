#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

// Mode "descent" of DynamicThresholdScheme: see docs/dts.md for the method
// and its measured behaviour. Everything here is backend-neutral host code;
// the engine only simulates the samples.

namespace loki::detection::detail {

namespace {

// Salts of the samples, all derived from the engine's seed. The signal
// sample of the descent, every noise genealogy, the final pin and the report
// are independent of each other.
constexpr uint64_t kSignalDescent = 0xDE5C0001ULL;
constexpr uint64_t kNoiseDescent  = 0xDE5C1000ULL; // + pass index
constexpr uint64_t kBankPin       = 0xDE5C0002ULL;
constexpr uint64_t kBankReport    = 0xDE5C0003ULL;
constexpr uint64_t kNoiseReport   = 0xDE5C0004ULL;
// A move must keep every threshold this far above the genealogy's schedule.
constexpr float kMarginGuard = 0.02F;
// A set of moves is kept only if the cost on the genealogy drops by this.
constexpr double kAccept = 1.0 - 1e-4;

struct Block {
    SizeType a;
    SizeType b; // [a, b)
};

/// Disjoint blocks of @p size stages over [0, top), the first one starting
/// at -offset (clipped).
std::vector<Block> make_blocks(SizeType top, SizeType size, SizeType offset) {
    std::vector<Block> out;
    SizeType lo = 0;
    SizeType hi = offset > 0 ? size - offset : size;
    while (lo < top) {
        out.push_back({lo, std::min(hi, top)});
        lo = hi;
        hi += size;
    }
    return out;
}

/// Unpruned signal scores on the host, trial-major.
class SignalSample {
public:
    SignalSample(const std::vector<float>& stage_major,
                 SizeType nstages,
                 SizeType n)
        : m_nstages(nstages),
          m_n(n),
          m_scores(nstages * n) {
        for (SizeType s = 0; s < nstages; ++s) {
            for (SizeType t = 0; t < n; ++t) {
                m_scores[(t * nstages) + s] = stage_major[(s * n) + t];
            }
        }
    }

    SizeType size() const { return m_n; }

    /// Mean and standard deviation of every stage's scores.
    std::pair<std::vector<float>, std::vector<float>> moments() const {
        std::vector<double> sum(m_nstages, 0.0);
        std::vector<double> sq(m_nstages, 0.0);
        for (SizeType t = 0; t < m_n; ++t) {
            const float* row = m_scores.data() + (t * m_nstages);
            for (SizeType s = 0; s < m_nstages; ++s) {
                sum[s] += row[s];
                sq[s] += static_cast<double>(row[s]) * row[s];
            }
        }
        std::vector<float> mean(m_nstages);
        std::vector<float> sd(m_nstages);
        const auto n = static_cast<double>(m_n);
        for (SizeType s = 0; s < m_nstages; ++s) {
            const double mu = sum[s] / n;
            mean[s]         = static_cast<float>(mu);
            sd[s]           = static_cast<float>(
                std::sqrt(std::max((sq[s] / n) - (mu * mu), 1e-12)));
        }
        return {std::move(mean), std::move(sd)};
    }

    /// Fraction of the first @p ntrials trials above every threshold.
    double pd(std::span<const float> g, SizeType ntrials) const {
        SizeType count = 0;
        for (SizeType t = 0; t < ntrials; ++t) {
            const float* row = m_scores.data() + (t * m_nstages);
            bool alive       = true;
            for (SizeType s = 0; s < m_nstages && alive; ++s) {
                alive = row[s] > g[s];
            }
            count += alive ? 1 : 0;
        }
        return static_cast<double>(count) / static_cast<double>(ntrials);
    }

    /// Fraction of the trials whose last score exceeds @p threshold.
    double ceiling(float threshold) const {
        SizeType count = 0;
        for (SizeType t = 0; t < m_n; ++t) {
            count +=
                m_scores[(t * m_nstages) + m_nstages - 1] > threshold ? 1 : 0;
        }
        return static_cast<double>(count) / static_cast<double>(m_n);
    }

    /// The common shift (either sign) of stages [0, nfree) after which
    /// ceil(target n) trials survive; the stages beyond stay as they are.
    float
    pin_shift(std::span<const float> g, double target, SizeType nfree) const {
        std::vector<float> margin(m_n);
        for (SizeType t = 0; t < m_n; ++t) {
            const float* row = m_scores.data() + (t * m_nstages);
            float m          = std::numeric_limits<float>::max();
            for (SizeType s = 0; s < nfree; ++s) {
                m = std::min(m, row[s] - g[s]);
            }
            for (SizeType s = nfree; s < m_nstages; ++s) {
                if (row[s] <= g[s]) {
                    m = std::numeric_limits<float>::lowest();
                }
            }
            margin[t] = m;
        }
        return shift_of(margin, target);
    }

    static float shift_of(std::vector<float>& margin, double target) {
        const SizeType n        = margin.size();
        const SizeType required = std::clamp<SizeType>(
            static_cast<SizeType>(std::ceil(target * static_cast<double>(n))),
            1, n);
        const auto rank =
            margin.begin() + static_cast<std::ptrdiff_t>(n - required);
        std::ranges::nth_element(margin, rank);
        return std::nextafter(*rank, std::numeric_limits<float>::lowest());
    }

    /// Exact change of Pd when the thresholds of one block move by +d
    /// (@p up, <= 0) or -d (@p dn, >= 0), every other threshold fixed.
    void block_deltas(std::span<const float> g,
                      std::span<const Block> blocks,
                      float d,
                      std::vector<double>& up,
                      std::vector<double>& dn,
                      int nthreads) const {
        std::vector<SizeType> n_up(blocks.size(), 0);
        std::vector<SizeType> n_dn(blocks.size(), 0);
        // Block of every stage (blocks.size(): none).
        std::vector<SizeType> block_of(m_nstages, blocks.size());
        for (SizeType k = 0; k < blocks.size(); ++k) {
            for (SizeType s = blocks[k].a; s < blocks[k].b; ++s) {
                block_of[s] = k;
            }
        }
        const SizeType nblocks = blocks.size();
        const SizeType nstages = m_nstages;
        const float* scores    = m_scores.data();
        const auto ntrials     = static_cast<std::int64_t>(m_n);
        // NOLINTNEXTLINE(openmp-use-default-none): spans are not listable
#pragma omp parallel num_threads(nthreads)
        {
            std::vector<SizeType> t_up(nblocks, 0);
            std::vector<SizeType> t_dn(nblocks, 0);
            std::vector<float> m(nstages);
#pragma omp for schedule(static) nowait
            for (std::int64_t trial = 0; trial < ntrials; ++trial) {
                const float* row =
                    scores + (static_cast<SizeType>(trial) * nstages);
                SizeType first = nstages;
                bool lost      = false; // cannot be revived by one block
                for (SizeType s = 0; s < nstages; ++s) {
                    const float v = row[s] - g[s];
                    m[s]          = v;
                    if (v <= 0.0F) {
                        if (v <= -d || block_of[s] == nblocks ||
                            (first != nstages &&
                             block_of[first] != block_of[s])) {
                            lost = true;
                            break;
                        }
                        first = std::min(first, s);
                    }
                }
                if (lost) {
                    continue;
                }
                if (first != nstages) {
                    ++t_dn[block_of[first]];
                    continue;
                }
                for (SizeType k = 0; k < nblocks; ++k) {
                    float inside = std::numeric_limits<float>::max();
                    for (SizeType s = blocks[k].a; s < blocks[k].b; ++s) {
                        inside = std::min(inside, m[s]);
                    }
                    t_up[k] += inside <= d ? 1 : 0;
                }
            }
#pragma omp critical
            for (SizeType k = 0; k < nblocks; ++k) {
                n_up[k] += t_up[k];
                n_dn[k] += t_dn[k];
            }
        }
        up.resize(blocks.size());
        dn.resize(blocks.size());
        const auto n = static_cast<double>(m_n);
        for (SizeType k = 0; k < blocks.size(); ++k) {
            up[k] = -static_cast<double>(n_up[k]) / n;
            dn[k] = static_cast<double>(n_dn[k]) / n;
        }
    }

private:
    SizeType m_nstages;
    SizeType m_n;
    std::vector<float> m_scores;
};

/// A noise genealogy with the cost accounting of a schedule above the one it
/// was simulated under.
class NoiseBank {
public:
    NoiseBank(NoiseGenealogy genealogy,
              std::span<const float> branching,
              double w_terminal)
        : m_g(std::move(genealogy)),
          m_nstages(branching.size()),
          m_n(m_g.ntrials),
          m_w(w_terminal),
          m_bp(branching.begin(), branching.end()),
          m_unit(m_nstages),
          m_offset(m_nstages + 1, 0),
          m_score(m_g.scores.data()),
          m_pre(m_nstages * m_n, 0),
          m_alive(m_nstages * m_n, 0),
          m_subtree(m_nstages * m_n, 0.0F),
          m_alive_list((m_nstages * m_n) + 1, 0),
          m_alive_count(m_nstages, 0),
          m_pre_spare(m_nstages * m_n, 0),
          m_alive_spare(m_nstages * m_n, 0),
          m_alive_list_spare((m_nstages * m_n) + 1, 0),
          m_alive_count_spare(m_nstages, 0) {
        // Stage s owns [s n, (s + 1) n); the parent of slot t of stage s is
        // parent[s - 1][t], a slot of stage s - 1 (m_parent is shifted so
        // that both use the same index).
        for (SizeType s = 0; s <= m_nstages; ++s) {
            m_offset[s] = s * m_n;
        }
        m_parent = m_g.parent.data() - m_n;
        // Population per trial alive after stage s:
        // prod_{j <= s} B_j * prod_{j < s} pass_rate_j / n.
        double scale = 1.0 / static_cast<double>(m_n);
        for (SizeType s = 0; s < m_nstages; ++s) {
            scale *= m_bp[s];
            m_unit[s] = scale;
            scale *= m_g.pass_rate[s];
        }
    }

    /// Total cost of @p g: 1 + sum_s B_s n_{s-1} + w n_{M-1}.
    double cost(std::span<const float> g) { return forward(g, false, 0.0F); }

    /// cost(g) with the forward caches of block_deltas() written to the
    /// spare buffers: accept(g) makes them current, reject() drops them.
    /// @p d is the largest downward move block_deltas() will be asked for.
    double evaluate(std::span<const float> g, float d) {
        swap_buffers();
        return forward(g, true, d);
    }
    void reject() { swap_buffers(); }

    /// Completes the caches of block_deltas() for the evaluated @p g: the
    /// trials alive before every stage are already there; this adds the cost
    /// E[s][t] that trial t of stage s adds by passing stage s (the scoring
    /// of its children and their subtrees).
    void accept(std::span<const float> g) {
        m_schedule.assign(g.begin(), g.end());
        const SizeType last = m_nstages - 1;
        std::fill(m_subtree.begin() +
                      static_cast<std::ptrdiff_t>(m_offset[last]),
                  m_subtree.end(), static_cast<float>(m_w * m_unit[last]));
        // Only the trials above their own threshold whose ancestry passes
        // the schedule lowered by d pass a subtree on: no move of
        // block_deltas() reads the subtree of another trial's parent.
        for (SizeType s = last; s-- > 0;) {
            float* e = m_subtree.data() + m_offset[s];
            std::fill_n(e, m_offset[s + 1] - m_offset[s],
                        static_cast<float>(m_bp[s + 1] * m_unit[s]));
            const uint32_t* list = m_alive_list.data() + m_offset[s + 1];
            const SizeType count = m_alive_count[s + 1];
            const SizeType base  = m_offset[s + 1];
            for (SizeType j = 0; j < count; ++j) {
                const SizeType i = base + list[j];
                e[m_parent[i]] += m_subtree[i];
            }
        }
    }

    /// Exact change of the cost when the thresholds of one block move by +d
    /// (@p up, <= 0) or -d (@p dn, >= 0), every other threshold as prepared.
    void block_deltas(std::span<const Block> blocks,
                      float d,
                      std::vector<double>& up,
                      std::vector<double>& dn,
                      int nthreads) {
        up.assign(blocks.size(), 0.0);
        dn.assign(blocks.size(), 0.0);
        // Bit 0: the prepared schedule, bit 1: block + d, bit 2: block - d.
        // The blocks are disjoint, and each writes m_alive at its own stages.
        const auto nblocks = static_cast<std::int64_t>(blocks.size());
        // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
        for (std::int64_t kb = 0; kb < nblocks; ++kb) {
            const auto k    = static_cast<SizeType>(kb);
            const Block blk = blocks[k];
            double c[3]     = {0.0, 0.0, 0.0};
            for (SizeType s = blk.a; s < blk.b; ++s) {
                const SizeType lo = m_offset[s];
                const SizeType hi = m_offset[s + 1];
                const float g0    = m_schedule[s];
                const float gu    = g0 + d;
                const float gd    = g0 - d;
                const bool first  = s == blk.a;
                const uint8_t* prv =
                    m_alive.data() + (first ? 0 : m_offset[s - 1]);
                if (s + 1 == blk.b) {
                    double e0 = 0.0;
                    double e1 = 0.0;
                    double e2 = 0.0;
                    for (SizeType i = lo; i < hi; ++i) {
                        const uint8_t p =
                            first ? (m_pre[i] != 0 ? uint8_t{7} : uint8_t{0})
                                  : prv[m_parent[i]];
                        const float v = m_score[i];
                        const float x = m_subtree[i];
                        e0 +=
                            static_cast<float>((p & 1U) & (v > g0 ? 1U : 0U)) *
                            x;
                        e1 += static_cast<float>(((p >> 1U) & 1U) &
                                                 (v > gu ? 1U : 0U)) *
                              x;
                        e2 += static_cast<float>(((p >> 2U) & 1U) &
                                                 (v > gd ? 1U : 0U)) *
                              x;
                    }
                    c[0] += e0;
                    c[1] += e1;
                    c[2] += e2;
                    break;
                }
                SizeType count[3] = {0, 0, 0};
                for (SizeType i = lo; i < hi; ++i) {
                    const uint8_t p =
                        first ? (m_pre[i] != 0 ? uint8_t{7} : uint8_t{0})
                              : prv[m_parent[i]];
                    const float v   = m_score[i];
                    const auto bits = static_cast<uint8_t>(
                        p & ((v > g0 ? 1U : 0U) | (v > gu ? 2U : 0U) |
                             (v > gd ? 4U : 0U)));
                    m_alive[i] = bits;
                    count[0] += bits & 1U;
                    count[1] += (bits >> 1U) & 1U;
                    count[2] += (bits >> 2U) & 1U;
                }
                const double scored = m_bp[s + 1] * m_unit[s];
                for (int j = 0; j < 3; ++j) {
                    c[j] += scored * static_cast<double>(count[j]);
                }
            }
            up[k] = c[1] - c[0];
            dn[k] = c[2] - c[0];
        }
    }

private:
    void swap_buffers() {
        m_pre.swap(m_pre_spare);
        m_alive.swap(m_alive_spare);
        m_alive_list.swap(m_alive_list_spare);
        m_alive_count.swap(m_alive_count_spare);
    }

    /// The cost of @p g; with @p keep, m_pre and the lists of the backward
    /// pass are filled (bit 1 of m_alive: the ancestry passes g - d).
    double forward(std::span<const float> g, bool keep, float d) {
        double c = 1.0 + m_bp[0];
        for (SizeType s = 0; s < m_nstages; ++s) {
            const SizeType lo     = m_offset[s];
            const SizeType hi     = m_offset[s + 1];
            const float threshold = g[s];
            const uint8_t* prv =
                s > 0 ? m_alive.data() + m_offset[s - 1] : nullptr;
            SizeType count = 0;
            if (keep) {
                const float low = threshold - d;
                uint32_t* list  = m_alive_list.data() + lo;
                SizeType k      = 0;
                for (SizeType i = lo; i < hi; ++i) {
                    const uint8_t p = s > 0 ? prv[m_parent[i]] : uint8_t{3};
                    const auto above =
                        static_cast<uint8_t>(m_score[i] > threshold);
                    const auto a = static_cast<uint8_t>(p & above);
                    const auto r = static_cast<uint8_t>(
                        (p & 2U) &
                        (static_cast<uint8_t>(m_score[i] > low) << 1U));
                    m_pre[i]   = p & 1U;
                    m_alive[i] = a | r;
                    count += a;
                    // Listed: above its threshold, ancestry above g - d
                    list[k] = static_cast<uint32_t>(i - lo);
                    k += (p >> 1U) & above;
                }
                m_alive_count[s] = k;
            } else if (s == 0) {
                for (SizeType i = lo; i < hi; ++i) {
                    const auto a = static_cast<uint8_t>(m_score[i] > threshold);
                    m_alive[i]   = a;
                    count += a;
                }
            } else {
                for (SizeType i = lo; i < hi; ++i) {
                    const auto a = static_cast<uint8_t>(
                        prv[m_parent[i]] &
                        static_cast<uint8_t>(m_score[i] > threshold));
                    m_alive[i] = a;
                    count += a;
                }
            }
            const double pop = m_unit[s] * static_cast<double>(count);
            c += s + 1 < m_nstages ? m_bp[s + 1] * pop : m_w * pop;
            if (count == 0 && !keep) {
                break;
            }
        }
        return c;
    }

    NoiseGenealogy m_g;
    SizeType m_nstages;
    SizeType m_n;
    double m_w;
    std::vector<double> m_bp;
    std::vector<double> m_unit;
    std::vector<SizeType> m_offset;
    const float* m_score;
    const uint32_t* m_parent;
    std::vector<float> m_schedule;
    std::vector<uint8_t> m_pre;
    std::vector<uint8_t> m_alive;
    std::vector<float> m_subtree;
    std::vector<uint32_t> m_alive_list;
    std::vector<SizeType> m_alive_count;
    std::vector<uint8_t> m_pre_spare;
    std::vector<uint8_t> m_alive_spare;
    std::vector<uint32_t> m_alive_list_spare;
    std::vector<SizeType> m_alive_count_spare;
};

/// Moves chosen by the multiplier @p lam (cost per unit Pd): a block rises
/// if the cost it saves exceeds the price of the Pd it loses, and falls if
/// the Pd it gains is worth more than the cost it adds. Returns the net
/// change of Pd.
double choose_moves(std::span<const double> dc_up,
                    std::span<const double> dp_up,
                    std::span<const double> dc_dn,
                    std::span<const double> dp_dn,
                    std::span<const uint8_t> room_dn,
                    double lam,
                    std::vector<int>& step) {
    double dp = 0.0;
    for (SizeType k = 0; k < step.size(); ++k) {
        const double gain_up = -dc_up[k] + (lam * dp_up[k]);
        const double gain_dn = room_dn[k] != 0
                                   ? (lam * dp_dn[k]) - dc_dn[k]
                                   : -std::numeric_limits<double>::infinity();
        step[k]              = 0;
        if (gain_up > 0.0 && gain_up >= gain_dn) {
            step[k] = 1;
            dp += dp_up[k];
        } else if (gain_dn > 0.0) {
            step[k] = -1;
            dp += dp_dn[k];
        }
    }
    return dp;
}

/// The same signal z-score at every stage, with Pd near @p target.
std::vector<float>
zscore_start(const SignalSample& signal, SizeType nstages, double target) {
    const auto [mean, sd]  = signal.moments();
    const SizeType ntrials = std::min<SizeType>(signal.size(), 4096);
    std::vector<float> g(nstages);
    double lo = -2.0; // Pd(lo) < target < Pd(hi)
    double hi = 8.0;
    for (int i = 0; i < 24; ++i) {
        const double z = 0.5 * (lo + hi);
        for (SizeType s = 0; s < nstages; ++s) {
            g[s] = mean[s] - (static_cast<float>(z) * sd[s]);
        }
        (signal.pd(g, ntrials) < target ? lo : hi) = z;
    }
    for (SizeType s = 0; s < nstages; ++s) {
        g[s] = mean[s] - (static_cast<float>(hi) * sd[s]);
    }
    return g;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

} // namespace

float fap_threshold(double fap, double ntemplates) {
    error_check::check(fap > 0.0 && fap < 1.0 && ntemplates >= 1.0,
                       "fap_threshold: fap must be in (0, 1)");
    // Q(x) = tail, by bisection on erfc (monotone; 60 steps are exact enough)
    const double tail = fap / ntemplates;
    double lo         = 0.0;
    double hi         = 40.0;
    for (int i = 0; i < 80; ++i) {
        const double x = 0.5 * (lo + hi);
        (0.5 * std::erfc(x / std::numbers::sqrt2) > tail ? lo : hi) = x;
    }
    return static_cast<float>(hi);
}

SchemeSolution descend_scheme(const ThresholdsEngine& engine,
                              float min_pd,
                              double w_terminal,
                              const DescentOptions& options) {
    error_check::check(min_pd > 0.0F && min_pd < 1.0F,
                       "descent: min_pd must be in (0, 1)");
    SchemeSolution out;
    out.min_pd             = min_pd;
    out.w_terminal         = static_cast<float>(w_terminal);
    const SizeType nstages = engine.get_nstages();
    const uint64_t seed    = engine.get_seed();
    const auto branching   = engine.get_branching_pattern();
    const auto budget      = engine.solve_budget();
    const double p         = static_cast<double>(min_pd);
    // As solve(): the pin aims 2 binomial standard errors above the target.
    const double target =
        std::min(1.0, p + (2.0 * std::sqrt(p * (1.0 - p) /
                                           static_cast<double>(budget.nbank))));
    const bool fixed_last = options.final_threshold > 0.0F;
    const int nthreads    = std::max(options.nthreads, 1);
    const SizeType nfree  = fixed_last ? nstages - 1 : nstages;
    const auto t_start    = std::chrono::steady_clock::now();
    double t_noise        = 0.0;
    double t_moves        = 0.0;

    // Coarse to fine: small samples and large steps first. The first pass
    // skips the finest blocks, the later ones the coarsest (measured:
    // docs/performance.md).
    constexpr SizeType kAll                       = SizeType{1} << 20U;
    const std::vector<DescentPass> default_passes = {
        {SizeType{1} << 12U, 0.4F, 0.2F, kAll, 4},
        {SizeType{1} << 12U, 0.4F, 0.2F, kAll, 1},
        {SizeType{1} << 13U, 0.3F, 0.1F, 32, 1},
        {SizeType{1} << 13U, 0.3F, 0.1F, 32, 1},
        {SizeType{1} << 13U, 0.2F, 0.05F, 8, 1},
        {SizeType{1} << 13U, 0.2F, 0.05F, 8, 1}};
    const auto& passes =
        options.passes.empty() ? default_passes : options.passes;
    const SignalSample signal(
        engine.simulate_h1_scores(options.nsignal,
                                  derive_seed(seed, kSignalDescent)),
        nstages, options.nsignal);
    if (fixed_last) {
        const double ceiling = signal.ceiling(options.final_threshold);
        out.diagnostics.emplace_back("pd_ceiling", ceiling);
        if (ceiling < target || nfree == 0) {
            // min_pd is above what the fixed last threshold lets through.
            out.pd       = ceiling;
            out.delivery = Delivery::kShort;
            return out;
        }
    }
    std::vector<float> g = options.start.empty()
                               ? zscore_start(signal, nstages, target)
                               : options.start;
    error_check::check_equal(g.size(), nstages,
                             "descent: one starting threshold per stage");
    if (fixed_last) {
        g[nstages - 1] = options.final_threshold;
    }
    const auto pin = [&](std::vector<float>& schedule) {
        const float shift = signal.pin_shift(schedule, target, nfree);
        for (SizeType s = 0; s < nfree; ++s) {
            schedule[s] += shift;
        }
    };
    pin(g);

    std::vector<SizeType> scales;
    for (SizeType size = std::bit_ceil(std::max<SizeType>(nfree, 2)) / 2;
         size >= 1; size /= 2) {
        scales.push_back(size);
    }
    SizeType nmoves = 0;
    std::vector<double> dc_up;
    std::vector<double> dc_dn;
    std::vector<double> dp_up;
    std::vector<double> dp_dn;
    std::vector<float> g_ref(nstages);
    std::vector<float> candidate(nstages);
    for (SizeType ipass = 0; ipass < passes.size(); ++ipass) {
        const auto& pass = passes[ipass];
        const float d    = pass.step;
        for (SizeType s = 0; s < nstages; ++s) {
            g_ref[s] = g[s] - pass.margin;
        }
        auto t0 = std::chrono::steady_clock::now();
        NoiseBank bank(
            engine.simulate_h0_genealogy(
                g_ref, pass.ntrials, derive_seed(seed, kNoiseDescent + ipass)),
            branching, w_terminal);
        t_noise += seconds_since(t0);
        t0          = std::chrono::steady_clock::now();
        double cost = bank.evaluate(g, d);
        bank.accept(g);
        for (const SizeType size : scales) {
            if (size > pass.max_block || size < pass.min_block) {
                continue;
            }
            const auto blocks = make_blocks(
                nfree, size, (ipass % 2 == 1 && size > 1) ? size / 2 : 0);
            bank.block_deltas(blocks, d, dc_up, dc_dn, nthreads);
            signal.block_deltas(g, blocks, d, dp_up, dp_dn, nthreads);
            std::vector<uint8_t> room(blocks.size());
            double saved = 0.0;
            double lost  = 0.0;
            for (SizeType k = 0; k < blocks.size(); ++k) {
                bool ok = true;
                for (SizeType s = blocks[k].a; s < blocks[k].b && ok; ++s) {
                    ok = g[s] - d > g_ref[s] + kMarginGuard;
                }
                room[k] = ok ? 1 : 0;
                saved -= dc_up[k];
                lost -= dp_up[k];
            }
            // Bisection of the multiplier: the net change of Pd of the chosen
            // moves rises with it; take the smallest one that keeps Pd.
            std::vector<int> step(blocks.size(), 0);
            double lo  = 0.0;
            double hi  = -1.0;
            double lam = std::max(saved / std::max(lost, 1e-12), 1e-9);
            for (int i = 0; i < 60; ++i) {
                const double dp =
                    choose_moves(dc_up, dp_up, dc_dn, dp_dn, room, lam, step);
                if (dp < 0.0) {
                    lo  = lam;
                    lam = hi < 0.0 ? lam * 2.0 : 0.5 * (lo + hi);
                } else {
                    hi  = lam;
                    lam = 0.5 * (lo + hi);
                }
            }
            choose_moves(dc_up, dp_up, dc_dn, dp_dn, room, hi < 0.0 ? lam : hi,
                         step);
            if (std::ranges::all_of(step, [](int s) { return s == 0; })) {
                continue;
            }
            candidate = g;
            for (SizeType k = 0; k < blocks.size(); ++k) {
                for (SizeType s = blocks[k].a; s < blocks[k].b; ++s) {
                    candidate[s] += d * static_cast<float>(step[k]);
                }
            }
            pin(candidate);
            bool above = true;
            for (SizeType s = 0; s < nstages && above; ++s) {
                above = candidate[s] > g_ref[s];
            }
            if (!above) {
                continue;
            }
            const double cost_candidate = bank.evaluate(candidate, d);
            if (cost_candidate < kAccept * cost) {
                bank.accept(candidate);
                g    = candidate;
                cost = cost_candidate;
                ++nmoves;
            } else {
                bank.reject();
            }
        }
        t_moves += seconds_since(t0);
    }
    out.search_thresholds = g;
    out.search_pd         = signal.pd(g, signal.size());

    // Pin on a fresh bank (either direction), report on another, as solve().
    const SizeType n = budget.nbank;
    float pin_shift  = 0.0F;
    {
        const auto bank =
            engine.make_signal_bank(n, derive_seed(seed, kBankPin));
        std::vector<float> probe = g;
        if (fixed_last) {
            probe[nstages - 1] = std::numeric_limits<float>::lowest();
        }
        auto margins = bank->margins(probe);
        if (fixed_last) {
            std::vector<float> last(nstages,
                                    std::numeric_limits<float>::lowest());
            last[nstages - 1] = g[nstages - 1];
            const auto above  = bank->margins(last);
            for (SizeType t = 0; t < n; ++t) {
                if (above[t] <= 0.0F) {
                    margins[t] = std::numeric_limits<float>::lowest();
                }
            }
        }
        pin_shift = SignalSample::shift_of(margins, target);
        if (std::isfinite(pin_shift) &&
            pin_shift > std::numeric_limits<float>::lowest() / 2) {
            for (SizeType s = 0; s < nfree; ++s) {
                g[s] += pin_shift;
            }
        }
    }
    {
        const auto report =
            engine.make_signal_bank(n, derive_seed(seed, kBankReport));
        const SizeType count = count_survivors(*report, g);
        out.pd    = static_cast<double>(count) / static_cast<double>(n);
        out.pd_se = std::sqrt(out.pd * (1.0 - out.pd) / static_cast<double>(n));
        out.pd_trials = n;
    }
    out.eval_states =
        engine.evaluate(g, budget.nnoise, derive_seed(seed, kNoiseReport),
                        /*h0_only=*/true);
    out.c = total_cost(out.eval_states.back(), w_terminal);
    out.cost =
        out.pd > 0.0 ? out.c / out.pd : std::numeric_limits<double>::max();
    out.search_c   = out.c;
    out.thresholds = std::move(g);
    out.diagnostics.emplace_back("descent_passes",
                                 static_cast<double>(passes.size()));
    out.diagnostics.emplace_back("descent_moves", static_cast<double>(nmoves));
    out.diagnostics.emplace_back("pin_shift", static_cast<double>(pin_shift));
    out.diagnostics.emplace_back("seconds_noise", t_noise);
    out.diagnostics.emplace_back("seconds_moves", t_moves);
    out.diagnostics.emplace_back("seconds_total", seconds_since(t_start));
    if (out.pd >= p) {
        out.delivery = Delivery::kMet;
    } else if (out.pd >= 0.9 * p) {
        out.delivery = Delivery::kWithinTolerance;
    } else {
        out.delivery = Delivery::kShort;
    }
    return out;
}

} // namespace loki::detection::detail
