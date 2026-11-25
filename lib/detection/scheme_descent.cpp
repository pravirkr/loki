#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"
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
constexpr uint64_t kNoiseStart    = 0xDE5C0005ULL;
// Noise trials of the evaluation that chooses between the starting schedules.
constexpr SizeType kStartTrials = SizeType{1} << 12U;
// A level of the default plan runs a second time when its first pass lowered
// the cost on its genealogy by more than this fraction.
constexpr double kRepeatDrop = 0.25;
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
          m_scores(nstages * n),
          m_mean(nstages),
          m_sd(nstages),
          m_margin(n) {
        m_active.reserve(n);
        // Moments from contiguous stage rows (same summation order over t as a
        // trial-major scan, so bit-identical).
        const auto n_d = static_cast<double>(n);
        for (SizeType s = 0; s < nstages; ++s) {
            const float* src = stage_major.data() + (s * n);
            double sum       = 0.0;
            double sq        = 0.0;
            for (SizeType t = 0; t < n; ++t) {
                const double v = src[t];
                sum += v;
                sq += v * v;
            }
            const double mu = sum / n_d;
            m_mean[s]       = static_cast<float>(mu);
            m_sd[s]         = static_cast<float>(
                std::sqrt(std::max((sq / n_d) - (mu * mu), 1e-12)));
        }
        // Cache-blocked stage-major -> trial-major transpose through an L1
        // tile: avoids power-of-2 cache-set aliasing at n = 2^15.
        constexpr SizeType kTileS = 32;
        constexpr SizeType kTileT = 64;
        std::array<std::array<float, kTileS + 1>, kTileT> tile{};
        for (SizeType s0 = 0; s0 < nstages; s0 += kTileS) {
            const SizeType ds = std::min(kTileS, nstages - s0);
            for (SizeType t0 = 0; t0 < n; t0 += kTileT) {
                const SizeType dt = std::min(kTileT, n - t0);
                for (SizeType ks = 0; ks < ds; ++ks) {
                    const float* src =
                        stage_major.data() + ((s0 + ks) * n) + t0;
                    for (SizeType kt = 0; kt < dt; ++kt) {
                        tile[kt][ks] = src[kt];
                    }
                }
                for (SizeType kt = 0; kt < dt; ++kt) {
                    float* dst = m_scores.data() + ((t0 + kt) * nstages) + s0;
                    std::copy_n(tile[kt].data(), ds, dst);
                }
            }
        }
    }

    SizeType size() const { return m_n; }

    /// Indexes the trials above @p g_ref at every stage, once per pass. Every
    /// candidate tested during the pass stays above @p g_ref, so trials at or
    /// below @p g_ref can never be revived by a single block move.
    void prepare_pass(std::span<const float> g_ref) {
        m_g_ref.assign(g_ref.begin(), g_ref.end());
        m_active.clear();
        const SizeType nstages = m_nstages;
        const float* scores    = m_scores.data();
        for (SizeType t = 0; t < m_n; ++t) {
            const float* row = scores + (t * nstages);
            bool ok          = true;
            for (SizeType s = 0; s < nstages && ok; ++s) {
                ok = row[s] > g_ref[s];
            }
            if (ok) {
                m_active.push_back(t);
            }
        }
    }

    /// Mean and standard deviation of every stage's scores.
    std::pair<std::vector<float>, std::vector<float>> moments() const {
        return {m_mean, m_sd};
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

    /// Fraction of all trials above @p g, scanning only m_active (exact when
    /// g[s] >= g_ref[s] for every stage s).
    double pd_active(std::span<const float> g) const {
        if (m_active.empty()) {
            return pd(g, m_n);
        }
        SizeType count = 0;
        for (const SizeType t : m_active) {
            const float* row = m_scores.data() + (t * m_nstages);
            bool alive       = true;
            for (SizeType s = 0; s < m_nstages && alive; ++s) {
                alive = row[s] > g[s];
            }
            count += alive ? 1 : 0;
        }
        return static_cast<double>(count) / static_cast<double>(m_n);
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
    /// Uses m_active when the resulting shifted schedule stays strictly above
    /// m_g_ref (exact: no inactive trial can survive such a schedule), falling
    /// back to a full vectorized scan otherwise.
    float
    pin_shift(std::span<const float> g, double target, SizeType nfree) const {
        const SizeType nstages  = m_nstages;
        const float* scores     = m_scores.data();
        const SizeType required = std::clamp<SizeType>(
            static_cast<SizeType>(std::ceil(target * static_cast<double>(m_n))),
            1, m_n);
        if (!m_g_ref.empty() && m_active.size() >= required) {
            bool fixed_above = true;
            for (SizeType s = nfree; s < nstages && fixed_above; ++s) {
                fixed_above = g[s] > m_g_ref[s];
            }
            if (fixed_above) {
                const SizeType nact = m_active.size();
                m_margin_act.resize(nact);
                for (SizeType k = 0; k < nact; ++k) {
                    const float* row = scores + (m_active[k] * nstages);
                    float m          = std::numeric_limits<float>::max();
                    for (SizeType s = 0; s < nfree; ++s) {
                        m = std::min(m, row[s] - g[s]);
                    }
                    for (SizeType s = nfree; s < nstages; ++s) {
                        if (row[s] <= g[s]) {
                            m = std::numeric_limits<float>::lowest();
                        }
                    }
                    m_margin_act[k] = m;
                }
                const auto rank = m_margin_act.begin() +
                                  static_cast<std::ptrdiff_t>(nact - required);
                std::ranges::nth_element(m_margin_act, rank);
                const float r = *rank;
                if (r > std::numeric_limits<float>::lowest() / 2.0F) {
                    const float shift =
                        std::nextafter(r, std::numeric_limits<float>::lowest());
                    bool free_above = true;
                    for (SizeType s = 0; s < nfree && free_above; ++s) {
                        free_above = (g[s] + shift) > m_g_ref[s];
                    }
                    if (free_above) {
                        return shift;
                    }
                }
            }
        }
        // Full vectorized scan (no inner branch so GCC vectorizes min-diff).
        for (SizeType t = 0; t < m_n; ++t) {
            const float* row = scores + (t * nstages);
            float m          = std::numeric_limits<float>::max();
            for (SizeType s = 0; s < nfree; ++s) {
                m = std::min(m, row[s] - g[s]);
            }
            for (SizeType s = nfree; s < nstages; ++s) {
                if (row[s] <= g[s]) {
                    m = std::numeric_limits<float>::lowest();
                }
            }
            m_margin[t] = m;
        }
        const float shift = shift_of(m_margin, target);
        // Fewer than ceil(target n) trials pass the fixed stages: no common
        // shift reaches the target, so leave the schedule as it is, as the
        // final pin does. Not reached from descend_scheme(): it fixes only a
        // last stage at the floor, and returns kShort first when fewer than
        // target n trials of this sample pass the floor (ceiling()).
        return shift > std::numeric_limits<float>::lowest() / 2.0F ? shift
                                                                   : 0.0F;
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
    /// (@p up, <= 0) or down by @p down per stage (@p dn, >= 0), every
    /// other threshold fixed. Scans m_active (from prepare_pass) when set.
    void block_deltas(std::span<const float> g,
                      std::span<const Block> blocks,
                      float d,
                      std::span<const float> down,
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
        const SizeType nblocks   = blocks.size();
        const SizeType nstages   = m_nstages;
        const float* scores      = m_scores.data();
        const bool use_active    = !m_active.empty();
        const SizeType* active_p = m_active.data();
        const auto nwork =
            static_cast<std::int64_t>(use_active ? m_active.size() : m_n);
        // NOLINTNEXTLINE(openmp-use-default-none): spans are not listable
#pragma omp parallel if (nthreads > 1 && nwork >= 4096) num_threads(nthreads)
        {
            std::vector<SizeType> t_up(nblocks, 0);
            std::vector<SizeType> t_dn(nblocks, 0);
            std::vector<float> m(nstages);
#pragma omp for schedule(static) nowait
            for (std::int64_t idx = 0; idx < nwork; ++idx) {
                const SizeType trial =
                    use_active ? active_p[idx] : static_cast<SizeType>(idx);
                const float* row = scores + (trial * nstages);
                SizeType first   = nstages;
                bool lost        = false; // cannot be revived by one block
                for (SizeType s = 0; s < nstages; ++s) {
                    const float v = row[s] - g[s];
                    m[s]          = v;
                    if (v <= 0.0F) {
                        if (v <= -down[s] || block_of[s] == nblocks ||
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
    std::vector<float> m_mean;
    std::vector<float> m_sd;
    mutable std::vector<float> m_margin;
    mutable std::vector<float> m_margin_act;
    std::vector<float> m_g_ref;
    std::vector<SizeType> m_active;
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
        // parent[s - 1][t], a slot of stage s - 1 (see parents_of()).
        for (SizeType s = 0; s <= m_nstages; ++s) {
            m_offset[s] = s * m_n;
        }
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
            const uint32_t* list   = m_alive_list.data() + m_offset[s + 1];
            const SizeType count   = m_alive_count[s + 1];
            const SizeType base    = m_offset[s + 1];
            const uint32_t* parent = parents_of(s + 1);
            for (SizeType j = 0; j < count; ++j) {
                e[parent[list[j]]] += m_subtree[base + list[j]];
            }
        }
    }

    /// Exact change of the cost when the thresholds of one block move by +d
    /// (@p up, <= 0) or down by @p down per stage (@p dn, >= 0, each at most
    /// the d of evaluate()), every other threshold as prepared.
    void block_deltas(std::span<const Block> blocks,
                      float d,
                      std::span<const float> down,
                      std::vector<double>& up,
                      std::vector<double>& dn,
                      int nthreads) {
        up.assign(blocks.size(), 0.0);
        dn.assign(blocks.size(), 0.0);
        // Bit 0: the prepared schedule, bit 1: block + d, bit 2: block - d.
        // The blocks are disjoint, and each writes m_alive at its own stages.
        const auto nblocks = static_cast<std::int64_t>(blocks.size());
        // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for if (nthreads > 1 && nblocks > 1)                      \
    num_threads(nthreads) schedule(dynamic)
        for (std::int64_t kb = 0; kb < nblocks; ++kb) {
            const auto k    = static_cast<SizeType>(kb);
            const Block blk = blocks[k];
            double c[3]     = {0.0, 0.0, 0.0};
            for (SizeType s = blk.a; s < blk.b; ++s) {
                const SizeType lo = m_offset[s];
                const SizeType hi = m_offset[s + 1];
                const float g0    = m_schedule[s];
                const float gu    = g0 + d;
                const float gd    = g0 - down[s];
                const bool first  = s == blk.a;
                const uint8_t* prv =
                    m_alive.data() + (first ? 0 : m_offset[s - 1]);
                const uint32_t* parent = first ? nullptr : parents_of(s);
                if (s + 1 == blk.b) {
                    double e0 = 0.0;
                    double e1 = 0.0;
                    double e2 = 0.0;
                    for (SizeType i = lo; i < hi; ++i) {
                        const uint8_t p =
                            first ? (m_pre[i] != 0 ? uint8_t{7} : uint8_t{0})
                                  : prv[parent[i - lo]];
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
                              : prv[parent[i - lo]];
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
    /// Parents of the slots of stage @p s >= 1: slots of stage s - 1.
    const uint32_t* parents_of(SizeType s) const {
        return m_g.parent.data() + m_offset[s - 1];
    }

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
            const uint32_t* parent = s > 0 ? parents_of(s) : nullptr;
            SizeType count         = 0;
            if (keep) {
                const float low = threshold - d;
                uint32_t* list  = m_alive_list.data() + lo;
                SizeType k      = 0;
                if (s == 0) {
                    for (SizeType i = lo; i < hi; ++i) {
                        const auto above =
                            static_cast<uint8_t>(m_score[i] > threshold);
                        const auto r = static_cast<uint8_t>(m_score[i] > low)
                                       << 1U;
                        m_pre[i]     = 1U;
                        m_alive[i]   = above | r;
                        count += above;
                        list[k] = static_cast<uint32_t>(i - lo);
                        k += above;
                    }
                } else {
                    for (SizeType i = lo; i < hi; ++i) {
                        const uint8_t p = prv[parent[i - lo]];
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
                        prv[parent[i - lo]] &
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
                                   : std::numeric_limits<double>::lowest();
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
std::vector<float> zscore_start(const SignalSample& signal,
                                std::span<const float> mean,
                                std::span<const float> sd,
                                double target) {
    const SizeType nstages = mean.size();
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

/// The shape the solutions of long branching patterns share, in z-scores
/// against the fraction of the stages done: loose at the very first stages,
/// at the mean signal score soon after, then rising steadily to the end
/// (measured: docs/performance.md). Only a candidate start: it is used when it
/// costs less than the flat one, and the descent moves it freely.
std::vector<float> smooth_start(std::span<const float> mean,
                                std::span<const float> sd) {
    constexpr std::array<float, 4> kX = {0.0F, 0.04F, 0.16F, 1.0F};
    constexpr std::array<float, 4> kZ = {2.0F, 0.0F, -0.1F, 1.7F};
    const SizeType nstages            = mean.size();
    std::vector<float> g(nstages);
    for (SizeType s = 0; s < nstages; ++s) {
        const float x = nstages > 1 ? static_cast<float>(s) /
                                          static_cast<float>(nstages - 1)
                                    : 0.0F;
        SizeType k    = 1;
        while (k + 1 < kX.size() && x > kX[k]) {
            ++k;
        }
        const float t = (x - kX[k - 1]) / (kX[k] - kX[k - 1]);
        const float z = kZ[k - 1] + (t * (kZ[k] - kZ[k - 1]));
        g[s]          = mean[s] - (z * sd[s]);
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

void validate_descent_options(const DescentOptions& options, SizeType nstages) {
    constexpr SizeType kMaxTrials = SizeType{1} << 20U;
    const auto fail               = [](const std::string& what) {
        throw std::invalid_argument("descent: " + what);
    };
    for (SizeType i = 0; i < options.passes.size(); ++i) {
        const auto& pass       = options.passes[i];
        const std::string name = "descent_passes[" + std::to_string(i) + "]";
        if (pass.ntrials == 0 || pass.ntrials > kMaxTrials) {
            fail(name + ": ntrials must be in [1, 2^20]");
        }
        if (!utils::is_finite(pass.margin) || pass.margin <= 0.0F) {
            fail(name + ": margin must be finite and > 0");
        }
        if (!utils::is_finite(pass.step) || pass.step <= 0.0F) {
            fail(name + ": step must be finite and > 0");
        }
        if (pass.step >= pass.margin) {
            fail(name + ": step must be below margin");
        }
        if (pass.margin - pass.step <= kMarginGuard) {
            // A down move needs room above the reference plus the guard.
            fail(name + ": margin - step must exceed 0.02 (no room for down "
                        "moves otherwise)");
        }
        if (pass.min_block == 0 || pass.min_block > pass.max_block) {
            fail(name + ": need 1 <= min_block <= max_block");
        }
    }
    if (options.nsignal == 0 || options.nsignal > kMaxTrials) {
        fail("the signal sample (descent_nsignal) must be in [1, 2^20]");
    }
    if (!options.start.empty()) {
        if (options.start.size() != nstages) {
            fail("one starting threshold per stage: got " +
                 std::to_string(options.start.size()) + " for " +
                 std::to_string(nstages) + " stages");
        }
        if (!std::ranges::all_of(options.start,
                                 [](float v) { return utils::is_finite(v); })) {
            fail("the start must be finite");
        }
    }
    if (!utils::is_finite(options.final_threshold) ||
        options.final_threshold < 0.0F) {
        fail("final_threshold must be finite and >= 0 (0: no floor)");
    }
}

std::vector<float> resample_zscores(std::span<const float> zscores,
                                    SizeType nstages) {
    if (zscores.size() == nstages) {
        return {zscores.begin(), zscores.end()};
    }
    if (zscores.size() < 2 || nstages < 2) {
        return {};
    }
    std::vector<float> out(nstages);
    const double scale = static_cast<double>(zscores.size() - 1) /
                         static_cast<double>(nstages - 1);
    for (SizeType s = 0; s < nstages; ++s) {
        const double x = static_cast<double>(s) * scale;
        const auto k   = std::min(static_cast<SizeType>(x), zscores.size() - 2);
        const auto t   = static_cast<float>(x - static_cast<double>(k));
        out[s]         = zscores[k] + (t * (zscores[k + 1] - zscores[k]));
    }
    return out;
}

SchemeSolution descend_scheme(const ThresholdsEngine& engine,
                              float min_pd,
                              double w_terminal,
                              const DescentOptions& options) {
    error_check::check(min_pd > 0.0F && min_pd < 1.0F,
                       "descent: min_pd must be in (0, 1)");
    validate_descent_options(options, engine.get_nstages());
    if (!utils::is_finite(w_terminal) || w_terminal < 0.0) {
        throw std::invalid_argument(
            "descent: w_terminal must be finite and >= 0");
    }
    SchemeSolution out;
    out.min_pd             = min_pd;
    out.w_terminal         = static_cast<float>(w_terminal);
    const SizeType nstages = engine.get_nstages();
    const uint64_t seed    = engine.get_seed();
    const auto branching   = engine.get_branching_pattern();
    auto budget            = engine.solve_budget();
    if (options.nbank > 0) {
        budget.nbank = options.nbank;
    }
    if (options.nnoise > 0) {
        budget.nnoise = options.nnoise;
    }
    const double p = static_cast<double>(min_pd);
    // As solve(): the pin aims 2 binomial standard errors above the target.
    const double target =
        std::min(1.0, p + (2.0 * std::sqrt(p * (1.0 - p) /
                                           static_cast<double>(budget.nbank))));
    // The last threshold never goes below the floor; the other stages make
    // up the Pd when it rests there.
    const bool has_floor = options.final_threshold > 0.0F;
    const float floor    = has_floor ? options.final_threshold
                                     : std::numeric_limits<float>::lowest();
    const int nthreads   = std::max(options.nthreads, 1);
    const bool free_pd   = options.pd_policy == PdPolicy::kFree;
    const auto t_start   = std::chrono::steady_clock::now();
    double t_noise       = 0.0;
    double t_moves       = 0.0;

    // Coarse to fine: small samples and large steps first, the later levels
    // without the coarsest blocks (measured: docs/performance.md). Without
    // given passes a level is repeated while it still gains.
    constexpr SizeType kAll         = SizeType{1} << 20U;
    std::vector<DescentPass> levels = {
        {SizeType{1} << 12U, 0.4F, 0.2F, kAll, 1},
        {SizeType{1} << 13U, 0.3F, 0.1F, 32, 1},
        {SizeType{1} << 13U, 0.2F, 0.05F, 8, 1}};
    const bool adaptive = options.passes.empty();
    SignalSample signal(engine.simulate_h1_scores(
                            options.nsignal, derive_seed(seed, kSignalDescent)),
                        nstages, options.nsignal);
    double ceiling = 1.0;
    if (has_floor) {
        ceiling = signal.ceiling(floor);
        out.diagnostics.emplace_back("pd_ceiling", ceiling);
        if (ceiling < target) {
            // min_pd is above what the fixed last threshold lets through.
            out.pd       = ceiling;
            out.delivery = Delivery::kShort;
            return out;
        }
    }
    // The schedule in units of the unpruned signal score carries over to
    // other patterns and bin counts: z_s = (mean_s - g_s) / sd_s.
    const auto moments = signal.moments();
    const auto& mean   = moments.first;
    const auto& sd     = moments.second;
    // Common shift of stages to @p target_pd; when the last threshold sits on
    // the active floor (or a downward shift would take it below), keep the
    // last stage at the floor and shift [0, nstages - 1) (KKT complementary
    // slackness).
    const auto pin_to = [&](std::vector<float>& schedule, double target_pd) {
        schedule[nstages - 1] = std::max(schedule[nstages - 1], floor);
        if (has_floor && schedule[nstages - 1] <= floor) {
            schedule[nstages - 1] = floor;
            const float rest =
                signal.pin_shift(schedule, target_pd, nstages - 1);
            for (SizeType s = 0; s + 1 < nstages; ++s) {
                schedule[s] += rest;
            }
            return;
        }
        const float shift = signal.pin_shift(schedule, target_pd, nstages);
        if (schedule[nstages - 1] + shift >= floor) {
            for (SizeType s = 0; s < nstages; ++s) {
                schedule[s] += shift;
            }
            return;
        }
        schedule[nstages - 1] = floor;
        const float rest = signal.pin_shift(schedule, target_pd, nstages - 1);
        for (SizeType s = 0; s + 1 < nstages; ++s) {
            schedule[s] += rest;
        }
    };
    const auto pin = [&](std::vector<float>& schedule) {
        pin_to(schedule, target);
    };
    const double se_search =
        std::sqrt(target * (1.0 - target) / static_cast<double>(signal.size()));
    const double interior_margin =
        std::max(0.15 * std::min(target, 1.0 - target), 6.0 * se_search);
    bool flat_or_interior_start = false;
    std::vector<float> g;
    if (options.start.empty()) {
        g = zscore_start(signal, mean, sd, target);
        pin(g);
        std::vector<float> smooth = smooth_start(mean, sd);
        pin(smooth);
        const uint64_t noise_seed = derive_seed(seed, kNoiseStart);
        const auto cost_of        = [&](const std::vector<float>& schedule) {
            return total_cost(
                engine.evaluate(schedule, kStartTrials, noise_seed, true)
                    .back(),
                w_terminal);
        };
        const double cost_flat   = cost_of(g);
        const double cost_smooth = cost_of(smooth);
        out.diagnostics.emplace_back("start_cost_ratio",
                                     cost_smooth / cost_flat);
        bool chose_smooth = false;
        double best_rank  = cost_flat;
        if (!free_pd) {
            if (cost_smooth < cost_flat) {
                g            = std::move(smooth);
                chose_smooth = true;
            }
        } else {
            const double pd_flat     = signal.pd(g, signal.size());
            const double pd_smooth   = signal.pd(smooth, signal.size());
            const double rank_flat   = cost_flat / std::max(pd_flat, 1e-9);
            const double rank_smooth = cost_smooth / std::max(pd_smooth, 1e-9);
            best_rank                = rank_flat;
            if (rank_smooth < best_rank) {
                g            = std::move(smooth);
                best_rank    = rank_smooth;
                chose_smooth = true;
            }
            // Flat starts at Pd above the target: a ladder of Pd values
            // (stopped once L = C / Pd is 2 % above the best), then three
            // bisection probes around the best point. Deep high-cost trees
            // fail on the first upward step and stop after one evaluation.
            const double p_max = std::min(0.995, has_floor ? ceiling : 1.0);
            const double p_first =
                std::min(p_max, std::max(target * 1.4, target + 0.03));
            const auto probe_flat = [&](double p_try) {
                std::vector<float> cand = zscore_start(signal, mean, sd, p_try);
                pin_to(cand, p_try);
                const double pd_c = signal.pd(cand, signal.size());
                const double rk   = cost_of(cand) / std::max(pd_c, 1e-9);
                return std::tuple{std::move(cand), pd_c, rk};
            };
            if (p_first > target + 1e-4) {
                auto [cand1, pd1, rk1] = probe_flat(p_first);
                if (rk1 < rank_flat) {
                    std::vector<std::pair<double, double>> pts = {
                        {target, rank_flat}, {p_first, rk1}};
                    double best_p = p_first;
                    if (rk1 < best_rank) {
                        g            = std::move(cand1);
                        best_rank    = rk1;
                        best_p       = p_first;
                        chose_smooth = false;
                    }
                    constexpr std::array<double, 9> kLadder = {
                        0.14, 0.22, 0.35, 0.50, 0.65, 0.80, 0.90, 0.96, 0.995};
                    for (double p_raw : kLadder) {
                        const double p_try = std::min(p_raw, p_max);
                        if (p_try <= pts.back().first + 0.01) {
                            continue;
                        }
                        auto [cand, pd_c, rk] = probe_flat(p_try);
                        pts.emplace_back(p_try, rk);
                        if (rk < best_rank) {
                            g            = std::move(cand);
                            best_rank    = rk;
                            best_p       = p_try;
                            chose_smooth = false;
                        } else if (rk > 1.02 * best_rank) {
                            break;
                        }
                        if (p_try >= p_max - 1e-4) {
                            break;
                        }
                    }
                    // Refine around the best bracket point with 3 bisection
                    // probes.
                    SizeType ibest = 0;
                    for (SizeType i = 1; i < pts.size(); ++i) {
                        if (pts[i].second < pts[ibest].second) {
                            ibest = i;
                        }
                    }
                    double p_lo = ibest > 0 ? pts[ibest - 1].first : target;
                    double p_hi =
                        ibest + 1 < pts.size() ? pts[ibest + 1].first : p_max;
                    double p_mid = pts[ibest].first;
                    for (int iter = 0; iter < 3; ++iter) {
                        const double p_try = (p_hi - p_mid > p_mid - p_lo)
                                                 ? 0.5 * (p_mid + p_hi)
                                                 : 0.5 * (p_lo + p_mid);
                        if (std::abs(p_try - p_mid) < 0.005) {
                            break;
                        }
                        auto [cand, pd_c, rk] = probe_flat(p_try);
                        if (rk < best_rank) {
                            if (p_try < p_mid) {
                                p_hi = p_mid;
                            } else {
                                p_lo = p_mid;
                            }
                            p_mid        = p_try;
                            g            = std::move(cand);
                            best_rank    = rk;
                            best_p       = p_try;
                            chose_smooth = false;
                        } else if (p_try < p_mid) {
                            p_lo = p_try;
                        } else {
                            p_hi = p_try;
                        }
                    }
                    // Also check the smooth profile shifted to best_p.
                    if (best_p > target + 1e-4) {
                        std::vector<float> smooth_hi = smooth_start(mean, sd);
                        pin_to(smooth_hi, best_p);
                        const double pd_sh =
                            signal.pd(smooth_hi, signal.size());
                        const double rk_sh =
                            cost_of(smooth_hi) / std::max(pd_sh, 1e-9);
                        if (rk_sh < best_rank) {
                            g            = std::move(smooth_hi);
                            best_rank    = rk_sh;
                            chose_smooth = true;
                        }
                    }
                }
            }
            const double pd_start = signal.pd(g, signal.size());
            flat_or_interior_start =
                !chose_smooth || (pd_start > target + interior_margin);
        }
    } else {
        g = options.start;
        if (options.start_zscores) {
            for (SizeType s = 0; s < nstages; ++s) {
                g[s] = mean[s] - (options.start[s] * sd[s]);
            }
        }
        if (free_pd) {
            // Free Pd: a warm start above the target is a feasible interior
            // point (often the previous optimum); pin it only below the
            // target, and keep the last threshold on the floor.
            g[nstages - 1] = std::max(g[nstages - 1], floor);
            if (signal.pd(g, signal.size()) < target) {
                pin(g);
            }
        } else {
            pin(g);
        }
    }

    // In free mode on short/flat starts, early stages often need to drop > 1
    // S/N below the flat start; prepend one wider coarse level so they are not
    // blocked by the 0.4 S/N margin wall.
    if (adaptive && free_pd && flat_or_interior_start) {
        levels.insert(levels.begin(),
                      {SizeType{1} << 12U, 0.8F, 0.4F, kAll, 1});
    }
    const auto& passes = adaptive ? levels : options.passes;
    const double repeat_drop =
        (free_pd && flat_or_interior_start) ? 0.05 : kRepeatDrop;

    std::vector<SizeType> scales;
    if (free_pd && nstages > 1) {
        scales.push_back(nstages);
    }
    for (SizeType size = std::bit_ceil(std::max<SizeType>(nstages, 2)) / 2;
         size >= 1; size /= 2) {
        scales.push_back(size);
    }
    SizeType nmoves = 0;
    bool repeat     = false;
    std::vector<double> dc_up;
    std::vector<double> dc_dn;
    std::vector<double> dp_up;
    std::vector<double> dp_dn;
    std::vector<float> g_ref(nstages);
    std::vector<float> candidate(nstages);
    SizeType ipass = 0;
    // The cost of the schedule on the last noise genealogy (in-run).
    double search_c = -1.0;
    for (SizeType ilevel = 0; ilevel < 2 * passes.size(); ++ilevel) {
        // Every level may run twice in the default plan (odd ilevel: the
        // repeat, entered only if the first pass gained enough).
        const auto& pass = passes[ilevel / 2];
        if (ilevel % 2 == 1 && !repeat) {
            continue;
        }
        const float d = pass.step;
        std::vector<float> down(nstages, d);
        for (SizeType s = 0; s < nstages; ++s) {
            g_ref[s] = g[s] - pass.margin;
        }
        auto t0 = std::chrono::steady_clock::now();
        NoiseBank bank(
            engine.simulate_h0_genealogy(
                g_ref, pass.ntrials, derive_seed(seed, kNoiseDescent + ipass)),
            branching, w_terminal);
        t_noise += seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        signal.prepare_pass(g_ref);
        double cost   = bank.evaluate(g, d);
        double pd_cur = free_pd ? signal.pd_active(g) : target;
        bank.accept(g);
        const double rank_begin =
            free_pd ? (cost / std::max(pd_cur, 1e-9)) : cost;
        for (const SizeType size : scales) {
            const bool is_global_shift = free_pd && size == nstages;
            if (!is_global_shift &&
                (size > pass.max_block || size < pass.min_block)) {
                continue;
            }
            // Odd passes offset the blocks by half a block; the global shift
            // under kFree stays one block over every stage.
            const auto blocks = make_blocks(
                nstages, size,
                (ipass % 2 == 1 && size > 1 && !is_global_shift) ? size / 2
                                                                 : 0);
            // The last stage moves down as far as the floor lets it.
            if (has_floor) {
                down[nstages - 1] = std::clamp(g[nstages - 1] - floor, 0.0F, d);
            }
            bank.block_deltas(blocks, d, down, dc_up, dc_dn, nthreads);
            signal.block_deltas(g, blocks, d, down, dp_up, dp_dn, nthreads);
            std::vector<uint8_t> room(blocks.size());
            double saved = 0.0;
            double lost  = 0.0;
            for (SizeType k = 0; k < blocks.size(); ++k) {
                bool ok = true;
                for (SizeType s = blocks[k].a; s < blocks[k].b && ok; ++s) {
                    ok = g[s] - down[s] > g_ref[s] + kMarginGuard;
                }
                room[k] = ok ? 1 : 0;
                saved -= dc_up[k];
                lost -= dp_up[k];
            }
            std::vector<int> step(blocks.size(), 0);
            bool binds = !free_pd;
            if (!free_pd) {
                double lo  = 0.0;
                double hi  = -1.0;
                double lam = std::max(saved / std::max(lost, 1e-12), 1e-9);
                for (int i = 0; i < 60; ++i) {
                    const double dp = choose_moves(dc_up, dp_up, dc_dn, dp_dn,
                                                   room, lam, step);
                    if (dp < 0.0) {
                        lo  = lam;
                        lam = hi < 0.0 ? lam * 2.0 : 0.5 * (lo + hi);
                    } else {
                        hi  = lam;
                        lam = 0.5 * (lo + hi);
                    }
                }
                choose_moves(dc_up, dp_up, dc_dn, dp_dn, room,
                             hi < 0.0 ? lam : hi, step);
            } else {
                // Dinkelbach fractional multiplier: lam >= C / Pd lowers
                // L = C / Pd whenever Pd stays >= target; otherwise bisect
                // lam above C / Pd to enforce Pd >= target.
                const double lam_lo  = cost / std::max(pd_cur, 1e-9);
                const double dp_free = choose_moves(dc_up, dp_up, dc_dn, dp_dn,
                                                    room, lam_lo, step);
                binds                = (pd_cur + dp_free) < target;
                if (binds) {
                    const double need = target - pd_cur;
                    double lo         = lam_lo;
                    double hi         = -1.0;
                    double lam =
                        std::max({lam_lo * 2.0, saved / std::max(lost, 1e-12),
                                  lo + 1e-9});
                    for (int i = 0; i < 60; ++i) {
                        const double dp = choose_moves(dc_up, dp_up, dc_dn,
                                                       dp_dn, room, lam, step);
                        if (dp < need) {
                            lo  = lam;
                            lam = hi < 0.0 ? lam * 2.0 : 0.5 * (lo + hi);
                        } else {
                            hi  = lam;
                            lam = 0.5 * (lo + hi);
                        }
                    }
                    choose_moves(dc_up, dp_up, dc_dn, dp_dn, room,
                                 hi < 0.0 ? lam : hi, step);
                }
            }
            if (std::ranges::all_of(step, [](int s) { return s == 0; })) {
                continue;
            }
            candidate = g;
            for (SizeType k = 0; k < blocks.size(); ++k) {
                for (SizeType s = blocks[k].a; s < blocks[k].b; ++s) {
                    candidate[s] +=
                        step[k] > 0 ? d : (step[k] < 0 ? -down[s] : 0.0F);
                }
            }
            candidate[nstages - 1] = std::max(candidate[nstages - 1], floor);
            double pd_cand         = target;
            if (!free_pd) {
                pin(candidate);
            } else if (binds) {
                pin(candidate);
            } else {
                bool above_pre = true;
                for (SizeType s = 0; s < nstages && above_pre; ++s) {
                    above_pre = candidate[s] > g_ref[s];
                }
                if (!above_pre) {
                    continue;
                }
                pd_cand = signal.pd_active(candidate);
                if (pd_cand < target) {
                    pin(candidate);
                    binds = true;
                }
            }
            bool above = true;
            for (SizeType s = 0; s < nstages && above; ++s) {
                above = candidate[s] > g_ref[s];
            }
            if (!above) {
                continue;
            }
            if (free_pd && binds) {
                pd_cand = signal.pd_active(candidate);
            }
            const double cost_candidate = bank.evaluate(candidate, d);
            const double rank_candidate =
                free_pd ? (cost_candidate / std::max(pd_cand, 1e-9))
                        : cost_candidate;
            const double rank_cur =
                free_pd ? (cost / std::max(pd_cur, 1e-9)) : cost;
            if (rank_candidate < kAccept * rank_cur) {
                bank.accept(candidate);
                g      = candidate;
                cost   = cost_candidate;
                pd_cur = pd_cand;
                ++nmoves;
            } else {
                bank.reject();
            }
        }
        t_moves += seconds_since(t0);
        ++ipass;
        search_c = cost;
        const double rank_end =
            free_pd ? (cost / std::max(pd_cur, 1e-9)) : cost;
        repeat = adaptive && ilevel % 2 == 0 &&
                 rank_end < (1.0 - repeat_drop) * rank_begin;
    }
    out.search_thresholds = g;
    out.search_pd         = signal.pd(g, signal.size());

    // Pin on a fresh bank (always two-sided under kPin; one-sided repair under
    // kFree when search Pd clears target by more than max(15% of min(target,
    // 1 - target), 6 SE)), then report on an independent bank, as solve().
    const SizeType n    = budget.nbank;
    const bool interior = free_pd && (out.search_pd > target + interior_margin);
    float pin_shift     = 0.0F;
    {
        constexpr float kPinCut = -0.5F;
        const uint64_t pin_seed = derive_seed(seed, kBankPin);
        if (!has_floor) {
            auto margins = engine.h1_margins_above(g, kPinCut, n, pin_seed);
            const SizeType required =
                std::clamp<SizeType>(static_cast<SizeType>(std::ceil(
                                         target * static_cast<double>(n))),
                                     1, n);
            const auto rank =
                margins.begin() + static_cast<std::ptrdiff_t>(n - required);
            std::ranges::nth_element(margins, rank);
            if (*rank > kPinCut) {
                pin_shift =
                    std::nextafter(*rank, std::numeric_limits<float>::lowest());
            } else {
                const auto bank = engine.make_signal_bank(n, pin_seed);
                auto full       = bank->margins(g);
                pin_shift       = SignalSample::shift_of(full, target);
            }
            if (interior) {
                pin_shift = std::min(0.0F, pin_shift);
            }
            for (SizeType s = 0; s < nstages; ++s) {
                g[s] += pin_shift;
            }
        } else {
            const auto bank      = engine.make_signal_bank(n, pin_seed);
            const bool lock_last = g[nstages - 1] <= floor;
            if (!lock_last) {
                auto margins = bank->margins(g);
                pin_shift    = SignalSample::shift_of(margins, target);
                if (interior) {
                    pin_shift = std::min(0.0F, pin_shift);
                }
            }
            if (!lock_last && g[nstages - 1] + pin_shift >= floor) {
                for (SizeType s = 0; s < nstages; ++s) {
                    g[s] += pin_shift;
                }
            } else {
                // The last threshold rests on the floor: shift the others over
                // the trials that pass it. pin_shift records the shift applied:
                // none with one stage, or when too few trials pass the floor
                // for any shift to reach the target.
                g[nstages - 1] = floor;
                pin_shift      = 0.0F;
                if (nstages > 1) {
                    std::vector<float> probe = g;
                    probe[nstages - 1] = std::numeric_limits<float>::lowest();
                    auto margins       = bank->margins(probe);
                    std::vector<float> last(
                        nstages, std::numeric_limits<float>::lowest());
                    last[nstages - 1] = floor;
                    const auto above  = bank->margins(last);
                    for (SizeType t = 0; t < n; ++t) {
                        if (above[t] <= 0.0F) {
                            margins[t] = std::numeric_limits<float>::lowest();
                        }
                    }
                    float shift = SignalSample::shift_of(margins, target);
                    if (interior) {
                        shift = std::min(0.0F, shift);
                    }
                    if (utils::is_finite(shift) &&
                        shift > std::numeric_limits<float>::lowest() / 2) {
                        pin_shift = shift;
                        for (SizeType s = 0; s + 1 < nstages; ++s) {
                            g[s] += pin_shift;
                        }
                    }
                }
            }
        }
    }
    {
        const SizeType count =
            engine.count_h1_survivors(g, n, derive_seed(seed, kBankReport));
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
    out.search_c = search_c >= 0.0 ? search_c : out.c;
    out.zscores.resize(nstages);
    for (SizeType s = 0; s < nstages; ++s) {
        out.zscores[s] = (mean[s] - g[s]) / sd[s];
    }
    out.thresholds = std::move(g);
    out.diagnostics.emplace_back("descent_passes", static_cast<double>(ipass));
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
