#include "lib/detail/math.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/math/special_functions/binomial.hpp>
#include <omp.h>

#include "loki/common/types.hpp"

#include "lib/detail/utils.hpp"

namespace loki::math {

namespace {

// Compute the connection coefficient S_{k,m}.
double compute_connection_coefficient_s(SizeType k, SizeType m) {
    // Check if k-m is even and m <= k
    if (m > k || ((k - m) % 2 != 0)) {
        return 0.0;
    }
    const SizeType n        = (k - m) / 2;
    const SizeType delta_m0 = (m == 0) ? 1 : 0;
    // 2^(1 - k - delta_m0)
    const double factor = std::ldexp(1.0, static_cast<int>(1 - k - delta_m0));
    return factor * boost::math::binomial_coefficient<double>(k, n);
}
} // namespace

std::vector<float> generate_cheb_table(SizeType order_max, SizeType n_derivs) {
    const SizeType dim1       = n_derivs + 1;
    const SizeType dim2       = order_max + 1;
    const SizeType dim3       = order_max + 1;
    const SizeType total_size = dim1 * dim2 * dim3;

    std::vector<float> tab(total_size, 0.0F);

    // Helper lambda for 3D indexing: tab(i, j, k)
    const auto idx3d = [=](SizeType i, SizeType j, SizeType k) -> SizeType {
        return (i * dim2 * dim3) + (j * dim3) + k;
    };

    // Base cases: T_0(x) = 1, T_1(x) = x
    tab[idx3d(0, 0, 0)] = 1.0F;
    tab[idx3d(0, 1, 1)] = 1.0F;

    // Generate Chebyshev polynomials using recurrence: T_{n+1}(x) = 2x T_n(x) -
    // T_{n-1}(x)
    for (SizeType jorder = 2; jorder <= order_max; ++jorder) {
        for (SizeType k = 0; k <= order_max; ++k) {
            const float prev2 = tab[idx3d(0, jorder - 2, k)];

            // Shift coefficients right (multiply by x) and apply recurrence
            const float rolled =
                (k > 0) ? tab[idx3d(0, jorder - 1, k - 1)] : 0.0F;
            tab[idx3d(0, jorder, k)] = (2.0F * rolled) - prev2;
        }
    }

    // Generate derivatives
    std::vector<float> factor(order_max + 2);
    for (SizeType i = 0; i < order_max + 2; ++i) {
        factor[i] = static_cast<float>(i + 1);
    }

    for (SizeType ideriv = 1; ideriv <= n_derivs; ++ideriv) {
        for (SizeType jorder = 1; jorder <= order_max; ++jorder) {
            for (SizeType k = 0; k <= order_max; ++k) {
                // Shift coefficients left and multiply by factor
                const float prev_deriv =
                    (k < order_max) ? tab[idx3d(ideriv - 1, jorder, k + 1)]
                                    : 0.0F;
                tab[idx3d(ideriv, jorder, k)] = prev_deriv * factor[k];
            }
            // Ensure last coefficient is zero
            tab[idx3d(ideriv, jorder, order_max)] = 0.0F;
        }
    }

    return tab;
}

std::vector<float>
generalized_cheb_pols(SizeType poly_order, float t0, float scale) {
    // Get the base Chebyshev polynomials (no derivatives needed)
    const auto cheb_table = generate_cheb_table(poly_order, 0);
    const SizeType dim2   = poly_order + 1;
    const SizeType dim3   = poly_order + 1;

    // Extract the 2D slice [0, :, :] from the 3D table
    std::vector<float> cheb_pols(dim2 * dim3);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            const SizeType idx_3d =
                (0 * dim2 * dim3) + (i * dim3) + j; // [0, i, j]
            const SizeType idx_2d = (i * dim3) + j;
            cheb_pols[idx_2d]     = cheb_table[idx_3d];
        }
    }

    // Scale the polynomials: scale_factor = (1/scale)^k for k=0..poly_order
    std::vector<float> scale_factor(poly_order + 1);
    float scale_power     = 1.0F;
    const float inv_scale = 1.0F / scale;
    for (SizeType k = 0; k <= poly_order; ++k) {
        scale_factor[k] = scale_power;
        scale_power *= inv_scale;
    }

    // Apply scaling: cheb_pols * scale_factor (broadcast along columns)
    std::vector<float> scaled_pols(dim2 * dim3);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            const SizeType idx = (i * dim3) + j;
            scaled_pols[idx]   = cheb_pols[idx] * scale_factor[j];
        }
    }

    // Shift the origin to t0 using binomial expansion
    std::vector<float> shifted_pols(dim2 * dim3, 0.0F);
    const float neg_t0_over_scale = -t0 / scale;

    for (SizeType iorder = 0; iorder <= poly_order; ++iorder) {
        for (SizeType iterm = 0; iterm <= iorder; ++iterm) {
            const auto binom_coef =
                boost::math::binomial_coefficient<float>(iorder, iterm);
            const float power_term =
                std::pow(neg_t0_over_scale, static_cast<float>(iorder - iterm));

            const SizeType idx = (iorder * dim3) + iterm;
            shifted_pols[idx]  = binom_coef * power_term;
        }
    }

    // Matrix multiplication: result = scaled_pols * shifted_pols
    std::vector<float> result(dim2 * dim3, 0.0F);
    for (SizeType i = 0; i < dim2; ++i) {
        for (SizeType j = 0; j < dim3; ++j) {
            float sum = 0.0F;
            for (SizeType k = 0; k < dim3; ++k) {
                const SizeType scaled_idx  = (i * dim3) + k;
                const SizeType shifted_idx = (k * dim3) + j;
                sum += scaled_pols[scaled_idx] * shifted_pols[shifted_idx];
            }
            result[(i * dim3) + j] = sum;
        }
    }

    return result;
}

std::vector<double> compute_connection_matrix_s(SizeType k_max,
                                                SizeType coeff_order) {

    const SizeType n_params = k_max + 1;
    std::vector<double> mat(n_params * n_params, 0.0);

    for (SizeType k = 0; k < n_params; ++k) {
        for (SizeType m = 0; m <= k; ++m) {
            const double val = compute_connection_coefficient_s(k, m);
            // ascending: m = 0 → k
            // descending: m = k → 0
            const SizeType col = (coeff_order == 0) ? m : (k - m);

            mat[(k * n_params) + col] = val;
        }
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Running filter
// ---------------------------------------------------------------------------

namespace detail {
Tuning& tuning() noexcept {
    static Tuning instance;
    return instance;
}
} // namespace detail

namespace {

/// Loops shorter than this are not worth spawning a thread team for.
constexpr SizeType kParallelThreshold = 1U << 15U;
/// Minimum samples handled by one thread in the sliding-window kernels.
constexpr SizeType kMinChunk = 1U << 14U;
/// Chunks must be at least this many windows long so that the per-chunk
/// window initialisation stays negligible.
constexpr SizeType kMinChunkWindows = 8;

/**
 * @brief Index of the sample that numpy's "symmetric" padding places at
 * position \p k of the padded series (period 2n, edge sample repeated).
 */
[[nodiscard]] inline SizeType reflect_index(std::int64_t k,
                                            SizeType n) noexcept {
    const std::int64_t period = 2 * static_cast<std::int64_t>(n);
    auto m                    = k % period;
    if (m < 0) {
        m += period;
    }
    return std::cmp_less(m, static_cast<std::int64_t>(n))
               ? static_cast<SizeType>(m)
               : static_cast<SizeType>(period - 1 - m);
}

/// Copy \p count samples of the reflected series starting at \p first.
void gather_reflected(const float* x,
                      SizeType n,
                      std::int64_t first,
                      SizeType count,
                      float* dst) {
    if (first >= 0 && static_cast<SizeType>(first) + count <= n) {
        std::copy_n(x + first, count, dst);
        return;
    }
    for (SizeType t = 0; t < count; ++t) {
        dst[t] = x[reflect_index(first + static_cast<std::int64_t>(t), n)];
    }
}

/// Window statistic: running mean (double accumulator).
class MeanWindow {
public:
    void init(const float* ring, SizeType w) {
        m_sum = 0.0;
        for (SizeType i = 0; i < w; ++i) {
            m_sum += static_cast<double>(ring[i]);
        }
        m_inv_w = 1.0 / static_cast<double>(w);
    }
    [[nodiscard]] float value() const {
        return static_cast<float>(m_sum * m_inv_w);
    }
    void replace(float* ring, SizeType pos, float v) {
        m_sum += static_cast<double>(v) - static_cast<double>(ring[pos]);
        ring[pos] = v;
    }

private:
    double m_sum{0.0};
    double m_inv_w{1.0};
};

[[nodiscard]] inline float mid_mean(float a, float b) noexcept {
    return static_cast<float>(
        0.5 * (static_cast<double>(a) + static_cast<double>(b)));
}

/**
 * @brief Window statistic: median kept in a contiguous sorted array.
 *
 * A step finds the leaving and entering values by binary search and moves
 * only the samples between the two slots with one memmove. Fastest for
 * windows up to a few thousand samples.
 */
class SortedWindow {
public:
    explicit SortedWindow(SizeType w) : m_sorted(w) {}

    void init(const float* ring, SizeType w) {
        std::copy_n(ring, w, m_sorted.begin());
        std::ranges::sort(m_sorted);
    }
    [[nodiscard]] float value() const {
        const SizeType w = m_sorted.size();
        const SizeType h = w / 2;
        return (w % 2 == 1) ? m_sorted[h]
                            : mid_mean(m_sorted[h - 1], m_sorted[h]);
    }
    void replace(float* ring, SizeType pos, float v) {
        const float old = ring[pos];
        ring[pos]       = v;
        float* first    = m_sorted.data();
        float* last     = first + m_sorted.size();
        float* at_old   = std::lower_bound(first, last, old);
        if (v > old) {
            float* ins = std::lower_bound(at_old + 1, last, v);
            std::memmove(at_old, at_old + 1,
                         static_cast<SizeType>(ins - at_old - 1) *
                             sizeof(float));
            *(ins - 1) = v;
        } else {
            float* ins = std::lower_bound(first, at_old, v);
            std::memmove(ins + 1, ins,
                         static_cast<SizeType>(at_old - ins) * sizeof(float));
            *ins = v;
        }
    }

private:
    std::vector<float> m_sorted;
};

/**
 * @brief Window statistic: median kept in two indexed 4-ary heaps.
 *
 * The lower half lives in a max-heap and the upper half in a min-heap. Heap
 * sizes never change; a step replaces one node (ring slot) in place and, if
 * the halves now cross, swaps the two roots. O(log w) per step, which wins
 * over SortedWindow for large windows.
 */
class HeapWindow {
public:
    explicit HeapWindow(SizeType w)
        : m_n_lo((w + 1) / 2),
          m_lo(m_n_lo),
          m_hi(w / 2),
          m_pos(w),
          m_in_lo(w) {
        if (w > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("median window is too large");
        }
    }

    void init(const float* ring, SizeType w) {
        m_val = ring;
        std::vector<std::uint32_t> order(w);
        std::iota(order.begin(), order.end(), std::uint32_t{0});
        std::ranges::sort(order, [ring](std::uint32_t a, std::uint32_t b) {
            return ring[a] < ring[b];
        });
        // A descending array is a valid max-heap, an ascending one a min-heap.
        for (SizeType i = 0; i < m_lo.size(); ++i) {
            const std::uint32_t id = order[m_n_lo - 1 - i];
            m_lo[i]                = id;
            m_pos[id]              = static_cast<std::uint32_t>(i);
            m_in_lo[id]            = 1;
        }
        for (SizeType i = 0; i < m_hi.size(); ++i) {
            const std::uint32_t id = order[m_n_lo + i];
            m_hi[i]                = id;
            m_pos[id]              = static_cast<std::uint32_t>(i);
            m_in_lo[id]            = 0;
        }
    }

    [[nodiscard]] float value() const {
        const float lo_top = m_val[m_lo[0]];
        return m_hi.size() == m_lo.size() ? mid_mean(lo_top, m_val[m_hi[0]])
                                          : lo_top;
    }

    void replace(float* ring, SizeType pos, float v) {
        ring[pos]     = v;
        const auto id = static_cast<std::uint32_t>(pos);
        if (m_in_lo[id] != 0) {
            sift_up(m_lo, m_pos[id], kMaxFirst);
            sift_down(m_lo, m_pos[id], kMaxFirst);
        } else {
            sift_up(m_hi, m_pos[id], kMinFirst);
            sift_down(m_hi, m_pos[id], kMinFirst);
        }
        if (!m_hi.empty() && m_val[m_lo[0]] > m_val[m_hi[0]]) {
            const std::uint32_t a = m_lo[0];
            const std::uint32_t b = m_hi[0];
            m_lo[0]               = b;
            m_hi[0]               = a;
            m_in_lo[a]            = 0;
            m_in_lo[b]            = 1;
            m_pos[a]              = 0;
            m_pos[b]              = 0;
            sift_down(m_lo, 0, kMaxFirst);
            sift_down(m_hi, 0, kMinFirst);
        }
    }

private:
    static constexpr SizeType kArity = 4;
    static constexpr auto kMaxFirst  = [](float a, float b) { return a > b; };
    static constexpr auto kMinFirst  = [](float a, float b) { return a < b; };

    template <class Before>
    void
    sift_up(std::vector<std::uint32_t>& heap, SizeType i, Before before) const {
        const std::uint32_t id = heap[i];
        const float v          = m_val[id];
        while (i > 0) {
            const SizeType parent = (i - 1) / kArity;
            if (!before(v, m_val[heap[parent]])) {
                break;
            }
            heap[i]        = heap[parent];
            m_pos[heap[i]] = static_cast<std::uint32_t>(i);
            i              = parent;
        }
        heap[i]   = id;
        m_pos[id] = static_cast<std::uint32_t>(i);
    }

    template <class Before>
    void sift_down(std::vector<std::uint32_t>& heap,
                   SizeType i,
                   Before before) const {
        const SizeType n       = heap.size();
        const std::uint32_t id = heap[i];
        const float v          = m_val[id];
        for (;;) {
            const SizeType first = (kArity * i) + 1;
            if (first >= n) {
                break;
            }
            const SizeType last = std::min(first + kArity, n);
            SizeType best       = first;
            float best_v        = m_val[heap[first]];
            for (SizeType c = first + 1; c < last; ++c) {
                const float cv = m_val[heap[c]];
                if (before(cv, best_v)) {
                    best   = c;
                    best_v = cv;
                }
            }
            if (!before(best_v, v)) {
                break;
            }
            heap[i]        = heap[best];
            m_pos[heap[i]] = static_cast<std::uint32_t>(i);
            i              = best;
        }
        heap[i]   = id;
        m_pos[id] = static_cast<std::uint32_t>(i);
    }

    SizeType m_n_lo;
    const float* m_val{nullptr};
    std::vector<std::uint32_t> m_lo;
    std::vector<std::uint32_t> m_hi;
    mutable std::vector<std::uint32_t> m_pos;
    std::vector<std::uint8_t> m_in_lo;
};

/**
 * @brief Sliding-window kernel shared by all statistics.
 *
 * The output range is split into one chunk per thread. A thread keeps the
 * window in a ring buffer holding the original samples, so the sample that
 * leaves the window is never read back from \p x. Before any thread writes,
 * each thread copies the samples it needs from its neighbours (window
 * initialisation and the \c right samples past its chunk). That makes
 * \c dst == \c x safe, so the same code serves out-of-place filtering and
 * in-place baseline subtraction.
 *
 * @param subtract If true, <tt>dst[i] = x[i] - filter[i]</tt>, else
 * <tt>dst[i] = filter[i]</tt>.
 */
template <class MakeWindow>
void filter_chunks(const float* x,
                   float* dst,
                   SizeType n,
                   SizeType w,
                   bool subtract,
                   int nthreads,
                   MakeWindow make_window) {
    const SizeType left      = w / 2;
    const SizeType right     = w - 1 - left;
    const SizeType min_chunk = std::max(kMinChunk, kMinChunkWindows * w);
    // Never more threads than requested, and no thread without a worthwhile
    // chunk of work.
    int team = static_cast<int>(
        std::clamp(n / min_chunk, SizeType{1},
                   static_cast<SizeType>(std::max(nthreads, 1))));
    team = std::max(team, 1);
#pragma omp parallel num_threads(team) default(none)                           \
    shared(x, n, w, left, right, dst, subtract, make_window)
    {
        const auto nt       = static_cast<SizeType>(omp_get_num_threads());
        const auto tid      = static_cast<SizeType>(omp_get_thread_num());
        const SizeType base = n / nt;
        const SizeType rem  = n % nt;
        const SizeType b    = (tid * base) + std::min(tid, rem);
        const SizeType e    = b + base + (tid < rem ? 1 : 0);

        std::vector<float> ring(w);
        std::vector<float> tail(right);
        auto window = make_window();
        if (b < e) {
            gather_reflected(x, n,
                             static_cast<std::int64_t>(b) -
                                 static_cast<std::int64_t>(left),
                             w, ring.data());
            gather_reflected(x, n, static_cast<std::int64_t>(e), right,
                             tail.data());
            window.init(ring.data(), w);
        }
        // Every thread has copied what it needs from foreign chunks; from
        // here on only the own chunk of dst is written.
#pragma omp barrier
        if (b < e) {
            SizeType ring_pos = 0;
            for (SizeType j = b;; ++j) {
                const float v = window.value();
                dst[j]        = subtract ? x[j] - v : v;
                if (j + 1 == e) {
                    break;
                }
                const SizeType k = j + 1 + right;
                const float in   = k < e ? x[k] : tail[k - e];
                window.replace(ring.data(), ring_pos, in);
                if (++ring_pos == w) {
                    ring_pos = 0;
                }
            }
        }
    }
}

void filter_exact(const float* x,
                  float* dst,
                  SizeType n,
                  SizeType w,
                  FilterMethod method,
                  bool subtract,
                  int nthreads) {
    if (method == FilterMethod::kMean) {
        filter_chunks(x, dst, n, w, subtract, nthreads,
                      [] { return MeanWindow{}; });
    } else if (w >= detail::tuning().heap_window) {
        filter_chunks(x, dst, n, w, subtract, nthreads,
                      [w] { return HeapWindow(w); });
    } else {
        filter_chunks(x, dst, n, w, subtract, nthreads,
                      [w] { return SortedWindow(w); });
    }
}

void check_filter_args(SizeType n, SizeType window) {
    if (n == 0) {
        throw std::invalid_argument("running filter: input is empty");
    }
    if (window == 0) {
        throw std::invalid_argument("running filter: window must be >= 1");
    }
}

/// Block-averaging factor and block count of the fast filter.
struct FastPlan {
    SizeType ds;
    SizeType nds;
    bool use;
};

[[nodiscard]] FastPlan
plan_fast(SizeType n, SizeType window, SizeType min_points) {
    if (min_points == 0) {
        throw std::invalid_argument("running filter: min_points must be >= 1");
    }
    const SizeType ds  = std::max<SizeType>(1, window / min_points);
    const SizeType nds = n / ds;
    return {.ds = ds, .nds = nds, .use = ds > 1 && nds >= min_points};
}

/// Block means (double accumulation), then the exact filter on them.
[[nodiscard]] std::vector<float> lowres_baseline(const float* x,
                                                 SizeType ds,
                                                 SizeType nds,
                                                 SizeType min_points,
                                                 FilterMethod method,
                                                 int nthreads) {
    std::vector<float> coarse(nds);
    const double inv_ds = 1.0 / static_cast<double>(ds);
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    shared(x, ds, nds, inv_ds, coarse) if (nds * ds >= kParallelThreshold)
    for (SizeType b = 0; b < nds; ++b) {
        const float* blk = x + (b * ds);
        double acc       = 0.0;
        for (SizeType j = 0; j < ds; ++j) {
            acc += static_cast<double>(blk[j]);
        }
        coarse[b] = static_cast<float>(acc * inv_ds);
    }
    std::vector<float> filtered(nds);
    filter_exact(coarse.data(), filtered.data(), nds, min_points, method, false,
                 nthreads);
    return filtered;
}

/**
 * @brief Linear interpolation of the block-level baseline to every sample.
 *
 * Block \c b is centred at <tt>b*ds + (ds-1)/2</tt>; samples before the first
 * and after the last centre take the end values (np.interp semantics). The
 * body loop is branch free per sample. With \p subtract the baseline is
 * removed from \p x in place.
 */
void interpolate_baseline(const float* x,
                          float* dst,
                          SizeType n,
                          SizeType ds,
                          std::span<const float> lo,
                          bool subtract,
                          int nthreads) {
    const SizeType nds        = lo.size();
    const SizeType half       = ds / 2;
    const SizeType tail_start = (ds * (nds - 1)) + half;
    const float inv           = 1.0F / static_cast<float>(2 * ds);
    const float c0            = (ds % 2 == 0) ? 1.0F : 0.0F;
    nthreads                  = std::max(nthreads, 1);

    const auto emit = [&](SizeType i, float bl) {
        dst[i] = subtract ? x[i] - bl : bl;
    };
    for (SizeType i = 0; i < std::min(half, n); ++i) {
        emit(i, lo[0]);
    }
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    shared(x, dst, n, ds, nds, half, inv, c0, lo,                              \
               subtract) if (n >= kParallelThreshold)
    for (SizeType blk = 0; blk < nds - 1; ++blk) {
        const float a     = lo[blk];
        const float d     = lo[blk + 1] - a;
        const SizeType i0 = (ds * blk) + half;
#pragma omp simd
        for (SizeType m = 0; m < ds; ++m) {
            const float frac = (c0 + static_cast<float>(2 * m)) * inv;
            const float bl   = std::fma(frac, d, a);
            dst[i0 + m]      = subtract ? x[i0 + m] - bl : bl;
        }
    }
    for (SizeType i = tail_start; i < n; ++i) {
        emit(i, lo[nds - 1]);
    }
}

} // namespace

void running_filter(std::span<const float> in,
                    std::span<float> out,
                    SizeType window,
                    FilterMethod method,
                    int nthreads) {
    check_filter_args(in.size(), window);
    nthreads = std::max(nthreads, 1);
    if (out.size() != in.size()) {
        throw std::invalid_argument(
            "running filter: output size must match input size");
    }
    const auto* lo_in  = in.data();
    const auto* hi_in  = in.data() + in.size();
    const auto* lo_out = out.data();
    const auto* hi_out = out.data() + out.size();
    if (std::less<>{}(lo_in, hi_out) && std::less<>{}(lo_out, hi_in)) {
        throw std::invalid_argument(
            "running filter: input and output must not overlap");
    }
    filter_exact(in.data(), out.data(), in.size(), window, method, false,
                 nthreads);
}

void running_filter_fast(std::span<const float> in,
                         std::span<float> out,
                         SizeType window,
                         FilterMethod method,
                         SizeType min_points,
                         int nthreads) {
    check_filter_args(in.size(), window);
    nthreads        = std::max(nthreads, 1);
    const auto plan = plan_fast(in.size(), window, min_points);
    if (!plan.use) {
        running_filter(in, out, window, method, nthreads);
        return;
    }
    if (out.size() != in.size()) {
        throw std::invalid_argument(
            "running filter: output size must match input size");
    }
    const auto lo = lowres_baseline(in.data(), plan.ds, plan.nds, min_points,
                                    method, nthreads);
    // Interpolation reads and writes one sample at a time; overlap is only
    // safe for the exact in-place case, so require disjoint buffers here.
    const auto* hi_in  = in.data() + in.size();
    const auto* hi_out = out.data() + out.size();
    if (std::less<>{}(in.data(), hi_out) && std::less<>{}(out.data(), hi_in) &&
        in.data() != out.data()) {
        throw std::invalid_argument(
            "running filter: input and output must not overlap");
    }
    interpolate_baseline(in.data(), out.data(), in.size(), plan.ds, lo, false,
                         nthreads);
}

void subtract_running_filter(std::span<float> x,
                             SizeType window,
                             FilterMethod method,
                             bool fast,
                             SizeType min_points,
                             int nthreads) {
    check_filter_args(x.size(), window);
    nthreads = std::max(nthreads, 1);
    if (fast) {
        const auto plan = plan_fast(x.size(), window, min_points);
        if (plan.use) {
            const auto lo = lowres_baseline(x.data(), plan.ds, plan.nds,
                                            min_points, method, nthreads);
            interpolate_baseline(x.data(), x.data(), x.size(), plan.ds, lo,
                                 true, nthreads);
            return;
        }
    }
    filter_exact(x.data(), x.data(), x.size(), window, method, true, nthreads);
}

// ---------------------------------------------------------------------------
// Location, scale and z-score
// ---------------------------------------------------------------------------

namespace {

constexpr double kSqrtHalfPi = 1.2533141373155003; // sqrt(pi / 2)

void check_nonempty(std::span<const float> x, const char* what) {
    if (x.empty()) {
        throw std::invalid_argument(std::string("cannot estimate ") + what +
                                    " of an empty series");
    }
}

[[nodiscard]] double sum_of(std::span<const float> x, int nthreads) {
    const SizeType n = x.size();
    const float* p   = x.data();
    double acc       = 0.0;
    nthreads         = std::max(nthreads, 1);
#pragma omp parallel for simd num_threads(nthreads)                            \
    schedule(static) default(none) shared(n, p)                                \
    reduction(+ : acc) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        acc += static_cast<double>(p[i]);
    }
    return acc;
}

[[nodiscard]] double mean_of(std::span<const float> x, int nthreads) {
    return sum_of(x, nthreads) / static_cast<double>(x.size());
}

[[nodiscard]] double std_of(std::span<const float> x, int nthreads) {
    const SizeType n = x.size();
    const float* p   = x.data();
    const double mu  = mean_of(x, nthreads);
    double acc       = 0.0;
#pragma omp parallel for simd num_threads(std::max(nthreads, 1))               \
    schedule(static) default(none) shared(n, p, mu)                            \
    reduction(+ : acc) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        const double d = static_cast<double>(p[i]) - mu;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

/// Maps a float to a uint32 whose unsigned order equals the float order.
[[nodiscard]] constexpr std::uint32_t float_to_key(float f) noexcept {
    const auto u = std::bit_cast<std::uint32_t>(f);
    return (u & 0x80000000U) != 0U ? ~u : (u | 0x80000000U);
}

[[nodiscard]] constexpr float key_to_float(std::uint32_t k) noexcept {
    return std::bit_cast<float>((k & 0x80000000U) != 0U ? (k & 0x7FFFFFFFU)
                                                        : ~k);
}

/**
 * @brief Exact k-th smallest key of n keys by 3-pass (11/11/10 bit) radix
 * select. Reads the data three times, allocates nothing and parallelises the
 * histograms with OpenMP.
 */
template <class KeyFn>
[[nodiscard]] std::uint32_t
radix_select_key(SizeType n, SizeType k, int nthreads, KeyFn key_of) {
    constexpr std::array<unsigned, 3> kShift{21U, 10U, 0U};
    constexpr std::array<unsigned, 3> kWidth{11U, 11U, 10U};
    constexpr SizeType kBins = 1U << 11U;
    nthreads                 = std::max(nthreads, 1);
    std::uint32_t prefix     = 0;
    for (SizeType pass = 0; pass < 3; ++pass) {
        const unsigned shift      = kShift[pass];
        const unsigned width      = kWidth[pass];
        const std::uint32_t mask  = (1U << width) - 1U;
        const unsigned prev_shift = pass == 0 ? 0U : kShift[pass - 1];
        const bool filter         = pass != 0;

        std::array<SizeType, kBins> hist{};
        SizeType* h = hist.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    shared(n, shift, width, mask, prev_shift, filter, prefix, key_of, h)       \
    reduction(+ : h[ : kBins]) if (n >= kParallelThreshold)
        for (SizeType i = 0; i < n; ++i) {
            const std::uint32_t key = key_of(i);
            if (!filter || (key >> prev_shift) == prefix) {
                ++h[(key >> shift) & mask];
            }
        }
        std::uint32_t digit = 0;
        for (; digit < (1U << width); ++digit) {
            if (k < hist[digit]) {
                break;
            }
            k -= hist[digit];
        }
        prefix = (prefix << width) | digit;
    }
    return prefix;
}

/**
 * @brief Order statistics and linear-interpolated quantiles (numpy default)
 * of a series, or of |x - center| when \p center is given.
 *
 * Large series use the copy-free radix select, small ones one scratch copy
 * and std::nth_element.
 */
class OrderStatistics {
public:
    explicit OrderStatistics(std::span<const float> x,
                             int nthreads,
                             std::optional<double> center = std::nullopt)
        : m_x(x),
          m_center(center),
          m_nthreads(nthreads),
          m_radix(x.size() >= detail::tuning().radix_select_size) {
        if (!m_radix) {
            m_scratch.resize(x.size());
            for (SizeType i = 0; i < x.size(); ++i) {
                m_scratch[i] = value_at(i);
            }
        }
    }

    /// k-th smallest value (0-based).
    [[nodiscard]] float kth(SizeType k) {
        if (!m_radix) {
            const auto nth = m_scratch.begin() + static_cast<std::ptrdiff_t>(k);
            std::nth_element(m_scratch.begin(), nth, m_scratch.end());
            return *nth;
        }
        const SizeType n = m_x.size();
        if (m_center.has_value()) {
            return std::bit_cast<float>(
                radix_select_key(n, k, m_nthreads, [this](SizeType i) {
                    return std::bit_cast<std::uint32_t>(value_at(i));
                }));
        }
        const float* p = m_x.data();
        return key_to_float(radix_select_key(
            n, k, m_nthreads, [p](SizeType i) { return float_to_key(p[i]); }));
    }

    [[nodiscard]] double quantile(double q) {
        const SizeType n  = m_x.size();
        const double pos  = static_cast<double>(n - 1) * q;
        const auto lo     = static_cast<SizeType>(std::floor(pos));
        const SizeType hi = std::min(lo + 1, n - 1);
        const double frac = pos - static_cast<double>(lo);
        const auto a      = static_cast<double>(kth(lo));
        if (frac <= 0.0) {
            return a;
        }
        return std::lerp(a, static_cast<double>(kth(hi)), frac);
    }

private:
    [[nodiscard]] float value_at(SizeType i) const {
        const auto v = static_cast<double>(m_x[i]);
        return m_center.has_value()
                   ? static_cast<float>(std::abs(v - *m_center))
                   : m_x[i];
    }

    std::span<const float> m_x;
    std::optional<double> m_center;
    int m_nthreads;
    bool m_radix;
    std::vector<float> m_scratch;
};

[[nodiscard]] double median_of(std::span<const float> x, int nthreads) {
    return OrderStatistics(x, nthreads).quantile(0.5);
}

/// Median of a scratch array, reordering it.
[[nodiscard]] double median_inplace(std::vector<float>& v) {
    const SizeType m = v.size();
    const auto mid   = v.begin() + static_cast<std::ptrdiff_t>(m / 2);
    std::nth_element(v.begin(), mid, v.end());
    const auto upper = static_cast<double>(*mid);
    if (m % 2 == 1) {
        return upper;
    }
    const double lower = static_cast<double>(*std::max_element(v.begin(), mid));
    return 0.5 * (lower + upper);
}

/// Gaussian-consistent MAD about \p med with a mean-deviation fallback.
[[nodiscard]] double
mad_about(std::span<const float> x, double med, int nthreads) {
    OrderStatistics dev(x, nthreads, med);
    const double mad = dev.quantile(0.5);
    if (mad > 0.0) {
        return mad * kMadScale;
    }
    // More than half of the samples equal the median: use the mean absolute
    // deviation (aad = sigma * sqrt(2/pi) for a Gaussian).
    const SizeType n = x.size();
    const float* p   = x.data();
    double acc       = 0.0;
#pragma omp parallel for simd num_threads(std::max(nthreads, 1))               \
    schedule(static) default(none) shared(n, p, med)                           \
    reduction(+ : acc) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        acc += std::abs(static_cast<double>(p[i]) - med);
    }
    return acc / static_cast<double>(n) * kSqrtHalfPi;
}

/// MAD of one side of the median (<= or >=), with the same fallback.
[[nodiscard]] double half_mad(std::span<const float> x, double med, bool left) {
    std::vector<float> dev;
    dev.reserve(x.size());
    double acc = 0.0;
    for (const float s : x) {
        const auto v = static_cast<double>(s);
        if (left ? v <= med : v >= med) {
            const double d = std::abs(v - med);
            acc += d;
            dev.push_back(static_cast<float>(d));
        }
    }
    if (dev.empty()) {
        return 0.0;
    }
    const double mad = median_inplace(dev);
    if (mad > 0.0) {
        return mad * kMadScale;
    }
    return acc / static_cast<double>(dev.size()) * kSqrtHalfPi;
}

[[nodiscard]] ScaleEstimate scale_impl(std::span<const float> x,
                                       ScaleMethod method,
                                       std::optional<double> median,
                                       int nthreads) {
    const auto symmetric = [](double s) {
        return ScaleEstimate{.left = s, .right = s};
    };
    switch (method) {
    case ScaleMethod::kNone:
        return symmetric(1.0);
    case ScaleMethod::kStd:
        return symmetric(std_of(x, nthreads));
    case ScaleMethod::kIqr: {
        OrderStatistics stats(x, nthreads);
        const double q75 = stats.quantile(0.75);
        const double q25 = stats.quantile(0.25);
        return symmetric((q75 - q25) / kIqrScale);
    }
    case ScaleMethod::kMad:
        return symmetric(
            mad_about(x, median.value_or(median_of(x, nthreads)), nthreads));
    case ScaleMethod::kDoubleMad: {
        const double med = median.value_or(median_of(x, nthreads));
        return {
            .left  = half_mad(x, med, true),
            .right = half_mad(x, med, false),
        };
    }
    }
    throw std::invalid_argument("unknown scale method");
}

} // namespace

double estimate_loc(std::span<const float> x, LocMethod method, int nthreads) {
    check_nonempty(x, "location");
    nthreads = std::max(nthreads, 1);
    switch (method) {
    case LocMethod::kNone:
        return 0.0;
    case LocMethod::kMean:
        return mean_of(x, nthreads);
    case LocMethod::kMedian:
        return median_of(x, nthreads);
    }
    throw std::invalid_argument("unknown location method");
}

ScaleEstimate
estimate_scale(std::span<const float> x, ScaleMethod method, int nthreads) {
    check_nonempty(x, "scale");
    return scale_impl(x, method, std::nullopt, std::max(nthreads, 1));
}

ZScoreResult
zscore(std::span<float> x, LocMethod loc, ScaleMethod scale, int nthreads) {
    check_nonempty(x, "z-scores");
    const int team         = std::max(nthreads, 1);
    const double loc_value = estimate_loc(x, loc, team);
    const auto scale_value =
        scale_impl(x, scale,
                   loc == LocMethod::kMedian ? std::optional<double>(loc_value)
                                             : std::nullopt,
                   team);

    const auto inverse = [](double s) {
        return (s > 0.0 && utils::is_finite(s)) ? static_cast<float>(1.0 / s)
                                                : 1.0F;
    };
    const auto loc_f   = static_cast<float>(loc_value);
    const float inv_lo = inverse(scale_value.left);
    const float inv_hi = inverse(scale_value.right);
    float* p           = x.data();
    const SizeType n   = x.size();
#pragma omp parallel for simd num_threads(team) schedule(static) default(none) \
    shared(n, p, loc_f, inv_lo, inv_hi) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        const float d = p[i] - loc_f;
        p[i]          = d * (d < 0.0F ? inv_lo : inv_hi);
    }
    return {.loc = loc_value, .scale = scale_value};
}

} // namespace loki::math
