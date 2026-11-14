#include "loki/search/refold.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include <omp.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/psr_utils.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"
#include "lib/utils/fft_impl.hpp"

namespace loki::search {

namespace {

constexpr SizeType kParamStride = 2;
constexpr SizeType kMaxParams   = 5;
/// Scan points whose folds are inverted in one batch.
constexpr SizeType kScanBlock = 256;
/// Floor of a summed variance, as the reference implementation's.
constexpr double kVarianceFloor = 1e-12;

struct Boxcar {
    double snr{std::numeric_limits<double>::lowest()};
    SizeType width{0};
    SizeType start{0};
};

/// The best boxcar of a profile: the maximum over widths 1..wmax and start
/// bins of sum(e) / sqrt(sum(v)); the first width, then the first start
/// bin, wins a tie.
Boxcar boxcar_snr(std::span<const double> e,
                  std::span<const double> v,
                  SizeType wmax,
                  std::vector<double>& ce,
                  std::vector<double>& cv) {
    const SizeType n = e.size();
    ce.assign((2 * n) + 1, 0.0);
    cv.assign((2 * n) + 1, 0.0);
    for (SizeType i = 0; i < 2 * n; ++i) {
        ce[i + 1] = ce[i] + e[i % n];
        cv[i + 1] = cv[i] + v[i % n];
    }
    Boxcar best;
    for (SizeType w = 1; w <= wmax; ++w) {
        Boxcar row;
        for (SizeType k = 0; k < n; ++k) {
            const double sv = cv[w + k] - cv[k];
            if (!(sv > 0.0)) {
                continue;
            }
            const double s = (ce[w + k] - ce[k]) / std::sqrt(sv);
            if (s > row.snr) {
                row = {.snr = s, .width = w, .start = k};
            }
        }
        if (row.snr > best.snr) {
            best = row;
        }
    }
    return best;
}

/// numpy.linspace(lo, hi, n).
std::vector<double> linspace(double lo, double hi, SizeType n) {
    std::vector<double> out(n);
    if (n == 1) {
        out[0] = lo;
        return out;
    }
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (SizeType i = 0; i < n; ++i) {
        out[i] = lo + (static_cast<double>(i) * step);
    }
    out[n - 1] = hi;
    return out;
}

/// The phases, in cycles, of samples [begin, end) of the target.
std::string target_phase(const RefoldTarget& target,
                         SizeType begin,
                         SizeType end,
                         double tsamp,
                         double t_ref,
                         std::vector<double>& times,
                         std::vector<double>& phase) {
    const SizeType count = end - begin;
    times.resize(count);
    phase.resize(count);
    for (SizeType i = 0; i < count; ++i) {
        times[i] = static_cast<double>(begin + i) * tsamp;
    }
    if (target.model == RefoldModel::kLeaf) {
        const auto model =
            psr_utils::leaf_phase(target.leaf, target.n_params, times, t_ref,
                                  phase, target.propagator_significance);
        switch (model) {
        case psr_utils::LeafModel::kCircularSnap:
            return "circular_snap";
        case psr_utils::LeafModel::kCircularCrackle:
            return "circular_crackle";
        default:
            return "taylor";
        }
    }
    // phase = f * (dt - D(dt) / c), D = sum_k p_k dt^k / k!, k >= 2: the
    // Taylor branch of leaf_phase, for any number of derivatives.
    const SizeType np = target.n_params;
    const double freq = target.leaf[(np - 1) * kParamStride];
    std::array<double, kMaxParams - 1> deriv{};
    for (SizeType k = 0; k + 1 < np; ++k) {
        deriv[k] = target.leaf[(np - 2 - k) * kParamStride];
    }
    for (SizeType i = 0; i < count; ++i) {
        const double dt = times[i] - t_ref;
        double delay    = 0.0;
        for (SizeType k = np - 1; k-- > 0;) {
            delay = (delay + deriv[k]) * dt / static_cast<double>(k + 2);
        }
        delay *= dt;
        phase[i] = freq * (dt - (delay * utils::kInvCval));
    }
    return "taylor";
}

} // namespace

RefoldTarget refold_taylor_target(double freq, std::span<const double> derivs) {
    error_check::check(derivs.size() <= kMaxParams - 1,
                       "refold: at most four derivatives (accel to crackle)");
    error_check::check(freq > 0.0, "refold: the frequency must be positive");
    SizeType nderiv = derivs.size();
    while (nderiv > 0 && derivs[nderiv - 1] == 0.0) {
        --nderiv;
    }
    RefoldTarget target;
    target.n_params = nderiv + 1;
    target.model    = RefoldModel::kTaylor;
    target.leaf.assign((target.n_params + 2) * kParamStride, 0.0);
    // Highest derivative first, then the frequency, the delay and f0.
    for (SizeType k = 0; k < nderiv; ++k) {
        target.leaf[(nderiv - 1 - k) * kParamStride] = derivs[k];
    }
    target.leaf[nderiv * kParamStride]                = freq;
    target.leaf[(target.n_params + 1) * kParamStride] = freq;
    return target;
}

RefoldTarget refold_circular_target(double freq,
                                    std::span<const double, 4> derivs,
                                    double propagator_significance) {
    error_check::check(freq > 0.0, "refold: the frequency must be positive");
    RefoldTarget target;
    target.n_params                = kMaxParams;
    target.model                   = RefoldModel::kLeaf;
    target.propagator_significance = propagator_significance;
    target.leaf.assign((kMaxParams + 2) * kParamStride, 0.0);
    for (SizeType k = 0; k < derivs.size(); ++k) {
        target.leaf[(derivs.size() - 1 - k) * kParamStride] = derivs[k];
    }
    target.leaf[(kMaxParams - 1) * kParamStride] = freq;
    target.leaf[(kMaxParams + 1) * kParamStride] = freq;
    return target;
}

RefoldResult refold(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    double tsamp,
                    double t_ref,
                    const RefoldTarget& target,
                    const RefoldOptions& options) {
    error_check::check_equal(ts_e.size(), ts_v.size(),
                             "refold: ts_e and ts_v sizes differ");
    error_check::check(target.n_params >= 1 && target.n_params <= kMaxParams,
                       "refold: n_params must be 1 to 5");
    error_check::check_equal(target.leaf.size(),
                             (target.n_params + 2) * kParamStride,
                             "refold: the leaf must hold (n_params + 2) rows "
                             "of (value, error)");
    error_check::check(tsamp > 0.0, "refold: tsamp must be positive");
    error_check::check(options.nsub >= 1, "refold: nsub must be positive");
    const SizeType n    = ts_e.size();
    const SizeType nsub = options.nsub;
    error_check::check(n >= nsub,
                       "refold: the series is shorter than nsub samples");
    error_check::check(options.split <= nsub,
                       "refold: split must not exceed nsub");
    const double freq = target.leaf[(target.n_params - 1) * kParamStride];
    error_check::check(freq > 0.0, "refold: the frequency must be positive");
    const double accel = target.n_params >= 2
                             ? target.leaf[(target.n_params - 2) * kParamStride]
                             : 0.0;

    RefoldResult result;
    result.freq    = freq;
    result.fdot    = -freq * accel * utils::kInvCval;
    SizeType nbins = options.nbins;
    if (nbins == 0) {
        const double per_period = std::floor(1.0 / (freq * tsamp));
        nbins =
            static_cast<SizeType>(std::min(64.0, std::max(8.0, per_period)));
    }
    error_check::check(nbins >= 2, "refold: nbins must be at least 2");
    result.nbins        = nbins;
    const SizeType nk   = (nbins / 2) + 1;
    const SizeType wmax = nbins / 2;
    const double tobs   = static_cast<double>(n) * tsamp;
    const int nthreads  = std::max(options.nthreads, 1);

    // One exact fold per subintegration at the target.
    std::vector<SizeType> edges(nsub + 1);
    for (SizeType i = 0; i <= nsub; ++i) {
        edges[i] = (i * n) / nsub;
    }
    std::vector<double> tsub(nsub);
    std::vector<float> folds_e(nsub * nbins);
    std::vector<float> folds_v(nsub * nbins);
    std::vector<std::string> models(nsub);
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (SizeType s = 0; s < nsub; ++s) {
        const SizeType begin = edges[s];
        const SizeType end   = edges[s + 1];
        tsub[s]              = ((static_cast<double>(begin) * tsamp) +
                                (static_cast<double>(end - 1) * tsamp)) /
                                   2.0 -
                               t_ref;
        std::vector<double> times;
        std::vector<double> phase;
        models[s] =
            target_phase(target, begin, end, tsamp, t_ref, times, phase);
        const auto fold = algorithms::fold_phase(
            ts_e.subspan(begin, end - begin), ts_v.subspan(begin, end - begin),
            phase, nbins);
        std::copy_n(fold.begin(), nbins, folds_e.begin() + (s * nbins));
        std::copy_n(fold.begin() + nbins, nbins, folds_v.begin() + (s * nbins));
    }
    result.model = models.front();

    std::vector<ComplexType> spec_e(nsub * nk);
    std::vector<ComplexType> spec_v(nsub * nk);
    math::rfft_batch(folds_e, spec_e, nsub, nbins, nthreads);
    math::rfft_batch(folds_v, spec_v, nsub, nbins, nthreads);

    // Shifted sums of the subintegration spectra over [s0, s1), one row of
    // e and one of v per point.
    const auto shifted_sum = [&](double df, double dfd, SizeType s0,
                                 SizeType s1, std::span<ComplexType> row_e,
                                 std::span<ComplexType> row_v) {
        std::vector<std::complex<double>> acc_e(nk);
        std::vector<std::complex<double>> acc_v(nk);
        for (SizeType s = s0; s < s1; ++s) {
            const double dphi =
                (df * tsub[s]) + (0.5 * dfd * tsub[s] * tsub[s]);
            for (SizeType k = 0; k < nk; ++k) {
                const double arg =
                    -2.0 * std::numbers::pi * dphi * static_cast<double>(k);
                const std::complex<double> rot(std::cos(arg), std::sin(arg));
                acc_e[k] += std::complex<double>(spec_e[(s * nk) + k]) * rot;
                acc_v[k] += std::complex<double>(spec_v[(s * nk) + k]) * rot;
            }
        }
        for (SizeType k = 0; k < nk; ++k) {
            row_e[k] = ComplexType(acc_e[k]);
            row_v[k] = ComplexType(acc_v[k]);
        }
    };

    struct Point {
        double df{0.0};
        double dfd{0.0};
        SizeType s0{0};
        SizeType s1{0};
    };
    struct Scored {
        Boxcar box;
        std::vector<double> e;
        std::vector<double> v;
    };
    // Scores a block of points; keeps the folds when @p keep_folds.
    const auto score = [&](std::span<const Point> points, bool keep_folds) {
        const SizeType np = points.size();
        std::vector<ComplexType> rows(2 * np * nk);
        std::vector<float> profiles(2 * np * nbins);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < np; ++i) {
            const auto& p = points[i];
            shifted_sum(p.df, p.dfd, p.s0, p.s1,
                        std::span(rows).subspan(2 * i * nk, nk),
                        std::span(rows).subspan(((2 * i) + 1) * nk, nk));
        }
        math::irfft_batch(rows, profiles, 2 * np, nbins, nthreads);
        std::vector<Scored> out(np);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < np; ++i) {
            std::vector<double> e(nbins);
            std::vector<double> v(nbins);
            for (SizeType b = 0; b < nbins; ++b) {
                e[b] = profiles[(2 * i * nbins) + b];
                v[b] = std::max(
                    static_cast<double>(profiles[(((2 * i) + 1) * nbins) + b]),
                    kVarianceFloor);
            }
            std::vector<double> ce;
            std::vector<double> cv;
            out[i].box = boxcar_snr(e, v, wmax, ce, cv);
            if (keep_folds) {
                out[i].e = std::move(e);
                out[i].v = std::move(v);
            }
        }
        return out;
    };

    const Point origin{.df = 0.0, .dfd = 0.0, .s0 = 0, .s1 = nsub};
    auto best_scored = std::move(score(std::span(&origin, 1), true).front());
    Point best_point = origin;
    result.snr_in    = static_cast<float>(best_scored.box.snr);
    result.width_in  = best_scored.box.width;

    const double fstep =
        options.df_step_bins / (static_cast<double>(nbins) * tobs);
    const auto nfs = options.df_steps != 0
                         ? options.df_steps
                         : static_cast<SizeType>(
                               std::ceil(options.df_span_t / tobs / fstep));
    const auto fds = options.fd_steps == 0
                         ? std::vector<double>{0.0}
                         : linspace(-options.fd_span, options.fd_span,
                                    (2 * options.fd_steps) + 1);
    std::vector<Point> grid;
    grid.reserve(fds.size() * ((2 * nfs) + 1));
    for (const double dfd : fds) {
        for (SizeType j = 0; j <= 2 * nfs; ++j) {
            const double k = static_cast<double>(j) - static_cast<double>(nfs);
            grid.push_back({.df = k * fstep, .dfd = dfd, .s0 = 0, .s1 = nsub});
        }
    }
    // In grid order, a strictly better point replaces the best.
    for (SizeType start = 0; start < grid.size(); start += kScanBlock) {
        const auto block = std::span(grid).subspan(
            start, std::min(kScanBlock, grid.size() - start));
        auto scored = score(block, false);
        for (SizeType i = 0; i < block.size(); ++i) {
            if (scored[i].box.snr > best_scored.box.snr) {
                best_scored.box = scored[i].box;
                best_point      = block[i];
            }
        }
    }
    // The folds of the best point.
    best_scored  = std::move(score(std::span(&best_point, 1), true).front());
    result.snr   = static_cast<float>(best_scored.box.snr);
    result.width = best_scored.box.width;
    result.start_bin = best_scored.box.start;
    result.df        = best_point.df;
    result.dfd       = best_point.dfd;
    result.profile.resize(nbins);
    for (SizeType b = 0; b < nbins; ++b) {
        result.profile[b] =
            static_cast<float>(best_scored.e[b] / std::sqrt(best_scored.v[b]));
    }

    if (options.split > 1) {
        std::vector<Point> parts(options.split);
        for (SizeType j = 0; j < options.split; ++j) {
            parts[j] = {.df  = best_point.df,
                        .dfd = best_point.dfd,
                        .s0  = (j * nsub) / options.split,
                        .s1  = ((j + 1) * nsub) / options.split};
        }
        for (const auto& part : score(parts, false)) {
            result.split_snr.push_back(static_cast<float>(part.box.snr));
            result.split_start_bin.push_back(part.box.start);
        }
    }
    return result;
}

} // namespace loki::search
