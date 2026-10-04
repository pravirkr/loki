#include "loki/algorithms/prune_rfi.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

#include "loki/exceptions.hpp"
#include "loki/utils.hpp"

namespace loki::algorithms {

namespace {

void validate_harvest_geometry(double harvest_mask_ntiles) {
    error_check::check_greater(harvest_mask_ntiles, 0.0,
                               "PruneRFIConfig: harvest_mask_ntiles must "
                               "be positive");
    error_check::check(utils::is_finite(harvest_mask_ntiles),
                       "PruneRFIConfig: harvest_mask_ntiles must be "
                       "finite");
}

} // namespace

void PruneRFIConfig::validate(SizeType nsegments) const {
    for (const auto& w : pulsar_mask) {
        error_check::check(utils::is_finite(w.f_lo) && utils::is_finite(w.f_hi),
                           "PruneRFIConfig: pulsar_mask window frequencies "
                           "must be finite");
        error_check::check_less_equal(
            w.f_lo, w.f_hi,
            "PruneRFIConfig: pulsar_mask window requires f_lo <= f_hi");
        error_check::check(!utils::is_nan(w.a_lo) && !utils::is_nan(w.a_hi),
                           "PruneRFIConfig: pulsar_mask window accelerations "
                           "must not be NaN");
        error_check::check_less_equal(
            w.a_lo, w.a_hi,
            "PruneRFIConfig: pulsar_mask window requires a_lo <= a_hi");
    }
    if (!harvest_scheme.empty()) {
        error_check::check_equal(
            harvest_scheme.size(), nsegments - 1,
            std::format("PruneRFIConfig: harvest_scheme must have size "
                        "nsegments - 1 = {} (got {})",
                        nsegments - 1, harvest_scheme.size()));
        for (const auto v : harvest_scheme) {
            error_check::check(!utils::is_nan(v),
                               "PruneRFIConfig: harvest_scheme must not "
                               "contain NaN");
        }
        validate_harvest_geometry(harvest_mask_ntiles);
    }
    if (impulsive_veto) {
        error_check::check_greater(impulsive_kappa, 0.0,
                                   "PruneRFIConfig: impulsive_kappa must be "
                                   "positive");
        error_check::check(utils::is_finite(impulsive_kappa),
                           "PruneRFIConfig: impulsive_kappa must be finite");
        error_check::check_greater_equal(impulsive_min_level, SizeType{1},
                                         "PruneRFIConfig: impulsive_min_level "
                                         "must be >= 1");
        error_check::check(!utils::is_nan(impulsive_min_snr),
                           "PruneRFIConfig: impulsive_min_snr must not be NaN");
    }
}

ParamWindow make_pulsar_window(
    double f, double a, double tobs, double f_pad, double a_pad) {
    error_check::check_greater(f, 0.0,
                               "make_pulsar_window: f must be positive");
    error_check::check_greater_equal(tobs, 0.0,
                                     "make_pulsar_window: tobs must be >= 0");
    error_check::check_greater_equal(f_pad, 0.0,
                                     "make_pulsar_window: f_pad must be >= 0");
    // Full one-sided sweep in either direction so the reference epoch of f is
    // irrelevant.
    const double sweep = f * std::abs(a) * tobs * utils::kInvCval;
    ParamWindow w;
    w.f_lo = f - sweep - f_pad;
    w.f_hi = f + sweep + f_pad;
    if (a_pad < std::numeric_limits<double>::max()) {
        error_check::check_greater_equal(
            a_pad, 0.0, "make_pulsar_window: a_pad must be >= 0");
        w.a_lo = a - a_pad;
        w.a_hi = a + a_pad;
    } else {
        w.a_lo = std::numeric_limits<double>::lowest();
        w.a_hi = std::numeric_limits<double>::max();
    }
    return w;
}

ParamWindow make_birdie_window(double f, double f_pad, double a_pad) {
    error_check::check_greater(f, 0.0,
                               "make_birdie_window: f must be positive");
    error_check::check_greater_equal(f_pad, 0.0,
                                     "make_birdie_window: f_pad must be >= 0");
    error_check::check_greater_equal(a_pad, 0.0,
                                     "make_birdie_window: a_pad must be >= 0");
    error_check::check(utils::is_finite(a_pad),
                       "make_birdie_window: a_pad must be finite");
    return ParamWindow{
        .f_lo = f - f_pad, .f_hi = f + f_pad, .a_lo = -a_pad, .a_hi = a_pad,};
}

ParamWindow make_harvest_window(double f,
                                double a,
                                double df,
                                double da,
                                double t_ref,
                                double tobs,
                                double ntiles) {
    error_check::check_greater(f, 0.0,
                               "make_harvest_window: f must be positive");
    error_check::check_greater_equal(df, 0.0,
                                     "make_harvest_window: df must be >= 0");
    error_check::check_greater_equal(da, 0.0,
                                     "make_harvest_window: da must be >= 0");
    error_check::check_greater_equal(tobs, 0.0,
                                     "make_harvest_window: tobs must be >= 0");
    error_check::check(utils::is_finite(t_ref),
                       "make_harvest_window: t_ref must be finite");
    error_check::check_greater(ntiles, 0.0,
                               "make_harvest_window: ntiles must be positive");
    error_check::check(utils::is_finite(ntiles),
                       "make_harvest_window: ntiles must be finite");
    const auto reach = std::max(t_ref, tobs - t_ref);
    const auto sweep = f * std::abs(a) * reach * utils::kInvCval;
    const auto f_pad = (ntiles * df) + sweep;
    const auto a_pad = ntiles * da;
    return ParamWindow{.f_lo = f - f_pad,
                       .f_hi = f + f_pad,
                       .a_lo = a - a_pad,
                       .a_hi = a + a_pad,};
}

std::vector<float>
make_default_harvest_scheme(std::span<const float> threshold_scheme,
                            SizeType min_level,
                            float offset,
                            float min_snr) {
    std::vector<float> scheme(threshold_scheme.size());
    for (SizeType i = 0; i < threshold_scheme.size(); ++i) {
        const auto level = i + 1; // 1-indexed stage
        if (level < min_level) {
            scheme[i] = kHarvestDisabled;
        } else {
            scheme[i] = std::max(min_snr, threshold_scheme[i] + offset);
        }
    }
    return scheme;
}

bool is_impulsive_segment(float score_child,
                          float score_parent,
                          SizeType n_seg,
                          double kappa,
                          float min_snr) noexcept {
    if (score_child < min_snr) {
        return false;
    }
    const double sc    = score_child;
    const double sp    = std::max(0.0, static_cast<double>(score_parent));
    const double gain  = (sc * sc) - (sp * sp);
    const double share = kappa * sc * sc / static_cast<double>(n_seg + 1);
    return gain > share;
}

} // namespace loki::algorithms
