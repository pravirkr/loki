#include "loki/io/preprocess.hpp"

#include <format>
#include <functional>
#include <span>
#include <stdexcept>

#include "loki/common/backend.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/detail/utils.hpp"
#include "lib/io/preprocess_engine.hpp"

namespace loki::io {

void PreprocessOptions::validate() const {
    const PreprocessOptions& o = *this;
    const auto require         = [](bool ok, const char* what) {
        if (!ok) {
            throw std::invalid_argument(std::format("preprocess: {}", what));
        }
    };
    require(utils::is_finite(o.filter_window) && o.filter_window >= 0.0,
            "filter_window must be finite and >= 0");
    require(utils::is_finite(o.variance_window) && o.variance_window >= 0.0,
            "variance_window must be finite and >= 0");
    require(o.window_blocks >= 1, "window_blocks must be >= 1");
    for (const double s : o.block_scales) {
        require(utils::is_finite(s) && s > 0.0,
                "block_scales must be finite and > 0");
    }
    require(utils::is_finite(o.block_sigma) && o.block_sigma > 0.0,
            "block_sigma must be > 0");
    require(utils::is_finite(o.min_good_fraction) &&
                o.min_good_fraction >= 0.0 && o.min_good_fraction <= 1.0,
            "min_good_fraction must be in [0, 1]");
    require(utils::is_finite(o.clip_sigma) && o.clip_sigma >= 0.0,
            "clip_sigma must be >= 0 (0 disables)");
    require(utils::is_finite(o.zap_sigma) && o.zap_sigma > 0.0,
            "zap_sigma must be > 0");
    require(o.zap_whiten_bins >= 3, "zap_whiten_bins must be >= 3");
    for (const auto& b : o.birdies) {
        require(utils::is_finite(b.freq) && b.freq > 0.0 &&
                    utils::is_finite(b.width) && b.width >= 0.0,
                "birdies need freq > 0 and width >= 0");
    }
    require(o.fast_median_min_points >= 1,
            "fast_median_min_points must be >= 1");
}

PreprocessReport preprocess(std::span<const float> raw,
                            double tsamp,
                            std::span<float> ts_e,
                            std::span<float> ts_v,
                            const PreprocessOptions& options,
                            Exec exec) {
    if (raw.empty()) {
        throw std::invalid_argument("preprocess: input is empty");
    }
    if (ts_e.size() != raw.size() || ts_v.size() != raw.size()) {
        throw std::invalid_argument(
            "preprocess: ts_e and ts_v must have the input length");
    }
    if (!utils::is_finite(tsamp) || tsamp <= 0.0) {
        throw std::invalid_argument("preprocess: tsamp must be positive");
    }
    options.validate();
    const auto* rb       = raw.data();
    const auto* re       = rb + raw.size();
    const auto* eb       = ts_e.data();
    const auto* vb       = ts_v.data();
    const bool e_partial = rb != eb && std::less<>{}(eb, re) &&
                           std::less<>{}(rb, eb + ts_e.size());
    const bool v_overlap =
        std::less<>{}(vb, re) && std::less<>{}(rb, vb + ts_v.size());
    if (e_partial || v_overlap ||
        (std::less<>{}(vb, eb + ts_e.size()) &&
         std::less<>{}(eb, vb + ts_v.size()))) {
        throw std::invalid_argument(
            "preprocess: buffers overlap (only raw == ts_e is allowed)");
    }
    if (exec.backend == Backend::kCPU) {
        return detail::preprocess_cpu(raw, tsamp, ts_e, ts_v, options,
                                      exec.nthreads);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        loki::detail::throw_unimplemented("preprocess", exec.backend);
    }
#endif
    loki::detail::throw_unavailable("preprocess", exec.backend);
}

} // namespace loki::io
