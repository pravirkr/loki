#pragma once

/**
 * @file ffa_engine.hpp
 * @brief Backend engine interface for FFA. Internal.
 */

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

/**
 * @brief Receives the hits of a scored FFA execution, in score-index order,
 * one batch of final-level coordinates at a time
 * (`cpu_hit_batch_coords`). The span is valid only during the call.
 */
using HitSink = std::function<void(std::span<const detection::SnrHit>)>;

template <SupportedFoldType FoldType> class FFAEngine {
protected:
    FFAEngine() = default;

public:
    virtual ~FFAEngine() = default;

    virtual const plans::FFAPlan<FoldType>& get_plan() const noexcept = 0;
    virtual plans::FFAPlan<FoldType> extract_plan() && noexcept       = 0;

    virtual float get_brute_fold_timing() const noexcept = 0;
    virtual float get_brute_fold_init_timing() const noexcept { return 0.0F; }

    virtual void
    set_fuse_levels(std::optional<SizeType> /*fuse_levels*/) noexcept {}
    virtual SizeType get_last_fuse_levels() const noexcept { return 0; }
    virtual float get_last_score_timing() const noexcept { return 0.0F; }

    virtual void execute_scored(std::span<const float> /*ts_e*/,
                                std::span<const float> /*ts_v*/,
                                float /*threshold*/,
                                std::span<const SizeType> /*widths*/,
                                std::vector<detection::SnrHit>& /*hits*/) {
        throw std::logic_error("execute_scored not implemented on this engine");
    }
    /**
     * @brief As above, handing the hits to @p sink batch by batch instead of
     * collecting them, so they never take more than a batch of memory
     * (`cpu_hit_buffer_bytes`). @p scratch is the caller's second ping-pong
     * buffer, at least `cpu_scored_buffer_size` elements; this overload
     * allocates no fold-sized memory (the CPU sweep).
     */
    virtual void execute_scored(std::span<const float> /*ts_e*/,
                                std::span<const float> /*ts_v*/,
                                float /*threshold*/,
                                std::span<const SizeType> /*widths*/,
                                const HitSink& /*sink*/,
                                std::span<FoldType> /*scratch*/) {
        throw std::logic_error("execute_scored not implemented on this engine");
    }

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<FoldType> fold) = 0;

    virtual void execute(DeviceSpan<const float> ts_e,
                         DeviceSpan<const float> ts_v,
                         DeviceSpan<FoldType> fold,
                         Stream stream) = 0;

    virtual void execute_return_to_time(std::span<const float> /*ts_e*/,
                                        std::span<const float> /*ts_v*/,
                                        std::span<float> /*fold*/) {
        throw std::logic_error(
            "execute_return_to_time not implemented on this engine");
    }

    virtual void execute_return_to_time(DeviceSpan<const float> /*ts_e*/,
                                        DeviceSpan<const float> /*ts_v*/,
                                        DeviceSpan<float> /*fold*/,
                                        Stream /*stream*/) {
        throw std::logic_error(
            "execute_return_to_time not implemented on this engine");
    }

    FFAEngine(const FFAEngine&)            = delete;
    FFAEngine& operator=(const FFAEngine&) = delete;
    FFAEngine(FFAEngine&&)                 = delete;
    FFAEngine& operator=(FFAEngine&&)      = delete;
};

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(const search::FFASearchConfig& cfg, bool show_progress);

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(memory::FFAWorkspaceCPU<FoldType>& workspace,
             const search::FFASearchConfig& cfg,
             bool show_progress);

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_gpu(const search::FFASearchConfig& cfg, int device_id);

/// Shares the caller's GPU workspace and FFT manager (both handles must
/// be GPU handles on @p device_id; the facade checks).
template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_gpu(memory::FFAWorkspace<FoldType>& workspace,
             math::FFTManager& fft_manager,
             const search::FFASearchConfig& cfg,
             int device_id);

/// compute_ffa_scores on the GPU: the time series is copied in once, folded
/// and scored on @p device_id, and only the scores are copied back.
std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores_gpu(std::span<const float> ts_e,
                       std::span<const float> ts_v,
                       const search::FFASearchConfig& cfg,
                       int device_id);

} // namespace loki::algorithms::detail
