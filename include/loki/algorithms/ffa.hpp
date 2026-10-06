#pragma once

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

/**
 * @brief Hierarchial P-FFA folding algorithm for Pulsar Search
 *
 * @tparam FoldType The type of fold to use (float for time domain, ComplexType
 * for Fourier domain)
 */
template <SupportedFoldType FoldType> class FFA {
public:
    /**
     * @brief Owns its workspace and FFT plans.
     *
     * The CPU thread count comes from @p cfg; @p exec selects the backend
     * and device.
     */
    explicit FFA(const search::FFASearchConfig& cfg,
                 bool show_progress = true,
                 Exec exec          = {});

    explicit FFA(const search::FFASearchConfig& cfg, Exec exec)
        : FFA(cfg, /*show_progress=*/false, exec) {}

    /**
     * @brief Runs on caller-owned buffers and FFT plans, so several FFA
     * instances (e.g. one per frequency chunk) can share one allocation.
     *
     * @p workspace and @p fft_manager must be built for the same backend and
     * device as @p exec, and @p workspace must be large enough for @p cfg.
     */
    explicit FFA(memory::FFAWorkspace<FoldType>& workspace,
                 math::FFTManager& fft_manager,
                 const search::FFASearchConfig& cfg,
                 bool show_progress = true,
                 Exec exec          = {});

    explicit FFA(memory::FFAWorkspace<FoldType>& workspace,
                 math::FFTManager& fft_manager,
                 const search::FFASearchConfig& cfg,
                 Exec exec)
        : FFA(workspace, fft_manager, cfg, /*show_progress=*/false, exec) {}

    // --- Rule of five: PIMPL ---
    ~FFA();
    FFA(FFA&&) noexcept;
    FFA& operator=(FFA&&) noexcept;
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;

    const plans::FFAPlan<FoldType>& get_plan() const noexcept;
    // Transfer ownership of the plan
    [[nodiscard]] plans::FFAPlan<FoldType> extract_plan() && noexcept;

    /// Total brute-fold time (table build + execute), in seconds.
    float get_brute_fold_timing() const noexcept;
    /// Brute-fold table-build time only (included in the total), in seconds.
    float get_brute_fold_init_timing() const noexcept;
    /**
     * @brief Override the number of merge levels fused into the brute fold.
     *
     * By default the level count is chosen automatically (time-domain,
     * frequency-only FFA only; 0 disables fusion). Mainly useful for tests and
     * benchmarks. Values are clamped to the number of merge levels. Ignored
     * (fusion stays off) for paths where fusion does not apply.
     */
    void set_fuse_levels(std::optional<SizeType> fuse_levels) noexcept;
    /// Merge levels fused on the most recent execute() (0 if fusion was off).
    [[nodiscard]] SizeType get_last_fuse_levels() const noexcept;
    /// Boxcar time included in the most recent execute_scored(), in seconds.
    [[nodiscard]] float get_last_score_timing() const noexcept;
    /**
     * @brief Fold and emit thresholded boxcar hits without storing the final
     * fold.
     *
     * Time-domain, frequency-only searches score each top-level frequency
     * tile while it is still in the cone-band scratch. `hits` are appended in
     * score-index order (`profile * nwidths + width`). Other FFA paths
     * materialise the fold and score it the same way, so the hit list matches
     * `snr_boxcar_3d` followed by a threshold scan.
     */
    void execute_scored(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        float threshold,
                        std::span<const SizeType> widths,
                        std::vector<detection::SnrHit>& hits);
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold);

    void execute(DeviceSpan<const float> ts_e,
                 DeviceSpan<const float> ts_v,
                 DeviceSpan<FoldType> fold,
                 Stream stream = {});

    // This overload is ONLY enabled when FoldType is ComplexType
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold)
        requires(std::is_same_v<FoldType, ComplexType>);

    void execute(DeviceSpan<const float> ts_e,
                 DeviceSpan<const float> ts_v,
                 DeviceSpan<float> fold,
                 Stream stream = {})
        requires(std::is_same_v<FoldType, ComplexType>);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using FFATime    = FFA<float>;
using FFAFourier = FFA<ComplexType>;

// Convenience function to fold time series using P-FFA (both time and Fourier
// domains)
template <SupportedFoldType FoldType>
std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::FFASearchConfig& cfg,
            bool quiet         = false,
            bool show_progress = false,
            Exec exec          = {});

// Convenience function to fold time series using P-FFA in the Fourier domain
// and return the result in the time domain (floats)
std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time(std::span<const float> ts_e,
                                   std::span<const float> ts_v,
                                   const search::FFASearchConfig& cfg,
                                   bool quiet         = false,
                                   bool show_progress = false,
                                   Exec exec          = {});

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   const search::FFASearchConfig& cfg,
                   bool quiet         = false,
                   bool show_progress = false,
                   Exec exec          = {});

} // namespace loki::algorithms