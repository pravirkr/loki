#include "loki/algorithms/ffa.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/algorithms/ffa_engine.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detection/score_engine.hpp"
#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms {

namespace {

template <SupportedFoldType FoldType>
std::unique_ptr<detail::FFAEngine<FoldType>> make_ffa_engine(
    const search::FFASearchConfig& cfg, bool show_progress, Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "FFA");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ffa_cpu<FoldType>(cfg, show_progress);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_ffa_gpu<FoldType>(cfg, exec.device);
    }
#endif
    loki::detail::throw_unavailable("FFA", exec.backend);
}

template <SupportedFoldType FoldType>
std::unique_ptr<detail::FFAEngine<FoldType>>
make_ffa_engine(memory::FFAWorkspace<FoldType>& workspace,
                math::FFTManager& fft_manager,
                const search::FFASearchConfig& cfg,
                bool show_progress,
                Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "FFA");
    loki::detail::check_same_exec(workspace.exec(), exec, "FFA (workspace)");
    loki::detail::check_same_exec(fft_manager.exec(), exec,
                                  "FFA (fft_manager)");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ffa_cpu<FoldType>(
            memory::detail::cpu_workspace(workspace, "FFA"),
            math::detail::cpu_fft(fft_manager, "FFA"), cfg, show_progress);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_ffa_gpu<FoldType>(workspace, fft_manager, cfg,
                                              exec.device);
    }
#endif
    loki::detail::throw_unavailable("FFA", exec.backend);
}

} // namespace

template <SupportedFoldType FoldType> class FFA<FoldType>::Impl {
public:
    Impl(const search::FFASearchConfig& cfg, bool show_progress, Exec exec)
        : m_exec(exec),
          m_engine(make_ffa_engine<FoldType>(cfg, show_progress, exec)) {}

    Impl(memory::FFAWorkspace<FoldType>& workspace,
         math::FFTManager& fft_manager,
         const search::FFASearchConfig& cfg,
         bool show_progress,
         Exec exec)
        : m_exec(exec),
          m_engine(make_ffa_engine<FoldType>(
              workspace, fft_manager, cfg, show_progress, exec)) {}

    const plans::FFAPlan<FoldType>& get_plan() const noexcept {
        return m_engine->get_plan();
    }
    plans::FFAPlan<FoldType> extract_plan() && noexcept {
        return std::move(*m_engine).extract_plan();
    }
    float get_brute_fold_timing() const noexcept {
        return m_engine->get_brute_fold_timing();
    }
    float get_brute_fold_init_timing() const noexcept {
        return m_engine->get_brute_fold_init_timing();
    }
    void set_fuse_levels(std::optional<SizeType> fuse_levels) noexcept {
        m_engine->set_fuse_levels(fuse_levels);
    }
    SizeType get_last_fuse_levels() const noexcept {
        return m_engine->get_last_fuse_levels();
    }
    float get_last_score_timing() const noexcept {
        return m_engine->get_last_score_timing();
    }
    void execute_scored(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        float threshold,
                        std::span<const SizeType> widths,
                        std::vector<detection::SnrHit>& hits) {
        m_engine->execute_scored(ts_e, ts_v, threshold, widths, hits);
    }
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) {
        m_engine->execute(ts_e, ts_v, fold);
    }
    void execute(DeviceSpan<const float> ts_e,
                 DeviceSpan<const float> ts_v,
                 DeviceSpan<FoldType> fold,
                 Stream stream) {
        check_device(ts_e.device, "FFA::execute");
        check_device(ts_v.device, "FFA::execute");
        check_device(fold.device, "FFA::execute");
        m_engine->execute(ts_e, ts_v, fold, stream);
    }
    void execute_return_to_time(std::span<const float> ts_e,
                                std::span<const float> ts_v,
                                std::span<float> fold) {
        m_engine->execute_return_to_time(ts_e, ts_v, fold);
    }
    void execute_return_to_time(DeviceSpan<const float> ts_e,
                                DeviceSpan<const float> ts_v,
                                DeviceSpan<float> fold,
                                Stream stream) {
        check_device(ts_e.device, "FFA::execute");
        check_device(ts_v.device, "FFA::execute");
        check_device(fold.device, "FFA::execute");
        m_engine->execute_return_to_time(ts_e, ts_v, fold, stream);
    }

private:
    Exec m_exec;
    std::unique_ptr<detail::FFAEngine<FoldType>> m_engine;

    void check_device(const Device& view, std::string_view what) const {
        loki::detail::check_device(view, m_exec.backend, m_exec.device, what);
    }
};

// --- Definitions for FFA ---
template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(const search::FFASearchConfig& cfg,
                   bool show_progress,
                   Exec exec)
    : m_impl(std::make_unique<Impl>(cfg, show_progress, exec)) {}

template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(memory::FFAWorkspace<FoldType>& workspace,
                   math::FFTManager& fft_manager,
                   const search::FFASearchConfig& cfg,
                   bool show_progress,
                   Exec exec)
    : m_impl(std::make_unique<Impl>(
          workspace, fft_manager, cfg, show_progress, exec)) {}

template <SupportedFoldType FoldType> FFA<FoldType>::~FFA() = default;
template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(FFA&& other) noexcept = default;
template <SupportedFoldType FoldType>
FFA<FoldType>& FFA<FoldType>::operator=(FFA&& other) noexcept = default;

template <SupportedFoldType FoldType>
const plans::FFAPlan<FoldType>& FFA<FoldType>::get_plan() const noexcept {
    return m_impl->get_plan();
}
template <SupportedFoldType FoldType>
plans::FFAPlan<FoldType> FFA<FoldType>::extract_plan() && noexcept {
    return std::move(*m_impl).extract_plan();
}
template <SupportedFoldType FoldType>
float FFA<FoldType>::get_brute_fold_timing() const noexcept {
    return m_impl->get_brute_fold_timing();
}
template <SupportedFoldType FoldType>
float FFA<FoldType>::get_brute_fold_init_timing() const noexcept {
    return m_impl->get_brute_fold_init_timing();
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::set_fuse_levels(
    std::optional<SizeType> fuse_levels) noexcept {
    m_impl->set_fuse_levels(fuse_levels);
}
template <SupportedFoldType FoldType>
SizeType FFA<FoldType>::get_last_fuse_levels() const noexcept {
    return m_impl->get_last_fuse_levels();
}
template <SupportedFoldType FoldType>
float FFA<FoldType>::get_last_score_timing() const noexcept {
    return m_impl->get_last_score_timing();
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute_scored(std::span<const float> ts_e,
                                   std::span<const float> ts_v,
                                   float threshold,
                                   std::span<const SizeType> widths,
                                   std::vector<detection::SnrHit>& hits) {
    m_impl->execute_scored(ts_e, ts_v, threshold, widths, hits);
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            std::span<FoldType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(DeviceSpan<const float> ts_e,
                            DeviceSpan<const float> ts_v,
                            DeviceSpan<FoldType> fold,
                            Stream stream) {
    m_impl->execute(ts_e, ts_v, fold, stream);
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            std::span<float> fold)
    requires(std::is_same_v<FoldType, ComplexType>)
{
    m_impl->execute_return_to_time(ts_e, ts_v, fold);
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(DeviceSpan<const float> ts_e,
                            DeviceSpan<const float> ts_v,
                            DeviceSpan<float> fold,
                            Stream stream)
    requires(std::is_same_v<FoldType, ComplexType>)
{
    m_impl->execute_return_to_time(ts_e, ts_v, fold, stream);
}

template <SupportedFoldType FoldType>
std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::FFASearchConfig& cfg,
            bool quiet,
            bool show_progress,
            Exec exec) {
    const timing::ScopedLogLevel scoped_log_level(quiet);
    FFA<FoldType> ffa(cfg, show_progress, exec);
    const plans::FFAPlan<FoldType>& ffa_plan = ffa.get_plan();
    const auto buffer_size                   = ffa_plan.get_buffer_size();
    std::vector<FoldType> fold(buffer_size, FoldType{});
    ffa.execute(ts_e, ts_v, std::span<FoldType>(fold));
    // RESIZE to actual result size
    const auto fold_size = ffa_plan.get_fold_size();
    fold.resize(fold_size);
    return {std::move(fold), std::move(ffa).extract_plan()};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time(std::span<const float> ts_e,
                                   std::span<const float> ts_v,
                                   const search::FFASearchConfig& cfg,
                                   bool quiet,
                                   bool show_progress,
                                   Exec exec) {
    const timing::ScopedLogLevel scoped_log_level(quiet);
    FFA<ComplexType> ffa(cfg, show_progress, exec);
    const plans::FFAPlan<ComplexType>& ffa_plan = ffa.get_plan();
    const auto buffer_size_time = ffa_plan.get_buffer_size_time();
    std::vector<float> fold(buffer_size_time);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    // RESIZE to actual result size
    const auto fold_size_time = ffa_plan.get_fold_size_time();
    fold.resize(fold_size_time);
    // Get the plan for the time domain
    plans::FFAPlan<float> ffa_plan_time(cfg);
    return {std::move(fold), std::move(ffa_plan_time)};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   const search::FFASearchConfig& cfg,
                   bool quiet,
                   bool show_progress,
                   Exec exec) {
    const timing::ScopedLogLevel scoped_log_level(quiet);
    auto [fold, ffa_plan] =
        cfg.get_use_fourier()
            ? compute_ffa_fourier_return_to_time(ts_e, ts_v, cfg, quiet,
                                                 show_progress, exec)
            : compute_ffa<float>(ts_e, ts_v, cfg, quiet, show_progress, exec);
    const auto nsegments = ffa_plan.get_nsegments().back();
    const auto ncoords   = ffa_plan.get_ncoords().back();
    error_check::check_equal(
        nsegments, 1U, "compute_ffa_scores: nsegments must be 1 for scores");
    const auto& score_widths = cfg.get_scoring_widths();
    const auto nscores       = ncoords * score_widths.size();
    std::vector<float> scores(nscores);
    detection::detail::snr_boxcar_3d_cpu(fold, score_widths, scores, ncoords,
                                         cfg.get_nbins(), cfg.get_nthreads());
    return {std::move(scores), std::move(ffa_plan)};
}

// Explicit instantiation
template class FFA<float>;
template class FFA<ComplexType>;

template std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa(std::span<const float>,
            std::span<const float>,
            const search::FFASearchConfig&,
            bool,
            bool,
            Exec);
template std::tuple<std::vector<ComplexType>, plans::FFAPlan<ComplexType>>
compute_ffa(std::span<const float>,
            std::span<const float>,
            const search::FFASearchConfig&,
            bool,
            bool,
            Exec);
} // namespace loki::algorithms
