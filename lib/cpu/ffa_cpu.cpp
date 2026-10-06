#include "loki/algorithms/ffa.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ffa_engine.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/core/kernels.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/progress.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detection/score_engine.hpp"
#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms {

namespace {

struct ConeScoreCtx {
    std::span<const SizeType> widths;
    float threshold{0.0F};
    std::vector<std::vector<float>> psum;
    std::vector<std::vector<detection::SnrHit>> hits;
};

void cone_score_tile(const float* profiles,
                     SizeType coord_begin,
                     SizeType nprofiles,
                     SizeType nbins,
                     void* ctx) {
    auto* state    = static_cast<ConeScoreCtx*>(ctx);
    const auto tid = static_cast<SizeType>(omp_get_thread_num());
    detection::append_snr_boxcar_3d_hits(profiles, coord_begin, nprofiles,
                                         nbins, state->widths, state->threshold,
                                         state->psum[tid], state->hits[tid]);
}

} // namespace

// FFACpuEngine implementation
template <SupportedFoldType FoldType>
class FFACpuEngine : public detail::FFAEngine<FoldType> {
public:
    explicit FFACpuEngine(search::FFASearchConfig cfg, bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_workspace_storage(m_ffa_plan),
          m_workspace_ptr(&m_workspace_storage),
          m_fft_ptr(&m_fft_storage) {
        // Validate workspace
        const auto& ws = get_workspace();
        ws.validate(m_ffa_plan);
        // Initialize BruteFold (table build time is tracked separately)
        timing::SimpleTimer init_timer;
        init_timer.start();
        initialize_brute_fold();
        m_brutefold_init_time = init_timer.stop();
        apply_fuse_levels_env();
        apply_level_timing_env();
        log_info();
    }

    explicit FFACpuEngine(memory::FFAWorkspaceCPU<FoldType>& workspace,
                          math::FFTWManager& fft_manager,
                          search::FFASearchConfig cfg,
                          bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_workspace_storage(),
          m_workspace_ptr(&workspace),
          m_fft_ptr(&fft_manager) {
        // Validate workspace
        const auto& ws = get_workspace();
        ws.validate(m_ffa_plan);
        // Initialize BruteFold (table build time is tracked separately)
        timing::SimpleTimer init_timer;
        init_timer.start();
        initialize_brute_fold();
        m_brutefold_init_time = init_timer.stop();
        apply_fuse_levels_env();
        apply_level_timing_env();
        log_info();
    }

    ~FFACpuEngine() override                     = default;
    FFACpuEngine(const FFACpuEngine&)            = delete;
    FFACpuEngine& operator=(const FFACpuEngine&) = delete;
    FFACpuEngine(FFACpuEngine&&)                 = delete;
    FFACpuEngine& operator=(FFACpuEngine&&)      = delete;

    const plans::FFAPlan<FoldType>& get_plan() const noexcept override {
        return m_ffa_plan;
    }

    [[nodiscard]] plans::FFAPlan<FoldType> extract_plan() && noexcept override {
        return std::move(m_ffa_plan);
    }

    /// Brute-fold time: table construction plus all execute() calls.
    float get_brute_fold_timing() const noexcept override {
        return m_brutefold_init_time + m_brutefold_time;
    }
    /// Time spent building the brute-fold lookup tables at construction.
    float get_brute_fold_init_timing() const noexcept override {
        return m_brutefold_init_time;
    }
    void
    set_fuse_levels(std::optional<SizeType> fuse_levels) noexcept override {
        m_fuse_levels_override = fuse_levels;
    }
    /// Levels fused on the most recent execute() (0 if fusion did not run).
    SizeType get_last_fuse_levels() const noexcept override {
        return m_last_fuse_levels;
    }
    /// Boxcar time folded into the most recent scored execute, in seconds.
    float get_last_score_timing() const noexcept override {
        return m_last_score_time;
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) override {
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACpuEngine::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACpuEngine::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_buffer_size(),
            "FFACpuEngine::execute: fold must have size buffer_size");

        auto& ws = get_workspace();
        // Resolve the coordinates into the workspace for the FFA plan
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws.coords_freq);
        } else {
            m_ffa_plan.resolve_coordinates(ws.coords);
        }

        if constexpr (std::is_same_v<FoldType, float>) {
            if (m_is_freq_only &&
                run_cone_bands(ts_e, ts_v, fold.data(), nullptr, 0.0F, {})) {
                return;
            }
        }

        // Execute the FFA plan
        execute_unified(ts_e, ts_v, fold);
    }

    void execute(DeviceSpan<const float> /*ts_e*/,
                 DeviceSpan<const float> /*ts_v*/,
                 DeviceSpan<FoldType> /*fold*/,
                 Stream /*stream*/) override {
        loki::detail::throw_no_device_memory("FFA", Backend::kCPU);
    }

    void execute_return_to_time(std::span<const float> ts_e,
                                std::span<const float> ts_v,
                                std::span<float> fold) override {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            execute_return_to_time_impl(ts_e, ts_v, fold);
        } else {
            (void)ts_e;
            (void)ts_v;
            (void)fold;
            throw std::logic_error(
                "execute_return_to_time only valid for Fourier domain");
        }
    }

    void execute_return_to_time(DeviceSpan<const float> /*ts_e*/,
                                DeviceSpan<const float> /*ts_v*/,
                                DeviceSpan<float> /*fold*/,
                                Stream /*stream*/) override {
        loki::detail::throw_no_device_memory("FFA", Backend::kCPU);
    }

    void execute_scored(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        float threshold,
                        std::span<const SizeType> widths,
                        std::vector<detection::SnrHit>& hits) override {
        if constexpr (!std::is_same_v<FoldType, float>) {
            throw std::invalid_argument(
                "FFA::execute_scored requires a time-domain FFA");
        } else {
            error_check::check_equal(
                ts_e.size(), m_cfg.get_nsamps(),
                "FFA::execute_scored: ts_e must have size nsamps");
            error_check::check_equal(
                ts_v.size(), ts_e.size(),
                "FFA::execute_scored: ts_v must have size nsamps");
            error_check::check(!widths.empty(),
                               "FFA::execute_scored: widths must be non-empty");
            const auto nsegments = m_ffa_plan.get_nsegments().back();
            error_check::check_equal(
                nsegments, SizeType{1},
                "FFA::execute_scored: the top level must be one segment");
            auto& ws = get_workspace();
            if (m_is_freq_only) {
                m_ffa_plan.resolve_coordinates_freq(ws.coords_freq);
            } else {
                m_ffa_plan.resolve_coordinates(ws.coords);
            }
            hits.clear();
            m_last_score_time = 0.0F;
            if (m_is_freq_only &&
                run_cone_bands(ts_e, ts_v, nullptr, &hits, threshold, widths)) {
                return;
            }
            const auto buffer_size = m_ffa_plan.get_buffer_size();
            std::vector<float> fold(buffer_size, 0.0F);
            execute_unified(ts_e, ts_v, std::span<float>(fold));
            const auto ncoords = m_ffa_plan.get_ncoords().back();
            const auto nbins   = m_cfg.get_nbins();
            std::vector<float> scores(ncoords * widths.size());
            detection::detail::snr_boxcar_3d_cpu(
                std::span<const float>(fold).first(ncoords * 2 * nbins), widths,
                scores, ncoords, nbins, m_nthreads);
            for (SizeType i = 0; i < scores.size(); ++i) {
                if (scores[i] >= threshold) {
                    hits.push_back(detection::SnrHit{
                        .score_index = static_cast<uint32_t>(i),
                        .snr         = scores[i],
                    });
                }
            }
        }
    }

    void execute_return_to_time_impl(std::span<const float> ts_e,
                                     std::span<const float> ts_v,
                                     std::span<float> fold) {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            const auto fold_size_time      = m_ffa_plan.get_fold_size_time();
            const auto fold_size_fourier   = m_ffa_plan.get_fold_size();
            const auto buffer_size_fourier = m_ffa_plan.get_buffer_size();

            error_check::check_equal(
                ts_e.size(), m_cfg.get_nsamps(),
                "FFACpuEngine::execute: ts_e must have size nsamps");
            error_check::check_equal(
                ts_v.size(), ts_e.size(),
                "FFACpuEngine::execute: ts_v must have size nsamps");
            error_check::check_equal(fold.size(), 2 * buffer_size_fourier,
                                     "FFACpuEngine::execute: fold must have "
                                     "size 2*buffer_size_fourier");

            auto& ws = get_workspace();
            // Resolve the coordinates for the FFA plan
            if (m_is_freq_only) {
                m_ffa_plan.resolve_coordinates_freq(ws.coords_freq);
            } else {
                m_ffa_plan.resolve_coordinates(ws.coords);
            }

            auto const fold_complex = std::span<ComplexType>(
                reinterpret_cast<ComplexType*>(fold.data()),
                buffer_size_fourier);
            // Execute the FFA plan
            execute_unified(ts_e, ts_v, fold_complex,
                            /*output_in_internal_buffer=*/true);
            // IRFFT
            const auto nfft = fold_size_time / m_cfg.get_nbins();
            get_fft().irfft_batch(
                std::span(ws.fold_internal).first(fold_size_fourier),
                fold.first(fold_size_time), nfft, m_cfg.get_nbins(),
                m_nthreads);
        }
    }

private:
    search::FFASearchConfig m_cfg;
    bool m_show_progress;
    plans::FFAPlan<FoldType> m_ffa_plan;
    int m_nthreads;
    bool m_is_freq_only;

    // Brute fold for the initial time-domain folding
    std::unique_ptr<BruteFold<FoldType>> m_the_bf;
    std::unique_ptr<BruteFold<float>> m_the_bf_float; // For lossy init
    bool m_use_lossy_init{false};
    std::optional<SizeType> m_fuse_levels_override;
    SizeType m_last_fuse_levels{0};
    bool m_level_timing{false};
    float m_last_score_time{0.0F};
    float m_brutefold_time{0.0F};      // Accumulated execute() time
    float m_brutefold_init_time{0.0F}; // Table-build time (constructor)

    // FFA workspace ownership
    memory::FFAWorkspaceCPU<FoldType> m_workspace_storage;
    // The observer pointer that always points to the active workspace.
    memory::FFAWorkspaceCPU<FoldType>* m_workspace_ptr{nullptr};

    math::FFTWManager m_fft_storage;
    math::FFTWManager* m_fft_ptr{nullptr};

    [[nodiscard]] memory::FFAWorkspaceCPU<FoldType>& get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::FFAWorkspaceCPU<FoldType>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] math::FFTWManager& get_fft() noexcept { return *m_fft_ptr; }

    /// Benchmark hook: LOKI_FUSE_LEVELS=<n> forces the fusion depth (0
    /// disables).
    void apply_fuse_levels_env() {
        const char* env = std::getenv("LOKI_FUSE_LEVELS");
        if (env == nullptr || env[0] == '\0') {
            return;
        }
        char* end             = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0') {
            spdlog::warn("Ignoring invalid LOKI_FUSE_LEVELS='{}'", env);
            return;
        }
        m_fuse_levels_override = static_cast<SizeType>(v);
        spdlog::info("LOKI_FUSE_LEVELS override: {}", v);
    }

    /// Benchmark hook: LOKI_FFA_LEVEL_TIMING=1 logs per-level Gfloat/s.
    void apply_level_timing_env() {
        const char* env = std::getenv("LOKI_FFA_LEVEL_TIMING");
        m_level_timing  = env != nullptr && env[0] != '\0' && env[0] != '0';
        if (m_level_timing) {
            spdlog::info("LOKI_FFA_LEVEL_TIMING enabled");
        }
    }

    void log_level_timing(std::string_view kind,
                          SizeType level,
                          double floats,
                          double seconds) const {
        if (!m_level_timing) {
            return;
        }
        const double gflops = seconds > 0.0 ? (floats / seconds / 1.0e9) : 0.0;
        spdlog::info(
            "FFA level timing: kind={} level={} floats={:.3e} time_s={:.4f} "
            "Gfloat/s={:.2f}",
            kind, level, floats, seconds, gflops);
    }

    void account_cone_band(float wall_s,
                           const core::ConeBandThreadSeconds& stats) {
        const double thread_sum = stats.brute + stats.merge + stats.score;
        const auto wall         = static_cast<double>(wall_s);
        const double prefix     = stats.prefix_wall;
        const double rest       = std::max(0.0, wall - prefix);
        const double scale      = thread_sum > 0.0 ? rest / thread_sum : 0.0;
        double brute            = prefix + (stats.brute * scale);
        double score            = stats.score * scale;
        if (brute + score > wall && wall > 0.0) {
            const double renorm = wall / (brute + score);
            brute *= renorm;
            score *= renorm;
        }
        m_brutefold_time += static_cast<float>(brute);
        m_last_score_time += static_cast<float>(score);
    }

    /**
     * @brief Run the frequency-only time-domain FFA as cache-resident cone
     * bands.
     *
     * Each band keeps one frequency tile's dependency cone in per-thread
     * scratch for K merge levels. `coords[i].idx` must be monotone; otherwise
     * the band falls back to a plain level. Returns false when the first
     * level cannot use a cone, so the caller runs the ping-pong path.
     * A null `fold_result` with a non-null `hits` scores the top tile in
     * scratch and does not allocate the final fold.
     */
    [[nodiscard]] bool run_cone_bands(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      FoldType* fold_result,
                                      std::vector<detection::SnrHit>* hits,
                                      float threshold,
                                      std::span<const SizeType> widths) {
        if constexpr (!std::is_same_v<FoldType, float>) {
            return false;
        } else {
            if (!m_is_freq_only || m_use_lossy_init || !m_the_bf) {
                return false;
            }
            if (m_fuse_levels_override.has_value() &&
                *m_fuse_levels_override == 0) {
                return false;
            }
            const SizeType levels  = m_cfg.get_niters_ffa() + 1;
            const SizeType n_merge = levels - 1;
            if (n_merge < 1) {
                return false;
            }
            const auto ncoords         = m_ffa_plan.get_ncoords();
            const auto offsets         = m_ffa_plan.get_ncoords_offsets();
            const auto& shapes         = m_ffa_plan.get_fold_shapes_time();
            const SizeType nbins       = m_cfg.get_nbins();
            const SizeType seg_len     = m_ffa_plan.get_segment_lens().front();
            const SizeType budget      = cone_scratch_budget_bytes();
            const SizeType tile_forced = cone_tile_override();
            auto& ws                   = get_workspace();

            std::vector<ConeChoice> steps;
            SizeType done = 0;
            while (done < n_merge) {
                const ConeChoice step = choose_cone_step(
                    done, n_merge - done, ncoords, offsets, shapes, nbins,
                    seg_len, budget, tile_forced, ws.coords_freq.data());
                if (!step.cone && steps.empty()) {
                    return false;
                }
                steps.push_back(step);
                done += step.k;
            }
            if (steps.empty()) {
                return false;
            }
            m_last_fuse_levels = steps.front().cone ? steps.front().k : 0;

            const bool scoring     = hits != nullptr;
            const SizeType n_steps = steps.size();
            std::vector<float> extra;
            auto* internal =
                static_cast<float*>(get_workspace().fold_internal.data());
            auto* user       = static_cast<float*>(fold_result);
            float* secondary = user;
            if (scoring && n_steps >= 3) {
                extra.resize(m_ffa_plan.get_buffer_size());
                secondary = extra.data();
            }

            ConeScoreCtx score_ctx;
            if (scoring) {
                const auto nthreads =
                    static_cast<SizeType>(std::max(m_nthreads, 1));
                score_ctx.widths    = widths;
                score_ctx.threshold = threshold;
                score_ctx.psum.resize(nthreads);
                score_ctx.hits.resize(nthreads);
                SizeType wmax = 0;
                for (const SizeType width : widths) {
                    wmax = std::max(wmax, width);
                }
                for (SizeType thread = 0; thread < nthreads; ++thread) {
                    score_ctx.psum[thread].resize(nbins + wmax);
                }
            }

            progress::ProgressGuard const progress_guard(m_show_progress);
            auto bar = progress::make_ffa_bar("Computing FFA", n_merge);
            const auto run_span    = m_the_bf->runs();
            const auto offset_span = m_the_bf->run_offsets();
            float const* current   = nullptr;
            SizeType progressed    = 0;
            for (SizeType istep = 0; istep < n_steps; ++istep) {
                const ConeChoice& step   = steps[istep];
                const bool last          = istep + 1 == n_steps;
                float* dest              = nullptr;
                const bool score_in_band = scoring && last && step.cone;
                if (!score_in_band) {
                    if (!scoring) {
                        const bool to_user = ((n_steps - 1 - istep) % 2) == 0;
                        dest               = to_user ? user : internal;
                    } else if (n_steps < 3) {
                        dest = internal;
                    } else {
                        dest = (istep % 2 == 0) ? internal : secondary;
                    }
                }
                const float brute_before = m_brutefold_time;
                const float score_before = m_last_score_time;
                timing::SimpleTimer step_timer;
                step_timer.start();
                if (step.cone) {
                    std::array<const coord::FFACoordFreq*, 17> coords{};
                    std::array<SizeType, 17> counts{};
                    counts[0] = ncoords[step.done];
                    for (SizeType j = 1; j <= step.k; ++j) {
                        counts[j] = ncoords[step.done + j];
                        coords[j] =
                            ws.coords_freq.data() + offsets[step.done + j];
                    }
                    const bool bottom = step.done == 0;
                    core::ConeBandThreadSeconds stats;
                    core::ffa_cone_band_freq(
                        bottom ? nullptr : current, ts_e.data(), ts_v.data(),
                        run_span.data(), offset_span.data(), seg_len, dest,
                        coords.data(), counts.data(), shapes[step.done][0],
                        nbins, step.k, step.tile,
                        score_in_band ? &cone_score_tile : nullptr,
                        score_in_band ? &score_ctx : nullptr, &stats,
                        m_nthreads);
                    account_cone_band(step_timer.stop(), stats);
                } else {
                    execute_iter_freq(current, dest, step.done + 1);
                    (void)step_timer.stop();
                }
                if (m_level_timing) {
                    double floats = 0.0;
                    for (SizeType j = 1; j <= step.k; ++j) {
                        const SizeType level = step.done + j;
                        floats += static_cast<double>(shapes[level][0]) *
                                  static_cast<double>(ncoords[level]) * 2.0 *
                                  static_cast<double>(nbins);
                    }
                    log_level_timing(step.cone ? "cone" : "merge", step.done,
                                     floats,
                                     static_cast<double>(step_timer.stop()));
                    if (step.done == 0) {
                        const double brute_floats =
                            static_cast<double>(shapes[0][0]) *
                            static_cast<double>(ncoords[0]) * 2.0 *
                            static_cast<double>(nbins);
                        log_level_timing("brute", 0, brute_floats,
                                         static_cast<double>(m_brutefold_time -
                                                             brute_before));
                    }
                    if (score_in_band) {
                        log_level_timing("score", step.done + step.k, floats,
                                         static_cast<double>(m_last_score_time -
                                                             score_before));
                    }
                }
                if (dest != nullptr) {
                    current = dest;
                }
                progressed += step.k;
                if (m_show_progress) {
                    bar->set_leaves(m_ffa_plan.get_ncoords_lb()[progressed]);
                    bar->set_progress(progressed);
                }
            }
            if (scoring && !steps.back().cone) {
                const float* top           = current;
                const SizeType ncoords_top = ncoords.back();
                std::vector<float> scores(ncoords_top * widths.size());
                detection::detail::snr_boxcar_3d_cpu(
                    std::span<const float>(top, ncoords_top * 2 * nbins),
                    widths, scores, ncoords_top, nbins, m_nthreads);
                for (SizeType i = 0; i < scores.size(); ++i) {
                    if (scores[i] >= threshold) {
                        hits->push_back(detection::SnrHit{
                            .score_index = static_cast<uint32_t>(i),
                            .snr         = scores[i],
                        });
                    }
                }
            } else if (scoring) {
                for (const auto& part : score_ctx.hits) {
                    hits->insert(hits->end(), part.begin(), part.end());
                }
            }
            bar->mark_as_completed();
            return true;
        }
    }

    [[nodiscard]] static SizeType cone_scratch_budget_bytes() {
        const char* env  = std::getenv("LOKI_CONE_SCRATCH_KB");
        unsigned long kb = 2048;
        if (env != nullptr && env[0] != '\0') {
            char* end             = nullptr;
            const unsigned long v = std::strtoul(env, &end, 10);
            if (end != env && *end == '\0' && v > 0) {
                kb = v;
            }
        }
        return static_cast<SizeType>(kb) * 1024U;
    }

    [[nodiscard]] static SizeType cone_tile_override() {
        const char* env = std::getenv("LOKI_CONE_TILE");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        char* end             = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0' || v == 0) {
            return 0;
        }
        return static_cast<SizeType>(v);
    }

    struct ConeChoice {
        SizeType k{1};
        SizeType tile{0};
        bool cone{false};
        SizeType done{0};
    };

    [[nodiscard]] ConeChoice
    choose_cone_step(SizeType done,
                     SizeType remain,
                     std::span<const SizeType> ncoords,
                     std::span<const uint32_t> offsets,
                     const std::vector<std::vector<SizeType>>& shapes,
                     SizeType nbins,
                     SizeType seg_len,
                     SizeType budget,
                     SizeType tile_forced,
                     const coord::FFACoordFreq* coords_base) const {
        ConeChoice plain{.k = 1, .tile = 0, .cone = false, .done = done};
        SizeType k_cap      = std::min(remain, SizeType{16});
        const SizeType nseg = shapes[done][0];
        while (k_cap > 0 && (SizeType{1} << k_cap) > nseg) {
            --k_cap;
        }
        if (m_fuse_levels_override.has_value() && *m_fuse_levels_override > 0) {
            k_cap = std::min(k_cap, *m_fuse_levels_override);
        }
        if (k_cap == 0) {
            return plain;
        }
        const bool bottom = done == 0;
        constexpr std::array<SizeType, 5> kTiles{512, 256, 128, 64, 32};
        const SizeType ntiles = tile_forced == 0 ? kTiles.size() : 1;
        for (SizeType k = k_cap; k >= 1; --k) {
            std::array<const coord::FFACoordFreq*, 17> coords{};
            std::array<SizeType, 17> counts{};
            counts[0]      = ncoords[done];
            bool coords_ok = true;
            for (SizeType j = 1; j <= k; ++j) {
                counts[j] = ncoords[done + j];
                if (counts[j] == 0) {
                    coords_ok = false;
                    break;
                }
                coords[j] = coords_base + offsets[done + j];
            }
            if (!coords_ok) {
                continue;
            }
            for (SizeType itile = 0; itile < ntiles; ++itile) {
                const SizeType tile =
                    tile_forced == 0 ? kTiles[itile] : tile_forced;
                const SizeType floats = core::cone_band_working_floats(
                    coords.data(), counts.data(), k, tile, nbins);
                if (floats == 0) {
                    continue;
                }
                SizeType bytes = 2 * floats * sizeof(float);
                if (bottom) {
                    bytes +=
                        (SizeType{1} << k) * (seg_len + 1) * 2 * sizeof(double);
                }
                if (bytes <= budget) {
                    return ConeChoice{
                        .k    = k,
                        .tile = tile,
                        .cone = true,
                        .done = done,
                    };
                }
            }
        }
        return plain;
    }

    void log_info() {
        // Log iniital and final fold shapes
        const auto& fold_shapes = m_ffa_plan.get_fold_shapes();
        spdlog::info("P-FFA [{}] -> [{}]", fmt::join(fold_shapes.front(), ", "),
                     fmt::join(fold_shapes.back(), ", "));
        //  Log memory usage
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info("FFA Memory Usage: {:.2f} GB + {:.2f} GB (coords)",
                     memory_buffer_gb, memory_coord_gb);
    }

    void initialize_brute_fold() {
        const auto t_ref =
            m_is_freq_only ? 0.0 : m_ffa_plan.get_tsegments()[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.compute_param_grid(0).back();

        // Check if we need lossy initialization (ComplexType with large nbins)
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            if (m_cfg.get_nbins() > m_cfg.get_nbins_min_lossy_bf()) {
                m_use_lossy_init = true;
                m_the_bf_float   = std::make_unique<BruteFold<float>>(
                    freqs_arr, m_ffa_plan.get_segment_lens()[0],
                    m_cfg.get_nbins(), m_cfg.get_nsamps(), m_cfg.get_tsamp(),
                    t_ref, Exec::cpu(m_nthreads));
                spdlog::debug(
                    "Using lossy initialization (time->freq) for nbins={}",
                    m_cfg.get_nbins());
                return;
            }
        }

        // Normal initialization
        m_the_bf = std::make_unique<BruteFold<FoldType>>(
            freqs_arr, m_ffa_plan.get_segment_lens()[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref,
            Exec::cpu(m_nthreads));
    }

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    FoldType* init_buffer,
                    FoldType* temp_buffer) {
        timing::SimpleTimer timer;
        timer.start();
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            if (m_use_lossy_init) {
                // Lossy path: use time-domain BruteFold, then RFFT to frequency
                // domain
                const auto brute_fold_size_time =
                    m_the_bf_float->get_fold_size();

                // Use temp_buffer for time-domain output
                // temp_buffer is ComplexType*, reinterpret as float* for
                // time-domain data
                auto const real_temp_view =
                    std::span<float>(reinterpret_cast<float*>(temp_buffer),
                                     brute_fold_size_time);
                m_the_bf_float->execute(ts_e, ts_v, real_temp_view);

                // Out-of-place RFFT from temp_buffer (real) to init_buffer
                // (complex)
                const auto nfft = brute_fold_size_time / m_cfg.get_nbins();
                const auto brute_fold_size_fourier =
                    nfft * ((m_cfg.get_nbins() / 2) + 1);
                get_fft().rfft_batch(real_temp_view,
                                     std::span<ComplexType>(
                                         init_buffer, brute_fold_size_fourier),
                                     nfft, m_cfg.get_nbins(), m_nthreads);
                m_brutefold_time += timer.stop();
                return;
            }
        }
        // Normal path (float or ComplexType with nbins <= 64)
        m_the_bf->execute(ts_e, ts_v,
                          std::span(init_buffer, m_the_bf->get_fold_size()));
        m_brutefold_time += timer.stop();
    }

    /**
     * @brief Number of FFA merge levels to fuse into the brute fold.
     *
     * Fusion processes tiles of 2^k adjacent brute-fold segments entirely in
     * a per-thread cache-resident scratch (see
     * BruteFold::execute_fused_freq), saving the DRAM round trips of the first
     * k merge levels. It is limited to the time-domain, frequency-only path,
     * and k is the largest value for which one thread's two scratch buffers
     * stay within kFuseScratchBytes while keeping at least one tile per
     * thread.
     *
     * @return 0 when fusion does not apply.
     */
    [[nodiscard]] SizeType get_fuse_levels(SizeType levels) const {
        if constexpr (!std::is_same_v<FoldType, float>) {
            return 0;
        } else {
            // Per-thread scratch budget (each of the two ping-pong buffers).
            constexpr SizeType kFuseScratchBytes = 512ULL * 1024ULL;
            constexpr SizeType kFuseMaxLevels    = 6;
            if (!m_is_freq_only || levels < 2 || m_use_lossy_init ||
                !m_the_bf) {
                return 0;
            }
            if (m_fuse_levels_override.has_value()) {
                const SizeType nseg0 = m_ffa_plan.get_fold_shapes_time()[0][0];
                SizeType forced = std::min(*m_fuse_levels_override, levels - 1);
                while (forced > 0 && (nseg0 % (SizeType{1} << forced)) != 0) {
                    --forced;
                }
                return forced;
            }
            const auto ncoords     = m_ffa_plan.get_ncoords();
            const auto nbins       = m_cfg.get_nbins();
            const SizeType nseg0   = m_ffa_plan.get_fold_shapes_time()[0][0];
            const auto nthreads    = static_cast<SizeType>(m_nthreads);
            const SizeType k_limit = std::min(kFuseMaxLevels, levels - 1);
            SizeType best          = 0;
            for (SizeType k = 1; k <= k_limit; ++k) {
                if ((nseg0 >> k) < nthreads ||
                    (nseg0 % (SizeType{1} << k)) != 0) {
                    break;
                }
                SizeType tile_floats = 0;
                for (SizeType j = 0; j <= k; ++j) {
                    tile_floats =
                        std::max(tile_floats, (SizeType{1} << (k - j)) *
                                                  ncoords[j] * 2 * nbins);
                }
                if (tile_floats * sizeof(float) > kFuseScratchBytes) {
                    break;
                }
                best = k;
            }
            return best;
        }
    }

    /// Fused brute fold + first `fuse_levels` merges (time domain, freq-only).
    void initialize_fused(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          FoldType* level_out,
                          SizeType fuse_levels) {
        if constexpr (std::is_same_v<FoldType, float>) {
            timing::SimpleTimer timer;
            timer.start();
            const auto& ws = get_workspace();
            std::vector<const coord::FFACoordFreq*> coords_levels(fuse_levels +
                                                                  1);
            const auto ncoords         = m_ffa_plan.get_ncoords();
            const auto ncoords_offsets = m_ffa_plan.get_ncoords_offsets();
            for (SizeType j = 1; j <= fuse_levels; ++j) {
                coords_levels[j] = ws.coords_freq.data() + ncoords_offsets[j];
            }
            const SizeType nseg_out =
                m_ffa_plan.get_fold_shapes_time()[0][0] >> fuse_levels;
            const SizeType out_size =
                nseg_out * ncoords[fuse_levels] * 2 * m_cfg.get_nbins();
            m_the_bf->execute_fused_freq(
                ts_e, ts_v, std::span<float>(level_out, out_size),
                coords_levels, ncoords.first(fuse_levels + 1), fuse_levels);
            // Reported as brute-fold time: it covers the brute fold plus the
            // first `fuse_levels` merges.
            m_brutefold_time += timer.stop();
        }
    }

    void execute_unified(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<FoldType> fold,
                         bool output_in_internal_buffer = false) {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        error_check::check_greater_equal(
            levels, 2,
            "FFA::execute_unified: levels must be greater than or equal "
            "to 2");

        auto& ws = get_workspace();
        // Use fold_internal from workspace and output fold for ping-pong
        FoldType* fold_internal_ptr = ws.fold_internal.data();
        FoldType* fold_result_ptr   = fold.data();

        FoldType* current_in_ptr  = nullptr;
        FoldType* current_out_ptr = nullptr;

        // Number of internal ping-pong iterations (excluding the final write)
        const SizeType internal_iters = levels - 2;
        // Determine starting configuration to ensure final result lands in the
        // correct side of the ping-pong table
        const bool odd_swaps        = (internal_iters % 2) == 1;
        const bool init_in_internal = (odd_swaps == output_in_internal_buffer);
        if (init_in_internal) {
            // init -> internal,
            // odd swaps -> ends in result, even swaps -> ends in internal
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        } else {
            // init -> result,
            // even swaps -> ends in result, odd swaps -> ends in internal
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        }

        // Level to resume the plain level-by-level merge from.
        SizeType first_level       = 1;
        const SizeType fuse_levels = get_fuse_levels(levels);
        m_last_fuse_levels         = fuse_levels;
        if (fuse_levels > 0) {
            // Fused brute fold + first `fuse_levels` merges. In the plain flow
            // level i writes to `out_i = (i odd) ? out0 : in0`, so the fused
            // result must land there for level fuse_levels + 1 to read it.
            FoldType* fused_out =
                (fuse_levels % 2 == 1) ? current_out_ptr : current_in_ptr;
            FoldType* fused_other = (fused_out == current_in_ptr)
                                        ? current_out_ptr
                                        : current_in_ptr;
            initialize_fused(ts_e, ts_v, fused_out, fuse_levels);
            current_in_ptr  = fused_out;
            current_out_ptr = fused_other;
            first_level     = fuse_levels + 1;
        } else {
            // Initialize in the current buffer (using optional temp buffer)
            const float brute_before = m_brutefold_time;
            initialize(ts_e, ts_v, current_in_ptr, current_out_ptr);
            if (m_level_timing && m_is_freq_only) {
                const auto nseg0    = m_ffa_plan.get_fold_shapes_time()[0][0];
                const auto ncoord0  = m_ffa_plan.get_ncoords().front();
                const double floats = static_cast<double>(nseg0) * ncoord0 *
                                      2.0 * m_cfg.get_nbins();
                log_level_timing(
                    "brute", 0, floats,
                    static_cast<double>(m_brutefold_time - brute_before));
            }
        }

        progress::ProgressGuard const progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        if (m_is_freq_only) {
            for (SizeType i_level = first_level; i_level < levels; ++i_level) {
                timing::SimpleTimer level_timer;
                level_timer.start();
                execute_iter_freq(current_in_ptr, current_out_ptr, i_level);
                if (m_level_timing) {
                    const auto nseg =
                        m_ffa_plan.get_fold_shapes_time()[i_level][0];
                    const auto ncoord   = m_ffa_plan.get_ncoords()[i_level];
                    const double floats = static_cast<double>(nseg) * ncoord *
                                          2.0 * m_cfg.get_nbins();
                    log_level_timing("merge", i_level, floats,
                                     static_cast<double>(level_timer.stop()));
                }
                // Ping-pong buffers (unless it's the final iteration)
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
                if (m_show_progress) {
                    bar->set_leaves(m_ffa_plan.get_ncoords_lb()[i_level]);
                    bar->set_progress(i_level);
                }
            }
        } else {
            for (SizeType i_level = first_level; i_level < levels; ++i_level) {
                execute_iter(current_in_ptr, current_out_ptr, i_level);
                // Ping-pong buffers (unless it's the final iteration)
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
                if (m_show_progress) {
                    bar->set_leaves(m_ffa_plan.get_ncoords_lb()[i_level]);
                    bar->set_progress(i_level);
                }
            }
        }
        bar->mark_as_completed();
    }

    void execute_iter_freq(const FoldType* __restrict__ fold_in,
                           FoldType* __restrict__ fold_out,
                           SizeType i_level) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        const auto& ws = get_workspace();
        auto const coords_cur_span =
            std::span(ws.coords_freq).subspan(ncoords_offset, ncoords_cur);
        if constexpr (std::is_same_v<FoldType, float>) {
            core::ffa_iter_freq(fold_in, fold_out, coords_cur_span.data(),
                                ncoords_cur, ncoords_prev, nsegments, nbins,
                                m_nthreads);
        } else {
            core::ffa_complex_iter_freq(
                fold_in, fold_out, coords_cur_span.data(), ncoords_cur,
                ncoords_prev, nsegments, nbins_f, nbins, m_nthreads);
        }
    }

    void execute_iter(const FoldType* __restrict__ fold_in,
                      FoldType* __restrict__ fold_out,
                      SizeType i_level) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        const auto& ws = get_workspace();
        auto const coords_cur_span =
            std::span(ws.coords).subspan(ncoords_offset, ncoords_cur);

        if constexpr (std::is_same_v<FoldType, float>) {
            core::ffa_iter(fold_in, fold_out, coords_cur_span.data(),
                           ncoords_cur, ncoords_prev, nsegments, nbins,
                           m_nthreads);

        } else {
            core::ffa_complex_iter(fold_in, fold_out, coords_cur_span.data(),
                                   ncoords_cur, ncoords_prev, nsegments,
                                   nbins_f, nbins, m_nthreads);
        }
    }
}; // End FFACpuEngine definition

namespace detail {

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(const search::FFASearchConfig& cfg, bool show_progress) {
    return std::make_unique<FFACpuEngine<FoldType>>(cfg, show_progress);
}

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(memory::FFAWorkspaceCPU<FoldType>& workspace,
             math::FFTWManager& fft_manager,
             const search::FFASearchConfig& cfg,
             bool show_progress) {
    return std::make_unique<FFACpuEngine<FoldType>>(workspace, fft_manager, cfg,
                                                    show_progress);
}

template std::unique_ptr<FFAEngine<float>>
make_ffa_cpu<float>(const search::FFASearchConfig&, bool);
template std::unique_ptr<FFAEngine<ComplexType>>
make_ffa_cpu<ComplexType>(const search::FFASearchConfig&, bool);

template std::unique_ptr<FFAEngine<float>>
make_ffa_cpu<float>(memory::FFAWorkspaceCPU<float>&,
                    math::FFTWManager&,
                    const search::FFASearchConfig&,
                    bool);
template std::unique_ptr<FFAEngine<ComplexType>>
make_ffa_cpu<ComplexType>(memory::FFAWorkspaceCPU<ComplexType>&,
                          math::FFTWManager&,
                          const search::FFASearchConfig&,
                          bool);

} // namespace detail

} // namespace loki::algorithms
