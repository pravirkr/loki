#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ffa_engine.hpp"
#include "lib/algorithms/planner_memory.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/core/kernels.hpp"
#include "lib/core/taylor_ffa.hpp"
#include "lib/cpu/boxcar_kernels.hpp"
#include "lib/cpu/lane_fft.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/progress.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/score_engine.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms {

namespace {

struct ConeScoreCtx {
    std::span<const SizeType> widths;
    float threshold{0.0F};
    // One scorer per thread, built on its first tile.
    std::vector<std::optional<detection::detail::BoxcarScorer3D>> scorers;
    std::vector<std::vector<detection::SnrHit>> hits;
    // Per-thread scoring seconds, kept only when `timed` is set.
    std::vector<double> seconds;
    bool timed{false};

    ConeScoreCtx() = default;
    ConeScoreCtx(std::span<const SizeType> widths_in,
                 float threshold_in,
                 int nthreads,
                 bool timed_in)
        : widths(widths_in),
          threshold(threshold_in),
          scorers(static_cast<SizeType>(std::max(nthreads, 1))),
          hits(static_cast<SizeType>(std::max(nthreads, 1))),
          seconds(static_cast<SizeType>(std::max(nthreads, 1)), 0.0),
          timed(timed_in) {}

    /// Hits of every thread, in thread order (coordinate order for static
    /// contiguous partitions).
    void collect(std::vector<detection::SnrHit>& out) const {
        for (const auto& part : hits) {
            out.insert(out.end(), part.begin(), part.end());
        }
    }
    [[nodiscard]] double max_seconds() const {
        return seconds.empty() ? 0.0 : *std::ranges::max_element(seconds);
    }
};

void cone_score_tile(const float* profiles,
                     SizeType coord_begin,
                     SizeType nprofiles,
                     SizeType nbins,
                     void* ctx) {
    auto* state    = static_cast<ConeScoreCtx*>(ctx);
    const auto tid = static_cast<SizeType>(omp_get_thread_num());
    timing::SimpleTimer timer;
    if (state->timed) {
        timer.start();
    }
    auto& scorer = state->scorers[tid];
    if (!scorer.has_value()) {
        scorer.emplace(nbins, state->widths.data(), state->widths.size());
    }
    detection::detail::append_boxcar_hits(*scorer, profiles, coord_begin,
                                          nprofiles, state->threshold,
                                          state->hits[tid]);
    if (state->timed) {
        state->seconds[tid] += static_cast<double>(timer.stop());
    }
}

/// A score callback whose coordinates start at `first` of the level (a batch
/// of final-level rows resolved on the fly).
struct OffsetScoreCtx {
    core::ConeScoreFn fn;
    void* ctx;
    SizeType first;
};

void offset_score_tile(const float* profiles,
                       SizeType coord_begin,
                       SizeType nprofiles,
                       SizeType nbins,
                       void* ctx) {
    const auto& state = *static_cast<const OffsetScoreCtx*>(ctx);
    state.fn(profiles, state.first + coord_begin, nprofiles, nbins, state.ctx);
}

struct OffsetTileCtx {
    core::ComplexTileFn fn;
    void* ctx;
    SizeType first;
};

void offset_tile(ComplexType* tile,
                 SizeType coord_begin,
                 SizeType count,
                 void* ctx) {
    const auto& state = *static_cast<const OffsetTileCtx*>(ctx);
    state.fn(tile, state.first + coord_begin, count, state.ctx);
}

/// Inverse-FFT stage of the fused Fourier final level: the lane FFT, whose
/// output the shared boxcar scorer of ConeScoreCtx reads in place.
struct FourierScoreCtx {
    ConeScoreCtx* score;
    SizeType nbins;
    SizeType nbins_f;
    std::vector<std::optional<lane_fft::EVTransform>> lanes;

    FourierScoreCtx(ConeScoreCtx* score_in,
                    SizeType nbins_in,
                    SizeType nbins_f_in,
                    int nthreads)
        : score(score_in),
          nbins(nbins_in),
          nbins_f(nbins_f_in),
          lanes(static_cast<SizeType>(std::max(nthreads, 1))) {}
};

void fourier_score_tile(ComplexType* tile,
                        SizeType coord_begin,
                        SizeType count,
                        void* ctx) {
    auto& state         = *static_cast<FourierScoreCtx*>(ctx);
    ConeScoreCtx& score = *state.score;
    const auto tid      = static_cast<SizeType>(omp_get_thread_num());
    timing::SimpleTimer timer;
    if (score.timed) {
        timer.start();
    }
    auto& transform = state.lanes[tid];
    if (!transform.has_value()) {
        transform.emplace(state.nbins);
    }
    auto& scorer = score.scorers[tid];
    if (!scorer.has_value()) {
        scorer.emplace(state.nbins, score.widths.data(), score.widths.size());
    }
    auto& hits             = score.hits[tid];
    const float threshold  = score.threshold;
    const SizeType nwidths = scorer->nwidths();
    for (SizeType p0 = 0; p0 < count; p0 += lane_fft::kLanes) {
        const SizeType nvalid = std::min(lane_fft::kLanes, count - p0);
        transform->inverse(tile + (p0 * 2 * state.nbins_f), nvalid);
        const SizeType base = coord_begin + p0;
        scorer->score_lanes(transform->e_lanes(), transform->v_lanes(), nvalid,
                            [&hits, threshold, base,
                             nwidths](SizeType p, SizeType iw, float snr) {
                                if (snr >= threshold) {
                                    hits.push_back(detection::SnrHit{
                                        .score_index = static_cast<uint32_t>(
                                            ((base + p) * nwidths) + iw),
                                        .snr = snr,
                                    });
                                }
                            });
    }
    if (score.timed) {
        score.seconds[tid] += static_cast<double>(timer.stop());
    }
}

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
          m_workspace_ptr(&m_workspace_storage) {
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
        log_info(/*scored=*/false);
    }

    explicit FFACpuEngine(memory::FFAWorkspaceCPU<FoldType>& workspace,
                          search::FFASearchConfig cfg,
                          bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_workspace_storage(),
          m_workspace_ptr(&workspace) {
        // A shared (sweep) workspace may be sized for scored execution only;
        // each execute path checks the fold buffer it needs.
        const auto& ws = get_workspace();
        ws.validate_coords(m_ffa_plan,
                           detail::cpu_scored_coord_size(m_ffa_plan));
        // Initialize BruteFold (table build time is tracked separately)
        timing::SimpleTimer init_timer;
        init_timer.start();
        initialize_brute_fold();
        m_brutefold_init_time = init_timer.stop();
        apply_fuse_levels_env();
        apply_level_timing_env();
        log_info(/*scored=*/true);
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
        ws.validate_buffer(m_ffa_plan.get_buffer_size());
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
        execute_scored(ts_e, ts_v, threshold, widths, hits, {});
    }

    void execute_scored(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        float threshold,
                        std::span<const SizeType> widths,
                        std::vector<detection::SnrHit>& hits,
                        std::span<FoldType> scratch) override {
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
        // Hits carry a 32-bit score index (coord * nwidths + iw).
        error_check::check_less_equal(
            static_cast<std::uint64_t>(m_ffa_plan.get_ncoords().back()) *
                widths.size(),
            static_cast<std::uint64_t>(std::numeric_limits<uint32_t>::max()),
            "FFA::execute_scored: ncoords * nwidths exceeds the 32-bit "
            "score index");
        const bool fusion_off =
            m_fuse_levels_override.has_value() && *m_fuse_levels_override == 0;
        // Time-domain frequency-only plans that the cone bands decline (e.g.
        // base segments too long for the cone scratch) have no fused final
        // kernel: run_final_scored reads accel coordinates. Use the stored
        // fold; the planner already sizes the full buffer for these plans.
        const bool freq_only_time =
            std::is_same_v<FoldType, float> && m_is_freq_only;
        const bool fused =
            m_cfg.get_niters_ffa() >= 1 && !fusion_off && !freq_only_time;
        auto& ws = get_workspace();
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws.coords_freq);
        } else if (fused && detail::cpu_final_coords_on_the_fly(m_ffa_plan)) {
            // run_final_scored resolves the final level batch by batch.
            resolve_coordinates_but_final(ws.coords);
        } else {
            ws.validate_coords(m_ffa_plan);
            m_ffa_plan.resolve_coordinates(ws.coords);
        }
        hits.clear();
        m_last_score_time = 0.0F;
        if constexpr (std::is_same_v<FoldType, float>) {
            if (m_is_freq_only &&
                run_cone_bands(ts_e, ts_v, nullptr, &hits, threshold, widths)) {
                return;
            }
        }
        // Fused: the final level is never stored (detail::
        // cpu_scored_buffer_size, which the sweep planner also uses).
        const auto buffer_size =
            fused ? detail::cpu_scored_buffer_size(m_ffa_plan)
                  : m_ffa_plan.get_buffer_size();
        ws.validate_buffer(buffer_size);
        utils::UninitVector<FoldType> owned;
        if (scratch.size() < buffer_size) {
            owned.resize(buffer_size);
            scratch = std::span<FoldType>(owned);
        }
        scratch = scratch.first(buffer_size);
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            if (m_is_freq_only &&
                run_cone_bands_fourier(ts_e, ts_v, scratch, threshold, widths,
                                       hits)) {
                return;
            }
        }
        if (fused) {
            run_final_scored(ts_e, ts_v, scratch, threshold, widths, hits);
            return;
        }
        // Store the final fold and scan it (the reference path).
        const auto ncoords = m_ffa_plan.get_ncoords().back();
        const auto nbins   = m_cfg.get_nbins();
        utils::UninitVector<float> fold_time;
        std::span<const float> fold;
        if constexpr (std::is_same_v<FoldType, float>) {
            execute_unified(ts_e, ts_v, scratch);
            fold = std::span<const float>(scratch.data(), ncoords * 2 * nbins);
        } else {
            fold_time.resize(2 * buffer_size);
            execute_return_to_time_impl(ts_e, ts_v, fold_time);
            fold = std::span<const float>(fold_time).first(ncoords * 2 * nbins);
        }
        std::vector<float> scores(ncoords * widths.size());
        detection::detail::snr_boxcar_3d_cpu(fold, widths, scores, ncoords,
                                             nbins, m_nthreads);
        for (SizeType i = 0; i < scores.size(); ++i) {
            if (scores[i] >= threshold) {
                hits.push_back(detection::SnrHit{
                    .score_index = static_cast<uint32_t>(i),
                    .snr         = scores[i],
                });
            }
        }
    }

    /**
     * @brief Merge every level but the last, then merge the final level tile
     * by tile into scratch and score it there; the final fold is never
     * stored.
     *
     * Time domain (accel): the same sums and scorer as execute() +
     * snr_boxcar_3d, so the hits are bit-identical to a threshold scan of the
     * stored fold. Fourier domain (accel and frequency-only): each tile is
     * inverted with the lane FFT and scored straight from its lane layout.
     */
    void run_final_scored(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          std::span<FoldType> scratch,
                          float threshold,
                          std::span<const SizeType> widths,
                          std::vector<detection::SnrHit>& hits) {
        const SizeType levels = m_cfg.get_niters_ffa() + 1;
        const SizeType final  = levels - 1;
        // Accel: levels L-1 and L both merged in scratch (two_levels).
        const bool two_levels =
            detail::cpu_scored_fused_levels(m_ffa_plan) == 2;
        // Levels merged in scratch and never stored (reported per chunk).
        m_last_fuse_levels = two_levels ? 2 : 1;
        const FoldType* src =
            execute_unified(ts_e, ts_v, scratch, false,
                            /*end_level=*/two_levels ? final - 1 : final);
        const auto nbins       = m_cfg.get_nbins();
        const auto ncoords_cur = m_ffa_plan.get_ncoords()[final];
        const auto offset      = m_ffa_plan.get_ncoords_offsets()[final];
        auto& ws               = get_workspace();
        // Final coordinates are rows of frequencies (one per accel or jerk
        // trial); the kernels tile rows that share parent rows together.
        const auto row_len = m_ffa_plan.get_param_counts()[final].back();
        const auto row_len_prev =
            m_ffa_plan.get_param_counts()[final - 1].back();
        // Accel plans resolve the final level a batch of rows at a time into
        // the slot after the stored levels (detail::cpu_scored_coord_size);
        // other plans read the stored level in one batch.
        const bool on_the_fly = detail::cpu_final_coords_on_the_fly(m_ffa_plan);
        const SizeType nrows  = ncoords_cur / row_len;
        const SizeType batch_rows =
            on_the_fly ? final_coord_batch_rows() : nrows;
        ConeScoreCtx score_ctx(widths, threshold, m_nthreads,
                               /*timed=*/true);
        timing::SimpleTimer timer;
        timer.start();
        for (SizeType row0 = 0; row0 < nrows; row0 += batch_rows) {
            const SizeType row1   = std::min(nrows, row0 + batch_rows);
            const SizeType nbatch = (row1 - row0) * row_len;
            const SizeType first  = row0 * row_len;
            // Frequency-only plans read coords_freq (ws.coords is empty).
            coord::FFACoord* coords =
                m_is_freq_only ? nullptr : ws.coords.data() + offset;
            if (on_the_fly) {
                resolve_final_rows(std::span(coords, nbatch), row0, row1);
            } else if (coords != nullptr) {
                coords += first;
            }
            run_final_batch(src, coords, nbatch, first, two_levels, row_len,
                            row_len_prev, score_ctx);
        }
        const auto wall = static_cast<double>(timer.stop());
        log_level_timing("final+score", final,
                         static_cast<double>(ncoords_cur) * 2.0 *
                             static_cast<double>(nbins),
                         wall);
        score_ctx.collect(hits);
        if (!m_is_freq_only) {
            // Accel tiles are visited row group by row group.
            std::ranges::sort(hits, {}, &detection::SnrHit::score_index);
        }
        m_last_score_time =
            static_cast<float>(std::min(wall, score_ctx.max_seconds()));
    }

    /// Final-level rows per on-the-fly batch: the planned batch, or fewer
    /// with LOKI_FINAL_COORD_BATCH_KB (test hook for the batch edges).
    [[nodiscard]] SizeType final_coord_batch_rows() const {
        const SizeType planned = detail::cpu_final_coord_batch_rows(m_ffa_plan);
        // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
        const char* env = std::getenv("LOKI_FINAL_COORD_BATCH_KB");
        if (env == nullptr || env[0] == '\0') {
            return planned;
        }
        char* end = nullptr;
        // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
        const unsigned long kb = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0' || kb == 0) {
            spdlog::warn("Ignoring invalid LOKI_FINAL_COORD_BATCH_KB='{}'",
                         env);
            return planned;
        }
        return std::min(planned,
                        detail::cpu_final_coord_batch_rows(
                            m_ffa_plan, static_cast<SizeType>(kb) << 10U));
    }

    /// Resolve every level's coordinates except the final one (accel plans
    /// that resolve the final level on the fly).
    void resolve_coordinates_but_final(std::span<coord::FFACoord> coords) {
        const SizeType final = m_ffa_plan.get_n_levels() - 1;
        error_check::check_greater_equal(
            coords.size(), detail::cpu_scored_coord_size(m_ffa_plan),
            "FFA::execute_scored: coordinates not allocated for enough "
            "levels");
        const auto& counts  = m_ffa_plan.get_param_counts();
        const auto ncoords  = m_ffa_plan.get_ncoords();
        const auto offsets  = m_ffa_plan.get_ncoords_offsets();
        const auto n_params = m_ffa_plan.get_n_params();
        for (SizeType i_level = 1; i_level < final; ++i_level) {
            const auto level_coords =
                coords.subspan(offsets[i_level], ncoords[i_level]);
            for (SizeType latter = 0; latter < 2; ++latter) {
                core::ffa_taylor_resolve_poly_batch(
                    counts[i_level], counts[i_level - 1],
                    m_cfg.get_param_limits(), level_coords, i_level, latter,
                    m_cfg.get_tseg_brute(), m_cfg.get_nbins(), n_params,
                    m_nthreads);
            }
        }
    }

    /// Resolve rows [row0, row1) of the final level into `coords`.
    void resolve_final_rows(std::span<coord::FFACoord> coords,
                            SizeType row0,
                            SizeType row1) const {
        const SizeType final = m_ffa_plan.get_n_levels() - 1;
        const auto& counts   = m_ffa_plan.get_param_counts();
        core::ffa_taylor_resolve_accel_rows(
            counts[final], counts[final - 1], m_cfg.get_param_limits(), coords,
            final, m_cfg.get_tseg_brute(), m_cfg.get_nbins(), row0, row1,
            m_nthreads);
    }

    /**
     * @brief Merge (and score) `nbatch` final-level coordinates, starting at
     * coordinate `first` of the level; `coords` holds exactly those.
     */
    void run_final_batch(const FoldType* src,
                         const coord::FFACoord* coords,
                         SizeType nbatch,
                         SizeType first,
                         bool two_levels,
                         SizeType row_len,
                         SizeType row_len_prev,
                         ConeScoreCtx& score_ctx) {
        const SizeType final    = m_cfg.get_niters_ffa();
        const auto nbins        = m_cfg.get_nbins();
        const auto ncoords_cur  = nbatch;
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[final - 1];
        const auto& ws          = get_workspace();
        if constexpr (std::is_same_v<FoldType, float>) {
            error_check::check(!m_is_freq_only,
                               "FFA::run_final_scored: no fused final level "
                               "for time-domain frequency-only plans");
            OffsetScoreCtx shifted{
                .fn = &cone_score_tile, .ctx = &score_ctx, .first = first};
            // About 64 KiB of profiles per tile: L2-resident while scored.
            const SizeType tile =
                std::max<SizeType>(16, (SizeType{16} * 1024) / (2 * nbins));
            if (two_levels) {
                core::ffa_final2_scored(
                    src,
                    ws.coords.data() +
                        m_ffa_plan.get_ncoords_offsets()[final - 1],
                    coords, ncoords_cur, m_ffa_plan.get_ncoords()[final - 2],
                    nbins, row_len, row_len_prev, tile, &offset_score_tile,
                    &shifted, m_nthreads);
            } else {
                core::ffa_final_scored(src, coords, ncoords_cur, ncoords_prev,
                                       nbins, row_len, row_len_prev, tile,
                                       &offset_score_tile, &shifted,
                                       m_nthreads);
            }
        } else {
            // A multiple of the merge step block and of the lane FFT width;
            // about 128 KiB of real profiles.
            const SizeType tile = std::clamp<SizeType>(
                std::bit_floor((SizeType{32} * 1024) / (2 * nbins)), 32, 512);
            FourierScoreCtx ctx(&score_ctx, nbins, m_cfg.get_nbins_f(),
                                m_nthreads);
            OffsetTileCtx shifted{
                .fn = &fourier_score_tile, .ctx = &ctx, .first = first};
            if (m_is_freq_only) {
                const auto offset = m_ffa_plan.get_ncoords_offsets()[final];
                core::ffa_complex_final_tiles_freq(
                    src, ws.coords_freq.data() + offset, ncoords_cur,
                    ncoords_prev, m_cfg.get_nbins_f(), nbins, tile,
                    &fourier_score_tile, &ctx, m_nthreads);
            } else if (two_levels) {
                core::ffa_complex_final2_tiles(
                    src,
                    ws.coords.data() +
                        m_ffa_plan.get_ncoords_offsets()[final - 1],
                    coords, ncoords_cur, ncoords_prev,
                    m_ffa_plan.get_ncoords()[final - 2], m_cfg.get_nbins_f(),
                    nbins, row_len, row_len_prev, tile, &offset_tile, &shifted,
                    m_nthreads);
            } else {
                core::ffa_complex_final_tiles(
                    src, coords, ncoords_cur, ncoords_prev, m_cfg.get_nbins_f(),
                    nbins, row_len, row_len_prev, tile, &offset_tile, &shifted,
                    m_nthreads);
            }
        }
    }

    void execute_return_to_time_impl(std::span<const float> ts_e,
                                     std::span<const float> ts_v,
                                     std::span<float> fold) {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            const auto fold_size_time      = m_ffa_plan.get_fold_size_time();
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
            ws.validate_buffer(buffer_size_fourier);
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
            timing::SimpleTimer irfft_timer;
            irfft_timer.start();
            const auto nfft = fold_size_time / m_cfg.get_nbins();
            // The fused final level's transform, so its scores match a scan
            // of this fold bit for bit.
            lane_fft::inverse_batch(ws.fold_internal.data(), fold.data(), nfft,
                                    m_cfg.get_nbins(), m_nthreads);
            log_level_timing("irfft", m_cfg.get_niters_ffa(),
                             static_cast<double>(fold_size_time),
                             static_cast<double>(irfft_timer.stop()));
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

    [[nodiscard]] memory::FFAWorkspaceCPU<FoldType>& get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::FFAWorkspaceCPU<FoldType>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    /// Benchmark hook: LOKI_FUSE_LEVELS=<n> forces the fusion depth (0
    /// disables).
    void apply_fuse_levels_env() {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
        const char* env = std::getenv("LOKI_FUSE_LEVELS");
        if (env == nullptr || env[0] == '\0') {
            return;
        }
        char* end = nullptr;
        // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
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
        // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
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
                score_ctx = ConeScoreCtx(widths, threshold, m_nthreads,
                                         /*timed=*/false);
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
                score_ctx.collect(*hits);
            }
            bar->mark_as_completed();
            return true;
        }
    }

    [[nodiscard]] static SizeType cone_scratch_budget_bytes() {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
        const char* env = std::getenv("LOKI_CONE_SCRATCH_KB");
        // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
        unsigned long kb = 2048;
        if (env != nullptr && env[0] != '\0') {
            char* end = nullptr;
            // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
            const unsigned long v = std::strtoul(env, &end, 10);
            if (end != env && *end == '\0' && v > 0) {
                kb = v;
            }
        }
        return static_cast<SizeType>(kb) * 1024U;
    }

    [[nodiscard]] static SizeType cone_tile_override() {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
        const char* env = std::getenv("LOKI_CONE_TILE");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
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

    /**
     * @brief Fourier frequency-only scored execution by cone bands (temporal
     * tiling): the brute fold is stored, then groups of levels are merged
     * tile by tile in cache and the top band is scored in place. Same merges
     * and phase steps as the level-by-level path (bit-identical). Returns
     * false (nothing done) when no band fits the scratch budget or the top
     * band would not reach the final level.
     */
    [[nodiscard]] bool
    run_cone_bands_fourier(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           std::span<FoldType> scratch,
                           float threshold,
                           std::span<const SizeType> widths,
                           std::vector<detection::SnrHit>& hits) {
        if constexpr (!std::is_same_v<FoldType, ComplexType>) {
            return false;
        } else {
            if (!m_is_freq_only) {
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
            const SizeType nbins_f     = m_cfg.get_nbins_f();
            const SizeType budget      = cone_scratch_budget_bytes();
            const SizeType tile_forced = cone_tile_override();
            auto& ws                   = get_workspace();

            std::vector<ConeChoice> steps;
            SizeType done = 0;
            while (done < n_merge) {
                const ConeChoice step = choose_cone_step_complex(
                    done, n_merge - done, ncoords, offsets, shapes, nbins_f,
                    budget, tile_forced, ws.coords_freq.data());
                steps.push_back(step);
                done += step.k;
            }
            // The top band scores the final level in cache.
            if (!steps.back().cone ||
                std::ranges::none_of(steps, &ConeChoice::cone)) {
                return false;
            }
            m_last_fuse_levels = steps.front().cone ? steps.front().k : 0;

            ConeScoreCtx score_ctx(widths, threshold, m_nthreads,
                                   /*timed=*/true);
            FourierScoreCtx ctx(&score_ctx, nbins, nbins_f, m_nthreads);
            ComplexType* buf_a = ws.fold_internal.data();
            ComplexType* buf_b = scratch.data();
            initialize(ts_e, ts_v, buf_a, buf_b);
            const ComplexType* current = buf_a;
            for (SizeType istep = 0; istep < steps.size(); ++istep) {
                const ConeChoice& step = steps[istep];
                const bool last        = istep + 1 == steps.size();
                ComplexType* dest =
                    last ? nullptr : (current == buf_a ? buf_b : buf_a);
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
                    core::ffa_complex_cone_band_freq(
                        current, dest, coords.data(), counts.data(),
                        shapes[step.done][0], nbins_f, nbins, step.k, step.tile,
                        last ? &fourier_score_tile : nullptr,
                        last ? &ctx : nullptr, m_nthreads);
                } else {
                    execute_iter_freq(current, dest, step.done + 1);
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
                }
                if (dest != nullptr) {
                    current = dest;
                }
            }
            score_ctx.collect(hits);
            m_last_score_time = static_cast<float>(score_ctx.max_seconds());
            return true;
        }
    }

    /// choose_cone_step for complex folds: the band starts from a stored
    /// level, so only the two scratch buffers count.
    [[nodiscard]] ConeChoice
    choose_cone_step_complex(SizeType done,
                             SizeType remain,
                             std::span<const SizeType> ncoords,
                             std::span<const uint32_t> offsets,
                             const std::vector<std::vector<SizeType>>& shapes,
                             SizeType nbins_f,
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
        constexpr std::array<SizeType, 5> kTiles{512, 256, 128, 64, 32};
        const SizeType ntiles = tile_forced == 0 ? kTiles.size() : 1;
        for (SizeType k = k_cap; k >= 1; --k) {
            std::array<const coord::FFACoordFreq*, 17> coords{};
            std::array<SizeType, 17> counts{};
            counts[0] = ncoords[done];
            for (SizeType j = 1; j <= k; ++j) {
                counts[j] = ncoords[done + j];
                coords[j] = coords_base + offsets[done + j];
            }
            for (SizeType itile = 0; itile < ntiles; ++itile) {
                const SizeType tile =
                    tile_forced == 0 ? kTiles[itile] : tile_forced;
                const SizeType elems = core::cone_band_working_floats(
                    coords.data(), counts.data(), k, tile, nbins_f);
                if (elems != 0 && 2 * elems * sizeof(ComplexType) <= budget) {
                    return ConeChoice{
                        .k = k, .tile = tile, .cone = true, .done = done};
                }
            }
        }
        return plain;
    }

    /**
     * @brief Log the fold shapes and the memory this engine's path uses.
     *
     * A standalone engine (`scored` false) serves execute(): its workspace
     * and the caller's fold, each the plan's full buffer. A sweep engine on a
     * shared workspace (`scored` true) runs execute_scored() only, whose
     * ping-pong buffers exclude the levels merged in scratch and whose
     * coordinates exclude the final accel level (detail::
     * cpu_scored_buffer_size and cpu_scored_coord_size, as the sweep
     * planner sizes them).
     */
    void log_info(bool scored) {
        const auto& fold_shapes = m_ffa_plan.get_fold_shapes();
        spdlog::info("P-FFA [{}] -> [{}]", fmt::join(fold_shapes.front(), ", "),
                     fmt::join(fold_shapes.back(), ", "));
        const SizeType buffer_size =
            scored ? detail::cpu_scored_buffer_size(m_ffa_plan)
                   : m_ffa_plan.get_buffer_size();
        const auto memory_buffer_gb =
            static_cast<float>(2 * buffer_size * sizeof(FoldType)) /
            static_cast<float>(1ULL << 30U);
        // Accel scored execution holds one batch of final-level rows in
        // place of the final level (detail::cpu_scored_coord_size).
        const SizeType coord_size =
            scored ? detail::cpu_scored_coord_size(m_ffa_plan)
                   : m_ffa_plan.get_coord_size();
        const auto memory_coord_gb =
            m_ffa_plan.get_coord_size() == 0
                ? 0.0F
                : m_ffa_plan.get_coord_memory_usage() *
                      static_cast<float>(coord_size) /
                      static_cast<float>(m_ffa_plan.get_coord_size());
        spdlog::info("FFA Memory Usage{}: {:.2f} GB + {:.2f} GB (coords)",
                     scored ? " (scored)" : "", memory_buffer_gb,
                     memory_coord_gb);
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

                // Forward transform from temp_buffer (real) to init_buffer
                // (complex)
                const auto nfft = brute_fold_size_time / m_cfg.get_nbins();
                lane_fft::forward_batch(real_temp_view.data(), init_buffer,
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

    /**
     * @brief Brute fold and merge levels [1, end_level) by ping-pong between
     * the workspace and `fold`; returns the buffer holding the last level.
     *
     * end_level = 0 runs every level. The ping-pong parity puts the full
     * result in `fold` (or in the workspace with output_in_internal_buffer).
     */
    FoldType* execute_unified(std::span<const float> ts_e,
                              std::span<const float> ts_v,
                              std::span<FoldType> fold,
                              bool output_in_internal_buffer = false,
                              SizeType end_level             = 0) {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        if (end_level == 0) {
            end_level = levels;
        }
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
            if (m_level_timing) {
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

        FoldType* last_written = current_in_ptr;
        if (m_is_freq_only) {
            for (SizeType i_level = first_level; i_level < end_level;
                 ++i_level) {
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
                last_written = current_out_ptr;
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
            for (SizeType i_level = first_level; i_level < end_level;
                 ++i_level) {
                timing::SimpleTimer level_timer;
                level_timer.start();
                execute_iter(current_in_ptr, current_out_ptr, i_level);
                if (m_level_timing) {
                    const auto nseg =
                        m_ffa_plan.get_fold_shapes_time()[i_level][0];
                    const auto ncoord   = m_ffa_plan.get_ncoords()[i_level];
                    const double floats = static_cast<double>(nseg) * ncoord *
                                          2.0 * m_cfg.get_nbins();
                    log_level_timing("merge", i_level, floats,
                                     static_cast<double>(level_timer.stop()));
                }
                last_written = current_out_ptr;
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
        return last_written;
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

} // namespace

namespace detail {

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(const search::FFASearchConfig& cfg, bool show_progress) {
    return std::make_unique<FFACpuEngine<FoldType>>(cfg, show_progress);
}

template <SupportedFoldType FoldType>
std::unique_ptr<FFAEngine<FoldType>>
make_ffa_cpu(memory::FFAWorkspaceCPU<FoldType>& workspace,
             const search::FFASearchConfig& cfg,
             bool show_progress) {
    return std::make_unique<FFACpuEngine<FoldType>>(workspace, cfg,
                                                    show_progress);
}

template std::unique_ptr<FFAEngine<float>>
make_ffa_cpu<float>(const search::FFASearchConfig&, bool);
template std::unique_ptr<FFAEngine<ComplexType>>
make_ffa_cpu<ComplexType>(const search::FFASearchConfig&, bool);

template std::unique_ptr<FFAEngine<float>> make_ffa_cpu<float>(
    memory::FFAWorkspaceCPU<float>&, const search::FFASearchConfig&, bool);
template std::unique_ptr<FFAEngine<ComplexType>>
make_ffa_cpu<ComplexType>(memory::FFAWorkspaceCPU<ComplexType>&,
                          const search::FFASearchConfig&,
                          bool);

} // namespace detail

} // namespace loki::algorithms
