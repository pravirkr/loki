#include "lib/algorithms/ep_chunking.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/common/plans.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/algorithms/prune_engine.hpp"

namespace loki::algorithms::detail {

namespace {

constexpr double kBisectionToleranceHz = 1.0e-2;
constexpr double kMinChunkWidthHz      = 1.0e-2;
constexpr double kRelativeTolerance    = 1.0e-4;
constexpr double kSliverFactor         = 4.0;
constexpr SizeType kMaxBisectionSteps  = 50;
/// Planning passes spent searching a smaller fixed shared FFA size.
constexpr SizeType kMaxCapSearchSteps = 8;

struct EvaluatedChunk {
    search::PulsarSearchConfig cfg;
    SizeType ncoords{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    SizeType nsegments{0};
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    /// Transient scratch of the chunk's FFA (CUDA policy only).
    SizeType ffa_transient_bytes{0};
    double memory_gb{0.0}; ///< The chunk alone, with its own FFA buffers.
};

/// FFA workspace and fold buffer: shared by every chunk of a sweep, so sized
/// from the global maxima.
struct SharedFFA {
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    /// Largest transient FFA scratch of any chunk. Live while a group's
    /// workspace is, so it is sized and capped like a shared buffer.
    SizeType ffa_transient_bytes{0};

    void absorb(const EvaluatedChunk& c) noexcept {
        fold_size   = std::max(fold_size, c.fold_size);
        buffer_size = std::max(buffer_size, c.buffer_size);
        coord_size  = std::max(coord_size, c.coord_size);
        ffa_transient_bytes =
            std::max(ffa_transient_bytes, c.ffa_transient_bytes);
    }
    [[nodiscard]] bool contains(const EvaluatedChunk& c) const noexcept {
        return c.fold_size <= fold_size && c.buffer_size <= buffer_size &&
               c.coord_size <= coord_size &&
               c.ffa_transient_bytes <= ffa_transient_bytes;
    }
    [[nodiscard]] SharedFFA scaled(double s) const noexcept {
        const auto scale = [s](SizeType v) {
            return static_cast<SizeType>(
                std::floor(static_cast<double>(v) * s));
        };
        return {.fold_size           = scale(fold_size),
                .buffer_size         = scale(buffer_size),
                .coord_size          = scale(coord_size),
                .ffa_transient_bytes = scale(ffa_transient_bytes)};
    }
};

/// Per-worker buffers: EPFreqSweep allocates them once per contiguous run of
/// chunks with the same nbins, from that run's own maxima.
struct GroupMaxima {
    bool active{false};
    SizeType nbins{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    SizeType ncoords{0};
    SizeType nsegments{0};

    void absorb(const EvaluatedChunk& c) noexcept {
        max_sugg   = std::max(max_sugg, c.max_sugg);
        branch_max = std::max(branch_max, c.branch_max);
        ncoords    = std::max(ncoords, c.ncoords);
        nsegments  = std::max(nsegments, c.nsegments);
    }
};

/// Result of a memory check: `required_gb` is the binding requirement.
struct FitResult {
    bool ok{false};
    double required_gb{0.0};
    /// The chunk's own FFA buffers exceed the fixed shared size.
    bool outside_cap{false};
};

/// Why a pass could not place a minimum-width chunk.
struct PassFailure {
    std::string message;
    /// The fixed shared size was too small for the chunk's own FFA buffers
    /// (a larger cap may help); otherwise the memory limit bound.
    bool outside_cap{false};
};

/// State of one planning pass.
struct PassState {
    SharedFFA shared;                    ///< Maxima over all chunks placed.
    std::optional<SharedFFA> shared_cap; ///< Fixed shared size (replan).
    GroupMaxima group;                   ///< Current nbins run.
    ChunkPlan plan;
    std::optional<PassFailure> failure;
};

template <SupportedFoldType FoldType> class Chunker {
public:
    Chunker(search::PulsarSearchConfig base_cfg,
            std::string_view poly_basis,
            double max_drift,
            double effective_limit_gb,
            const EPMemoryContext& memory)
        : m_base_cfg(std::move(base_cfg)),
          m_poly_basis(poly_basis),
          m_max_drift(max_drift),
          m_limit_gb(effective_limit_gb),
          m_memory(memory) {}

    ChunkPlan run(std::span<const RegionDesign> designs) {
        auto state = run_pass(designs, std::nullopt);
        if (state.failure) {
            throw std::runtime_error(state.failure->message);
        }
        finalize(state);
        if (state.plan.peak_memory_gb <= m_limit_gb) {
            return std::move(state.plan);
        }

        // The shared FFA size grew after an earlier run was planned, and that
        // run no longer fits next to the final shared buffers: replan with the
        // shared size fixed.
        const auto shared_final = state.shared;
        spdlog::info(
            "EPRegionPlanner: sweep peak {:.2f} GB exceeds the limit {:.2f} "
            "GB once the shared FFA buffers reach buffer_size={}, "
            "coord_size={}; replanning with them fixed",
            state.plan.peak_memory_gb, m_limit_gb, shared_final.buffer_size,
            shared_final.coord_size);
        auto replan = run_pass(designs, shared_final);
        if (!replan.failure) {
            return finish_replan(std::move(replan), 1.0);
        }

        // An earlier run does not fit next to the full shared size even at
        // the minimum width. A smaller fixed size leaves it more room and
        // makes the chunks that set the size narrower: search the largest
        // scale of the shared size that fits, in a bounded number of passes.
        const auto full_failure = std::move(*replan.failure);
        spdlog::info("EPRegionPlanner: replan with the full shared FFA size "
                     "is infeasible; searching a smaller fixed size");
        double lo = 0.0;
        double hi = 1.0;
        std::optional<std::pair<PassState, double>> best;
        for (SizeType step = 0; step < kMaxCapSearchSteps; ++step) {
            const double mid = std::midpoint(lo, hi);
            auto trial       = run_pass(designs, shared_final.scaled(mid));
            if (!trial.failure) {
                lo   = mid;
                best = {std::move(trial), mid};
            } else if (trial.failure->outside_cap) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (!best) {
            throw std::runtime_error(full_failure.message);
        }
        return finish_replan(std::move(best->first), best->second);
    }

private:
    using MemoKey = std::tuple<SizeType, double, double>;

    search::PulsarSearchConfig m_base_cfg;
    std::string_view m_poly_basis;
    double m_max_drift;
    double m_limit_gb;
    EPMemoryContext m_memory;
    // Chunk evaluations are pure functions of (region, nominal band): reuse
    // them across planning passes (each builds an FFAPlan).
    std::map<MemoKey, EvaluatedChunk> m_memo;

    ChunkPlan finish_replan(PassState state, double cap_scale) const {
        finalize(state);
        state.plan.replanned        = true;
        state.plan.shared_cap_scale = cap_scale;
        // Every run was fitted next to the fixed shared size, which bounds
        // the final one: the peak fits by construction.
        if (state.plan.peak_memory_gb > m_limit_gb) {
            throw std::logic_error(std::format(
                "EPRegionPlanner: replanned sweep peak {:.4f} GB exceeds the "
                "limit {:.4f} GB",
                state.plan.peak_memory_gb, m_limit_gb));
        }
        if (cap_scale < 1.0) {
            spdlog::info("EPRegionPlanner: shared FFA size fixed to {:.3f} of "
                         "the first pass's maximum",
                         cap_scale);
        }
        return std::move(state.plan);
    }

    PassState run_pass(std::span<const RegionDesign> designs,
                       const std::optional<SharedFFA>& shared_cap) {
        PassState state;
        state.shared_cap = shared_cap;
        for (SizeType i = 0; i < designs.size() && !state.failure; ++i) {
            if (designs[i].f_end > designs[i].f_start) {
                subdivide_region(i, designs[i], state);
            }
        }
        return state;
    }

    void finalize(PassState& state) const {
        auto& plan       = state.plan;
        plan.buffer_size = state.shared.buffer_size;
        plan.coord_size  = state.shared.coord_size;
        plan.fold_size   = state.shared.fold_size;
        plan.peak_memory_gb =
            plan.chunk_cfgs.empty()
                ? 0.0
                : ep_sweep_peak_gb<FoldType>(plan.chunk_cfgs, m_memory);
    }

    [[nodiscard]] double chunk_alone_gb(SizeType nbins,
                                        const EvaluatedChunk& c) const {
        return ep_total_gb(
            m_memory.n_workers,
            ep_thread_bytes<FoldType>(m_memory, nbins, c.nsegments, c.ncoords,
                                      c.max_sugg, c.branch_max),
            ep_fixed_bytes<FoldType>(m_memory, c.buffer_size, c.coord_size,
                                     c.ffa_transient_bytes));
    }

    const EvaluatedChunk& evaluate_chunk(SizeType design_idx,
                                         const RegionDesign& design,
                                         double nominal_start,
                                         double nominal_end) {
        const MemoKey key{design_idx, nominal_start, nominal_end};
        if (const auto it = m_memo.find(key); it != m_memo.end()) {
            return it->second;
        }
        const auto nbins       = design.nbins;
        const double act_start = nominal_start * (1.0 - m_max_drift);
        const double act_end   = nominal_end * (1.0 + m_max_drift);
        auto chunk_cfg = m_base_cfg.get_updated_ep_config(nbins, design.eta,
                                                          act_start, act_end);
        const plans::FFAPlan<FoldType> plan(chunk_cfg);
        // The branching pattern depends on the chunk's own band (not
        // monotone in width), so branch_max must come from the chunk's
        // plan, exactly as EPMultiPass computes it.
        EvaluatedChunk c{
            .cfg     = std::move(chunk_cfg),
            .ncoords = plan.get_ncoords().back(),
            .branch_max =
                compute_branch_max(plan.get_branching_pattern(m_poly_basis)),
            .nsegments   = design.nsegments,
            .fold_size   = plan.get_fold_size(),
            .buffer_size = plan.get_buffer_size(),
            .coord_size  = plan.get_coord_size(),
            .ffa_transient_bytes =
                ep_ffa_transient_bytes<FoldType>(m_memory, plan),
        };
        c.max_sugg = std::max(
            SizeType{1024}, static_cast<SizeType>(std::ceil(
                                static_cast<double>(c.ncoords) *
                                static_cast<double>(design.safe_complexity))));
        if (m_memory.kind == EPMemoryKind::kCuda) {
            // The device world tree needs a capacity above its largest batch.
            c.max_sugg = ep_cuda_effective_max_sugg(m_memory.batch_size,
                                                    c.branch_max, c.max_sugg);
        }
        c.memory_gb = chunk_alone_gb(nbins, c);
        return m_memo.emplace(key, std::move(c)).first->second;
    }

    void subdivide_region(SizeType design_idx,
                          const RegionDesign& design,
                          PassState& state) {
        const double f_start = design.f_start;
        const double f_end   = design.f_end;
        const auto nbins     = design.nbins;

        auto evaluate = [&](double nominal_start,
                            double nominal_end) -> const EvaluatedChunk& {
            return evaluate_chunk(design_idx, design, nominal_start,
                                  nominal_end);
        };

        // Memory of the sweep if chunk `c` is placed next: per-worker
        // buffers of the chunk's nbins run plus the shared FFA buffers and
        // the inputs.
        auto fits = [&](const EvaluatedChunk& c) -> FitResult {
            GroupMaxima group =
                (state.group.active && state.group.nbins == nbins)
                    ? state.group
                    : GroupMaxima{};
            group.absorb(c);
            SharedFFA shared =
                state.shared_cap ? *state.shared_cap : state.shared;
            shared.absorb(c);
            const double sweep_gb = ep_total_gb(
                m_memory.n_workers,
                ep_thread_bytes<FoldType>(m_memory, nbins, group.nsegments,
                                          group.ncoords, group.max_sugg,
                                          group.branch_max),
                ep_fixed_bytes<FoldType>(m_memory, shared.buffer_size,
                                         shared.coord_size,
                                         shared.ffa_transient_bytes));
            // With a fixed shared size the chunk must not enlarge it.
            const bool within_cap =
                !state.shared_cap || state.shared_cap->contains(c);
            return {
                .ok          = within_cap && sweep_gb <= m_limit_gb,
                .required_gb = sweep_gb,
                .outside_cap = !within_cap,
            };
        };

        const double region_span = f_end - f_start;
        const double boundary_tolerance =
            std::max(kBisectionToleranceHz, kRelativeTolerance * region_span);

        auto bisect_chunk =
            [&](double current_f_end, double min_width, EvaluatedChunk min_eval,
                double max_width) -> std::pair<double, EvaluatedChunk> {
            double lo_width     = min_width;
            double hi_width     = max_width;
            EvaluatedChunk best = std::move(min_eval);

            for (SizeType step = 0; step < kMaxBisectionSteps &&
                                    (hi_width - lo_width) > boundary_tolerance;
                 ++step) {
                const double mid_width = std::midpoint(lo_width, hi_width);
                const auto& probe =
                    evaluate(current_f_end - mid_width, current_f_end);
                if (fits(probe).ok) {
                    lo_width = mid_width;
                    best     = probe;
                } else {
                    hi_width = mid_width;
                }
            }
            return {current_f_end - lo_width, std::move(best)};
        };

        const auto find_largest_fitting = [&](double current_f_end)
            -> std::optional<std::pair<double, EvaluatedChunk>> {
            const double remaining = current_f_end - f_start;
            const auto& full_eval  = evaluate(f_start, current_f_end);
            if (fits(full_eval).ok) {
                return std::pair{f_start, full_eval};
            }

            const double min_width = std::min(remaining, kMinChunkWidthHz);
            const double min_start = current_f_end - min_width;
            const auto& min_eval   = evaluate(min_start, current_f_end);
            if (const auto min_fit = fits(min_eval); !min_fit.ok) {
                state.failure = PassFailure{
                    .message = std::format(
                        "EPRegionPlanner: Cannot fit minimum viable chunk at "
                        "[{:08.3f}, {:08.3f}] Hz (nbins={}).\n"
                        "  Required memory: {:.2f} GB (per-thread workspaces "
                        "for this nbins run plus the shared FFA buffers and "
                        "the input series), Available: {:.2f} GB\n{}"
                        "  Suggestion: Increase max_process_memory_gb.",
                        min_start, current_f_end, nbins, min_fit.required_gb,
                        m_limit_gb,
                        state.shared_cap
                            ? "  The shared FFA buffers are fixed to the "
                              "largest size found while planning all "
                              "regions.\n"
                            : ""),
                    .outside_cap = min_fit.outside_cap,
                };
                return std::nullopt;
            }
            return bisect_chunk(current_f_end, min_width, min_eval, remaining);
        };

        // Sliver absorption: merge a tiny leftover into the last chunk.
        const auto try_absorb_sliver =
            [&](double current_f_end,
                double nominal_start) -> std::optional<EvaluatedChunk> {
            const double remainder = nominal_start - f_start;
            if (remainder <= 0.0 ||
                remainder > (kSliverFactor * boundary_tolerance)) {
                return std::nullopt;
            }
            const auto& merged = evaluate(f_start, current_f_end);
            if (fits(merged).ok) {
                return merged;
            }
            return std::nullopt;
        };

        double current_f_end = f_end;
        while (current_f_end > f_start) {
            auto found = find_largest_fitting(current_f_end);
            if (!found) {
                return;
            }
            auto [nominal_start, eval] = std::move(*found);
            if (auto absorbed =
                    try_absorb_sliver(current_f_end, nominal_start)) {
                nominal_start = f_start;
                eval          = std::move(*absorbed);
            }
            if (nominal_start >= current_f_end) {
                throw std::runtime_error(std::format(
                    "EPRegionPlanner: no progress in [{:08.3f}, {:08.3f}] Hz.",
                    f_start, current_f_end));
            }

            const double nominal_end   = current_f_end;
            const double nominal_width = nominal_end - nominal_start;
            const double actual_start  = nominal_start * (1.0 - m_max_drift);
            const double actual_end    = nominal_end * (1.0 + m_max_drift);
            const double actual_width  = actual_end - actual_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            auto& plan          = state.plan;
            const auto chunk_id = plan.chunk_cfgs.size();
            plan.chunk_cfgs.push_back(EPChunkConfig{
                .cfg                 = eval.cfg,
                .threshold_scheme    = design.threshold_scheme,
                .branching_pattern   = design.bp_float,
                .max_sugg            = eval.max_sugg,
                .branch_max          = eval.branch_max,
                .nominal_f_start     = nominal_start,
                .nominal_f_end       = nominal_end,
                .actual_f_start      = actual_start,
                .actual_f_end        = actual_end,
                .peak_complexity     = design.peak_complexity,
                .chunk_memory_gb     = eval.memory_gb,
                .nsegments           = design.nsegments,
                .ncoords             = eval.ncoords,
                .buffer_size         = eval.buffer_size,
                .coord_size          = eval.coord_size,
                .fold_size           = eval.fold_size,
                .ffa_transient_bytes = eval.ffa_transient_bytes,
            });
            plan.chunk_stats.push_back(EPChunkStats{
                .chunk_id         = chunk_id,
                .nominal_f_start  = nominal_start,
                .nominal_f_end    = nominal_end,
                .actual_f_start   = actual_start,
                .actual_f_end     = actual_end,
                .nominal_width    = nominal_width,
                .actual_width     = actual_width,
                .nbins            = nbins,
                .eta              = design.eta,
                .ncoords          = eval.ncoords,
                .max_sugg         = eval.max_sugg,
                .branch_max       = eval.branch_max,
                .peak_complexity  = design.peak_complexity,
                .memory_gb        = eval.memory_gb,
                .overlap_fraction = overlap_fraction,
            });

            // A new run starts when nbins changes (same rule as the sweep).
            if (!state.group.active || state.group.nbins != nbins) {
                state.group        = GroupMaxima{};
                state.group.active = true;
                state.group.nbins  = nbins;
            }
            state.group.absorb(eval);
            state.shared.absorb(eval);
            plan.max_sugg    = std::max(plan.max_sugg, eval.max_sugg);
            plan.max_ncoords = std::max(plan.max_ncoords, eval.ncoords);
            current_f_end    = nominal_start;
        }
    }
};

} // namespace

template <SupportedFoldType FoldType>
ChunkPlan plan_chunks(const search::PulsarSearchConfig& base_cfg,
                      std::string_view poly_basis,
                      std::span<const RegionDesign> designs,
                      double max_drift,
                      double effective_limit_gb,
                      const EPMemoryContext& memory) {
    return Chunker<FoldType>(base_cfg, poly_basis, max_drift,
                             effective_limit_gb, memory)
        .run(designs);
}

template ChunkPlan plan_chunks<float>(const search::PulsarSearchConfig&,
                                      std::string_view,
                                      std::span<const RegionDesign>,
                                      double,
                                      double,
                                      const EPMemoryContext&);
template ChunkPlan plan_chunks<ComplexType>(const search::PulsarSearchConfig&,
                                            std::string_view,
                                            std::span<const RegionDesign>,
                                            double,
                                            double,
                                            const EPMemoryContext&);

} // namespace loki::algorithms::detail
