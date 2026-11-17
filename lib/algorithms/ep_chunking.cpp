#include "lib/algorithms/ep_chunking.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "loki/common/plans.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/algorithms/prune_engine.hpp"
#include "lib/common/plans_detail.hpp"
#include "lib/detail/utils.hpp"

namespace loki::algorithms::detail {

namespace {

constexpr double kBisectionToleranceHz = 1.0e-2;
constexpr double kMinChunkWidthHz      = 1.0e-2;
constexpr double kRelativeTolerance    = 1.0e-4;
constexpr double kSliverFactor         = 4.0;
/// A region's leftover below this fraction of the chunk above it is not
/// planned as its own chunk: the two share the band evenly instead. A narrow
/// chunk repeats a whole FFA and its Doppler padding for a sliver of band.
constexpr double kRebalanceFraction   = 0.25;
constexpr SizeType kMaxBisectionSteps = 50;
/// Planning passes spent searching a smaller fixed shared FFA size.
constexpr SizeType kMaxCapSearchSteps = 8;
/// Most seed slices of a chunk wider than the minimum width. Past it the
/// reserve below dominates each slice's tree, so more slices save little
/// memory and cost a pass over every pruning level each: the width search
/// narrows the chunk instead. A minimum-width chunk takes as many slices as
/// it needs (up to one per frequency cell), so the limit is honoured.
constexpr SizeType kMaxSeedSlices = 64;
/// Largest share of the memory limit that the FFA buffers of a chunk with
/// seed slices may take, unless the chunk has the minimum width. The width
/// search would otherwise widen a sliced chunk until its FFA buffers fill
/// the room the slices free; those buffers are shared by the whole sweep,
/// so every other chunk would need more slices or fail to fit.
constexpr double kSlicedFFAShare = 0.25;
/// Reserve of every seed slice's tree: the live forecast of this many of the
/// chunk's busiest cells, at most kSeedSliceReserveFraction of the whole
/// chunk's. The forecast is a noise population spread over the cells by
/// their branching; a pulsar or RFI puts its subtrees into a few cells, so
/// one slice can hold far more than its share. One slice for the whole
/// chunk keeps the forecast itself as that headroom. Measured: a bright
/// source added 65-100 cells' worth above its slice's share, where the room
/// for a CUDA batch covered the rest; at high frequency the busiest slice
/// stayed below half its noise forecast (docs/performance.md).
constexpr double kSeedSliceReserveCells = 32.0;
/// Search-time model of a CUDA plan, per run: one batch iteration of one
/// level (kernel launches and synchronisation) and one forecast leaf. A
/// chunk costs kCudaIterationSeconds * (parents / batch + slices * levels)
/// + kCudaLeafSeconds * leaves. Fitted on one chunk timed at batches 8192 to
/// 65536 and 1 to 3062 slices (docs/performance.md); the forecasts run
/// about 1.65 times the measured populations, which these constants absorb.
constexpr double kCudaIterationSeconds     = 50.0e-6;
constexpr double kCudaLeafSeconds          = 1.0e-9;
constexpr double kSeedSliceReserveFraction = 1.0 / 16.0;

struct EvaluatedChunk {
    search::PulsarSearchConfig cfg;
    SizeType ncoords{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    SizeType batch_size{0};
    SizeType nsegments{0};
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    /// Transient scratch of the chunk's FFA (CUDA policy only).
    SizeType ffa_transient_bytes{0};
    SizeType predicted_peak{0};
    SizeType predicted_live{0};
    double peak_complexity{1.0};
    std::vector<float> branching_pattern;
    bool indexable{true};
    double memory_gb{0.0}; ///< The chunk alone, with its own FFA buffers.
    /// Unclamped ring live forecast of every seed at once.
    double live_forecast{0.0};
    /// Forecast parents and survivors summed over the stages.
    double total_parents{0.0};
    double total_population{0.0};
    /// Forecast final leaves of a run (with margin).
    SizeType final_leaves{0};
    /// Seed slice edges (empty: one slice) and the final leaves a CPU worker
    /// holds between them.
    std::vector<SizeType> seed_slice_edges{};
    SizeType held_leaves{0};
    /// The region schedule the chunk runs.
    std::vector<float> threshold_scheme{};
    double pd{0.0};
    double pd_se{0.0};
};

/// Chunk geometry: everything of a chunk that does not depend on its
/// thresholds or on the batch.
struct ChunkGeometry {
    search::PulsarSearchConfig cfg;
    SizeType ncoords{0};
    SizeType branch_max{0};
    SizeType nsegments{0};
    SizeType fold_size{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    SizeType ffa_transient_bytes{0};
    std::vector<float> branching_pattern;
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
    SizeType batch_size{0};
    SizeType ncoords{0};
    SizeType nsegments{0};
    SizeType held_leaves{0};

    void absorb(const EvaluatedChunk& c) noexcept {
        max_sugg    = std::max(max_sugg, c.max_sugg);
        branch_max  = std::max(branch_max, c.branch_max);
        ncoords     = std::max(ncoords, c.ncoords);
        nsegments   = std::max(nsegments, c.nsegments);
        held_leaves = std::max(held_leaves, c.held_leaves);
        batch_size  = c.batch_size;
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
    /// The chunk fits next to its group except that it raises the group's
    /// branch_max: (nbins, branch_max) to start such groups with.
    std::optional<std::pair<SizeType, SizeType>> branch_max_floor;
};

/// State of one planning pass.
struct PassState {
    SharedFFA shared;                    ///< Maxima over all chunks placed.
    std::optional<SharedFFA> shared_cap; ///< Fixed shared size (replan).
    GroupMaxima group;                   ///< Current nbins run.
    ChunkPlan plan;
    std::optional<PassFailure> failure;
};

/**
 * @brief Live forecast of a chunk per frequency cell of its seeds.
 *
 * Cell i holds n_accel seeds; its population after stage s is n_accel times
 * its leaves per seed w_s[i] (the branching simulation) times the stage's
 * pass-rate product and margin. The cells' populations sum to the chunk's
 * ring forecast, so a slice of cells is forecast by summing them. A cell's
 * live is the largest sum of two consecutive populations (the ring's
 * occupancy); a slice's live is at most the sum of its cells'.
 */
struct CellForecast {
    /// Prefix sums over the cells of the live and of the largest stage
    /// population: `live[b] - live[a]` covers cells [a, b).
    std::vector<double> live;
    std::vector<double> peak;
    /// Largest live of one cell.
    double max_cell_live{0.0};

    [[nodiscard]] SizeType n_cells() const noexcept {
        return live.empty() ? 0 : live.size() - 1;
    }
};

/// Seed slices of a chunk: edges of the frequency cells and the largest
/// slice's forecast.
struct SlicePartition {
    std::vector<SizeType> edges;
    double max_live{0.0};
    double max_peak{0.0};
};

/// Slices in greedy order whose live is at most @p cap each, with one cell
/// per slice at least; n_cells + 1 if a single cell exceeds @p cap.
SizeType greedy_slices(const CellForecast& cells,
                       double cap,
                       std::vector<SizeType>* edges) {
    const auto n  = cells.n_cells();
    const auto& p = cells.live;
    if (edges != nullptr) {
        edges->assign(1, 0);
    }
    SizeType count = 0;
    SizeType a     = 0;
    while (a < n) {
        const auto it = std::upper_bound(
            p.begin() + static_cast<IndexType>(a + 1), p.end(), p[a] + cap);
        const auto b = static_cast<SizeType>(it - p.begin()) - 1;
        if (b <= a) {
            return n + 1;
        }
        a = b;
        ++count;
        if (edges != nullptr) {
            edges->push_back(b);
        }
    }
    return count;
}

/**
 * @brief At most @p k slices whose largest live is (nearly) the smallest
 * possible: the smallest cap the greedy partition meets in @p k slices.
 */
SlicePartition balanced_slices(const CellForecast& cells, SizeType k) {
    double lo = cells.max_cell_live;
    double hi = cells.live.back();
    if (greedy_slices(cells, lo, nullptr) > k) {
        for (SizeType step = 0; step < 64 && hi - lo > 0.5; ++step) {
            const double mid = std::midpoint(lo, hi);
            if (greedy_slices(cells, mid, nullptr) <= k) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    } else {
        hi = lo;
    }
    SlicePartition out;
    greedy_slices(cells, hi, &out.edges);
    for (SizeType s = 0; s + 1 < out.edges.size(); ++s) {
        const auto a = out.edges[s];
        const auto b = out.edges[s + 1];
        out.max_live = std::max(out.max_live, cells.live[b] - cells.live[a]);
        out.max_peak = std::max(out.max_peak, cells.peak[b] - cells.peak[a]);
    }
    return out;
}

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
        const SizeType default_batch = m_memory.batch_size;
        auto plan                    = plan_sweep(designs);
        log_seed_slices(plan);
        if (m_memory.batch_size < default_batch) {
            log_slow_batches(plan, default_batch);
        }
        return plan;
    }

private:
    /// Says once per plan how many chunks prune their seeds in slices.
    static void log_seed_slices(const ChunkPlan& plan) {
        SizeType n_sliced = 0;
        SizeType worst    = 1;
        for (const auto& c : plan.chunk_stats) {
            if (c.n_seed_slices > 1) {
                ++n_sliced;
                worst = std::max(worst, c.n_seed_slices);
            }
        }
        if (n_sliced == 0) {
            return;
        }
        spdlog::warn(
            "EPRegionPlanner: to fit the memory limit {} of {} chunks prune "
            "each run's seeds in up to {} sequential slices; every slice "
            "repeats all pruning levels, so they search slower (see \"Seed "
            "slices of a run\" in docs/performance.md)",
            n_sliced, plan.chunk_stats.size(), worst);
    }

    /// After the batch ladder: says once per plan how many chunks run with a
    /// smaller batch than the default would give them, and by how much.
    void log_slow_batches(const ChunkPlan& plan, SizeType default_batch) const {
        SizeType n_lowered    = 0;
        SizeType worst_factor = 1;
        for (const auto& c : plan.chunk_stats) {
            const SizeType full =
                ep_cuda_batch_for_nbins(c.nbins, default_batch);
            if (c.batch_size > 0 && c.batch_size < full) {
                ++n_lowered;
                worst_factor = std::max(worst_factor, full / c.batch_size);
            }
        }
        if (n_lowered == 0) {
            return;
        }
        spdlog::warn(
            "EPRegionPlanner: to fit the memory limit (or with fewer seed "
            "slices, where that searches faster) the CUDA batch is at most "
            "{} (default {}): {} of {} chunks run with up to {}x fewer "
            "parents per batch (see \"CUDA batch and branch_max\" in "
            "docs/performance.md)",
            m_memory.batch_size, default_batch, n_lowered,
            plan.chunk_stats.size(), worst_factor);
    }

    ChunkPlan plan_sweep(std::span<const RegionDesign> designs) {
        // On CUDA a smaller batch leaves more room: halve it until a pass
        // places every chunk and its plan (replanned with the shared FFA size
        // fixed if needed) fits.
        std::optional<ChunkPlan> first;
        while (!first) {
            auto state = run_pass(designs, std::nullopt);
            const bool can_halve =
                m_memory.kind == EPMemoryKind::kCuda && m_memory.batch_size > 1;
            std::string failure;
            if (!state.failure) {
                try {
                    first = plan_from_pass(designs, std::move(state));
                    break;
                } catch (const std::runtime_error& e) {
                    if (!can_halve) {
                        throw;
                    }
                    failure = e.what();
                }
            } else if (state.failure->outside_cap || !can_halve) {
                throw std::runtime_error(state.failure->message);
            }
            m_memory.batch_size /= 2;
            spdlog::warn("EPRegionPlanner: {}; retrying with CUDA batch={}",
                         failure.empty() ? "minimum chunk does not fit"
                                         : "the sweep does not fit",
                         m_memory.batch_size);
        }
        auto plan = std::move(*first);
        if (m_memory.kind != EPMemoryKind::kCuda) {
            return plan;
        }
        // Seed slices before a smaller batch, unless the smaller batch is
        // cheaper: halving it can remove slices, each of which repeats every
        // level. Keep halving while the forecast search time drops.
        SizeType best_batch = m_memory.batch_size;
        while (uses_seed_slices(plan) && m_memory.batch_size > 1) {
            m_memory.batch_size /= 2;
            std::optional<ChunkPlan> trial;
            auto pass = run_pass(designs, std::nullopt);
            if (!pass.failure) {
                try {
                    trial = plan_from_pass(designs, std::move(pass));
                } catch (const std::runtime_error&) {
                    trial.reset();
                }
            }
            if (!trial ||
                trial->estimated_search_s >= plan.estimated_search_s) {
                break;
            }
            spdlog::info("EPRegionPlanner: CUDA batch {} instead of {} saves "
                         "seed slices (forecast search {:.3g} s against "
                         "{:.3g} s per run)",
                         m_memory.batch_size, best_batch,
                         trial->estimated_search_s, plan.estimated_search_s);
            best_batch = m_memory.batch_size;
            plan       = std::move(*trial);
        }
        m_memory.batch_size = best_batch;
        return plan;
    }

    [[nodiscard]] static bool uses_seed_slices(const ChunkPlan& plan) {
        return std::ranges::any_of(plan.chunk_stats, [](const auto& c) {
            return c.n_seed_slices > 1;
        });
    }

    /// The plan of a first pass that placed every chunk: as it is if its
    /// peak fits, else replanned with the shared FFA size fixed.
    ChunkPlan plan_from_pass(std::span<const RegionDesign> designs,
                             PassState state) {
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
    // Geometry and designs are pure functions of (region, nominal band) and
    // do not depend on the batch: reuse them across planning passes and
    // batch retries.
    std::map<MemoKey, ChunkGeometry> m_geometry;
    std::map<MemoKey, ChunkDesign> m_designs;
    std::map<MemoKey, CellForecast> m_cells;
    /// Starting branch_max of the groups of an nbins (see run_pass).
    std::map<SizeType, SizeType> m_branch_max_floor;

    ChunkPlan finish_replan(PassState state, double cap_scale) const {
        finalize(state);
        state.plan.replanned        = true;
        state.plan.shared_cap_scale = cap_scale;
        // Every run was fitted next to the fixed shared size, which bounds
        // the final one: the peak fits by construction.
        if (state.plan.peak_memory_gb > m_limit_gb) {
            throw std::logic_error(fmt::format(
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

    /// One planning pass. A pass that fails only because a chunk raises its
    /// group's branch_max is redone with that branch_max for every group of
    /// its nbins from the start: the earlier chunks of the group then leave
    /// room for the larger branch workspace instead of filling the limit with
    /// tree. Floors persist across passes (they only make plans safer).
    PassState run_pass(std::span<const RegionDesign> designs,
                       const std::optional<SharedFFA>& shared_cap) {
        while (true) {
            PassState state;
            state.shared_cap = shared_cap;
            for (SizeType i = 0; i < designs.size() && !state.failure; ++i) {
                if (designs[i].f_end > designs[i].f_start) {
                    subdivide_region(i, designs[i], state);
                }
            }
            if (!state.failure || !state.failure->branch_max_floor) {
                return state;
            }
            const auto [nbins, branch_max] = *state.failure->branch_max_floor;
            spdlog::info("EPRegionPlanner: a minimum chunk raises the "
                         "branch_max of its nbins={} group to {}; replanning "
                         "with groups that start there",
                         nbins, branch_max);
            m_branch_max_floor[nbins] = branch_max;
        }
    }

    /// A new group of @p nbins, with the branch_max it starts with.
    [[nodiscard]] GroupMaxima fresh_group(SizeType nbins) const {
        GroupMaxima group;
        if (const auto it = m_branch_max_floor.find(nbins);
            it != m_branch_max_floor.end()) {
            group.branch_max = it->second;
        }
        return group;
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
            ep_thread_bytes<FoldType>(chunk_memory(c.batch_size, c.buffer_size),
                                      nbins, c.nsegments, c.ncoords, c.max_sugg,
                                      c.branch_max, c.held_leaves),
            ep_fixed_bytes<FoldType>(m_memory, c.buffer_size, c.coord_size,
                                     c.ffa_transient_bytes));
    }

    /// The FFA buffers of chunk @p c alone, in GB: the shared bytes it
    /// needs, without the input series.
    [[nodiscard]] double ffa_gb(const EvaluatedChunk& c) const {
        const auto fixed = ep_fixed_bytes<FoldType>(
            m_memory, c.buffer_size, c.coord_size, c.ffa_transient_bytes);
        return static_cast<double>(fixed - ep_input_bytes(m_memory.nsamps)) /
               utils::kBytesPerGB;
    }

    EPMemoryContext chunk_memory(SizeType batch,
                                 SizeType buffer_size = 0) const {
        auto ctx              = m_memory;
        ctx.batch_size        = batch;
        ctx.shared_fold_bytes = buffer_size * fold_bytes<FoldType>();
        return ctx;
    }

    const ChunkGeometry& geometry(SizeType design_idx,
                                  const RegionDesign& design,
                                  double nominal_start,
                                  double nominal_end) {
        const MemoKey key{design_idx, nominal_start, nominal_end};
        if (const auto it = m_geometry.find(key); it != m_geometry.end()) {
            return it->second;
        }
        const double act_start = nominal_start * (1.0 - m_max_drift);
        const double act_end   = nominal_end * (1.0 + m_max_drift);
        auto chunk_cfg         = m_base_cfg.get_updated_ep_config(
            design.nbins, design.eta, act_start, act_end);
        const plans::FFAPlan<FoldType> plan(chunk_cfg);
        // The branching pattern depends on the chunk's own band (not
        // monotone in width), so branch_max must come from the chunk's
        // plan, exactly as EPMultiPass computes it. It does not depend on
        // the reference segment (the snail scheme enters only through the
        // span and the step to the next epoch, the same for every start;
        // tests/cpp/api/ep_branching_t.cpp), so one forecast serves every
        // run.
        const auto forecast = plan.forecast_branching(m_poly_basis);
        ChunkGeometry g{
            .cfg         = std::move(chunk_cfg),
            .ncoords     = plan.get_ncoords().back(),
            .branch_max  = compute_branch_max(forecast.max_children),
            .nsegments   = design.nsegments,
            .fold_size   = plan.get_fold_size(),
            .buffer_size = plan.get_buffer_size(),
            .coord_size  = plan.get_coord_size(),
            .ffa_transient_bytes =
                ep_ffa_transient_bytes<FoldType>(m_memory, plan),
            .branching_pattern =
                std::vector<float>(forecast.mean.begin(), forecast.mean.end()),
        };
        return m_geometry.emplace(key, std::move(g)).first->second;
    }

    /// The design of a band (the whole region's, in practice); synthetic and
    /// exact when the region has no designer.
    const ChunkDesign& chunk_design(SizeType design_idx,
                                    const RegionDesign& design,
                                    double nominal_start,
                                    double nominal_end,
                                    const ChunkGeometry& g) {
        const MemoKey key{design_idx, nominal_start, nominal_end};
        if (const auto it = m_designs.find(key); it != m_designs.end()) {
            return it->second;
        }
        ChunkDesign d;
        if (design.design) {
            d = design.design(g.branching_pattern, nominal_start, nominal_end);
            if (d.thresholds.size() != g.branching_pattern.size()) {
                throw std::logic_error("EPRegionPlanner: a chunk design "
                                       "needs one threshold per stage");
            }
        } else {
            d = ChunkDesign{
                .thresholds = std::vector<float>(g.branching_pattern.size()),
                .forecast   = {.complexity = {static_cast<double>(
                                   design.safe_complexity)}},
            };
        }
        return m_designs.emplace(key, std::move(d)).first->second;
    }

    /// Sizes a chunk of geometry @p g for a per-stage complexity forecast
    /// and its ring margin.
    EvaluatedChunk sized(const ChunkGeometry& g,
                         SizeType nbins,
                         std::span<const double> complexity,
                         std::span<const double> margin) const {
        EvaluatedChunk c{
            .cfg        = g.cfg,
            .ncoords    = g.ncoords,
            .branch_max = g.branch_max,
            .batch_size =
                m_memory.kind == EPMemoryKind::kCuda
                    ? ep_cuda_batch_for_nbins(nbins, m_memory.batch_size)
                    : m_memory.batch_size,
            .nsegments           = g.nsegments,
            .fold_size           = g.fold_size,
            .buffer_size         = g.buffer_size,
            .coord_size          = g.coord_size,
            .ffa_transient_bytes = g.ffa_transient_bytes,
            .branching_pattern   = g.branching_pattern,
        };
        const auto ring    = ep_ring_forecast(c.ncoords, complexity, margin);
        c.peak_complexity  = ring.peak_complexity;
        c.live_forecast    = ring.live;
        c.total_parents    = ring.total_parents;
        c.total_population = ring.total_population;
        c.final_leaves     = static_cast<SizeType>(
            std::min(ring.final_population,
                     static_cast<double>(ep_tree_index_limit(m_memory.kind))));
        // An oversized probe must fail fitting, rather than aborting the
        // width search before it can try a narrower chunk.
        const auto bound =
            static_cast<double>(ep_tree_index_limit(m_memory.kind));
        c.indexable      = ring.live < bound;
        c.predicted_peak = static_cast<SizeType>(std::min(ring.peak, bound));
        c.predicted_live = static_cast<SizeType>(std::min(ring.live, bound));
        try {
            c.max_sugg = ep_tree_capacity(c.predicted_live, c.batch_size,
                                          c.branch_max, m_memory.kind);
        } catch (const std::overflow_error&) {
            // No tree is counted, so the memory below is a lower bound.
            c.indexable = false;
            c.max_sugg  = 0;
        }
        c.memory_gb = chunk_alone_gb(nbins, c);
        return c;
    }

    /// A chunk sized from the region design: the region schedule's noise
    /// population on the chunk's own branching pattern. A schedule's pass
    /// rates do not depend on the branching, which only multiplies the leaves
    /// of each stage (State::gen_next on both backends), so they carry over
    /// exactly: c'_s = c_s * prod_{j <= s} bp'_j / bp_j, with the design's Pd
    /// and ring margin. Without a designer the synthetic complexity is used
    /// as is.
    EvaluatedChunk scaled(SizeType design_idx,
                          const RegionDesign& design,
                          double nominal_start,
                          double nominal_end) {
        const auto& g =
            geometry(design_idx, design, nominal_start, nominal_end);
        const auto& region =
            geometry(design_idx, design, design.f_start, design.f_end);
        const auto& d = chunk_design(design_idx, design, design.f_start,
                                     design.f_end, region);
        std::vector<double> complexity = d.forecast.complexity;
        if (design.design) {
            const auto& bp_region = region.branching_pattern;
            const auto& bp_chunk  = g.branching_pattern;
            if (bp_chunk.size() != bp_region.size() ||
                complexity.size() > bp_region.size()) {
                throw std::logic_error("EPRegionPlanner: a chunk's stages "
                                       "differ from its region's");
            }
            double ratio = 1.0;
            for (SizeType s = 0; s < complexity.size(); ++s) {
                if (!(bp_region[s] > 0.0F)) {
                    throw std::logic_error("EPRegionPlanner: the region's "
                                           "branching must be positive");
                }
                ratio *= static_cast<double>(bp_chunk[s]) /
                         static_cast<double>(bp_region[s]);
                complexity[s] *= ratio;
            }
        }
        auto c = sized(g, design.nbins, complexity, ep_ring_margin(d.forecast));
        c.threshold_scheme = d.thresholds;
        c.pd               = d.pd;
        c.pd_se            = d.pd_se;
        return c;
    }

    /// The per-cell live forecast of a chunk (CellForecast), from the same
    /// region design and branching simulation as scaled().
    const CellForecast& cell_forecast(SizeType design_idx,
                                      const RegionDesign& design,
                                      double nominal_start,
                                      double nominal_end) {
        const MemoKey key{design_idx, nominal_start, nominal_end};
        if (const auto it = m_cells.find(key); it != m_cells.end()) {
            return it->second;
        }
        const auto& g =
            geometry(design_idx, design, nominal_start, nominal_end);
        const auto& region =
            geometry(design_idx, design, design.f_start, design.f_end);
        const auto& d     = chunk_design(design_idx, design, design.f_start,
                                         design.f_end, region);
        const auto margin = ep_ring_margin(d.forecast);
        const auto& complexity = d.forecast.complexity;
        // Per-seed population of stage s is w_s[i] * q[s]: the region's
        // complexity divided by its branching so far leaves the pass rates.
        std::vector<double> q(complexity.size());
        double bp_prod = 1.0;
        for (SizeType s = 0; s < q.size(); ++s) {
            const double m = s < margin.size() ? margin[s] : 1.0;
            if (design.design) {
                bp_prod *= static_cast<double>(region.branching_pattern[s]);
            }
            q[s] = complexity[s] * m / bp_prod;
        }

        const plans::FFAPlan<FoldType> plan(g.cfg);
        const SizeType n_freq = plan.get_param_counts().back().back();
        if (n_freq == 0 || g.ncoords % n_freq != 0) {
            throw std::logic_error("EPRegionPlanner: seeds are not a grid of "
                                   "frequency cells");
        }
        std::vector<double> live(n_freq, 1.0);
        std::vector<double> peak(n_freq, 1.0);
        std::vector<double> prev(n_freq, 1.0);
        const auto add_stage = [&](SizeType s, auto&& leaves_per_seed) {
            if (s >= q.size()) {
                return;
            }
            for (SizeType i = 0; i < n_freq; ++i) {
                const double cur = leaves_per_seed(i) * q[s];
                live[i]          = std::max(live[i], prev[i] + cur);
                peak[i]          = std::max(peak[i], cur);
                prev[i]          = cur;
            }
        };
        if (design.design) {
            const auto forecast = plans::detail::simulate_branching_stages(
                plan, m_poly_basis, [&](SizeType s, std::span<const double> w) {
                    add_stage(s, [w](SizeType i) { return w[i]; });
                });
            if (forecast.mean.size() != g.branching_pattern.size()) {
                throw std::logic_error("EPRegionPlanner: a chunk's stages "
                                       "differ from its branching pattern");
            }
        } else {
            // Synthetic: the same complexity for every cell.
            for (SizeType s = 0; s < q.size(); ++s) {
                add_stage(s, [](SizeType) { return 1.0; });
            }
        }

        const auto n_accel = static_cast<double>(g.ncoords / n_freq);
        CellForecast cells;
        cells.live.assign(n_freq + 1, 0.0);
        cells.peak.assign(n_freq + 1, 0.0);
        for (SizeType i = 0; i < n_freq; ++i) {
            cells.live[i + 1] = cells.live[i] + (n_accel * live[i]);
            cells.peak[i + 1] = cells.peak[i] + (n_accel * peak[i]);
            cells.max_cell_live =
                std::max(cells.max_cell_live, n_accel * live[i]);
        }
        return m_cells.emplace(key, std::move(cells)).first->second;
    }

    /// @p base pruned in at most @p k balanced seed slices: the tree holds
    /// the largest slice's forecast plus a reserve (kSeedSliceReserveCells),
    /// and a CPU worker the run's final leaves. @p base itself if one slice
    /// results.
    EvaluatedChunk sliced(const EvaluatedChunk& base,
                          const CellForecast& cells,
                          SizeType k,
                          SizeType nbins) const {
        auto part = balanced_slices(cells, k);
        if (part.edges.size() <= 2) {
            return base;
        }
        EvaluatedChunk c = base;
        const double reserve =
            std::min(kSeedSliceReserveCells * cells.max_cell_live,
                     kSeedSliceReserveFraction * base.live_forecast);
        const double live = std::ceil(part.max_live) + 2.0 + std::ceil(reserve);
        const double peak = std::min(std::ceil(part.max_peak) + 1.0, live);
        const auto bound =
            static_cast<double>(ep_tree_index_limit(m_memory.kind));
        c.indexable      = live < bound;
        c.predicted_live = static_cast<SizeType>(std::min(live, bound));
        c.predicted_peak = static_cast<SizeType>(std::min(peak, bound));
        try {
            c.max_sugg = ep_tree_capacity(c.predicted_live, c.batch_size,
                                          c.branch_max, m_memory.kind);
        } catch (const std::overflow_error&) {
            c.indexable = false;
            c.max_sugg  = 0;
        }
        c.seed_slice_edges = std::move(part.edges);
        c.held_leaves      = c.final_leaves;
        c.memory_gb        = chunk_alone_gb(nbins, c);
        return c;
    }

    /// @p base (which does not fit) with the fewest seed slices that fit by
    /// @p ok, at most @p k_limit; the most slices if none fits.
    EvaluatedChunk
    fit_slices(EvaluatedChunk base,
               const CellForecast& cells,
               SizeType nbins,
               SizeType k_limit,
               const std::function<bool(const EvaluatedChunk&)>& ok) const {
        const SizeType k_max = std::min(cells.n_cells(), k_limit);
        if (k_max < 2) {
            return base;
        }
        auto most = sliced(base, cells, k_max, nbins);
        if (!ok(most)) {
            return most.memory_gb < base.memory_gb ? std::move(most)
                                                   : std::move(base);
        }
        // The largest slice shrinks as k grows: bisect the fewest that fit.
        SizeType lo = 1;
        SizeType hi = k_max;
        while (hi - lo > 1) {
            const SizeType mid = std::midpoint(lo, hi);
            auto probe         = sliced(base, cells, mid, nbins);
            if (ok(probe)) {
                hi   = mid;
                most = std::move(probe);
            } else {
                lo = mid;
            }
        }
        return most;
    }

    void subdivide_region(SizeType design_idx,
                          const RegionDesign& design,
                          PassState& state) {
        using EvalFn         = std::function<EvaluatedChunk(double, double)>;
        const double f_start = design.f_start;
        const double f_end   = design.f_end;
        const auto nbins     = design.nbins;

        // Memory of the sweep if chunk `c` is placed next: per-worker
        // buffers of the chunk's nbins run plus the shared FFA buffers and
        // the inputs.
        auto fits = [&](const EvaluatedChunk& c) -> FitResult {
            GroupMaxima group =
                (state.group.active && state.group.nbins == nbins &&
                 state.group.batch_size == c.batch_size)
                    ? state.group
                    : fresh_group(nbins);
            group.absorb(c);
            SharedFFA shared =
                state.shared_cap ? *state.shared_cap : state.shared;
            shared.absorb(c);
            const double sweep_gb = ep_total_gb(
                m_memory.n_workers,
                ep_thread_bytes<FoldType>(
                    chunk_memory(group.batch_size, shared.buffer_size), nbins,
                    group.nsegments, group.ncoords, group.max_sugg,
                    group.branch_max, group.held_leaves),
                ep_fixed_bytes<FoldType>(m_memory, shared.buffer_size,
                                         shared.coord_size,
                                         shared.ffa_transient_bytes));
            // With a fixed shared size the chunk must not enlarge it.
            const bool within_cap =
                !state.shared_cap || state.shared_cap->contains(c);
            return {
                .ok = c.indexable && within_cap && sweep_gb <= m_limit_gb,
                .required_gb = sweep_gb,
                .outside_cap = !within_cap,
            };
        };

        const double region_span = f_end - f_start;
        const double boundary_tolerance =
            std::max(kBisectionToleranceHz, kRelativeTolerance * region_span);
        // Start of a chunk of @p width ending at @p end: exactly f_start
        // when the chunk reaches it.
        const auto start_of = [&](double end, double width) {
            const double start = end - width;
            return start - f_start <= kRelativeTolerance * boundary_tolerance
                       ? f_start
                       : start;
        };

        auto bisect_chunk =
            [&](const EvalFn& evaluate, double current_f_end, double min_width,
                EvaluatedChunk min_eval, double max_width,
                double tolerance) -> std::pair<double, EvaluatedChunk> {
            double lo_width     = min_width;
            double hi_width     = max_width;
            EvaluatedChunk best = std::move(min_eval);

            for (SizeType step = 0;
                 step < kMaxBisectionSteps && (hi_width - lo_width) > tolerance;
                 ++step) {
                const double mid_width = std::midpoint(lo_width, hi_width);
                auto probe =
                    evaluate(start_of(current_f_end, mid_width), current_f_end);
                if (fits(probe).ok) {
                    lo_width = mid_width;
                    best     = std::move(probe);
                } else {
                    hi_width = mid_width;
                }
            }
            // Branching changes discretely with the band. Probe beyond the
            // first failing bracket before accepting a locally maximal width.
            const double anchor = lo_width;
            for (SizeType probe_idx = 1; probe_idx <= 8; ++probe_idx) {
                const double width =
                    anchor +
                    (max_width - anchor) * static_cast<double>(probe_idx) / 8.0;
                if (width <= lo_width) {
                    continue;
                }
                auto probe =
                    evaluate(start_of(current_f_end, width), current_f_end);
                if (fits(probe).ok) {
                    lo_width = width;
                    best     = std::move(probe);
                }
            }
            return {start_of(current_f_end, lo_width), std::move(best)};
        };

        // The widest chunk ending at current_f_end, at most @p max_width
        // wide, that fits by @p evaluate. Sets state.failure if none does.
        const auto find_largest_fitting =
            [&](const EvalFn& evaluate, double current_f_end, double max_width,
                double tolerance)
            -> std::optional<std::pair<double, EvaluatedChunk>> {
            {
                auto full =
                    evaluate(start_of(current_f_end, max_width), current_f_end);
                if (fits(full).ok) {
                    return std::pair{start_of(current_f_end, max_width),
                                     std::move(full)};
                }
            }
            const double min_width = std::min(max_width, kMinChunkWidthHz);
            const double min_start = start_of(current_f_end, min_width);
            auto min_eval          = evaluate(min_start, current_f_end);
            if (const auto min_fit = fits(min_eval); !min_fit.ok) {
                // A wider band can have a cheaper discrete branching pattern.
                // Do not conclude infeasibility from the minimum probe alone.
                std::optional<std::pair<double, EvaluatedChunk>> wider;
                for (SizeType step = 1; step <= 8; ++step) {
                    const double width =
                        min_width + (max_width - min_width) *
                                        static_cast<double>(step) / 8.0;
                    auto probe =
                        evaluate(start_of(current_f_end, width), current_f_end);
                    if (fits(probe).ok) {
                        wider = {width, std::move(probe)};
                    }
                }
                if (wider) {
                    return bisect_chunk(evaluate, current_f_end, wider->first,
                                        std::move(wider->second), max_width,
                                        tolerance);
                }
                state.failure = PassFailure{
                    .message = fmt::format(
                        "EPRegionPlanner: Cannot fit minimum viable chunk at "
                        "[{:08.3f}, {:08.3f}] Hz (nbins={}).\n"
                        "  Required memory: {}{:.2f} GB (per-thread workspaces "
                        "for this nbins run plus the shared FFA buffers and "
                        "the input series), Available: {:.2f} GB\n{}"
                        "{}",
                        min_start, current_f_end, nbins,
                        min_eval.indexable ? "" : "at least ",
                        min_fit.required_gb, m_limit_gb,
                        state.shared_cap
                            ? "  The shared FFA buffers are fixed to the "
                              "largest size found while planning all "
                              "regions.\n"
                            : "",
                        min_eval.indexable
                            ? "  Suggestion: Increase max_process_memory_gb."
                            : "  Forecast exceeds the tree index range; "
                              "the memory estimate is a lower bound."),
                    .outside_cap      = min_fit.outside_cap,
                    .branch_max_floor = std::nullopt,
                };
                // Would it fit without raising its group's branch_max?
                const bool joins =
                    state.group.active && state.group.nbins == nbins &&
                    state.group.batch_size == min_eval.batch_size;
                if (!min_fit.outside_cap && joins &&
                    min_eval.branch_max > state.group.branch_max) {
                    auto at_group       = min_eval;
                    at_group.branch_max = state.group.branch_max;
                    if (fits(at_group).ok) {
                        state.failure->branch_max_floor =
                            std::pair{nbins, min_eval.branch_max};
                    }
                }
                return std::nullopt;
            }
            return bisect_chunk(evaluate, current_f_end, min_width,
                                std::move(min_eval), max_width, tolerance);
        };

        // A chunk that does not fit with every seed at once tries the fewest
        // seed slices that fit, before the width search narrows it. A sliced
        // chunk widens only while its FFA buffers stay within
        // kSlicedFFAShare of the limit, and with at most kMaxSeedSlices
        // slices; a minimum-width chunk takes as many as it needs. Whether a
        // smaller CUDA batch beats the slices is decided per plan
        // (plan_sweep).
        const EvalFn size_fn = [&](double start, double end) {
            auto c         = scaled(design_idx, design, start, end);
            const auto fit = fits(c);
            if (fit.ok || fit.outside_cap) {
                return c;
            }
            // A minimum chunk (start_of may snap it to the region start).
            const bool min_width =
                end - start <= (kMinChunkWidthHz * (1.0 + kRelativeTolerance)) +
                                   (kRelativeTolerance * boundary_tolerance);
            if (!min_width && ffa_gb(c) > kSlicedFFAShare * m_limit_gb) {
                return c;
            }
            const auto& cells = cell_forecast(design_idx, design, start, end);
            return fit_slices(
                std::move(c), cells, nbins,
                min_width ? cells.n_cells() : kMaxSeedSlices,
                [&](const EvaluatedChunk& e) { return fits(e).ok; });
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
            auto merged = size_fn(f_start, current_f_end);
            if (fits(merged).ok) {
                return merged;
            }
            return std::nullopt;
        };

        // The group and shared maxima once @p eval is placed.
        const auto absorb = [&](const EvaluatedChunk& eval) {
            // A new run starts when nbins changes (same rule as the sweep).
            if (!state.group.active || state.group.nbins != nbins ||
                state.group.batch_size != eval.batch_size) {
                state.group        = fresh_group(nbins);
                state.group.active = true;
                state.group.nbins  = nbins;
            }
            state.group.absorb(eval);
            state.shared.absorb(eval);
        };

        const auto place = [&](double nominal_start, double nominal_end,
                               const EvaluatedChunk& eval) {
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
                .threshold_scheme    = eval.threshold_scheme,
                .branching_pattern   = eval.branching_pattern,
                .max_sugg            = eval.max_sugg,
                .branch_max          = eval.branch_max,
                .batch_size          = eval.batch_size,
                .predicted_peak      = eval.predicted_peak,
                .predicted_live      = eval.predicted_live,
                .nominal_f_start     = nominal_start,
                .nominal_f_end       = nominal_end,
                .actual_f_start      = actual_start,
                .actual_f_end        = actual_end,
                .peak_complexity     = eval.peak_complexity,
                .pd                  = eval.pd,
                .pd_se               = eval.pd_se,
                .chunk_memory_gb     = eval.memory_gb,
                .nsegments           = design.nsegments,
                .ncoords             = eval.ncoords,
                .buffer_size         = eval.buffer_size,
                .coord_size          = eval.coord_size,
                .fold_size           = eval.fold_size,
                .ffa_transient_bytes = eval.ffa_transient_bytes,
                .seed_slice_edges    = eval.seed_slice_edges,
                .held_final_leaves   = eval.held_leaves,
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
                .batch_size       = eval.batch_size,
                .predicted_peak   = eval.predicted_peak,
                .predicted_live   = eval.predicted_live,
                .peak_complexity  = eval.peak_complexity,
                .pd               = eval.pd,
                .pd_se            = eval.pd_se,
                .memory_gb        = eval.memory_gb,
                .overlap_fraction = overlap_fraction,
                .n_seed_slices    = eval.seed_slice_edges.empty()
                                        ? SizeType{1}
                                        : eval.seed_slice_edges.size() - 1,
            });

            absorb(eval);
            const double n_slices =
                eval.seed_slice_edges.empty()
                    ? 1.0
                    : static_cast<double>(eval.seed_slice_edges.size() - 1);
            plan.estimated_search_s +=
                (kCudaIterationSeconds *
                 ((eval.total_parents / static_cast<double>(std::max<SizeType>(
                                            eval.batch_size, 1))) +
                  (n_slices * static_cast<double>(design.nsegments - 1)))) +
                (kCudaLeafSeconds * eval.total_population);
            plan.max_sugg    = std::max(plan.max_sugg, eval.max_sugg);
            plan.max_ncoords = std::max(plan.max_ncoords, eval.ncoords);
        };

        double current_f_end = f_end;
        while (current_f_end > f_start) {
            auto found = find_largest_fitting(size_fn, current_f_end,
                                              current_f_end - f_start,
                                              boundary_tolerance);
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
                throw std::runtime_error(fmt::format(
                    "EPRegionPlanner: no progress in [{:08.3f}, {:08.3f}] Hz.",
                    f_start, current_f_end));
            }
            // A leftover far narrower than this chunk would be planned as
            // its own sliver: split the rest of the region evenly between
            // the two instead, if both halves fit. The lower half is fitted
            // next to the upper one (as placed), and neither is placed
            // unless both fit.
            const double remainder = nominal_start - f_start;
            if (remainder > 0.0 &&
                remainder <
                    kRebalanceFraction * (current_f_end - nominal_start)) {
                const double mid = f_start + (0.5 * (current_f_end - f_start));
                auto upper       = size_fn(mid, current_f_end);
                if (fits(upper).ok) {
                    const auto group_before  = state.group;
                    const auto shared_before = state.shared;
                    absorb(upper);
                    auto lower            = size_fn(f_start, mid);
                    const bool lower_fits = fits(lower).ok;
                    state.group           = group_before;
                    state.shared          = shared_before;
                    if (lower_fits) {
                        place(mid, current_f_end, upper);
                        place(f_start, mid, lower);
                        return;
                    }
                }
            }
            place(nominal_start, current_f_end, eval);
            current_f_end = nominal_start;
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
