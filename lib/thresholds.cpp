#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <highfive/highfive.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/progress.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"

namespace loki::detection {
namespace {

class DualPoolFoldManager;

/**
 * Handle to a pre-allocated FoldVector. Its destructor automatically returns
 * the memory to the manager.
 */
class FoldVectorHandle {
public:
    FoldVectorHandle(float* data,
                     SizeType actual_ntrials,
                     SizeType capacity_ntrials,
                     SizeType nbins,
                     float variance,
                     DualPoolFoldManager* manager)
        : m_data(data),
          m_actual_ntrials(actual_ntrials),
          m_capacity_ntrials(capacity_ntrials),
          m_nbins(nbins),
          m_variance(variance),
          m_manager(manager) {}

    // Move-only semantics
    FoldVectorHandle(const FoldVectorHandle&)            = delete;
    FoldVectorHandle& operator=(const FoldVectorHandle&) = delete;

    FoldVectorHandle(FoldVectorHandle&& other) noexcept
        : m_data(other.m_data),
          m_actual_ntrials(other.m_actual_ntrials),
          m_capacity_ntrials(other.m_capacity_ntrials),
          m_nbins(other.m_nbins),
          m_variance(other.m_variance),
          m_manager(other.m_manager) {
        other.m_manager = nullptr; // Prevent double deallocation
    }

    FoldVectorHandle& operator=(FoldVectorHandle&& other) noexcept {
        if (this != &other) {
            release();
            m_data             = other.m_data;
            m_actual_ntrials   = other.m_actual_ntrials;
            m_capacity_ntrials = other.m_capacity_ntrials;
            m_nbins            = other.m_nbins;
            m_variance         = other.m_variance;
            m_manager          = other.m_manager;
            other.m_manager    = nullptr;
        }
        return *this;
    }

    ~FoldVectorHandle() { release(); }

    // Interface
    std::span<float> data() { return {m_data, m_actual_ntrials * m_nbins}; }
    std::span<const float> data() const {
        return {m_data, m_actual_ntrials * m_nbins};
    }
    SizeType ntrials() const { return m_actual_ntrials; }
    SizeType nbins() const { return m_nbins; }
    float variance() const { return m_variance; }
    void set_ntrials(SizeType ntrials) {
        assert(ntrials <= m_capacity_ntrials);
        m_actual_ntrials = ntrials;
    }
    void set_variance(float variance) { m_variance = variance; }

private:
    void release() noexcept; // Implemented after DualPoolFoldManager

    float* m_data;
    SizeType m_actual_ntrials;
    SizeType m_capacity_ntrials;
    SizeType m_nbins;
    float m_variance;
    DualPoolFoldManager* m_manager;
};

/**
 * A helper struct to encapsulate all data for a single memory pool.
 */
struct Pool {
    std::vector<float> data;
    // Using 'char' as a boolean flag is safer in concurrent contexts
    std::vector<char> slot_occupied;
    std::vector<SizeType> free_slots;
    SizeType free_count;

    Pool(SizeType slots_per_pool, SizeType slot_size_in_floats)
        : data(slots_per_pool * slot_size_in_floats),
          slot_occupied(slots_per_pool, 0),
          free_slots(slots_per_pool),
          free_count(slots_per_pool) {
        std::iota(free_slots.begin(), free_slots.end(), 0);
    }

    std::span<float> get_slot(SizeType idx, SizeType slot_size) noexcept {
        return {data.data() + (idx * slot_size), slot_size};
    }
};

/**
 * A thread-safe, pre-allocated memory manager using a dual-pool (ping-pong)
 * buffering strategy to improve cache locality.
 */
class DualPoolFoldManager {
public:
    DualPoolFoldManager(SizeType nbins,
                        SizeType max_ntrials_per_slot,
                        SizeType slots_per_pool)
        : m_nbins(nbins),
          m_max_ntrials(max_ntrials_per_slot),
          m_slot_size_floats(m_max_ntrials * nbins),
          m_slots_per_pool(slots_per_pool),
          m_pool_a(slots_per_pool, m_slot_size_floats),
          m_pool_b(slots_per_pool, m_slot_size_floats),
          m_pool_out(&m_pool_a),
          m_pool_in(&m_pool_b) {}

    DualPoolFoldManager(const DualPoolFoldManager&)            = delete;
    DualPoolFoldManager& operator=(const DualPoolFoldManager&) = delete;
    DualPoolFoldManager(DualPoolFoldManager&&)                 = delete;
    DualPoolFoldManager& operator=(DualPoolFoldManager&&)      = delete;

    ~DualPoolFoldManager() = default;

    /**
     * Allocates a new FoldVector from the current "out" pool.
     */
    [[nodiscard]] std::unique_ptr<FoldVectorHandle>
    allocate(SizeType ntrials = 0, float variance = 0.0F) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_pool_out->free_count == 0) {
            spdlog::error("Pool exhausted! Active slots: {}, Max slots: {}",
                          m_slots_per_pool - m_pool_out->free_count,
                          m_slots_per_pool);
            throw std::runtime_error(
                "DualPoolFoldManager 'out' pool exhausted!");
        }

        const auto slot_idx = m_pool_out->free_slots[--m_pool_out->free_count];
        m_pool_out->slot_occupied[slot_idx] = 1; // 1 for true

        auto slot_span = m_pool_out->get_slot(slot_idx, m_slot_size_floats);
        return std::make_unique<FoldVectorHandle>(
            slot_span.data(), ntrials, m_max_ntrials, m_nbins, variance, this);
    }

    /**
     * Deallocates a handle's memory, returning it to the correct pool's free
     * list.
     */
    void deallocate(const float* data_ptr) noexcept {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Determine which pool the pointer belongs to and release it.
        Pool* target_pool = nullptr;
        if (data_ptr >= m_pool_a.data.data() &&
            data_ptr < m_pool_a.data.data() + m_pool_a.data.size()) {
            target_pool = &m_pool_a;
        } else if (data_ptr >= m_pool_b.data.data() &&
                   data_ptr < m_pool_b.data.data() + m_pool_b.data.size()) {
            target_pool = &m_pool_b;
        } else {
            assert(false &&
                   "Attempted to deallocate memory not owned by this manager.");
            std::terminate();
        }
        deallocate_from_pool(data_ptr, *target_pool);
    }

    /**
     * Swaps the roles of the "in" and "out" pools.
     */
    void swap_pools() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::swap(m_pool_in, m_pool_out);
    }

    /**
     * Returns the total memory size of the two pools.
     */
    [[nodiscard]] SizeType get_memory_size() const noexcept {
        return 2 * m_slots_per_pool * m_slot_size_floats * sizeof(float);
    }

private:
    void deallocate_from_pool(const float* data_ptr,
                              Pool& pool) const noexcept {
        // Calculate slot index from pointer offset
        const auto float_offset = std::distance(
            static_cast<const float*>(pool.data.data()), data_ptr);
        const auto slot_idx =
            static_cast<SizeType>(float_offset) / m_slot_size_floats;

        assert(slot_idx < m_slots_per_pool);
        assert(pool.slot_occupied[slot_idx] &&
               "Attempting to deallocate a free slot");

        pool.slot_occupied[slot_idx]       = 0;        // 0 for false
        pool.free_slots[pool.free_count++] = slot_idx; // LIFO push
    }

    // Config
    SizeType m_nbins;
    SizeType m_max_ntrials;
    SizeType m_slot_size_floats;
    SizeType m_slots_per_pool;
    mutable std::mutex m_mutex;
    // Pool data
    Pool m_pool_a;
    Pool m_pool_b;

    // Pointers to current in/out pools
    Pool* m_pool_out;
    Pool* m_pool_in;
};

inline void FoldVectorHandle::release() noexcept {
    if (m_manager != nullptr) {
        try {
            m_manager->deallocate(m_data);
        } catch (...) {
            assert(false && "Exception in FoldVectorHandle::release");
            std::terminate();
        }
        m_manager = nullptr;
    }
}

struct FoldsType {
    std::unique_ptr<FoldVectorHandle> folds_h0;
    std::unique_ptr<FoldVectorHandle> folds_h1;

    FoldsType() = default;
    FoldsType(std::unique_ptr<FoldVectorHandle> h0,
              std::unique_ptr<FoldVectorHandle> h1)
        : folds_h0(std::move(h0)),
          folds_h1(std::move(h1)) {}

    bool is_valid() const noexcept { return folds_h0 && folds_h1; }

    bool is_empty() const noexcept {
        return !is_valid() || folds_h0->data().empty() ||
               folds_h1->data().empty();
    }

    void invalidate() noexcept {
        folds_h0.reset();
        folds_h1.reset();
    }

    FoldsType(const FoldsType&)            = delete;
    FoldsType& operator=(const FoldsType&) = delete;
    FoldsType(FoldsType&&)                 = default;
    FoldsType& operator=(FoldsType&&)      = default;

    ~FoldsType() = default;
};

// Alias for clarity
using FoldGrid = std::vector<FoldsType>;

/**
 * A collection of reusable buffers for a single thread to avoid repeated
 * memory allocations in performance-critical loops.
 */
struct ThreadLocalBuffers {
    std::vector<float> profile_scaled;
    std::vector<float> noise;
    std::vector<float> scores;
    FoldsType folds_h0;
    FoldsType folds_h1;

    ThreadLocalBuffers(SizeType nbins, SizeType max_ntrials)
        : profile_scaled(nbins),
          noise(max_ntrials * nbins),
          scores(max_ntrials) {}
};

IndexType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::ranges::upper_bound(bins, value);
    return std::distance(bins.begin(), it) - 1;
}

std::unique_ptr<FoldVectorHandle>
simulate_folds(const FoldVectorHandle& folds_in,
               std::span<const float> profile,
               math::ThreadLocalNormalRNG& rng,
               DualPoolFoldManager& manager,
               ThreadLocalBuffers& buffers,
               float bias_snr   = 0.0F,
               float var_add    = 1.0F,
               SizeType ntrials = 1024) {
    const auto ntrials_in = folds_in.ntrials();
    const auto nbins      = folds_in.nbins();

    if (ntrials_in == 0) {
        throw std::invalid_argument("No trials in the input folds");
    }
    // Allocate output
    auto folds_out   = manager.allocate(ntrials);
    auto input_data  = folds_in.data();
    auto output_data = folds_out->data();

    // Scale profile by bias_snr
    auto profile_scaled = std::span(buffers.profile_scaled).first(nbins);
    std::ranges::transform(profile, profile_scaled.begin(),
                           [bias_snr](float x) { return x * bias_snr; });
    // Generate noise
    auto noise = std::span(buffers.noise).first(ntrials * nbins);
    rng.generate(noise, 0.0F, std::sqrt(var_add));
    const float var_new = folds_in.variance() + var_add;

    // Fill output data
    for (SizeType i = 0; i < ntrials; ++i) {
        const SizeType source_trial_idx =
            (i < ntrials_in) ? i : rng.uniform_index(ntrials_in - 1);
        const auto in_offset  = source_trial_idx * nbins;
        const auto out_offset = i * nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            output_data[out_offset + j] = input_data[in_offset + j] +
                                          noise[out_offset + j] +
                                          profile_scaled[j];
        }
    }
    folds_out->set_variance(var_new);
    return folds_out;
}

std::pair<std::unique_ptr<FoldVectorHandle>, float>
simulate_score_prune_fused(const FoldVectorHandle& folds_in,
                           std::span<const float> profile,
                           math::ThreadLocalNormalRNG& rng,
                           DualPoolFoldManager& manager,
                           ThreadLocalBuffers& buffers,
                           BoxcarWidthsCache& box_cache,
                           float threshold,
                           cands::TimerStats::TimerMap& thread_timers,
                           float bias_snr   = 0.0F,
                           float var_add    = 1.0F,
                           SizeType ntrials = 1024) {
    timing::SimpleTimer timer;
    const auto ntrials_in = folds_in.ntrials();
    const auto nbins      = folds_in.nbins();

    if (ntrials_in == 0) {
        throw std::invalid_argument("No trials in the input folds");
    }
    // Allocate output
    timer.start();
    auto folds_out   = manager.allocate(ntrials);
    auto input_data  = folds_in.data();
    auto output_data = folds_out->data();

    // Scale profile by bias_snr
    auto profile_scaled = std::span(buffers.profile_scaled).first(nbins);
    for (SizeType j = 0; j < nbins; ++j) {
        profile_scaled[j] = profile[j] * bias_snr;
    }
    thread_timers["io"] += timer.stop();
    // Generate noise
    timer.start();
    auto noise = std::span(buffers.noise).first(ntrials * nbins);
    rng.generate(noise, 0.0F, std::sqrt(var_add));
    thread_timers["random"] += timer.stop();

    // Fill output data
    timer.start();
    const float var_new      = folds_in.variance() + var_add;
    const float stdnoise     = std::sqrt(var_new);
    SizeType ntrials_success = 0;

    float* __restrict__ out_ptr = output_data.data();
    for (SizeType i = 0; i < ntrials; ++i) {
        const SizeType source_trial_idx =
            (i < ntrials_in) ? i : rng.uniform_index(ntrials_in - 1);
        const auto in_offset    = source_trial_idx * nbins;
        const auto trial_offset = i * nbins;
        const auto out_offset   = ntrials_success * nbins;

        // Generate fold directly into output slot
        for (SizeType j = 0; j < nbins; ++j) {
            out_ptr[out_offset + j] = input_data[in_offset + j] +
                                      noise[trial_offset + j] +
                                      profile_scaled[j];
        }
        // Compute SNR
        const bool snr_above_threshold =
            detection::snr_boxcar_threshold_with_cache(
                std::span<const float>(out_ptr + out_offset, nbins), nbins,
                box_cache, threshold, stdnoise);
        // Prune if SNR is below threshold
        if (snr_above_threshold) {
            ++ntrials_success;
        }
    }
    thread_timers["add_score"] += timer.stop();
    const float success =
        static_cast<float>(ntrials_success) / static_cast<float>(ntrials);
    folds_out->set_ntrials(ntrials_success);
    folds_out->set_variance(var_new);
    return {std::move(folds_out), success};
}

float compute_threshold_survival(std::span<const float> scores,
                                 float survive_prob) {
    if (scores.empty()) {
        throw std::invalid_argument("Scores array is empty");
    }
    auto n_surviving =
        static_cast<SizeType>(survive_prob * static_cast<float>(scores.size()));
    n_surviving = std::max(static_cast<SizeType>(1),
                           std::min(n_surviving, scores.size()));

    std::vector<float> top_scores(n_surviving);
    std::ranges::partial_sort_copy(scores, top_scores, std::greater<>());
    return top_scores.back();
}

void prune_folds(FoldVectorHandle& folds_in,
                 std::span<const float> scores,
                 float threshold) {
    const auto ntrials = folds_in.ntrials();
    const auto nbins   = folds_in.nbins();
    error_check::check_equal(scores.size(), ntrials,
                             "Scores size does not match number of trials");

    std::span<float> data    = folds_in.data();
    SizeType ntrials_success = 0;
    for (SizeType i = 0; i < ntrials; ++i) {
        if (scores[i] > threshold) {
            if (i != ntrials_success) { // Avoid self-copy
                const auto input_offset  = i * nbins;
                const auto output_offset = ntrials_success * nbins;
                std::copy_n(
                    data.begin() + static_cast<IndexType>(input_offset), nbins,
                    data.begin() + static_cast<IndexType>(output_offset));
            }
            ++ntrials_success;
        }
    }
    // Set correct number of trials in output
    folds_in.set_ntrials(ntrials_success);
}

std::pair<State, FoldsType>
gen_next_using_thresh(const State& state_cur,
                      const FoldsType& folds_cur,
                      float threshold,
                      float nbranches,
                      float bias_snr,
                      std::span<const float> profile,
                      math::ThreadLocalNormalRNG& rng,
                      DualPoolFoldManager& manager,
                      ThreadLocalBuffers& buffers,
                      BoxcarWidthsCache& box_cache,
                      cands::TimerStats::TimerMap& thread_timers,
                      float var_add    = 1.0F,
                      SizeType ntrials = 1024) {
    auto [folds_h0_pruned, success_h0] = simulate_score_prune_fused(
        *folds_cur.folds_h0, profile, rng, manager, buffers, box_cache,
        threshold, thread_timers, 0.0F, var_add, ntrials);
    auto [folds_h1_pruned, success_h1] = simulate_score_prune_fused(
        *folds_cur.folds_h1, profile, rng, manager, buffers, box_cache,
        threshold, thread_timers, bias_snr, var_add, ntrials);
    const auto state_next =
        state_cur.gen_next(threshold, success_h0, success_h1, nbranches);
    return {state_next,
            FoldsType{std::move(folds_h0_pruned), std::move(folds_h1_pruned)}};
}

std::pair<State, FoldsType>
gen_next_using_surv_prob(const State& state_cur,
                         const FoldsType& folds_cur,
                         float surv_prob_h0,
                         float nbranches,
                         float bias_snr,
                         std::span<const float> profile,
                         std::span<const SizeType> box_score_widths,
                         math::ThreadLocalNormalRNG& rng,
                         DualPoolFoldManager& manager,
                         ThreadLocalBuffers& buffers,
                         BoxcarWidthsCache& box_cache,
                         cands::TimerStats::TimerMap& thread_timers,
                         float var_add    = 1.0F,
                         SizeType ntrials = 1024) {
    auto folds_h0_sim =
        simulate_folds(*folds_cur.folds_h0, profile, rng, manager, buffers,
                       0.0F, var_add, ntrials);
    auto scores_h0 = std::span(buffers.scores).first(folds_h0_sim->ntrials());
    detection::snr_boxcar_2d_max(folds_h0_sim->data(), box_score_widths,
                                 scores_h0, folds_h0_sim->ntrials(),
                                 folds_h0_sim->nbins(),
                                 std::sqrt(folds_h0_sim->variance()));
    const auto threshold_h0 =
        compute_threshold_survival(scores_h0, surv_prob_h0);
    const auto folds_h0_sim_ntrials = folds_h0_sim->ntrials();
    prune_folds(*folds_h0_sim, scores_h0, threshold_h0);
    const auto success_h0 = static_cast<float>(folds_h0_sim->ntrials()) /
                            static_cast<float>(folds_h0_sim_ntrials);

    auto [folds_h1_pruned, success_h1] = simulate_score_prune_fused(
        *folds_cur.folds_h1, profile, rng, manager, buffers, box_cache,
        threshold_h0, thread_timers, bias_snr, var_add, ntrials);

    const auto state_next =
        state_cur.gen_next(threshold_h0, success_h0, success_h1, nbranches);
    return {state_next,
            FoldsType{std::move(folds_h0_sim), std::move(folds_h1_pruned)}};
}

// Create a compound type for State
HighFive::CompoundType create_compound_state() {
    return {{"success_h0", HighFive::create_datatype<float>()},
            {"success_h1", HighFive::create_datatype<float>()},
            {"complexity", HighFive::create_datatype<float>()},
            {"complexity_cumul", HighFive::create_datatype<float>()},
            {"success_h1_cumul", HighFive::create_datatype<float>()},
            {"nbranches", HighFive::create_datatype<float>()},
            {"threshold", HighFive::create_datatype<float>()},
            {"cost", HighFive::create_datatype<float>()},
            {"threshold_prev", HighFive::create_datatype<float>()},
            {"success_h1_cumul_prev", HighFive::create_datatype<float>()},
            {"is_empty", HighFive::create_datatype<bool>()}};
}

} // namespace

// CPU-specific implementation
class DynamicThresholdScheme::Impl {
public:
    Impl(std::span<const float> branching_pattern,
         float ref_ducy,
         SizeType nbins,
         SizeType ntrials,
         SizeType nprobs,
         float prob_min,
         float snr_final,
         SizeType nthresholds,
         float ducy_max,
         float wtsp,
         float beam_width,
         SizeType trials_start,
         int nthreads)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_trials_start(trials_start),
          m_nthreads(nthreads) {
        if (m_branching_pattern.empty()) {
            throw std::invalid_argument("Branching pattern is empty");
        }
        m_profile = simulation::generate_folded_profile(nbins, ref_ducy);
        m_thresholds =
            detection::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs       = detection::compute_probs(nprobs, prob_min);
        m_nprobs      = m_probs.size();
        m_nbins       = m_profile.size();
        m_nstages     = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        m_box_score_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        m_nthreads    = std::clamp(m_nthreads, 1, omp_get_max_threads());
        m_timer_stats = cands::TimerStats(m_nthreads);

        m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_guess_path = detection::guess_scheme(
            m_nstages, snr_final, m_branching_pattern, m_trials_start);

        m_rng = std::make_unique<math::ThreadLocalNormalRNG>(
            std::random_device{}());

        const auto slots_per_pool = compute_max_allocations_needed();
        m_manager = std::make_unique<DualPoolFoldManager>(m_nbins, m_ntrials,
                                                          slots_per_pool);
        const auto pool_memory_size = m_manager->get_memory_size();
        const auto pool_memory_size_gb =
            static_cast<float>(pool_memory_size) / 1024.0F / 1024.0F / 1024.0F;
        spdlog::info("Pre-allocated 2 pools of {} slots ({:.2f} GB)",
                     slots_per_pool, pool_memory_size_gb);
        m_folds_current.resize(m_nthresholds * m_nprobs);
        m_folds_next.resize(m_nthresholds * m_nprobs);
        m_states.resize(m_nstages * m_nthresholds * m_nprobs, State{});
        init_states();
    }
    ~Impl()                          = default;
    Impl(const Impl&)                = delete;
    Impl& operator=(const Impl&)     = delete;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;

    // Getters
    std::vector<float> get_branching_pattern() const {
        return m_branching_pattern;
    }
    std::vector<float> get_profile() const { return m_profile; }
    std::vector<float> get_thresholds() const { return m_thresholds; }
    std::vector<float> get_probs() const { return m_probs; }
    SizeType get_nstages() const { return m_nstages; }
    SizeType get_nthresholds() const { return m_nthresholds; }
    SizeType get_nprobs() const { return m_nprobs; }
    std::vector<SizeType> get_box_score_widths() const {
        return m_box_score_widths;
    }
    std::vector<State> get_states() const { return m_states; }

    // Methods
    void run(SizeType thres_neigh = 10) {
        spdlog::info("Running dynamic threshold scheme");

        m_timer_stats.reset();
        timing::SimpleTimer total_timer;
        total_timer.start();

        progress::ProgressGuard progress_guard(true);
        auto bar =
            progress::make_standard_bar("Computing scheme", m_nstages - 1);

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            run_segment(istage, thres_neigh);
            m_manager->swap_pools();
            std::swap(m_folds_current, m_folds_next);
            // Release memory slots
            for (auto& fold_opt : m_folds_next) {
                fold_opt.invalidate();
            }
            bar.set_progress(istage);
        }
        bar.mark_as_completed();

        const float total_time = total_timer.stop();
        spdlog::info("Dynamic threshold scheme complete: {}",
                     m_timer_stats.summary(total_time));
    }

    // Save
    std::string save(const std::string& outdir = "./") const {
        const std::filesystem::path filebase = std::format(
            "dynscheme_nstages_{:03d}_nthresh_{:03d}_nprobs_{:03d}_"
            "ntrials_{:04d}_snr_{:04.1f}_ducy_{:04.2f}_beam_{:03.1f}.h5",
            m_nstages, m_nthresholds, m_nprobs, m_ntrials, m_thresholds.back(),
            m_ref_ducy, m_beam_width);
        const std::filesystem::path filepath =
            std::filesystem::path(outdir) / filebase;
        HighFive::File file(filepath, HighFive::File::Overwrite);
        // Save simple attributes
        file.createAttribute("ntrials", m_ntrials);
        file.createAttribute("snr_final", m_thresholds.back());
        file.createAttribute("ref_ducy", m_ref_ducy);
        file.createAttribute("ducy_max", m_ducy_max);
        file.createAttribute("wtsp", m_wtsp);
        file.createAttribute("beam_width", m_beam_width);

        // Create dataset creation property list and enable compression
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1024}));
        props.add(HighFive::Deflate(9));

        // Save arrays
        file.createDataSet("branching_pattern", m_branching_pattern);
        file.createDataSet("profile", m_profile);
        file.createDataSet("thresholds", m_thresholds);
        file.createDataSet("probs", m_probs);
        file.createDataSet("guess_path", m_guess_path);
        // Define the 3D dataspace for states
        std::vector<SizeType> dims = {m_nstages, m_nthresholds, m_nprobs};
        HighFive::DataSetCreateProps props_states;
        std::vector<hsize_t> chunk_dims(dims.begin(), dims.end());
        props_states.add(HighFive::Chunking(chunk_dims));
        auto dataset =
            file.createDataSet("states", HighFive::DataSpace(dims),
                               create_compound_state(), props_states);
        dataset.write_raw(m_states.data());
        spdlog::info("Saved dynamic threshold scheme to {}", filepath.string());
        return filepath.string();
    }

private:
    std::vector<float> m_branching_pattern;
    float m_ref_ducy;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    SizeType m_trials_start;
    int m_nthreads;

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    std::vector<SizeType> m_box_score_widths;
    float m_bias_snr;
    std::vector<float> m_guess_path;
    std::vector<State> m_states;

    cands::TimerStats m_timer_stats;
    std::unique_ptr<math::ThreadLocalNormalRNG> m_rng;
    std::unique_ptr<DualPoolFoldManager> m_manager;
    FoldGrid m_folds_current;
    FoldGrid m_folds_next;

    SizeType compute_max_allocations_needed() {
        SizeType max_active_per_stage = 0;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            auto active_thresholds = get_current_thresholds_idx(istage);
            max_active_per_stage =
                std::max(max_active_per_stage, active_thresholds.size());
        }
        // h0 + h1 per cell
        const auto max_persistent = max_active_per_stage * m_nprobs * 2;
        // 2 simulated folds per transition
        const auto max_temporary  = m_nthreads * 2;
        const auto slots_per_pool = max_persistent + max_temporary;
        spdlog::info("Allocation analysis: {} active thresholds max, {} prob "
                     "bins, {} threads",
                     max_active_per_stage, m_nprobs, m_nthreads);
        spdlog::info("Need {} persistent + {} temporary = {} slots per pool",
                     max_persistent, max_temporary, slots_per_pool);
        return slots_per_pool;
    }

    void init_states() {
        const float var_init = 1.0F;
        // Create initial fold vectors
        auto folds_h0_init = m_manager->allocate(m_ntrials);
        auto folds_h1_init = m_manager->allocate(m_ntrials);
        std::ranges::fill(folds_h0_init->data(), 0.0F);
        std::ranges::fill(folds_h1_init->data(), 0.0F);

        auto buffers_ptr =
            std::make_unique<ThreadLocalBuffers>(m_nbins, m_ntrials);
        auto boxcar_widths_cache_ptr =
            std::make_unique<detection::BoxcarWidthsCache>(m_box_score_widths,
                                                           m_nbins);
        // Simulate the initial folds
        auto folds_h0_sim =
            simulate_folds(*folds_h0_init, m_profile, *m_rng, *m_manager,
                           *buffers_ptr, 0.0F, var_init, m_ntrials);
        auto folds_h1_sim =
            simulate_folds(*folds_h1_init, m_profile, *m_rng, *m_manager,
                           *buffers_ptr, m_bias_snr, var_init, m_ntrials);

        State initial_state = State::initial();
        FoldsType fold_state{std::move(folds_h0_sim), std::move(folds_h1_sim)};
        const auto thresholds_idx = get_current_thresholds_idx(0);
        // Dummy timer object to avoid compiler warning
        cands::TimerStats::TimerMap thread_timers;
        for (SizeType ithres : thresholds_idx) {
            auto [cur_state, cur_fold_state] = gen_next_using_thresh(
                initial_state, fold_state, m_thresholds[ithres],
                m_branching_pattern[0], m_bias_snr, m_profile, *m_rng,
                *m_manager, *buffers_ptr, *boxcar_widths_cache_ptr,
                thread_timers, 1.0F, m_ntrials);

            const auto iprob =
                find_bin_index(m_probs, cur_state.success_h1_cumul);
            if (iprob < 0 || iprob >= static_cast<IndexType>(m_nprobs)) {
                continue;
            }
            const auto fold_idx       = (ithres * m_nprobs) + iprob;
            m_states[fold_idx]        = cur_state;
            m_folds_current[fold_idx] = std::move(cur_fold_state);
        }
    }

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const {
        const auto guess       = m_guess_path[istage];
        const auto half_extent = m_beam_width;
        const auto lower_bound = std::max(0.0F, guess - half_extent);
        const auto upper_bound =
            std::min(m_thresholds.back(), guess + half_extent);

        std::vector<SizeType> result;
        for (SizeType i = 0; i < m_thresholds.size(); ++i) {
            if (m_thresholds[i] >= lower_bound &&
                m_thresholds[i] <= upper_bound) {
                result.push_back(i);
            }
        }

        return result;
    }

    // Run a segment of the dynamic threshold scheme
    void run_segment(SizeType istage, SizeType thres_neigh = 10) {
        const auto beam_idx_cur      = get_current_thresholds_idx(istage);
        const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
        const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
        const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;

        // Local stats for this segment
        cands::TimerStats segment_stats(m_nthreads);

#pragma omp parallel num_threads(m_nthreads)
        {
            static thread_local std::unique_ptr<ThreadLocalBuffers> buffers_ptr;
            static thread_local std::unique_ptr<detection::BoxcarWidthsCache>
                boxcar_widths_cache_ptr;
            if (!buffers_ptr) {
                buffers_ptr =
                    std::make_unique<ThreadLocalBuffers>(m_nbins, m_ntrials);
            }
            if (!boxcar_widths_cache_ptr) {
                boxcar_widths_cache_ptr =
                    std::make_unique<detection::BoxcarWidthsCache>(
                        m_box_score_widths, m_nbins);
            }

            auto& thread_timers = segment_stats.get_thread_local();
#pragma omp for schedule(dynamic)
            for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
                const auto ithres = beam_idx_cur[i];
                // Find nearest neighbors in the previous beam
                const auto neighbour_beam_indices =
                    utils::find_neighbouring_indices(beam_idx_prev, ithres,
                                                     thres_neigh);
                for (SizeType jthresh : neighbour_beam_indices) {
                    for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                        const auto prev_fold_idx = (jthresh * m_nprobs) + kprob;
                        const auto prev_state =
                            m_states[stage_offset_prev + prev_fold_idx];

                        if (prev_state.is_empty) {
                            continue;
                        }
                        const auto& prev_fold_state =
                            m_folds_current[prev_fold_idx];
                        if (!prev_fold_state.is_valid() ||
                            prev_fold_state.is_empty()) {
                            continue;
                        }
                        auto [cur_state, cur_fold_state] =
                            gen_next_using_thresh(
                                prev_state, prev_fold_state,
                                m_thresholds[ithres],
                                m_branching_pattern[istage], m_bias_snr,
                                m_profile, *m_rng, *m_manager, *buffers_ptr,
                                *boxcar_widths_cache_ptr, thread_timers, 1.0F,
                                m_ntrials);

                        const auto iprob =
                            find_bin_index(m_probs, cur_state.success_h1_cumul);
                        if (iprob < 0 ||
                            iprob >= static_cast<IndexType>(m_nprobs)) {
                            continue;
                        }

                        const auto cur_idx       = (ithres * m_nprobs) + iprob;
                        const auto cur_state_idx = stage_offset_cur + cur_idx;

                        State& existing_state     = m_states[cur_state_idx];
                        FoldsType& existing_folds = m_folds_next[cur_idx];

                        if (existing_state.is_empty ||
                            cur_state.complexity_cumul <
                                existing_state.complexity_cumul) {
                            existing_state = cur_state;
                            existing_folds = std::move(cur_fold_state);
                        }
                    }
                }
            }
        }
        // Merge this segment's stats into global
        m_timer_stats.merge(segment_stats);
    }
}; // End DynamicThresholdScheme::Impl definition

DynamicThresholdScheme::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    float ref_ducy,
    SizeType nbins,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType trials_start,
    int nthreads)
    : m_impl(std::make_unique<Impl>(branching_pattern,
                                    ref_ducy,
                                    nbins,
                                    ntrials,
                                    nprobs,
                                    prob_min,
                                    snr_final,
                                    nthresholds,
                                    ducy_max,
                                    wtsp,
                                    beam_width,
                                    trials_start,
                                    nthreads)) {}
DynamicThresholdScheme::~DynamicThresholdScheme() = default;
DynamicThresholdScheme::DynamicThresholdScheme(
    DynamicThresholdScheme&&) noexcept = default;
DynamicThresholdScheme&
DynamicThresholdScheme::operator=(DynamicThresholdScheme&&) noexcept = default;

std::vector<float> DynamicThresholdScheme::get_branching_pattern() const {
    return m_impl->get_branching_pattern();
}
std::vector<float> DynamicThresholdScheme::get_profile() const {
    return m_impl->get_profile();
}
std::vector<float> DynamicThresholdScheme::get_thresholds() const {
    return m_impl->get_thresholds();
}
std::vector<float> DynamicThresholdScheme::get_probs() const {
    return m_impl->get_probs();
}
SizeType DynamicThresholdScheme::get_nstages() const {
    return m_impl->get_nstages();
}
SizeType DynamicThresholdScheme::get_nthresholds() const {
    return m_impl->get_nthresholds();
}
SizeType DynamicThresholdScheme::get_nprobs() const {
    return m_impl->get_nprobs();
}
std::vector<SizeType> DynamicThresholdScheme::get_box_score_widths() const {
    return m_impl->get_box_score_widths();
}
std::vector<State> DynamicThresholdScheme::get_states() const {
    return m_impl->get_states();
}
void DynamicThresholdScheme::run(SizeType thres_neigh) {
    m_impl->run(thres_neigh);
}
std::string DynamicThresholdScheme::save(const std::string& outdir) const {
    return m_impl->save(outdir);
}

std::vector<State> evaluate_scheme(std::span<const float> thresholds,
                                   std::span<const float> branching_pattern,
                                   float ref_ducy,
                                   SizeType nbins,
                                   SizeType ntrials,
                                   float snr_final,
                                   float ducy_max,
                                   float wtsp) {
    error_check::check(!thresholds.empty(), "thresholds cannot be empty");
    error_check::check(!branching_pattern.empty(),
                       "branching_pattern cannot be empty");
    error_check::check_equal(thresholds.size(), branching_pattern.size(),
                             "Number of thresholds must match the number of "
                             "stages");

    const auto nstages = branching_pattern.size();
    const auto profile = simulation::generate_folded_profile(nbins, ref_ducy);
    const auto box_score_widths =
        detection::generate_box_width_trials(nbins, ducy_max, wtsp);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);

    const auto slots_per_pool = 10;
    auto manager =
        std::make_unique<DualPoolFoldManager>(nbins, ntrials, slots_per_pool);
    std::vector<FoldsType> folds_current(1);
    std::vector<FoldsType> folds_next(1);

    math::ThreadLocalNormalRNG rng(std::random_device{}());
    // Create initial fold vectors
    auto folds_h0_init = manager->allocate(ntrials);
    auto folds_h1_init = manager->allocate(ntrials);
    std::ranges::fill(folds_h0_init->data(), 0.0F);
    std::ranges::fill(folds_h1_init->data(), 0.0F);
    auto buffers_ptr = std::make_unique<ThreadLocalBuffers>(nbins, 2 * ntrials);
    auto boxcar_widths_cache_ptr =
        std::make_unique<detection::BoxcarWidthsCache>(box_score_widths, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0_sim = simulate_folds(*folds_h0_init, profile, rng, *manager,
                                       *buffers_ptr, 0.0F, var_init, ntrials);
    auto folds_h1_sim =
        simulate_folds(*folds_h1_init, profile, rng, *manager, *buffers_ptr,
                       bias_snr, var_init, ntrials);
    FoldsType initial_fold_state{std::move(folds_h0_sim),
                                 std::move(folds_h1_sim)};
    // Dummy timer object to avoid compiler warning
    cands::TimerStats::TimerMap thread_timers;
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : folds_current[0];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info(
                "Path not viable, No trials survived, stopping at stage {}",
                istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_thresh(
            prev_state, prev_fold_state, thresholds[istage],
            branching_pattern[istage], bias_snr, profile, rng, *manager,
            *buffers_ptr, *boxcar_widths_cache_ptr, thread_timers, 1.0F,
            ntrials);
        states[istage] = cur_state;
        if (istage == 0) {
            folds_current[0] = std::move(cur_fold_state);
        } else {
            folds_next[0] = std::move(cur_fold_state);
        }
        if (istage > 0) {
            // Swap the folds
            manager->swap_pools();
            std::swap(folds_current, folds_next);
            for (auto& fold_opt : folds_next) {
                fold_opt.invalidate();
            }
        }
    }
    return states;
}

std::vector<State> determine_scheme(std::span<const float> survive_probs,
                                    std::span<const float> branching_pattern,
                                    float ref_ducy,
                                    SizeType nbins,
                                    SizeType ntrials,
                                    float snr_final,
                                    float ducy_max,
                                    float wtsp) {
    error_check::check(!survive_probs.empty(), "thresholds cannot be empty");
    error_check::check(!branching_pattern.empty(),
                       "branching_pattern cannot be empty");
    error_check::check_equal(survive_probs.size(), branching_pattern.size(),
                             "Number of thresholds must match the number of "
                             "stages");

    const auto nstages = branching_pattern.size();
    const auto profile = simulation::generate_folded_profile(nbins, ref_ducy);
    const auto box_score_widths =
        detection::generate_box_width_trials(nbins, ducy_max, wtsp);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);

    const auto slots_per_pool = 10;
    auto manager =
        std::make_unique<DualPoolFoldManager>(nbins, ntrials, slots_per_pool);
    std::vector<FoldsType> folds_current(1);
    std::vector<FoldsType> folds_next(1);

    math::ThreadLocalNormalRNG rng(std::random_device{}());
    // Create initial fold vectors
    auto folds_h0_init = manager->allocate(ntrials);
    auto folds_h1_init = manager->allocate(ntrials);
    std::ranges::fill(folds_h0_init->data(), 0.0F);
    std::ranges::fill(folds_h1_init->data(), 0.0F);
    auto buffers_ptr = std::make_unique<ThreadLocalBuffers>(nbins, 2 * ntrials);
    auto boxcar_widths_cache_ptr =
        std::make_unique<detection::BoxcarWidthsCache>(box_score_widths, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0_sim = simulate_folds(*folds_h0_init, profile, rng, *manager,
                                       *buffers_ptr, 0.0F, var_init, ntrials);
    auto folds_h1_sim =
        simulate_folds(*folds_h1_init, profile, rng, *manager, *buffers_ptr,
                       bias_snr, var_init, ntrials);
    FoldsType initial_fold_state{std::move(folds_h0_sim),
                                 std::move(folds_h1_sim)};
    // Dummy timer object to avoid compiler warning
    cands::TimerStats::TimerMap thread_timers;
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : folds_current[0];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info(
                "Path not viable, No trials survived, stopping at stage {}",
                istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_surv_prob(
            prev_state, prev_fold_state, survive_probs[istage],
            branching_pattern[istage], bias_snr, profile, box_score_widths, rng,
            *manager, *buffers_ptr, *boxcar_widths_cache_ptr, thread_timers,
            1.0F, ntrials);
        states[istage] = cur_state;
        if (istage == 0) {
            folds_current[0] = std::move(cur_fold_state);
        } else {
            folds_next[0] = std::move(cur_fold_state);
        }
        if (istage > 0) {
            // Swap the folds
            manager->swap_pools();
            std::swap(folds_current, folds_next);
            for (auto& fold_opt : folds_next) {
                fold_opt.invalidate();
            }
        }
    }
    return states;
}

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)