#include "loki/detection/thresholds.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <highfive/highfive.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/simulation/simulation.hpp"
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
 * A thread-safe, pre-allocated memory manager using a dual-pool (ping-pong)
 * buffering strategy to improve cache locality.
 */
class DualPoolFoldManager {
public:
    DualPoolFoldManager(SizeType nbins,
                        SizeType ntrials_min,
                        SizeType slots_per_pool)
        : m_nbins(nbins),
          m_max_ntrials(2 * ntrials_min),
          m_slot_size(m_max_ntrials * nbins),
          m_slots_per_pool(slots_per_pool) {

        // Pre-allocate all memory for both pools
        m_data_a.resize(m_slots_per_pool * m_slot_size);
        m_slot_occupied_a.resize(m_slots_per_pool, false);
        m_data_b.resize(m_slots_per_pool * m_slot_size);
        m_slot_occupied_b.resize(m_slots_per_pool, false);

        // Initialize free slots for both pools
        for (SizeType i = 0; i < m_slots_per_pool; ++i) {
            m_free_slots_a.push(i);
            m_free_slots_b.push(i);
        }

        // Start with A as the "out" pool and B as the "in" pool
        // We only ever allocate from the "out" pool.
        set_pools_a_out_b_in();
    }

    DualPoolFoldManager(const DualPoolFoldManager&)            = delete;
    DualPoolFoldManager& operator=(const DualPoolFoldManager&) = delete;
    DualPoolFoldManager(DualPoolFoldManager&&)                 = delete;
    DualPoolFoldManager& operator=(DualPoolFoldManager&&)      = delete;

    ~DualPoolFoldManager() = default;

    /**
     * Allocates a new FoldVector from the current "out" pool.
     */
    [[nodiscard]] std::unique_ptr<FoldVectorHandle>
    allocate(SizeType initial_ntrials = 0, float variance = 0.0F) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_free_slots_out->empty()) {
            throw std::runtime_error(
                "DualPoolFoldManager 'out' pool exhausted!");
        }

        const auto slot_idx = m_free_slots_out->front();
        m_free_slots_out->pop();
        (*m_slot_occupied_out)[slot_idx] = true;
        float* slot_data = m_data_out->data() + (slot_idx * m_slot_size);
        return std::make_unique<FoldVectorHandle>(
            slot_data, initial_ntrials, m_max_ntrials, m_nbins, variance, this);
    }

    /**
     * Deallocates a handle's memory, returning it to the correct pool's free
     * list.
     */
    void deallocate(const float* data_ptr) noexcept {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Determine which pool the pointer belongs to and release it.
        if (data_ptr >= m_data_a.data() &&
            data_ptr < m_data_a.data() + m_data_a.size()) {
            deallocate_from_pool(data_ptr, m_data_a, m_slot_occupied_a,
                                 m_free_slots_a);
        } else if (data_ptr >= m_data_b.data() &&
                   data_ptr < m_data_b.data() + m_data_b.size()) {
            deallocate_from_pool(data_ptr, m_data_b, m_slot_occupied_b,
                                 m_free_slots_b);
        } else {
            // This should not happen if used correctly
            assert(false &&
                   "Attempted to deallocate memory not owned by this manager.");
            std::terminate();
        }
    }

    /**
     * Swaps the roles of the "in" and "out" pools.
     */
    void swap_pools() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_data_out == &m_data_a) {
            set_pools_b_out_a_in();
        } else {
            set_pools_a_out_b_in();
        }
    }

private:
    void deallocate_from_pool(const float* data_ptr,
                              const std::vector<float>& data_pool,
                              std::vector<bool>& occupied_pool,
                              std::queue<SizeType>& free_pool) const {
        // Calculate slot index from pointer offset
        const auto byte_offset = static_cast<SizeType>(
            reinterpret_cast<const char*>(data_ptr) -
            reinterpret_cast<const char*>(data_pool.data()));
        const auto slot_idx = byte_offset / (m_slot_size * sizeof(float));

        assert(slot_idx < m_slots_per_pool);
        assert(occupied_pool[slot_idx]);

        occupied_pool[slot_idx] = false;
        free_pool.push(slot_idx);
    }

    void set_pools_a_out_b_in() {
        m_data_out          = &m_data_a;
        m_slot_occupied_out = &m_slot_occupied_a;
        m_free_slots_out    = &m_free_slots_a;
        m_data_in           = &m_data_b;
        m_slot_occupied_in  = &m_slot_occupied_b;
        m_free_slots_in     = &m_free_slots_b;
    }

    void set_pools_b_out_a_in() {
        m_data_out          = &m_data_b;
        m_slot_occupied_out = &m_slot_occupied_b;
        m_free_slots_out    = &m_free_slots_b;
        m_data_in           = &m_data_a;
        m_slot_occupied_in  = &m_slot_occupied_a;
        m_free_slots_in     = &m_free_slots_a;
    }

    // Pool A
    std::vector<float> m_data_a;
    std::vector<bool> m_slot_occupied_a;
    std::queue<SizeType> m_free_slots_a;

    // Pool B
    std::vector<float> m_data_b;
    std::vector<bool> m_slot_occupied_b;
    std::queue<SizeType> m_free_slots_b;

    // Pointers to current in/out pools
    std::vector<float>* m_data_out         = nullptr;
    std::vector<bool>* m_slot_occupied_out = nullptr;
    std::queue<SizeType>* m_free_slots_out = nullptr;
    std::vector<float>* m_data_in          = nullptr;
    std::vector<bool>* m_slot_occupied_in  = nullptr;
    std::queue<SizeType>* m_free_slots_in  = nullptr;

    // Config
    SizeType m_nbins;
    SizeType m_max_ntrials;
    SizeType m_slot_size;
    SizeType m_slots_per_pool;
    mutable std::mutex m_mutex;
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

    bool is_empty() const {
        return !folds_h0 || !folds_h1 || folds_h0->data().empty() ||
               folds_h1->data().empty();
    }

    FoldsType(const FoldsType&)            = delete;
    FoldsType& operator=(const FoldsType&) = delete;
    FoldsType(FoldsType&&)                 = default;
    FoldsType& operator=(FoldsType&&)      = default;

    ~FoldsType() = default;
};

// Alias for clarity
using FoldGrid = std::vector<std::optional<FoldsType>>;

IndexType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::ranges::upper_bound(bins, value);
    return std::distance(bins.begin(), it) - 1;
}

std::unique_ptr<FoldVectorHandle>
simulate_folds(const FoldVectorHandle& folds_in,
               std::span<const float> profile,
               math::ThreadSafeRNGBase& rng,
               DualPoolFoldManager& manager,
               float bias_snr       = 0.0F,
               float var_add        = 1.0F,
               SizeType ntrials_min = 1024) {
    const auto ntrials_in = folds_in.ntrials();
    const auto nbins      = folds_in.nbins();

    if (ntrials_in == 0) {
        throw std::invalid_argument("No trials in the input folds");
    }
    // Calculate output size
    const auto repeat_factor = static_cast<SizeType>(std::ceil(
        static_cast<float>(ntrials_min) / static_cast<float>(ntrials_in)));
    const auto ntrials_out   = repeat_factor * ntrials_in;

    // Allocate output
    auto folds_out =
        manager.allocate(ntrials_out, folds_in.variance() + var_add);
    auto input_data  = folds_in.data();
    auto output_data = folds_out->data();

    // Scale profile by bias_snr
    std::vector<float> profile_scaled(nbins);
    std::ranges::transform(profile, profile_scaled.begin(),
                           [bias_snr](float x) { return x * bias_snr; });
    // Generate noise
    std::vector<float> noise(ntrials_out * nbins);
    rng.generate_normal_dist_range(std::span<float>(noise), 0.0F,
                                   std::sqrt(var_add));

    // Fill output data
    for (SizeType i = 0; i < ntrials_out; ++i) {
        const auto orig_trial   = i % ntrials_in;
        const auto trial_offset = i * nbins;
        const auto orig_offset  = orig_trial * nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            const auto out_idx = trial_offset + j;
            const auto in_idx  = orig_offset + j;
            output_data[out_idx] =
                input_data[in_idx] + noise[out_idx] + profile_scaled[j];
        }
    }
    return folds_out;
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

std::unique_ptr<FoldVectorHandle> prune_folds(const FoldVectorHandle& folds_in,
                                              std::span<const float> scores,
                                              float threshold,
                                              DualPoolFoldManager& manager) {
    const auto ntrials = folds_in.ntrials();
    const auto nbins   = folds_in.nbins();
    error_check::check_equal(scores.size(), ntrials,
                             "Scores size does not match number of trials");
    std::vector<SizeType> good_indices;
    good_indices.reserve(ntrials);
    for (SizeType i = 0; i < ntrials; ++i) {
        if (scores[i] > threshold) {
            good_indices.push_back(i);
        }
    }
    const auto ntrials_success = good_indices.size();
    // Allocate output
    auto folds_out = manager.allocate(ntrials_success, folds_in.variance());
    // Copy successful trials
    auto input_data  = folds_in.data();
    auto output_data = folds_out->data();

    for (SizeType i = 0; i < ntrials_success; ++i) {
        const auto input_offset  = good_indices[i] * nbins;
        const auto output_offset = i * nbins;
        std::copy_n(
            input_data.begin() + static_cast<IndexType>(input_offset), nbins,
            output_data.begin() + static_cast<IndexType>(output_offset));
    }
    return folds_out;
}

State gen_next_state(const State& state_cur,
                     float threshold,
                     float success_h0,
                     float success_h1,
                     float nbranches) {
    const auto nleaves_next     = state_cur.complexity * nbranches;
    const auto nleaves_surv     = nleaves_next * success_h0;
    const auto complexity_cumul = state_cur.complexity_cumul + nleaves_next;
    const auto success_h1_cumul = state_cur.success_h1_cumul * success_h1;

    // Create a new state struct
    State state_next;
    state_next.success_h0       = success_h0;
    state_next.success_h1       = success_h1;
    state_next.complexity       = nleaves_surv;
    state_next.complexity_cumul = complexity_cumul;
    state_next.success_h1_cumul = success_h1_cumul;
    state_next.nbranches        = nbranches;
    state_next.threshold        = threshold;
    state_next.cost             = complexity_cumul / success_h1_cumul;
    state_next.is_empty         = false;
    // For backtracking
    state_next.threshold_prev        = state_cur.threshold;
    state_next.success_h1_cumul_prev = state_cur.success_h1_cumul;
    return state_next;
}

std::tuple<State, FoldsType>
gen_next_using_thresh(const State& state_cur,
                      const FoldsType& folds_cur,
                      float threshold,
                      float nbranches,
                      float bias_snr,
                      std::span<const float> profile,
                      std::span<const SizeType> box_score_widths,
                      math::ThreadSafeRNGBase& rng,
                      DualPoolFoldManager& manager,
                      float var_add    = 1.0F,
                      SizeType ntrials = 1024) {
    auto folds_h0_sim = simulate_folds(*folds_cur.folds_h0, profile, rng,
                                       manager, 0.0F, var_add, ntrials);
    std::vector<float> scores_h0(folds_h0_sim->ntrials());
    detection::snr_boxcar_2d_max(folds_h0_sim->data(), folds_h0_sim->ntrials(),
                                 box_score_widths, scores_h0,
                                 std::sqrt(folds_h0_sim->variance()));
    auto folds_h0_pruned =
        prune_folds(*folds_h0_sim, scores_h0, threshold, manager);
    const auto success_h0 = static_cast<float>(folds_h0_pruned->ntrials()) /
                            static_cast<float>(folds_h0_sim->ntrials());

    auto folds_h1_sim = simulate_folds(*folds_cur.folds_h1, profile, rng,
                                       manager, bias_snr, var_add, ntrials);
    std::vector<float> scores_h1(folds_h1_sim->ntrials());
    detection::snr_boxcar_2d_max(folds_h1_sim->data(), folds_h1_sim->ntrials(),
                                 box_score_widths, scores_h1,
                                 std::sqrt(folds_h1_sim->variance()));
    auto folds_h1_pruned =
        prune_folds(*folds_h1_sim, scores_h1, threshold, manager);
    const auto success_h1 = static_cast<float>(folds_h1_pruned->ntrials()) /
                            static_cast<float>(folds_h1_sim->ntrials());

    const auto state_next =
        gen_next_state(state_cur, threshold, success_h0, success_h1, nbranches);
    return {state_next,
            FoldsType{std::move(folds_h0_pruned), std::move(folds_h1_pruned)}};
}

std::tuple<State, FoldsType>
gen_next_using_surv_prob(const State& state_cur,
                         const FoldsType& folds_cur,
                         float surv_prob_h0,
                         float nbranches,
                         float bias_snr,
                         std::span<const float> profile,
                         std::span<const SizeType> box_score_widths,
                         math::ThreadSafeRNGBase& rng,
                         DualPoolFoldManager& manager,
                         float var_add    = 1.0F,
                         SizeType ntrials = 1024) {
    auto folds_h0_sim = simulate_folds(*folds_cur.folds_h0, profile, rng,
                                       manager, 0.0F, var_add, ntrials);
    std::vector<float> scores_h0(folds_h0_sim->ntrials());
    detection::snr_boxcar_2d_max(folds_h0_sim->data(), folds_h0_sim->ntrials(),
                                 box_score_widths, scores_h0,
                                 std::sqrt(folds_h0_sim->variance()));
    const auto threshold_h0 =
        compute_threshold_survival(scores_h0, surv_prob_h0);
    auto folds_h0_pruned =
        prune_folds(*folds_h0_sim, scores_h0, threshold_h0, manager);
    const auto success_h0 = static_cast<float>(folds_h0_pruned->ntrials()) /
                            static_cast<float>(folds_h0_sim->ntrials());

    auto folds_h1_sim = simulate_folds(*folds_cur.folds_h1, profile, rng,
                                       manager, bias_snr, var_add, ntrials);
    std::vector<float> scores_h1(folds_h1_sim->ntrials());
    detection::snr_boxcar_2d_max(folds_h1_sim->data(), folds_h1_sim->ntrials(),
                                 box_score_widths, scores_h1,
                                 std::sqrt(folds_h1_sim->variance()));
    auto folds_h1_pruned =
        prune_folds(*folds_h1_sim, scores_h1, threshold_h0, manager);
    const auto success_h1 = static_cast<float>(folds_h1_pruned->ntrials()) /
                            static_cast<float>(folds_h1_sim->ntrials());

    const auto state_next = gen_next_state(state_cur, threshold_h0, success_h0,
                                           success_h1, nbranches);
    return {state_next,
            FoldsType{std::move(folds_h0_pruned), std::move(folds_h1_pruned)}};
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
         bool use_lut_rng,
         int nthreads)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_trials_start(trials_start),
          m_use_lut_rng(use_lut_rng),
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

        m_nthreads = std::clamp(m_nthreads, 1, omp_get_max_threads());

        m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
        m_guess_path = detection::guess_scheme(
            m_nstages, snr_final, m_branching_pattern, m_trials_start);

        if (m_use_lut_rng) {
            m_rng = std::make_unique<math::ThreadSafeLUTRNG>(
                std::random_device{}());
        } else {
            m_rng =
                std::make_unique<math::ThreadSafeRNG>(std::random_device{}());
        }

        const auto slots_per_pool = compute_max_allocations_needed();
        m_manager = std::make_unique<DualPoolFoldManager>(m_nbins, m_ntrials,
                                                          slots_per_pool);
        spdlog::info("Pre-allocated 2 pools of {} slots each", slots_per_pool);
        m_folds_current.resize(m_nthresholds * m_nprobs);
        m_folds_next.resize(m_nthresholds * m_nprobs);
        m_states.resize(m_nstages * m_nthresholds * m_nprobs, State{});
        init_states();
    }

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
        utils::ProgressGuard progress_guard(true);
        auto bar = utils::make_standard_bar("Computing scheme...");

        for (SizeType istage = 1; istage < m_nstages; ++istage) {
            run_segment(istage, thres_neigh);
            m_manager->swap_pools();
            std::swap(m_folds_current, m_folds_next);
            for (auto& fold_opt : m_folds_next) {
                fold_opt.reset();
            }
            const auto progress = static_cast<float>(istage) /
                                  static_cast<float>(m_nstages - 1) * 100.0F;
            bar.set_progress(static_cast<SizeType>(progress));
        }
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
    bool m_use_lut_rng;
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

    std::unique_ptr<math::ThreadSafeRNGBase> m_rng;
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
        // 2 simulated and 2 pruned folds per transition
        const auto max_temporary  = m_nthreads * 4;
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
        auto folds_h0_init = m_manager->allocate(m_ntrials, 0.0F);
        auto folds_h1_init = m_manager->allocate(m_ntrials, 0.0F);
        std::fill(folds_h0_init->data().begin(), folds_h0_init->data().end(),
                  0.0F);
        std::fill(folds_h1_init->data().begin(), folds_h1_init->data().end(),
                  0.0F);

        // Simulate the initial folds
        auto folds_h0_sim =
            simulate_folds(*folds_h0_init, m_profile, *m_rng, *m_manager, 0.0F,
                           var_init, m_ntrials);
        auto folds_h1_sim =
            simulate_folds(*folds_h1_init, m_profile, *m_rng, *m_manager,
                           m_bias_snr, var_init, m_ntrials);

        State initial_state;
        FoldsType fold_state{std::move(folds_h0_sim), std::move(folds_h1_sim)};
        const auto thresholds_idx = get_current_thresholds_idx(0);
        for (SizeType ithres : thresholds_idx) {
            auto [cur_state, cur_fold_state] = gen_next_using_thresh(
                initial_state, fold_state, m_thresholds[ithres],
                m_branching_pattern[0], m_bias_snr, m_profile,
                m_box_score_widths, *m_rng, *m_manager, 1.0F, m_ntrials);

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

#pragma omp parallel for num_threads(m_nthreads)
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
                    const auto& prev_fold_state_opt =
                        m_folds_current[prev_fold_idx];
                    if (!prev_fold_state_opt.has_value() ||
                        prev_fold_state_opt->is_empty()) {
                        continue;
                    }
                    auto [cur_state, cur_fold_state] = gen_next_using_thresh(
                        prev_state, *prev_fold_state_opt, m_thresholds[ithres],
                        m_branching_pattern[istage], m_bias_snr, m_profile,
                        m_box_score_widths, *m_rng, *m_manager, 1.0F,
                        m_ntrials);

                    const auto iprob =
                        find_bin_index(m_probs, cur_state.success_h1_cumul);
                    if (iprob < 0 ||
                        iprob >= static_cast<IndexType>(m_nprobs)) {
                        continue;
                    }

                    const auto cur_idx       = (ithres * m_nprobs) + iprob;
                    const auto cur_state_idx = stage_offset_cur + cur_idx;

                    auto& existing_state = m_states[cur_state_idx];
                    auto& existing_folds = m_folds_next[cur_idx];

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
    bool use_lut_rng,
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
                                    use_lut_rng,
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
    if (thresholds.empty() || branching_pattern.empty()) {
        throw std::invalid_argument("Input arrays cannot be empty");
    }
    if (thresholds.size() != branching_pattern.size()) {
        throw std::invalid_argument(
            "Number of thresholds must match the number of stages");
    }

    const auto nstages = branching_pattern.size();
    const auto profile = simulation::generate_folded_profile(nbins, ref_ducy);
    const auto box_score_widths =
        detection::generate_box_width_trials(nbins, ducy_max, wtsp);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    math::ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 =
        simulate_folds(folds_init, profile, rng, 0.0F, var_init, ntrials);
    auto folds_h1 =
        simulate_folds(folds_init, profile, rng, bias_snr, var_init, ntrials);
    FoldsType initial_fold_state{std::move(folds_h0), std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info(
                "Path not viable, No trials survived, stopping at stage {}",
                istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_thresh(
            prev_state, prev_fold_state, thresholds[istage],
            branching_pattern[istage], bias_snr, profile, box_score_widths, rng,
            1.0F, ntrials);
        states[istage]      = cur_state;
        fold_states[istage] = std::move(cur_fold_state);
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
    if (survive_probs.empty() || branching_pattern.empty()) {
        throw std::invalid_argument("Input arrays cannot be empty");
    }
    if (survive_probs.size() != branching_pattern.size()) {
        throw std::invalid_argument(
            "Number of survival probabilities must match the number of stages");
    }

    const auto nstages = branching_pattern.size();
    const auto profile = simulation::generate_folded_profile(nbins, ref_ducy);
    const auto box_score_widths =
        detection::generate_box_width_trials(nbins, ducy_max, wtsp);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    math::ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 =
        simulate_folds(folds_init, profile, rng, 0.0F, var_init, ntrials);
    auto folds_h1 =
        simulate_folds(folds_init, profile, rng, bias_snr, var_init, ntrials);
    const FoldsType initial_fold_state{std::move(folds_h0),
                                       std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info(
                "Path not viable, No trials survived, stopping at stage {}",
                istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_surv_prob(
            prev_state, prev_fold_state, survive_probs[istage],
            branching_pattern[istage], bias_snr, profile, box_score_widths, rng,
            1.0F, ntrials);
        states[istage]      = cur_state;
        fold_states[istage] = cur_fold_state;
    }
    return states;
}

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)