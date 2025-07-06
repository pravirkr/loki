#include "loki/detection/thresholds.hpp"

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/scheme.hpp"
#include "loki/detection/score.hpp"
#include "loki/simulation/simulation.hpp"
#include "loki/utils.hpp"

#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <random>

#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

namespace loki::detection {

namespace {

#ifndef LOKI_CHECK_CUDA_ERROR
#define LOKI_CHECK_CUDA_ERROR(msg)                                             \
    do {                                                                       \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s: %s\n", msg,                     \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

#ifndef CURAND_CHECK
#define CURAND_CHECK(expr)                                                     \
    do {                                                                       \
        curandStatus_t status = (expr);                                        \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "CURAND error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status);                                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

struct FoldVector {
    thrust::device_vector<float> data;
    float variance;
    SizeType ntrials;
    SizeType nbins;

    FoldVector(SizeType ntrials = 0, SizeType nbins = 0, float variance = 0.0F)
        : data(ntrials * nbins),
          variance(variance),
          ntrials(ntrials),
          nbins(nbins) {}

    bool is_empty() const { return ntrials == 0; }
};

struct FoldsType {
    FoldVector folds_h0;
    FoldVector folds_h1;
    bool is_empty_flag = true;

    FoldsType(SizeType ntrials = 0, SizeType nbins = 0, float variance = 0.0F)
        : folds_h0(ntrials, nbins, variance),
          folds_h1(ntrials, nbins, variance),
          is_empty_flag(ntrials == 0) {}

    template <typename FoldVector0, typename FoldVector1>
    FoldsType(FoldVector0&& folds_h0, FoldVector1&& folds_h1)
        : folds_h0(std::forward<FoldVector0>(folds_h0)),
          folds_h1(std::forward<FoldVector1>(folds_h1)) {}

    FoldsType(FoldsType&&)                 = default;
    FoldsType& operator=(FoldsType&&)      = default;
    FoldsType(const FoldsType&)            = delete;
    FoldsType& operator=(const FoldsType&) = delete;

    bool is_empty() const { return is_empty_flag; }
};

struct FoldInfo {
    float* data_h0;
    float* data_h1;
    SizeType ntrials_h0;
    SizeType ntrials_h1;
    SizeType nbins;
    float variance_h0;
    float variance_h1;
    bool is_empty;
};

IndexType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::ranges::upper_bound(bins, value);
    return std::distance(bins.begin(), it) - 1;
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

__global__ void simulate_folds_kernel(float* __restrict__ folds_out,
                                      const float* __restrict__ folds_in,
                                      const float* __restrict__ profile,
                                      const float* __restrict__ noise,
                                      SizeType ntrials_out,
                                      SizeType ntrials_in,
                                      SizeType nbins,
                                      float bias_snr) {
    const SizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntrials_out * nbins)
        return;

    const SizeType trial_idx = i / nbins;
    const SizeType bin_idx   = i % nbins;

    if (trial_idx >= ntrials_out)
        return;

    const SizeType orig_trial =
        (ntrials_in == 0) ? 0 : (trial_idx % ntrials_in);
    const SizeType orig_offset = orig_trial * nbins + bin_idx;

    float fold_in_val = (ntrials_in == 0) ? 0.0f : folds_in[orig_offset];
    folds_out[i]      = fold_in_val + noise[i] + profile[bin_idx] * bias_snr;
}

FoldVector simulate_folds_cuda(const FoldVector& folds_in,
                               const thrust::device_vector<float>& profile,
                               curandGenerator_t& generator,
                               float bias_snr,
                               float var_add,
                               SizeType ntrials_min) {
    const auto ntrials_in = folds_in.ntrials;
    const auto nbins      = folds_in.nbins;

    const auto repeat_factor =
        (ntrials_in == 0)
            ? 1
            : static_cast<SizeType>(std::ceil(static_cast<float>(ntrials_min) /
                                              static_cast<float>(ntrials_in)));
    const auto ntrials_out = std::max(ntrials_min, repeat_factor * ntrials_in);

    FoldVector folds_out(ntrials_out, nbins, folds_in.variance + var_add);

    thrust::device_vector<float> noise(ntrials_out * nbins);
    CURAND_CHECK(curandGenerateNormal(generator,
                                      thrust::raw_pointer_cast(noise.data()),
                                      noise.size(), 0.0f, std::sqrt(var_add)));

    const float* folds_in_ptr =
        ntrials_in > 0 ? thrust::raw_pointer_cast(folds_in.data.data())
                       : nullptr;

    const int threads = 256;
    const int blocks  = (ntrials_out * nbins + threads - 1) / threads;
    simulate_folds_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(folds_out.data.data()), folds_in_ptr,
        thrust::raw_pointer_cast(profile.data()),
        thrust::raw_pointer_cast(noise.data()), ntrials_out, ntrials_in, nbins,
        bias_snr);
    LOKI_CHECK_CUDA_ERROR("simulate_folds_kernel");

    return folds_out;
}

struct is_above_threshold {
    float threshold;
    is_above_threshold(float t) : threshold(t) {}
    __device__ bool operator()(float score) const { return score > threshold; }
};

FoldVector prune_folds_cuda(const FoldVector& folds_in,
                            const thrust::device_vector<float>& scores,
                            float threshold) {
    const auto ntrials_in = folds_in.ntrials;
    const auto nbins      = folds_in.nbins;

    // Find indices of trials that survive the threshold
    thrust::device_vector<SizeType> trial_indices(ntrials_in);
    thrust::sequence(trial_indices.begin(), trial_indices.end());

    auto new_end = thrust::copy_if(trial_indices.begin(), trial_indices.end(),
                                   scores.begin(), trial_indices.begin(),
                                   is_above_threshold(threshold));

    SizeType ntrials_out = thrust::distance(trial_indices.begin(), new_end);
    trial_indices.resize(ntrials_out);

    FoldVector folds_out(ntrials_out, nbins, folds_in.variance);

    if (ntrials_out > 0) {
        // Copy surviving trials
        for (SizeType i = 0; i < ntrials_out; ++i) {
            SizeType old_trial_idx = trial_indices[i];
            thrust::copy_n(folds_in.data.begin() + old_trial_idx * nbins, nbins,
                           folds_out.data.begin() + i * nbins);
        }
    }

    return folds_out;
}

std::tuple<State, FoldsType> gen_next_using_thresh_cuda(
    const State& state_cur,
    const FoldsType& folds_cur,
    float threshold,
    float nbranches,
    float bias_snr,
    const thrust::device_vector<float>& profile,
    const thrust::device_vector<SizeType>& box_score_widths,
    curandGenerator_t& generator,
    float var_add,
    SizeType ntrials_min) {

    // H0 simulation
    const auto folds_h0 = simulate_folds_cuda(
        folds_cur.folds_h0, profile, generator, 0.0f, var_add, ntrials_min);

    thrust::device_vector<float> scores_h0(folds_h0.ntrials);
    if (folds_h0.ntrials > 0) {
        // Create cuda::std::span for the scoring function
        cuda::std::span<const float> folds_span(
            thrust::raw_pointer_cast(folds_h0.data.data()),
            folds_h0.data.size());
        cuda::std::span<const SizeType> widths_span(
            thrust::raw_pointer_cast(box_score_widths.data()),
            box_score_widths.size());
        cuda::std::span<float> scores_span(
            thrust::raw_pointer_cast(scores_h0.data()), scores_h0.size());

        detection::snr_boxcar_2d_max_cuda_d(folds_span, folds_h0.ntrials,
                                            widths_span, scores_span,
                                            std::sqrt(folds_h0.variance));
    }

    const auto folds_h0_pruned =
        prune_folds_cuda(folds_h0, scores_h0, threshold);
    const float success_h0 = (folds_h0.ntrials == 0)
                                 ? 0.0f
                                 : static_cast<float>(folds_h0_pruned.ntrials) /
                                       static_cast<float>(folds_h0.ntrials);

    // H1 simulation
    const auto folds_h1 = simulate_folds_cuda(
        folds_cur.folds_h1, profile, generator, bias_snr, var_add, ntrials_min);

    thrust::device_vector<float> scores_h1(folds_h1.ntrials);
    if (folds_h1.ntrials > 0) {
        // Create cuda::std::span for the scoring function
        cuda::std::span<const float> folds_span(
            thrust::raw_pointer_cast(folds_h1.data.data()),
            folds_h1.data.size());
        cuda::std::span<const SizeType> widths_span(
            thrust::raw_pointer_cast(box_score_widths.data()),
            box_score_widths.size());
        cuda::std::span<float> scores_span(
            thrust::raw_pointer_cast(scores_h1.data()), scores_h1.size());

        detection::snr_boxcar_2d_max_cuda_d(folds_span, folds_h1.ntrials,
                                            widths_span, scores_span,
                                            std::sqrt(folds_h1.variance));
    }

    const auto folds_h1_pruned =
        prune_folds_cuda(folds_h1, scores_h1, threshold);
    const float success_h1 = (folds_h1.ntrials == 0)
                                 ? 0.0f
                                 : static_cast<float>(folds_h1_pruned.ntrials) /
                                       static_cast<float>(folds_h1.ntrials);

    const auto state_next =
        gen_next_state(state_cur, threshold, success_h0, success_h1, nbranches);

    FoldsType new_folds;
    new_folds.folds_h0 = std::move(folds_h0_pruned);
    new_folds.folds_h1 = std::move(folds_h1_pruned);
    new_folds.is_empty_flag =
        new_folds.folds_h0.is_empty() || new_folds.folds_h1.is_empty();

    return {state_next, std::move(new_folds)};
}

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

// CUDA-specific implementation
class DynamicThresholdSchemeCUDA::Impl {
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
         int device_id);
    ~Impl();
    Impl(const Impl&)                = delete;
    Impl& operator=(const Impl&)     = delete;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;

    // Methods
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    void init_states();
    void run_segment(SizeType istage, SizeType thres_neigh);
    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;

    // Host-side parameters and metadata
    std::vector<float> m_branching_pattern;
    float m_ref_ducy;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    SizeType m_trials_start;
    curandGenerator_t m_generator;
    std::vector<float> m_profile_h;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    std::vector<float> m_guess_path;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    float m_bias_snr;
    int m_device_id;

    // Device-side data
    thrust::device_vector<float> m_profile_d;
    thrust::device_vector<SizeType> m_box_score_widths_d;

    // State management
    std::vector<State> m_states;
    std::vector<std::optional<FoldsType>> m_folds_in;
    std::vector<std::optional<FoldsType>> m_folds_out;
};

DynamicThresholdSchemeCUDA::Impl::Impl(std::span<const float> branching_pattern,
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
                                       int device_id)
    : m_branching_pattern(branching_pattern.begin(), branching_pattern.end()),
      m_ref_ducy(ref_ducy),
      m_ntrials(ntrials),
      m_ducy_max(ducy_max),
      m_wtsp(wtsp),
      m_beam_width(beam_width),
      m_trials_start(trials_start),
      m_device_id(device_id) {

    cudaSetDevice(m_device_id);

    unsigned int seed = std::random_device{}();
    CURAND_CHECK(
        curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(m_generator, seed));

    if (m_branching_pattern.empty()) {
        throw std::invalid_argument("Branching pattern is empty");
    }

    // Host-side computations
    m_profile_h   = simulation::generate_folded_profile(nbins, ref_ducy);
    m_thresholds  = detection::compute_thresholds(0.1F, snr_final, nthresholds);
    m_probs       = detection::compute_probs(nprobs, prob_min);
    m_nprobs      = m_probs.size();
    m_nbins       = m_profile_h.size();
    m_nstages     = m_branching_pattern.size();
    m_nthresholds = m_thresholds.size();
    auto box_score_widths_h =
        detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
    m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
    m_guess_path = detection::guess_scheme(m_nstages, snr_final,
                                           m_branching_pattern, m_trials_start);

    // Copy data to device
    m_profile_d          = m_profile_h;
    m_box_score_widths_d = box_score_widths_h;

    // Initialize state management
    State initial_state;
    m_states.resize(m_nstages * m_nthresholds * m_nprobs, initial_state);
    m_folds_in.resize(m_nthresholds * m_nprobs);
    m_folds_out.resize(m_nthresholds * m_nprobs);

    init_states();
}

DynamicThresholdSchemeCUDA::Impl::~Impl() {
    CURAND_CHECK(curandDestroyGenerator(m_generator));
}

void DynamicThresholdSchemeCUDA::Impl::init_states() {
    const float var_init = 1.0F;
    const FoldVector folds_init(m_ntrials, m_nbins);

    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 = simulate_folds_cuda(folds_init, m_profile_d, m_generator,
                                        0.0F, var_init, m_ntrials);
    auto folds_h1 = simulate_folds_cuda(folds_init, m_profile_d, m_generator,
                                        m_bias_snr, var_init, m_ntrials);

    State initial_state;
    FoldsType fold_state{std::move(folds_h0), std::move(folds_h1)};

    const auto thresholds_idx = get_current_thresholds_idx(0);
    for (SizeType ithres : thresholds_idx) {
        auto [cur_state, cur_fold_state] = gen_next_using_thresh_cuda(
            initial_state, fold_state, m_thresholds[ithres],
            m_branching_pattern[0], m_bias_snr, m_profile_d,
            m_box_score_widths_d, m_generator, 1.0F, m_ntrials);

        const auto iprob = find_bin_index(m_probs, cur_state.success_h1_cumul);
        if (iprob < 0 || iprob >= static_cast<IndexType>(m_nprobs)) {
            continue;
        }

        m_states[(ithres * m_nprobs) + iprob]   = cur_state;
        m_folds_in[(ithres * m_nprobs) + iprob] = std::move(cur_fold_state);
    }
}

void DynamicThresholdSchemeCUDA::Impl::run_segment(SizeType istage,
                                                   SizeType thres_neigh) {
    const auto beam_idx_cur      = get_current_thresholds_idx(istage);
    const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
    const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
    const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;

    for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
        const auto ithres = beam_idx_cur[i];
        // Find nearest neighbors in the previous beam
        const auto neighbour_beam_indices = utils::find_neighbouring_indices(
            beam_idx_prev, ithres, thres_neigh);

        for (SizeType jthresh : neighbour_beam_indices) {
            for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                const auto prev_fold_idx = (jthresh * m_nprobs) + kprob;
                const auto prev_state =
                    m_states[stage_offset_prev + prev_fold_idx];

                if (prev_state.is_empty) {
                    continue;
                }

                const auto& prev_fold_state = m_folds_in[prev_fold_idx];
                if (prev_fold_state.has_value() &&
                    !prev_fold_state->is_empty()) {
                    auto [cur_state, cur_fold_state] =
                        gen_next_using_thresh_cuda(
                            prev_state, *prev_fold_state, m_thresholds[ithres],
                            m_branching_pattern[istage], m_bias_snr,
                            m_profile_d, m_box_score_widths_d, m_generator,
                            1.0F, m_ntrials);

                    const auto iprob =
                        find_bin_index(m_probs, cur_state.success_h1_cumul);
                    if (iprob < 0 ||
                        iprob >= static_cast<IndexType>(m_nprobs)) {
                        continue;
                    }

                    const auto cur_idx       = (ithres * m_nprobs) + iprob;
                    const auto cur_state_idx = stage_offset_cur + cur_idx;
                    auto& existing_state     = m_states[cur_state_idx];

                    if (existing_state.is_empty ||
                        cur_state.complexity_cumul <
                            existing_state.complexity_cumul) {
                        existing_state       = cur_state;
                        m_folds_out[cur_idx] = std::move(cur_fold_state);
                    }
                }
            }
        }
    }
}

void DynamicThresholdSchemeCUDA::Impl::run(SizeType thres_neigh) {
    spdlog::info("Running dynamic threshold scheme on CUDA");
    utils::ProgressGuard progress_guard(true);
    auto bar = utils::make_standard_bar("Computing scheme...");

    for (SizeType istage = 1; istage < m_nstages; ++istage) {
        run_segment(istage, thres_neigh);
        // swap fold buffers
        std::swap(m_folds_in, m_folds_out);
        std::ranges::fill(m_folds_out, std::nullopt);

        const auto progress = static_cast<float>(istage) /
                              static_cast<float>(m_nstages - 1) * 100.0F;
        bar.set_progress(static_cast<SizeType>(progress));
    }
}

std::string
DynamicThresholdSchemeCUDA::Impl::save(const std::string& outdir) const {
    const std::filesystem::path filebase = std::format(
        "dynscheme_cuda_nstages_{:03d}_nthresh_{:03d}_nprobs_{:03d}_"
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
    file.createDataSet("profile", m_profile_h);
    file.createDataSet("thresholds", m_thresholds);
    file.createDataSet("probs", m_probs);
    file.createDataSet("guess_path", m_guess_path);
    // Define the 3D dataspace for states
    std::vector<SizeType> dims = {m_nstages, m_nthresholds, m_nprobs};
    HighFive::DataSetCreateProps props_states;
    std::vector<hsize_t> chunk_dims(dims.begin(), dims.end());
    props_states.add(HighFive::Chunking(chunk_dims));
    auto dataset = file.createDataSet("states", HighFive::DataSpace(dims),
                                      create_compound_state(), props_states);
    dataset.write_raw(m_states.data());
    spdlog::info("Saved dynamic threshold scheme to {}", filepath.string());
    return filepath.string();
}

std::vector<SizeType>
DynamicThresholdSchemeCUDA::Impl::get_current_thresholds_idx(
    SizeType istage) const {
    const auto guess       = m_guess_path[istage];
    const auto half_extent = m_beam_width;
    const auto lower_bound = std::max(0.0F, guess - half_extent);
    const auto upper_bound = std::min(m_thresholds.back(), guess + half_extent);

    std::vector<SizeType> result;
    for (SizeType i = 0; i < m_thresholds.size(); ++i) {
        if (m_thresholds[i] >= lower_bound && m_thresholds[i] <= upper_bound) {
            result.push_back(i);
        }
    }
    return result;
}

DynamicThresholdSchemeCUDA::DynamicThresholdSchemeCUDA(
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
    int device_id)
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
                                    device_id)) {}
DynamicThresholdSchemeCUDA::~DynamicThresholdSchemeCUDA() = default;
DynamicThresholdSchemeCUDA::DynamicThresholdSchemeCUDA(
    DynamicThresholdSchemeCUDA&&) noexcept = default;
DynamicThresholdSchemeCUDA& DynamicThresholdSchemeCUDA::operator=(
    DynamicThresholdSchemeCUDA&&) noexcept = default;

void DynamicThresholdSchemeCUDA::run(SizeType thres_neigh) {
    m_impl->run(thres_neigh);
}
std::string DynamicThresholdSchemeCUDA::save(const std::string& outdir) const {
    return m_impl->save(outdir);
}

} // namespace loki::detection

HIGHFIVE_REGISTER_TYPE(loki::detection::State,
                       loki::detection::create_compound_state)