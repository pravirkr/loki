#include "loki/thresholds.hpp"

#include <thrust/device_vector.h>

namespace loki {

namespace {
// CUDA-specific helpers
struct FoldVectorGPU {
    thrust::device_vector<float> data;
    // Other members...
};

struct FoldsTypeGPU {
    FoldVectorGPU folds_h0;
    FoldVectorGPU folds_h1;
    // Methods...
};
} // anonymous namespace

// CUDA-specific implementation
template <> class DynamicThresholdScheme<CUDAExecutionTag>::Impl {
public:
    Impl(std::span<const float> branching_pattern,
         std::span<const float> profile,
         SizeType ntrials,
         SizeType nprobs,
         float prob_min,
         float snr_final,
         SizeType nthresholds,
         float ducy_max,
         float wtsp,
         float beam_width,
         int device_id);

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;
    std::vector<float> get_branching_pattern() const {
        return m_branching_pattern;
    }
    std::vector<float> get_profile() const { return m_profile; }
    std::vector<float> get_thresholds() const { return m_thresholds; }
    std::vector<float> get_probs() const { return m_probs; }
    SizeType get_nstages() const { return m_nstages; }
    SizeType get_nthresholds() const { return m_nthresholds; }
    SizeType get_nprobs() const { return m_nprobs; }
    std::vector<State> get_states() const { return m_states; }
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    thrust::device_vector<float> m_branching_pattern;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    int m_device_id;

    thrust::device_vector<float> m_profile;
    thrust::device_vector<float> m_thresholds;
    thrust::device_vector<float> m_probs;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    float m_bias_snr;
    thrust::device_vector<float> m_guess_path;
    std::vector<std::optional<FoldsTypeGPU>> m_folds_in;
    std::vector<std::optional<FoldsTypeGPU>> m_folds_out;
    thrust::device_vector<State> m_states;

    // Host copies for return values
    std::vector<float> m_branching_pattern_host;
    std::vector<float> m_profile_host;
    std::vector<float> m_thresholds_host;
    std::vector<float> m_probs_host;
    std::vector<State> m_states_host;

    void run_segment_cuda(SizeType istage, SizeType thres_neigh = 10);
    void init_states_cuda();
};

// CUDA implementation of methods
DynamicThresholdScheme<CUDAExecutionTag>::Impl::Impl(
    std::span<const float> branching_pattern,
    std::span<const float> profile,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    int device_id)
    : m_ntrials(ntrials),
      m_ducy_max(ducy_max),
      m_wtsp(wtsp),
      m_beam_width(beam_width),
      m_device_id(device_id),
      m_nprobs(nprobs),
      m_nthresholds(nthresholds) {

    // Set CUDA device
    cudaSetDevice(device_id);

    // Copy data to device
    m_branching_pattern_host.assign(branching_pattern.begin(),
                                    branching_pattern.end());
    m_profile_host.assign(profile.begin(), profile.end());

    m_branching_pattern = thrust::device_vector<float>(
        m_branching_pattern_host.begin(), m_branching_pattern_host.end());
    m_profile = thrust::device_vector<float>(m_profile_host.begin(),
                                             m_profile_host.end());

    // Initialize other members...
    init_states_cuda();
}

void DynamicThresholdScheme<CUDAExecutionTag>::Impl::run(SizeType thres_neigh) {
    // CUDA implementation code
    // Launch CUDA kernels, etc.
}

// CUDA-specific constructor implementation
template <>
template <std::same_as<CUDAExecutionTag> P>
DynamicThresholdScheme<CUDAExecutionTag>::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    std::span<const float> profile,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    int device_id)
    : m_impl(std::make_unique<Impl>(branching_pattern,
                                    profile,
                                    ntrials,
                                    nprobs,
                                    prob_min,
                                    snr_final,
                                    nthresholds,
                                    ducy_max,
                                    wtsp,
                                    beam_width,
                                    device_id)) {}

// Method implementations for CUDA version
template <>
std::vector<float>
DynamicThresholdScheme<CUDAExecutionTag>::get_branching_pattern() const {
    return m_impl->get_branching_pattern();
}
template <>
std::vector<float>
DynamicThresholdScheme<CUDAExecutionTag>::get_profile() const {
    return m_impl->get_profile();
}
template <>
std::vector<float>
DynamicThresholdScheme<CUDAExecutionTag>::get_thresholds() const {
    return m_impl->get_thresholds();
}
template <>
std::vector<float> DynamicThresholdScheme<CUDAExecutionTag>::get_probs() const {
    return m_impl->get_probs();
}
template <>
SizeType DynamicThresholdScheme<CUDAExecutionTag>::get_nstages() const {
    return m_impl->get_nstages();
}
template <>
SizeType DynamicThresholdScheme<CUDAExecutionTag>::get_nthresholds() const {
    return m_impl->get_nthresholds();
}
template <>
SizeType DynamicThresholdScheme<CUDAExecutionTag>::get_nprobs() const {
    return m_impl->get_nprobs();
}
template <>
std::vector<State>
DynamicThresholdScheme<CUDAExecutionTag>::get_states() const {
    return m_impl->get_states();
}
template <>
void DynamicThresholdScheme<CUDAExecutionTag>::run(SizeType thres_neigh) {
    m_impl->run(thres_neigh);
}
template <>
std::string DynamicThresholdScheme<CUDAExecutionTag>::save(
    const std::string& outdir) const {
    return m_impl->save(outdir);
}

} // namespace loki