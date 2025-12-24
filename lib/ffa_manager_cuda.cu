#include "loki/pipelines/ffa_manager.hpp"

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/plans.hpp"
#include "loki/cands.hpp"
#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

class FFAManagerCUDA::BaseImpl {
public:
    BaseImpl()                           = default;
    virtual ~BaseImpl()                  = default;
    BaseImpl(const BaseImpl&)            = delete;
    BaseImpl& operator=(const BaseImpl&) = delete;
    BaseImpl(BaseImpl&&)                 = delete;
    BaseImpl& operator=(BaseImpl&&)      = delete;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) = 0;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class FFAManagerCUDATypedImpl final : public FFAManagerCUDA::BaseImpl {
public:
    using HostFoldType   = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = typename FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    FFAManagerCUDATypedImpl(search::PulsarSearchConfig cfg, int device_id)
        : m_base_cfg(std::move(cfg)),
          m_region_planner(m_base_cfg),
          m_device_id(device_id) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = FFAWorkspaceCUDA<FoldTypeCUDA>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        m_fold_d_time.resize(planner_stats.get_max_buffer_size_time());
        m_scores.resize(planner_stats.get_max_scores_size());
        m_scores_d.resize(planner_stats.get_max_scores_size());
        m_ffa_param_sets.resize(planner_stats.get_max_param_sets_size());

        // Copy scoring widths to device
        m_widths_d = m_base_cfg.get_scoring_widths();

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAManager allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_manager_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_extra_memory_usage());
        spdlog::info("FFAManager will process {} chunks",
                     m_region_planner.get_nregions());
    }

    ~FFAManagerCUDATypedImpl() final                        = default;
    FFAManagerCUDATypedImpl(const FFAManagerCUDATypedImpl&) = delete;
    FFAManagerCUDATypedImpl& operator=(const FFAManagerCUDATypedImpl&) = delete;
    FFAManagerCUDATypedImpl(FFAManagerCUDATypedImpl&&)                 = delete;
    FFAManagerCUDATypedImpl& operator=(FFAManagerCUDATypedImpl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        // timing::ScopeTimer timer("FFAManager::execute");
        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_scoring_widths());

        // Copy input data to device
        cudaStream_t stream = nullptr;
        m_ts_e_d.resize(ts_e.size());
        m_ts_v_d.resize(ts_v.size());
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()), ts_e.data(),
                        ts_e.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()), ts_v.data(),
                        ts_v.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);

        for (const auto& cfg_cur : m_region_planner.get_cfgs()) {
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.front(), freq_limits.back());
            execute_ffa_region(cfg_cur, writer, stream);
        }
        spdlog::info("FFA Pipeline completed.");
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    plans::FFARegionPlanner<HostFoldType> m_region_planner;
    int m_device_id;

    algorithms::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    std::vector<float> m_scores;
    std::vector<double> m_ffa_param_sets; // includes width

    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_d_time;
    thrust::device_vector<float> m_scores_d;
    thrust::device_vector<SizeType> m_widths_d;

    void execute_ffa_region(const search::PulsarSearchConfig& cfg,
                            cands::FFAResultWriter& result_writer,
                            cudaStream_t stream) {
        // Create FFA with shared workspace
        auto the_ffa = FFACUDA<FoldTypeCUDA>(cfg, m_ffa_workspace, m_device_id);
        const plans::FFAPlan<HostFoldType>& ffa_plan = the_ffa.get_plan();
        const auto buffer_size_time = ffa_plan.get_buffer_size_time();
        const auto fold_size_time   = ffa_plan.get_fold_size_time();

        the_ffa.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_e_d.data()), m_ts_e_d.size()),
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_v_d.data()), m_ts_v_d.size()),
            cuda::std::span<float>(
                thrust::raw_pointer_cast(m_fold_d_time.data()),
                buffer_size_time),
            stream);

        // Compute scores
        const auto nsegments       = ffa_plan.get_nsegments().back();
        const auto ncoords         = ffa_plan.get_ncoords().back();
        const auto n_params        = m_base_cfg.get_nparams();
        const auto& scoring_widths = m_base_cfg.get_scoring_widths();
        const SizeType n_widths    = scoring_widths.size();
        const auto n_scores        = ncoords * n_widths;
        error_check::check_equal(nsegments, 1U,
                                 "FFAManager::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        detection::snr_boxcar_3d_cuda_d(
            cuda::std::span(thrust::raw_pointer_cast(m_fold_d_time.data()),
                            fold_size_time),
            ncoords,
            cuda::std::span<const SizeType>(
                thrust::raw_pointer_cast(m_widths_d.data()), m_widths_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(m_scores_d.data()),
                                   n_scores),
            m_device_id);

        // Copy scores to host
        cudaMemcpyAsync(
            m_scores.data(), thrust::raw_pointer_cast(m_scores_d.data()),
            n_scores * sizeof(float), cudaMemcpyDeviceToHost, stream);
        auto scores_span = std::span(m_scores).first(n_scores);

        // Filter scores and param sets
        const SizeType total_params = n_params + 1;
        auto param_sets_span =
            std::span(m_ffa_param_sets).first(n_scores * total_params);
        const auto& param_arr = ffa_plan.get_params().back();
        const auto snr_min    = m_base_cfg.get_snr_min();
        // Write results
        SizeType p_set_idx    = 0;
        SizeType filtered_idx = 0;
        for (const auto& p_set_view :
             utils::cartesian_product_view(param_arr)) {
            for (SizeType w_idx = 0; w_idx < n_widths; ++w_idx) {
                const SizeType score_idx = (p_set_idx * n_widths) + w_idx;
                const float score        = scores_span[score_idx];
                if (score >= snr_min) {
                    for (SizeType j = 0; j < n_params; ++j) {
                        param_sets_span[(filtered_idx * total_params) + j] =
                            p_set_view[j];
                    }
                    param_sets_span[(filtered_idx * total_params) + n_params] =
                        static_cast<double>(scoring_widths[w_idx]);
                    scores_span[filtered_idx] = score;
                    filtered_idx++;
                }
            }
            ++p_set_idx;
        }
        auto scores_span_filtered = scores_span.first(filtered_idx);
        auto param_sets_span_filtered =
            param_sets_span.first(filtered_idx * total_params);
        result_writer.write_results(param_sets_span_filtered,
                                    scores_span_filtered, filtered_idx,
                                    total_params);
    }

}; // End FFAManagerCUDATypedImpl definition

FFAManagerCUDA::FFAManagerCUDA(const search::PulsarSearchConfig& cfg,
                               int device_id) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<FFAManagerCUDATypedImpl<ComplexTypeCUDA>>(
            cfg, device_id);
    } else {
        m_impl =
            std::make_unique<FFAManagerCUDATypedImpl<float>>(cfg, device_id);
    }
}
FFAManagerCUDA::~FFAManagerCUDA()                               = default;
FFAManagerCUDA::FFAManagerCUDA(FFAManagerCUDA&& other) noexcept = default;
FFAManagerCUDA&
FFAManagerCUDA::operator=(FFAManagerCUDA&& other) noexcept = default;

void FFAManagerCUDA::execute(std::span<const float> ts_e,
                             std::span<const float> ts_v,
                             const std::filesystem::path& outdir,
                             std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms