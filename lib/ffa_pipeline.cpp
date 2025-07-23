#include "loki/pipelines/ffa_pipeline.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/progress.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"

/*
namespace loki::search {

struct SearchRegion {
    std::map<std::string, std::pair<double, double>> param_limits;
    int fold_bins;
    int tol_bins;
};

class SearchCampaign {
public:
    void add_region(SearchRegion region);
    const std::vector<SearchRegion>& get_regions() const;
    double get_max_memory_gb() const;
    void set_max_memory_gb(double max_gb);

private:
    std::vector<SearchRegion> m_regions;
    double m_max_memory_gb = 8.0;
};

void FFASearchOrchestrator::plan_chunks() {

    const auto& limits      = m_base_config.get_param_limits().at("f0");
    double current_f0_start = limits.first;

    while (current_f0_start < limits.second) {
        // Find the largest f0_end that fits in memory for the chunk
        // starting at current_f0_start.

        double low         = current_f0_start;
        double high        = limits.second;
        double best_f0_end = current_f0_start;

        double probe_f0_end = std::min(
            current_f0_start + 1.0, limits.second);

        while (probe_f0_end <= limits.second) {
            PulsarSearchConfig chunk_config = m_base_config;
            auto chunk_limits               = chunk_config.get_param_limits();
            chunk_limits["f0"]              = {current_f0_start, probe_f0_end};
            chunk_config.set_param_limits(chunk_limits);

            plans::FFAPlan plan(chunk_config);
            if (plan.get_memory_usage() > m_max_memory_bytes) {
                break;
            }

            best_f0_end = probe_f0_end;

            if (probe_f0_end == limits.second)
                break;

            // Increase probe, maybe by doubling the current chunk width
            probe_f0_end =
                std::min(probe_f0_end + (probe_f0_end - current_f0_start),
                         limits.second);
        }

        // Finalize and store this chunk
        PulsarSearchConfig final_chunk_config = m_base_config;
        auto final_chunk_limits  = final_chunk_config.get_param_limits();
        final_chunk_limits["f0"] = {current_f0_start, best_f0_end};
        final_chunk_config.set_param_limits(final_chunk_limits);
        m_chunks.push_back(final_chunk_config);

        // Move to the next chunk
        current_f0_start = best_f0_end;
    }
}
} // namespace loki::search

namespace loki::algorithms {

class FFAPipeline::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg,
                  float max_memory_gb,
                  bool show_progress)
        : m_base_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_nthreads(m_base_cfg.get_nthreads()),
          m_plan_partitioner(m_base_cfg, max_memory_gb) {
        // Allocate buffers once, sized for the largest chunk
        const auto buffer_size = m_plan_partitioner.get_max_buffer_size();
        m_fold_in.resize(buffer_size, 0.0F);
        m_fold_out.resize(buffer_size, 0.0F);
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);

        // Log the actual memory usage for the reused buffers
        const auto memory_bytes = buffer_size * 2 * sizeof(float);
        const auto memory_gb_alloc =
            static_cast<float>(memory_bytes) / (1024.0F * 1024.0F * 1024.0F);
        spdlog::info("FFAPipeline allocated {:.2f} GB for FFA buffers.",
                     memory_gb_alloc);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) {
        error_check::check_equal(
            ts_e.size(), m_base_cfg.get_nsamps(),
            "FFA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFA::Impl::execute: ts_v must have size nsamps");

        timing::ScopeTimer timer("FFA::execute");

        const auto& plan_chunks  = m_plan_partitioner.get_plans();
        const auto& score_widths = m_base_cfg.get_score_widths();
        const auto nscores_max =
            plan_chunks.get_n_param_sets_max() * score_widths.size();
        std::vector<float> scores(nscores_max);
        spdlog::info("Starting FFA search with {} planned chunks.",
                     plan_chunks.size());
        for (const auto& ffa_plan_cur : plan_chunks) {
            const auto& freqs = ffa_plan_cur.params.back().back();
            spdlog::info("Processing chunk f0: [{}, {}]", freqs.front(),
                         freqs.back());
            const auto nscores =
                ffa_plan_cur.ncoords.back() * score_widths.size();
            auto scores_span = std::span(scores).first(nscores);
            execute_ffa_plan(ts_e, ts_v, ffa_plan_cur, scores_span);
            // Write scores to file outdir/file_prefix_f0.h5 (placeholder)
        }
        spdlog::info("FFA Pipeline completed.");
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    bool m_show_progress;
    plans::FFAPlanPartitioner m_plan_partitioner;
    int m_nthreads;
    std::unique_ptr<algorithms::BruteFold> m_the_bf;

    // Buffers for the FFA plan, shared across chunks
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    float* init_buffer) {
        timing::ScopeTimer timer("FFA::initialize");
        m_the_bf->execute(ts_e, ts_v,
                          std::span(init_buffer, m_the_bf->get_fold_size()));
    }

    void execute_ffa_plan(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          const plans::FFAPlan& ffa_plan,
                          std::span<float> scores) {
        // Create BruteFold for this specific chunk on the stack
        const auto t_ref =
            m_base_cfg.get_nparams() == 1 ? 0.0 : ffa_plan.tsegments[0] / 2.0;
        const auto& freqs_arr = ffa_plan.params[0].back();
        BruteFold the_bf(freqs_arr, ffa_plan.segment_lens[0],
                         m_base_cfg.get_nbins(), m_base_cfg.get_nsamps(),
                         m_base_cfg.get_tsamp(), t_ref, m_nthreads);

        initialize(ts_e, ts_v, m_fold_in.data());
        float* fold_in_ptr  = m_fold_in.data();
        float* fold_out_ptr = m_fold_out.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;
        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(fold_in_ptr, fold_out_ptr, i_level, ffa_plan);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(fold_in_ptr, fold_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
        // FFA output is in fold_out_ptr
        // Compute scores
        const auto nsegments    = ffa_plan.nsegments.back();
        const auto n_param_sets = ffa_plan.ncoords.back();
        error_check::check_equal(
            nsegments, 1U,
            "compute_ffa_scores: nsegments must be 1 for scores");
        const auto fold_span =
            std::span(fold_out_ptr, ffa_plan.get_fold_size());
        detection::snr_boxcar_3d(fold_span, n_param_sets,
                                 m_base_cfg.get_score_widths(), scores,
                                 m_nthreads);
    }

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level,
                      const plans::FFAPlan& ffa_plan) {
        const auto coords_cur   = ffa_plan.coordinates[i_level];
        const auto coords_prev  = ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = ffa_plan.fold_shapes[i_level].back();
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();

        // Choose strategy based on level characteristics
        if (nsegments >= 256) {
            execute_iter_segment(fold_in, fold_out, coords_cur, nsegments,
                                 nbins, ncoords_cur, ncoords_prev);
        } else {
            execute_iter_standard(fold_in, fold_out, coords_cur, nsegments,
                                  nbins, ncoords_cur, ncoords_prev);
        }
    }

    void execute_iter_segment(const float* __restrict__ fold_in,
                              float* __restrict__ fold_out,
                              const auto& coords_cur,
                              SizeType nsegments,
                              SizeType nbins,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev) {
        // Process one segment at a time to keep data in cache
        constexpr SizeType kBlockSize = 32;

        std::vector<float> temp_buffer(2 * nbins);
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev) firstprivate(temp_buffer)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            // Process coordinates in blocks within each segment
            for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
                 icoord_block += kBlockSize) {
                SizeType block_end =
                    std::min(icoord_block + kBlockSize, ncoords_cur);
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_tail * 2 * nbins);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_head * 2 * nbins);
                    const auto out_offset =
                        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);
                    const float* __restrict__ fold_tail = &fold_in[tail_offset];
                    const float* __restrict__ fold_head = &fold_in[head_offset];
                    float* __restrict__ fold_sum        = &fold_out[out_offset];
                    float* __restrict__ temp_buffer_ptr = temp_buffer.data();
                    kernels::shift_add_buffer(fold_tail, coord_cur.shift_tail,
                                              fold_head, coord_cur.shift_head,
                                              fold_sum, temp_buffer_ptr, nbins);
                }
            }
        }
    }

    void execute_iter_standard(const float* __restrict__ fold_in,
                               float* __restrict__ fold_out,
                               const auto& coords_cur,
                               SizeType nsegments,
                               SizeType nbins,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev) {
        constexpr SizeType kBlockSize = 32;

        std::vector<float> temp_buffer(2 * nbins);
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev) firstprivate(temp_buffer)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_tail * 2 * nbins);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_head * 2 * nbins);
                    const auto out_offset =
                        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

                    const float* __restrict__ fold_tail = &fold_in[tail_offset];
                    const float* __restrict__ fold_head = &fold_in[head_offset];
                    float* __restrict__ fold_sum        = &fold_out[out_offset];
                    float* __restrict__ temp_buffer_ptr = temp_buffer.data();
                    kernels::shift_add_buffer(fold_tail, coord_cur.shift_tail,
                                              fold_head, coord_cur.shift_head,
                                              fold_sum, temp_buffer_ptr, nbins);
                }
            }
        }
    }
}; // End FFAPipeline::Impl definition

FFAPipeline::FFAPipeline(const search::PulsarSearchConfig& cfg,
                         float max_memory_gb,
                         bool show_progress)
    : m_impl(std::make_unique<Impl>(cfg, show_progress)) {}
FFAPipeline::~FFAPipeline()                                       = default;
FFAPipeline::FFAPipeline(FFAPipeline&& other) noexcept            = default;
FFAPipeline& FFAPipeline::operator=(FFAPipeline&& other) noexcept = default;

void FFAPipeline::execute(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          const std::filesystem::path& outdir,
                          std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms
 */