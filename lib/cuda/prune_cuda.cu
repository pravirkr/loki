#include "lib/cuda/prune_cuda.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/algorithms/prune_engine.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/dynamic_cuda.cuh"
#include "lib/cuda/ffa_cuda.cuh"
#include "lib/cuda/interrupt_cuda.cuh"
#include "lib/cuda/world_tree_cuda.cuh"
#include "lib/detail/error_check.hpp"
#include "lib/detail/nvtx.hpp"
#include "lib/detail/psr_utils.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detail/utils.hpp"
#include "lib/search/cands.hpp"
#include "lib/search/ep_support.hpp"
#include "lib/search/prune_mask.hpp"

namespace loki::algorithms {

namespace {

struct ExecutionStream {
    cudaStream_t stream{nullptr};
    bool owns{false};

    explicit ExecutionStream(int device_id) {
        cuda_utils::CudaSetDeviceGuard guard(device_id);
        cuda_utils::check_cuda_call(
            cudaStreamCreate(&stream),
            "EPMultiPassCudaCore: cudaStreamCreate failed");
        owns = true;
    }

    explicit ExecutionStream(cudaStream_t external_stream)
        : stream(external_stream),
          owns(false) {
        if (stream == nullptr) {
            throw std::invalid_argument("EPMultiPassCudaCore: execution_stream "
                                        "must be non-null when using "
                                        "an external EPWorkspaceCUDA");
        }
    }

    ~ExecutionStream() noexcept {
        if (owns && stream != nullptr) {
            // Destructors may run while unwinding an exception; never throw.
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
            (void)cudaGetLastError();
        }
    }

    ExecutionStream(const ExecutionStream&)            = delete;
    ExecutionStream& operator=(const ExecutionStream&) = delete;
    ExecutionStream(ExecutionStream&&)                 = delete;
    ExecutionStream& operator=(ExecutionStream&&)      = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream; }
};

__global__ void kernel_cached_score_mask(const float* scores,
                                         const uint8_t* valid,
                                         uint8_t* flags,
                                         float threshold,
                                         SizeType n) {
    const SizeType i =
        static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        flags[i] =
            static_cast<uint8_t>(valid[i] != 0 && scores[i] >= threshold);
    }
}

/// Clears the validation flag of leaves whose base-grid cell is masked
/// (@p bits in the GridMask layout) and adds the number cleared to
/// @p n_masked (if not null).
__global__ void
kernel_mask_resolved(const uint32_t* __restrict__ param_idx,
                     uint8_t* __restrict__ valid,
                     const uint64_t* __restrict__ bits,
                     SizeType n_cells,
                     SizeType n,
                     unsigned long long* __restrict__ n_masked) {
    const SizeType i =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int masked = 0;
    if (i < n && valid[i] != 0) {
        const SizeType c = param_idx[i];
        if (c < n_cells && ((bits[c >> 6U] >> (c & 63U)) & 1ULL) != 0ULL) {
            valid[i] = 0;
            masked   = 1;
        }
    }
    const int block_masked = __syncthreads_count(masked);
    if (threadIdx.x == 0 && block_masked > 0 && n_masked != nullptr) {
        atomicAdd(n_masked, static_cast<unsigned long long>(block_masked));
    }
}

/// Lists the entries j < n (slot (start + j) mod capacity of @p scores) with
/// a score >= @p threshold and, if @p flags is not null, flags[j] set.
__global__ void kernel_list_harvest(const float* __restrict__ scores,
                                    const uint8_t* __restrict__ flags,
                                    SizeType start,
                                    SizeType capacity,
                                    SizeType n,
                                    float threshold,
                                    uint32_t* __restrict__ out,
                                    unsigned long long* __restrict__ count) {
    const SizeType j =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (j >= n || (flags != nullptr && flags[j] == 0)) {
        return;
    }
    const SizeType slot =
        start + j < capacity ? start + j : start + j - capacity;
    if (scores[slot] >= threshold) {
        out[atomicAdd(count, 1ULL)] = static_cast<uint32_t>(j);
    }
}

/// dst row r = src row (start + idx[r]) mod capacity, rows of @p stride.
template <typename T>
__global__ void kernel_gather_rows(const T* __restrict__ src,
                                   SizeType stride,
                                   SizeType start,
                                   SizeType capacity,
                                   const uint32_t* __restrict__ idx,
                                   SizeType n_rows,
                                   T* __restrict__ dst) {
    const SizeType e =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (e >= n_rows * stride) {
        return;
    }
    const SizeType r = e / stride;
    const SizeType k = e - (r * stride);
    const SizeType j = idx[r];
    const SizeType slot =
        start + j < capacity ? start + j : start + j - capacity;
    dst[e] = src[(slot * stride) + k];
}

/// valid[list[r]] = 0 for r < n.
__global__ void kernel_clear_listed(uint8_t* __restrict__ valid,
                                    const uint32_t* __restrict__ list,
                                    SizeType n) {
    const SizeType r =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (r < n) {
        valid[list[r]] = 0;
    }
}

/// flags[i] &= valid[i]; adds the set flags to @p count.
__global__ void kernel_and_count(uint8_t* __restrict__ flags,
                                 const uint8_t* __restrict__ valid,
                                 SizeType n,
                                 unsigned long long* __restrict__ count) {
    const SizeType i =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int set = 0;
    if (i < n) {
        flags[i] = static_cast<uint8_t>(flags[i] != 0 && valid[i] != 0);
        set      = flags[i];
    }
    const int block_set = __syncthreads_count(set);
    if (threadIdx.x == 0 && block_set > 0) {
        atomicAdd(count, static_cast<unsigned long long>(block_set));
    }
}

constexpr unsigned kRfiBlock = 256;

unsigned rfi_grid(SizeType n) {
    return static_cast<unsigned>((n + kRfiBlock - 1) / kRfiBlock);
}

// Number of individual harvests echoed to the log at info level per run, as
// on CPU.
constexpr SizeType kMaxHarvestLogLines = 16;

template <SupportedFoldTypeCUDA FoldTypeCUDA> class PruneCUDAImpl {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;
    PruneCUDAImpl(memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
                  search::PulsarSearchConfig cfg,
                  std::span<const float> threshold_scheme,
                  SizeType max_sugg,
                  SizeType batch_size,
                  SizeType branch_max,
                  std::string_view poly_basis,
                  int device_id,
                  cudaStream_t stream,
                  std::function<void(pipelines::detail::EPResultPayload)>
                      result_sink                   = {},
                  const PruneRFIConfig* rfi         = nullptr,
                  const search::GridMask* mask_base = nullptr)
        : m_workspace_ptr(&workspace),
          m_cfg(cfg.has_explicit_nsegments() ? std::move(cfg)
                                             : cfg.with_ep_default_nsegments()),
          m_ffa_plan(m_cfg),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_branch_max(branch_max),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()),
          m_device_id(device_id),
          m_stream(stream),
          m_result_sink(std::move(result_sink)),
          m_rfi(rfi != nullptr && rfi->is_active() ? rfi : nullptr),
          m_mask_base(mask_base) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        if (m_rfi != nullptr) {
            error_check::check(m_mask_base != nullptr,
                               "PruneCUDAImpl: RFI control needs a base mask");
            error_check::check(!m_rfi->impulsive_veto,
                               "PruneCUDAImpl: the impulsive veto is CPU only");
            m_counters_d.resize(2);
        }
        error_check::check_less_equal(m_cfg.get_nparams(), 5,
                                      "Pruning not supported for nparams > 5.");
        m_prune_funcs = core::create_prune_dp_functs_cuda<FoldTypeCUDA>(
            m_poly_basis, m_ffa_plan.get_param_counts().back(),
            m_ffa_plan.get_dparams_actual().back(),
            m_ffa_plan.get_nsegments_active(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max, m_device_id, m_stream);
    }

    ~PruneCUDAImpl()                               = default;
    PruneCUDAImpl(const PruneCUDAImpl&)            = delete;
    PruneCUDAImpl& operator=(const PruneCUDAImpl&) = delete;
    PruneCUDAImpl(PruneCUDAImpl&&)                 = delete;
    PruneCUDAImpl& operator=(PruneCUDAImpl&&)      = delete;

    [[nodiscard]] float get_memory_usage_gib() const noexcept {
        return get_workspace().get_memory_usage_gib() +
               m_prune_funcs->get_scorer_scratch_memory_gib();
    }

    void execute(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType ref_seg,
                 std::span<const SizeType> ascend_levels,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file,
                 int task_id) {
        timing::SimpleTimer timer;
        timer.start();
        const std::string run_name =
            fmt::format("{:03d}_{:02d}", ref_seg, task_id);

        const auto& ws                 = get_workspace();
        const auto memory_tree_gb      = ws.world_tree.get_memory_usage_gib();
        const auto memory_workspace_gb = ws.prune.get_memory_usage_gib();
        const auto memory_scorer_scratch_gb =
            m_prune_funcs->get_scorer_scratch_memory_gib();
        const auto memory_total_gb = get_memory_usage_gib();
        spdlog::debug(
            "Pruning run {:03d}: Memory Usage: {:.2f} GB total "
            "({:.2f} GB tree + {:.2f} GB workspace + {:.3f} GB scorer scratch)",
            ref_seg, memory_total_gb, memory_tree_gb, memory_workspace_gb,
            memory_scorer_scratch_gb);

        // Setup log and result files
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / fmt::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / fmt::format("tmp_{}_results.h5", run_name));

        const auto nsegments = m_ffa_plan.get_nsegments_active();

        initialize(ffa_fold, ref_seg, m_stream);
        spdlog::debug("Pruning run {:03d}: initialized", ref_seg);

        SizeType iterations_completed = 0;
        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            loki::interrupt::check(m_stream);
            execute_iteration(ffa_fold, m_stream);
            iterations_completed = iter + 1;
            // Check for early termination (no survivors)
            if (m_prune_complete) {
                break;
            }
            // Should we reintegrate survivors at this level?
            if (!ascend_levels.empty() &&
                std::ranges::find(ascend_levels, m_prune_level) !=
                    ascend_levels.end()) {
                ascend_survivors_batched(ffa_fold, m_stream);
            }
        }

        // Log early exit ONCE after loop if needed
        if (m_prune_complete && iterations_completed < nsegments - 1) {
            spdlog::debug(
                "Pruning terminated early at iteration {} - no survivors",
                iterations_completed);
        }

        // Final ascend if needed
        if (ascend_levels.empty() ||
            std::ranges::find(ascend_levels, m_prune_level) ==
                ascend_levels.end()) {
            ascend_survivors_batched(ffa_fold, m_stream);
        }

        // Write results (after transforming to middle of the data)
        report_survivors(actual_result_file, run_name);

        // Final log entries
        std::ofstream log(actual_log_file, std::ios::app);
        log << fmt::format("Pruning log for ref segment: {}\n", ref_seg);
        log << m_pstats.get_all_summaries();
        log << fmt::format("Pruning run complete for ref segment {}\n",
                           ref_seg);
        log.close();

        spdlog::debug(
            "Pruning run {:03d}: complete (peak live {}, overload trims {})",
            ref_seg, get_workspace().world_tree.peak_live(),
            get_workspace().world_tree.overload_count());
        spdlog::debug("Pruning run {:03d}: stats: {}", ref_seg,
                      m_pstats.get_stats_summary_cuda(timer.stop()));
    }

private:
    // The observer pointer that always points to the active workspace.
    memory::EPWorkspaceCUDA<FoldTypeCUDA>* m_workspace_ptr{nullptr};

    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan<HostFoldT> m_ffa_plan;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    SizeType m_branch_max;
    std::string_view m_poly_basis;
    SizeType m_total_levels;
    int m_device_id;
    cudaStream_t m_stream;
    std::function<void(pipelines::detail::EPResultPayload)> m_result_sink;

    bool m_prune_complete{false};
    SizeType m_prune_level{};
    psr_utils::MiddleOutScheme m_snail_scheme;
    search::PruneStatsCollection m_pstats;
    std::unique_ptr<core::PruneDPFunctsCUDA<FoldTypeCUDA>> m_prune_funcs;

    // RFI control (CPU semantics, see PruneRFIConfig; no impulsive veto).
    // The host mask is authoritative; m_mask_d is its copy for the kernels.
    const PruneRFIConfig* m_rfi{nullptr};
    const search::GridMask* m_mask_base{nullptr};
    search::GridMask m_mask;
    bool m_mask_on_device{false};
    search::HarvestBuffer<HostFoldT> m_harvest;
    SizeType m_n_harvested_total{0};
    thrust::device_vector<uint64_t> m_mask_d;
    thrust::device_vector<uint32_t> m_seed_keep_d;
    // [0]: leaves masked in the level, [1]: scratch count
    thrust::device_vector<unsigned long long> m_counters_d;
    thrust::device_vector<double> m_harvest_leaves_d;
    thrust::device_vector<float> m_harvest_scores_d;
    thrust::device_vector<FoldTypeCUDA> m_harvest_folds_d;

    [[nodiscard]] memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    [[nodiscard]] double get_tobs_scheme() const noexcept {
        return static_cast<double>(m_ffa_plan.get_nsegments_active()) *
               m_ffa_plan.get_tsegments().back();
    }

    [[nodiscard]] float get_harvest_threshold(SizeType level) const noexcept {
        if (m_rfi == nullptr || m_rfi->harvest_scheme.empty() ||
            level > m_rfi->harvest_scheme.size()) {
            return kHarvestDisabled;
        }
        return m_rfi->harvest_scheme[level - 1];
    }

    /// Copies the host mask to the device (ordered on m_stream; the host
    /// mask is only changed after a later synchronisation).
    void upload_mask() {
        const auto words = m_mask.words();
        if (m_mask_d.size() != words.size()) {
            m_mask_d.resize(words.size());
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_mask_d.data()),
                            words.data(), words.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, m_stream),
            "EP RFI mask upload failed");
        m_mask_on_device = !m_mask.empty();
    }

    /// Clears validation flags of the batch's resolved leaves in masked
    /// cells, counting them into the level's masked count if @p count.
    void mask_resolved(SizeType n_leaves, bool count) {
        auto& prune_ws = get_workspace().prune;
        kernel_mask_resolved<<<rfi_grid(n_leaves), kRfiBlock, 0, m_stream>>>(
            thrust::raw_pointer_cast(prune_ws.branched_param_idx_d.data()),
            thrust::raw_pointer_cast(prune_ws.validation_mask_d.data()),
            thrust::raw_pointer_cast(m_mask_d.data()), m_mask.get_n_cells(),
            n_leaves,
            count ? thrust::raw_pointer_cast(m_counters_d.data()) : nullptr);
        cuda_utils::check_last_cuda_error("EP RFI mask kernel failed");
    }

    /// Where harvest candidates are read from: entry j < n is slot
    /// (start + j) mod capacity of leaves/scores (and folds, when stored).
    struct HarvestSource {
        const double* leaves{nullptr};
        const float* scores{nullptr};
        const FoldTypeCUDA* folds{nullptr}; ///< null: computed per record
        const uint8_t* flags{nullptr};      ///< null: every entry
        /// Not null: validation flags of the entries, cleared for every
        /// listed entry (harvested, or skipped as already masked).
        uint8_t* clear_valid{nullptr};
        SizeType start{0};
        SizeType capacity{0};
        SizeType n{0};
    };

    /**
     * @brief Early harvest of the entries of @p src at or above
     * @p threshold, with the CPU sequence: strongest first, skip a source
     * already masked, else widen the mask (window and harmonics) and record
     * it (up to max_harvests). @p fold_of computes entry j's fold into
     * m_harvest_folds_d row r when src.folds is null.
     *
     * @return Number of harvests (0: the mask is unchanged); @p n_listed
     * receives the number of entries at or above @p threshold.
     */
    template <typename FoldFn>
    SizeType harvest_entries(const HarvestSource& src,
                             float threshold,
                             std::pair<double, double> coord_cur,
                             SizeType seg_idx_cur,
                             FoldFn&& fold_of,
                             SizeType* n_listed = nullptr) {
        auto& prune_ws           = get_workspace().prune;
        const auto leaves_stride = prune_ws.leaves_stride;
        const auto folds_stride  = prune_ws.folds_stride;
        auto* const count_d = thrust::raw_pointer_cast(m_counters_d.data()) + 1;
        auto* const list_d =
            thrust::raw_pointer_cast(prune_ws.passing_indices_d.data());
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(count_d, 0, sizeof(unsigned long long), m_stream),
            "EP harvest count reset failed");
        kernel_list_harvest<<<rfi_grid(src.n), kRfiBlock, 0, m_stream>>>(
            src.scores, src.flags, src.start, src.capacity, src.n, threshold,
            list_d, count_d);
        cuda_utils::check_last_cuda_error("EP harvest list kernel failed");
        unsigned long long n_cand = 0;
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(&n_cand, count_d, sizeof(n_cand),
                            cudaMemcpyDeviceToHost, m_stream),
            "EP harvest count copy failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream),
                                    "EP harvest count wait failed");
        if (n_listed != nullptr) {
            *n_listed = static_cast<SizeType>(n_cand);
        }
        if (n_cand == 0) {
            return 0;
        }
        const auto k = static_cast<SizeType>(n_cand);
        if (src.clear_valid != nullptr) {
            kernel_clear_listed<<<rfi_grid(k), kRfiBlock, 0, m_stream>>>(
                src.clear_valid, list_d, k);
            cuda_utils::check_last_cuda_error("EP harvest clear failed");
        }
        if (m_harvest_leaves_d.size() < k * leaves_stride) {
            m_harvest_leaves_d.resize(k * leaves_stride);
            m_harvest_scores_d.resize(k);
        }
        kernel_gather_rows<double>
            <<<rfi_grid(k * leaves_stride), kRfiBlock, 0, m_stream>>>(
                src.leaves, leaves_stride, src.start, src.capacity, list_d, k,
                thrust::raw_pointer_cast(m_harvest_leaves_d.data()));
        kernel_gather_rows<float><<<rfi_grid(k), kRfiBlock, 0, m_stream>>>(
            src.scores, 1, src.start, src.capacity, list_d, k,
            thrust::raw_pointer_cast(m_harvest_scores_d.data()));
        cuda_utils::check_last_cuda_error("EP harvest gather failed");
        // Physical parameters at t_ref = coord_cur, as CPU harvest_candidate.
        m_prune_funcs->report(
            cuda_utils::as_span(m_harvest_leaves_d).first(k * leaves_stride),
            coord_cur, k, m_stream);
        std::vector<double> leaves_h(k * leaves_stride);
        std::vector<float> scores_h(k);
        std::vector<uint32_t> entry_h(k);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_h.data(),
                            thrust::raw_pointer_cast(m_harvest_leaves_d.data()),
                            leaves_h.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, m_stream),
            "EP harvest leaves copy failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(scores_h.data(),
                            thrust::raw_pointer_cast(m_harvest_scores_d.data()),
                            k * sizeof(float), cudaMemcpyDeviceToHost,
                            m_stream),
            "EP harvest scores copy failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(entry_h.data(), list_d, k * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, m_stream),
            "EP harvest list copy failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream),
                                    "EP harvest copy wait failed");

        // Strongest first; ties in entry order (the CPU's stable sort).
        std::vector<SizeType> order(k);
        std::iota(order.begin(), order.end(), SizeType{0});
        std::ranges::sort(order, [&](SizeType x, SizeType y) {
            return scores_h[x] != scores_h[y] ? scores_h[x] > scores_h[y]
                                              : entry_h[x] < entry_h[y];
        });
        constexpr SizeType kParamStride = 2U;
        const auto n_params             = m_cfg.get_nparams();
        const auto a_lim = m_cfg.get_param_limits()[n_params - 2];
        const auto a_max = std::max(std::abs(a_lim.min), std::abs(a_lim.max));
        const auto t_ref = coord_cur.first;
        std::vector<SizeType> recorded;
        SizeType n_harvested = 0;
        for (const auto r : order) {
            const auto* leaf = leaves_h.data() + (r * leaves_stride);
            const auto a     = leaf[(n_params - 2) * kParamStride];
            const auto da = std::abs(leaf[((n_params - 2) * kParamStride) + 1]);
            const auto f  = leaf[(n_params - 1) * kParamStride];
            const auto df = std::abs(leaf[((n_params - 1) * kParamStride) + 1]);
            if (m_mask.is_masked_at(a, f)) {
                continue; // a source this batch already harvested
            }
            const auto window =
                make_harvest_window(f, a, df, da, t_ref, get_tobs_scheme(),
                                    m_rfi->harvest_mask_ntiles, a_max);
            m_mask.add_window(window, m_rfi->n_harmonics,
                              m_rfi->fractional_harmonics);
            ++m_n_harvested_total;
            ++n_harvested;
            if (m_harvest.size() + recorded.size() < m_rfi->max_harvests) {
                recorded.push_back(r);
                if (m_harvest.size() + recorded.size() <= kMaxHarvestLogLines) {
                    spdlog::info("Pruning level {:3d}: harvested candidate "
                                 "score {:.2f}, f = {:.9f} Hz, a = {:.4e} "
                                 "m/s^2 (t_ref = {:.1f} s)",
                                 m_prune_level, scores_h[r], f, a, t_ref);
                }
            }
        }
        if (n_harvested == 0) {
            return 0;
        }
        upload_mask();
        if (recorded.empty()) {
            return n_harvested;
        }
        std::vector<HostFoldT> folds_h;
        if (m_rfi->harvest_store_folds) {
            const auto n_rec = recorded.size();
            if (m_harvest_folds_d.size() < n_rec * folds_stride) {
                m_harvest_folds_d.resize(n_rec * folds_stride);
            }
            for (SizeType q = 0; q < n_rec; ++q) {
                const auto j = static_cast<SizeType>(entry_h[recorded[q]]);
                auto* dst = thrust::raw_pointer_cast(m_harvest_folds_d.data()) +
                            (q * folds_stride);
                if (src.folds != nullptr) {
                    const auto slot = src.start + j < src.capacity
                                          ? src.start + j
                                          : src.start + j - src.capacity;
                    cuda_utils::check_cuda_call(
                        cudaMemcpyAsync(dst, src.folds + (slot * folds_stride),
                                        folds_stride * sizeof(FoldTypeCUDA),
                                        cudaMemcpyDeviceToDevice, m_stream),
                        "EP harvest fold copy failed");
                } else {
                    fold_of(j, dst);
                }
            }
            folds_h.resize(n_rec * folds_stride);
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(
                    folds_h.data(),
                    thrust::raw_pointer_cast(m_harvest_folds_d.data()),
                    folds_h.size() * sizeof(FoldTypeCUDA),
                    cudaMemcpyDeviceToHost, m_stream),
                "EP harvest folds copy failed");
            cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream),
                                        "EP harvest folds wait failed");
        }
        if (m_harvest.empty()) {
            // One allocation of the cap, as the CPU's (ep_harvest_bytes).
            m_harvest.reserve(m_rfi->max_harvests);
        }
        for (SizeType q = 0; q < recorded.size(); ++q) {
            const auto r = recorded[q];
            m_harvest.push(std::span<const double>(leaves_h).subspan(
                               r * leaves_stride, leaves_stride),
                           m_rfi->harvest_store_folds
                               ? std::span<const HostFoldT>(folds_h).subspan(
                                     q * folds_stride, folds_stride)
                               : std::span<const HostFoldT>{},
                           scores_h[r], m_prune_level, seg_idx_cur, t_ref);
        }
        return n_harvested;
    }

    void initialize(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                    SizeType ref_seg,
                    cudaStream_t stream) {
        //  Reset the suggestion buffer state
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        world_tree.reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments_active();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme = psr_utils::MiddleOutScheme(nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;

        // Initialize the world tree with the first segment
        const auto fold_segment =
            m_prune_funcs->load_segment(ffa_fold, m_snail_scheme.get_ref_idx());
        const auto coord_init = m_snail_scheme.get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        // A sweep workspace is sized for the largest chunk in its group.
        // Seed only this chunk's active coordinates, not the whole allocation.
        const auto seed_leaves =
            cuda_utils::as_span(ws.seed_leaves_d)
                .first(n_leaves * world_tree.get_leaves_stride());
        const auto seed_scores =
            cuda_utils::as_span(ws.seed_scores_d).first(n_leaves);
        m_prune_funcs->seed(fold_segment, seed_leaves, seed_scores, coord_init,
                            stream);
        // Per-run RFI state: start from the chunk's static mask.
        SizeType n_seeds_masked = 0;
        m_mask_on_device        = false;
        if (m_rfi != nullptr) {
            m_mask.assign(*m_mask_base);
            m_harvest = search::HarvestBuffer<HostFoldT>(
                world_tree.get_leaves_stride(), world_tree.get_folds_stride(),
                m_rfi->harvest_store_folds);
            m_n_harvested_total = 0;
            if (!m_mask.empty()) {
                upload_mask();
            }
        }
        if (!m_mask_on_device) {
            world_tree.add_initial(seed_leaves, fold_segment, seed_scores,
                                   n_leaves, stream);
        } else {
            // Seed leaf i sits on base-grid cell i: drop the masked ones.
            std::vector<SizeType> keep(n_leaves);
            const auto n_keep = m_mask.select_seeds(keep, n_leaves);
            n_seeds_masked    = n_leaves - n_keep;
            std::vector<uint32_t> keep_u32(keep.begin(), keep.begin() + n_keep);
            m_seed_keep_d.resize(std::max<SizeType>(n_keep, 1));
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_seed_keep_d.data()),
                                keep_u32.data(), n_keep * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "EP RFI seed indices upload failed");
            world_tree.add_initial_scattered(
                seed_leaves, fold_segment, seed_scores,
                cuda_utils::as_span(m_seed_keep_d), n_keep, stream);
            // keep_u32 is pageable and dies here.
            cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                        "EP RFI seed upload wait failed");
            spdlog::debug("Pruning run {:03d}: pulsar mask removed {} of {} "
                          "seeds ({} grid cells masked)",
                          ref_seg, n_seeds_masked, n_leaves, m_mask.count());
        }

        // Initialize the prune stats
        m_pstats = search::PruneStatsCollection();
        const search::PruneStats pstats_cur{
            .level           = m_prune_level,
            .seg_idx         = m_snail_scheme.get_segment_idx(m_prune_level),
            .threshold       = 0,
            .score_min       = world_tree.get_score_min(stream),
            .score_max       = world_tree.get_score_max(stream),
            .n_branches      = n_leaves,
            .n_leaves        = n_leaves,
            .n_leaves_phy    = world_tree.get_size(),
            .n_leaves_surv   = world_tree.get_size(),
            .n_leaves_masked = n_seeds_masked,
        };
        m_pstats.update_stats(pstats_cur);
        if (world_tree.get_size() == 0) {
            m_prune_complete = true;
        }
    }

    void ascend_survivors_batched(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                                  cudaStream_t stream) {
        if (m_prune_complete) {
            return;
        }
        auto& ws               = get_workspace();
        auto& world_tree       = ws.world_tree;
        const auto n_survivors = world_tree.get_size();
        if (n_survivors == 0) {
            return;
        }
        auto& prune_ws       = ws.prune;
        const auto coord_mid = m_snail_scheme.get_coord(m_prune_level);
        const auto segment_coords_so_far =
            m_snail_scheme.get_segment_coords_so_far(m_prune_level);
        const auto& idx_segments   = segment_coords_so_far.first;
        const auto& coord_segments = segment_coords_so_far.second;
        const auto batch_cap =
            std::max(SizeType{1}, std::min(m_batch_size, n_survivors));
        const auto n_segments_so_far = idx_segments.size();

        if (n_segments_so_far > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(ws.idx_segments_d.data()),
                    idx_segments.data(), n_segments_so_far * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync idx_segments failed");
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(ws.coord_segments_d.data()),
                    coord_segments.data(),
                    n_segments_so_far * sizeof(std::pair<double, double>),
                    cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync coord_segments failed");
            // Both sources are pageable and die with this call. Wait until
            // the copies have read them.
            cuda_utils::check_cuda_call(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize segment upload failed");
        }

        memory::CircularViewCUDA<double> leaves_cv =
            world_tree.get_leaves_circular_view();
        memory::CircularViewCUDA<FoldTypeCUDA> folds_cv =
            world_tree.get_folds_circular_view();
        memory::CircularViewCUDA<float> scores_cv =
            world_tree.get_scores_circular_view();
        memory::CircularViewCUDA<float> scores_ep_cv =
            world_tree.get_scores_ep_circular_view();

        const auto leaves_stride = world_tree.get_leaves_stride();
        const auto folds_stride  = world_tree.get_folds_stride();

        auto process_region = [&](cuda::std::span<const double> leaves_region,
                                  cuda::std::span<FoldTypeCUDA> folds_region,
                                  cuda::std::span<float> scores_region,
                                  cuda::std::span<float> scores_ep_region) {
            if (leaves_region.empty()) {
                return;
            }
            error_check::check_equal(
                leaves_region.size() % leaves_stride, SizeType{0},
                "ascend_survivors_batched: leaves segment not stride-aligned");
            const auto n_leaves_region = leaves_region.size() / leaves_stride;
            error_check::check_equal(
                folds_region.size(), n_leaves_region * folds_stride,
                "ascend_survivors_batched: folds/leaves leaf count mismatch");
            error_check::check_equal(
                scores_region.size(), n_leaves_region,
                "ascend_survivors_batched: scores/leaves leaf count mismatch");
            error_check::check_equal(
                scores_ep_region.size(), n_leaves_region,
                "ascend_survivors_batched: scores_ep/leaves leaf count "
                "mismatch");

            for (SizeType off = 0; off < n_leaves_region; off += batch_cap) {
                const auto chunk = std::min(batch_cap, n_leaves_region - off);
                m_prune_funcs->ascend(
                    ffa_fold,
                    leaves_region.subspan(off * leaves_stride,
                                          chunk * leaves_stride),
                    folds_region.subspan(off * folds_stride,
                                         chunk * folds_stride),
                    scores_region.subspan(off, chunk),
                    scores_ep_region.subspan(off, chunk),
                    cuda_utils::as_span(ws.idx_segments_d)
                        .first(n_segments_so_far),
                    cuda_utils::as_span(ws.coord_segments_d)
                        .first(n_segments_so_far),
                    coord_mid,
                    cuda_utils::as_span(prune_ws.branched_param_idx_d),
                    cuda_utils::as_span(prune_ws.branched_phase_shift_d), chunk,
                    stream);
            }
        };

        process_region(leaves_cv.first, folds_cv.first, scores_cv.first,
                       scores_ep_cv.first);
        process_region(leaves_cv.second, folds_cv.second, scores_cv.second,
                       scores_ep_cv.second);
    }

    void report_survivors(const std::filesystem::path& actual_result_file,
                          std::string_view run_name) {
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;

        memory::CircularViewCUDA<double> leaves_view =
            world_tree.get_leaves_circular_view();
        memory::CircularViewCUDA<float> scores_view =
            world_tree.get_scores_circular_view();
        memory::CircularViewCUDA<float> scores_ep_view =
            world_tree.get_scores_ep_circular_view();

        const auto n_leaves      = world_tree.get_size();
        const auto leaves_stride = world_tree.get_leaves_stride();
        const auto coord_mid     = m_snail_scheme.get_coord(m_prune_level);
        const auto n1            = leaves_view.first.size() / leaves_stride;
        const auto n2            = leaves_view.second.size() / leaves_stride;

        std::vector<std::uint32_t> widths_report_h;
        if (m_poly_basis == "taylor") {
            widths_report_h.resize(n_leaves);
            const auto folds   = world_tree.get_folds_circular_view();
            auto& scratch      = ws.prune;
            SizeType written   = 0;
            const auto collect = [&](auto region) {
                const auto stride = world_tree.get_folds_stride();
                const auto count  = region.size() / stride;
                for (SizeType p = 0; p < count; p += m_batch_size) {
                    const auto n = std::min(m_batch_size, count - p);
                    auto widths =
                        cuda_utils::as_span(scratch.branched_indices_d)
                            .first(n);
                    auto scores =
                        cuda_utils::as_span(scratch.branched_scores_d).first(n);
                    m_prune_funcs->final_widths(
                        region.subspan(p * stride, n * stride), widths, scores,
                        m_stream);
                    cuda_utils::check_cuda_call(
                        cudaMemcpyAsync(widths_report_h.data() + written,
                                        widths.data(),
                                        n * sizeof(std::uint32_t),
                                        cudaMemcpyDeviceToHost, m_stream),
                        "copy final EP widths failed");
                    written += n;
                }
            };
            collect(folds.first);
            collect(folds.second);
        }

        if (n_leaves > 0) {
            // Transform the suggestion params to middle of the data
            search::detail::check_report_epoch(m_cfg, coord_mid.first,
                                               "EPMultiPass::report");
            m_prune_funcs->report(leaves_view.first, coord_mid, n1, m_stream);
            if (n2 > 0) {
                m_prune_funcs->report(leaves_view.second, coord_mid, n2,
                                      m_stream);
            }
        }

        // Copy the results to the host
        std::vector<double> leaves_report_h(n_leaves * leaves_stride);
        std::vector<float> scores_report_h(n_leaves);
        std::vector<float> scores_ep_report_h(n_leaves);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_report_h.data(), leaves_view.first.data(),
                            (n1 * leaves_stride) * sizeof(double),
                            cudaMemcpyDeviceToHost, m_stream),
            "cudaMemcpyAsync leaves_report failed");
        if (n2 > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(leaves_report_h.data() + (n1 * leaves_stride),
                                leaves_view.second.data(),
                                (n2 * leaves_stride) * sizeof(double),
                                cudaMemcpyDeviceToHost, m_stream),
                "cudaMemcpyAsync leaves_report failed");
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(scores_report_h.data(), scores_view.first.data(),
                            n1 * sizeof(float), cudaMemcpyDeviceToHost,
                            m_stream),
            "cudaMemcpyAsync scores_report part 1 failed");
        if (n2 > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(scores_report_h.data() + n1,
                                scores_view.second.data(), n2 * sizeof(float),
                                cudaMemcpyDeviceToHost, m_stream),
                "cudaMemcpyAsync scores_report part 2 failed");
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(scores_ep_report_h.data(),
                            scores_ep_view.first.data(), n1 * sizeof(float),
                            cudaMemcpyDeviceToHost, m_stream),
            "cudaMemcpyAsync scores_ep_report part 1 failed");
        if (n2 > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(scores_ep_report_h.data() + n1,
                                scores_ep_view.second.data(),
                                n2 * sizeof(float), cudaMemcpyDeviceToHost,
                                m_stream),
                "cudaMemcpyAsync scores_ep_report part 2 failed");
        }

        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(m_stream),
            "cudaStreamSynchronize write results failed");

        if (m_result_sink) {
            pipelines::detail::EPResultPayload payload{
                .run_name     = std::string(run_name),
                .snail_scheme = m_snail_scheme.get_data(),
                .leaves       = std::move(leaves_report_h),
                .scores       = std::move(scores_report_h),
                .scores_ep    = std::move(scores_ep_report_h),
                .widths       = std::move(widths_report_h),
                .gflops       = compute_total_prune_gflops(),
                .nparams      = m_cfg.get_nparams(),
                .stats        = m_pstats,
            };
            if (m_rfi != nullptr && m_rfi->has_harvest()) {
                payload.harvest           = std::move(m_harvest);
                payload.n_harvested_total = m_n_harvested_total;
            }
            m_result_sink(std::move(payload));
            return;
        }
        error_check::check(m_rfi == nullptr,
                           "PruneCUDAImpl: RFI control needs the sweep writer");
        // Both host vectors are now contiguous — wrap as single-span views
        memory::CircularView<double> leaves_report_view{
            std::span<double>(leaves_report_h), std::span<double>{}};
        memory::CircularView<float> scores_report_view{
            std::span<float>(scores_report_h), std::span<float>{}};
        memory::CircularView<float> scores_ep_report_view{
            std::span<float>(scores_ep_report_h), std::span<float>{}};

        const auto total_pruning_gflops = compute_total_prune_gflops();
        // Write results
        auto result_writer = search::PruneResultWriter(
            actual_result_file, search::PruneResultWriter::Mode::kAppend);
        result_writer.write_run_results(
            run_name, m_snail_scheme.get_data(), leaves_report_view,
            scores_report_view, scores_ep_report_view, total_pruning_gflops,
            n_leaves, m_cfg.get_nparams(), m_pstats, widths_report_h);
    }

    [[nodiscard]] double compute_total_prune_gflops() const {
        const auto packed_stats = m_pstats.get_packed_data();
        const auto& level_stats = packed_stats.first;
        if (level_stats.empty()) {
            return 0.0;
        }

        const auto n_params = static_cast<double>(m_cfg.get_nparams());
        const auto nbins    = static_cast<double>(m_cfg.get_nbins());
        const auto nbins_f  = static_cast<double>(m_cfg.get_nbins_f());
        const auto n_widths = static_cast<double>(m_cfg.get_n_scoring_widths());
        const auto conservative_tile =
            static_cast<double>(m_cfg.get_use_conservative_tile());

        auto score_flops = [&](double n_leaves) {
            return (n_leaves * 2.0) * (n_widths * 2.0 * nbins);
        };
        auto irfft_flops = [&](double n_leaves) {
            if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
                return (2.0 * n_leaves) * nbins * std::log2(nbins);
            } else {
                return 0.0;
            }
        };
        auto shift_add_flops = [&](double n_leaves) {
            if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
                return n_leaves * 2.0 * nbins_f * 8.0;
            } else {
                return n_leaves * 2.0 * nbins;
            }
        };

        auto branch_flops = [&](double n_branches, double n_leaves) {
            // Dominant Taylor branch arithmetic: per-parameter step/shift work
            // over input branches plus child-center generation over outputs.
            return (n_branches * n_params * 10.0) + (n_leaves * n_params * 2.0);
        };
        auto resolve_flops = [&](double n_leaves) {
            // Polynomial propagation to acceleration/frequency plus phase and
            // nearest-grid arithmetic. The order-dependent part scales with
            // the number of Taylor parameters.
            return n_leaves * ((6.0 * n_params) + 16.0);
        };
        auto transform_flops = [&](double n_leaves) {
            // Value propagation is triangular in the Taylor order; conservative
            // tiles also propagate uncertainty with squared terms and sqrt.
            const auto value_flops = n_params * (n_params + 1.0);
            const auto error_flops =
                conservative_tile * n_params * ((2.0 * n_params) + 1.0);
            return n_leaves * (value_flops + error_flops);
        };
        auto report_flops = [&](double n_leaves) {
            // Gauge transform and error propagation for all non-frequency
            // parameters, plus final frequency/error conversion.
            return n_leaves * (((n_params - 1.0) * 12.0) + 4.0);
        };

        double total_flops = 0.0;
        for (const auto& stats : level_stats) {
            const auto n_branches    = static_cast<double>(stats.n_branches);
            const auto n_leaves      = static_cast<double>(stats.n_leaves);
            const auto n_leaves_phy  = static_cast<double>(stats.n_leaves_phy);
            const auto n_leaves_surv = static_cast<double>(stats.n_leaves_surv);

            if (stats.level == 0) {
                total_flops += irfft_flops(n_leaves_surv);
                total_flops += score_flops(n_leaves_surv);
                continue;
            }

            total_flops += branch_flops(n_branches, n_leaves);
            total_flops += resolve_flops(n_leaves_phy);
            total_flops += shift_add_flops(n_leaves_phy);
            total_flops += irfft_flops(n_leaves_phy);
            total_flops += score_flops(n_leaves_phy);
            total_flops += transform_flops(n_leaves_surv);
        }

        const auto& final_stats = level_stats.back();
        const auto n_final_survivors =
            static_cast<double>(final_stats.n_leaves_surv);
        const auto segment_coords =
            m_snail_scheme.get_segment_coords_so_far(m_prune_level);
        const auto n_segments =
            static_cast<double>(segment_coords.first.size());

        total_flops += n_segments * resolve_flops(n_final_survivors);
        total_flops += n_segments * shift_add_flops(n_final_survivors);
        total_flops += irfft_flops(n_final_survivors);
        total_flops += score_flops(n_final_survivors);
        total_flops += report_flops(n_final_survivors);

        return total_flops * 1.0e-9;
    }

    void execute_iteration(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                           cudaStream_t stream) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        error_check::check_less_equal(
            m_prune_level, m_total_levels,
            fmt::format("Pruning complete - exceeded total levels at level "
                        "{}",
                        m_prune_level));
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        world_tree.prepare_in_place_update();

        search::PruneIterationStats stats;
        const auto seg_idx_cur = m_snail_scheme.get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches  = world_tree.get_size_old();
        const auto n_overloads = world_tree.overload_count();

        if (m_rfi != nullptr) {
            cuda_utils::check_cuda_call(
                cudaMemsetAsync(thrust::raw_pointer_cast(m_counters_d.data()),
                                0, sizeof(unsigned long long), stream),
                "EP RFI counter reset failed");
        }
        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats,
                                  stream);
        if (m_rfi != nullptr) {
            unsigned long long n_masked = 0;
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(
                    &n_masked, thrust::raw_pointer_cast(m_counters_d.data()),
                    sizeof(n_masked), cudaMemcpyDeviceToHost, stream),
                "EP RFI counter copy failed");
            cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                        "EP RFI counter wait failed");
            stats.n_leaves_masked += static_cast<SizeType>(n_masked);
            if (stats.n_harvested > 0) {
                spdlog::debug("Pruning level {:3d}: harvested {} candidate(s) "
                              "(run total {}, recorded {}, mask cells {})",
                              m_prune_level, stats.n_harvested,
                              m_n_harvested_total, m_harvest.size(),
                              m_mask.count());
            }
        }

        // Finalize: make new region active, defragment for contiguous access.
        world_tree.finalize_in_place_update();

        // Update statistics
        stats.norm_scores(world_tree.get_size());
        const search::PruneStats pstats_cur{
            .level           = m_prune_level,
            .seg_idx         = seg_idx_cur,
            .threshold       = threshold,
            .score_min       = stats.score_min,
            .score_max       = stats.score_max,
            .n_branches      = n_branches,
            .n_leaves        = stats.n_leaves,
            .n_leaves_phy    = stats.n_leaves_phy,
            .n_leaves_surv   = world_tree.get_size(),
            .n_leaves_masked = stats.n_leaves_masked,
            .n_harvested     = stats.n_harvested,
            .n_overloads     = world_tree.overload_count() - n_overloads,
        };
        m_pstats.update_stats(pstats_cur);

        // Check if no survivors
        if (world_tree.get_size() == 0) {
            m_prune_complete = true;
            return;
        }
    }

    // Iteration flow: Branch -> Validate -> Resolve -> Load/Shift/Add -> Score
    // -> Filter -> Transform -> Add to buffer.
    void execute_iteration_batched(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                                   SizeType seg_idx_cur,
                                   float threshold,
                                   search::PruneIterationStats& stats,
                                   cudaStream_t stream) {

        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        auto& prune_ws   = ws.prune;
        auto& scratch_ws = ws.scratch;

        // Get coordinates
        const auto coord_init = m_snail_scheme.get_coord(0);
        const auto coord_prev = m_snail_scheme.get_coord(m_prune_level - 1);
        const auto coord_next = m_snail_scheme.get_coord(m_prune_level);
        const auto coord_cur  = m_snail_scheme.get_current_coord(m_prune_level);
        const auto coord_add  = m_snail_scheme.get_segment_coord(m_prune_level);

        // Load fold segment for current level
        const auto ffa_fold_segment =
            m_prune_funcs->load_segment(ffa_fold, seg_idx_cur);

        auto current_threshold = threshold;

        const auto n_branches = world_tree.get_size_old();
        const auto batch_size =
            std::max(1UL, std::min(m_batch_size, n_branches));

        // Early harvest at this level (the batch's max score gates the work).
        const auto harvest_threshold = get_harvest_threshold(m_prune_level);
        const bool harvest_active    = is_harvest_enabled(harvest_threshold);

        MinMaxFloat batch_minmax{};
        auto accumulate_min_max = [&stats](const MinMaxFloat& minmax) {
            stats.score_min = std::min(stats.score_min, minmax.min);
            stats.score_max = std::max(stats.score_max, minmax.max);
        };
        // A direct-mode batch is collected once the next batch's branch is
        // enqueued (its count sync then also covers the fused kernel).
        bool direct_pending        = false;
        SizeType direct_write_head = 0;
        auto collect_direct        = [&]() {
            if (!direct_pending) {
                return;
            }
            SizeType n_valid = 0;
            auto n_passing =
                m_prune_funcs->collect_shift_add_score(batch_minmax, &n_valid);
            stats.n_leaves_phy += n_valid;
            accumulate_min_max(batch_minmax);
            if (harvest_active && n_passing > 0 &&
                batch_minmax.max >= harvest_threshold) {
                // The survivors sit in the tree's uncommitted region, not yet
                // transformed. Note collect_direct() runs after branch() for
                // the next batch, so keep branched_leaves_d, branched_indices_d
                // and validation_mask_d untouched here; use filtered_mask_d and
                // resolved scratch instead.
                const auto dw       = world_tree.get_direct_write();
                const auto capacity = world_tree.get_capacity();
                auto* const keep_d =
                    thrust::raw_pointer_cast(prune_ws.filtered_mask_d.data());
                cuda_utils::check_cuda_call(
                    cudaMemsetAsync(keep_d, 1, n_passing * sizeof(uint8_t),
                                    stream),
                    "EP harvest direct keep reset failed");
                SizeType n_listed = 0;
                const auto n_h    = harvest_entries(
                    HarvestSource{.leaves      = dw.leaves,
                                  .scores      = dw.scores,
                                  .folds       = dw.folds,
                                  .clear_valid = keep_d,
                                  .start       = direct_write_head,
                                  .capacity    = capacity,
                                  .n           = n_passing},
                    harvest_threshold, coord_cur, seg_idx_cur,
                    [](SizeType, FoldTypeCUDA*) {}, &n_listed);
                stats.n_harvested += n_h;
                if (n_listed > 0) {
                    if (n_h > 0) {
                        // Also drop any surviving leaf in this batch whose
                        // resolved cell fell into the newly widened mask.
                        const auto leaves_stride =
                            world_tree.get_leaves_stride();
                        const auto n1 =
                            std::min(n_passing, capacity - direct_write_head);
                        const auto n2 = n_passing - n1;
                        const cuda::std::span<uint8_t> keep_span(keep_d,
                                                                 n_passing);
                        auto param_idx_span =
                            cuda_utils::as_span(prune_ws.branched_param_idx_d);
                        auto phase_shift_span = cuda_utils::as_span(
                            prune_ws.branched_phase_shift_d);
                        m_prune_funcs->resolve(
                            cuda::std::span<const double>(
                                dw.leaves + (direct_write_head * leaves_stride),
                                n1 * leaves_stride),
                            keep_span.first(n1), param_idx_span.first(n1),
                            phase_shift_span.first(n1), coord_add, coord_cur,
                            coord_init, n1, stream);
                        if (n2 > 0) {
                            m_prune_funcs->resolve(
                                cuda::std::span<const double>(
                                    dw.leaves, n2 * leaves_stride),
                                keep_span.subspan(n1, n2),
                                param_idx_span.subspan(n1, n2),
                                phase_shift_span.subspan(n1, n2), coord_add,
                                coord_cur, coord_init, n2, stream);
                        }
                        kernel_mask_resolved<<<rfi_grid(n_passing), kRfiBlock,
                                               0, stream>>>(
                            param_idx_span.data(), keep_d,
                            thrust::raw_pointer_cast(m_mask_d.data()),
                            m_mask.get_n_cells(), n_passing, nullptr);
                        cuda_utils::check_last_cuda_error(
                            "EP RFI mask kernel failed");
                    }
                    n_passing = world_tree.commit_direct_write_filtered(
                        n_passing,
                        cuda::std::span<const uint8_t>(keep_d, n_passing),
                        stream);
                } else {
                    world_tree.commit_direct_write(n_passing);
                }
            } else {
                world_tree.commit_direct_write(n_passing);
            }
            if (n_passing > 0) {
                const auto capacity      = world_tree.get_capacity();
                const auto leaves_stride = world_tree.get_leaves_stride();
                auto* const tree_leaves  = world_tree.get_direct_write().leaves;
                const auto n1 =
                    std::min(n_passing, capacity - direct_write_head);
                const auto n2 = n_passing - n1;
                m_prune_funcs->transform(
                    cuda::std::span<double>(
                        tree_leaves + (direct_write_head * leaves_stride),
                        n1 * leaves_stride),
                    {}, coord_next, coord_cur, n1, stream);
                if (n2 > 0) {
                    m_prune_funcs->transform(
                        cuda::std::span<double>(tree_leaves,
                                                n2 * leaves_stride),
                        {}, coord_next, coord_cur, n2, stream);
                }
            }
            direct_pending = false;
        };

        // Process branches in batches
        // Process branches in potentially split batches to handle wraps
        auto branch_ws_view      = ws.branch.get_view();
        SizeType total_processed = 0;
        while (total_processed < n_branches) {
            const SizeType remaining       = n_branches - total_processed;
            const SizeType this_batch_size = std::min(batch_size, remaining);

            // Get contiguous span; it may be smaller if wrap occurs
            // Read from the beginning of unconsumed data
            auto [leaves_tree_span, current_batch_size] =
                world_tree.get_leaves_span(this_batch_size);
            if (current_batch_size == 0) {
                throw std::runtime_error(
                    fmt::format("Loaded batch size is 0: total_processed={}, "
                                "this_batch_size={}, remaining={}",
                                total_processed, this_batch_size, remaining));
            }
            total_processed += current_batch_size;

            // Branch
            const auto n_leaves_batch = m_prune_funcs->branch(
                leaves_tree_span,
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d), coord_cur,
                coord_prev, current_batch_size, branch_ws_view, scratch_ws,
                stream);
            collect_direct();
            stats.n_leaves += n_leaves_batch;
            if (n_leaves_batch == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_batch, prune_ws.max_branched_leaves,
                "Branch factor exceeded workspace size:n_leaves_batch <= "
                "max_branched_leaves");

            // Validation (asynchronous: valid count is accumulated in the
            // fused shift-add-score kernel and read back with its stats).
            m_prune_funcs->validate_async(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d), coord_cur,
                n_leaves_batch, stream);

            // Direct path: when even an all-passing batch fits the free
            // slots (add_batch_scattered's fast path), the fused kernel
            // writes the survivors into the tree itself, in the order of
            // the compaction below, and only those survivors are transformed
            // once collected.
            const bool direct = n_leaves_batch <= world_tree.get_space_left();
            if (direct) {
                m_prune_funcs->resolve(
                    cuda_utils::as_span(prune_ws.branched_leaves_d),
                    cuda_utils::as_span(prune_ws.validation_mask_d),
                    cuda_utils::as_span(prune_ws.branched_param_idx_d),
                    cuda_utils::as_span(prune_ws.branched_phase_shift_d),
                    coord_add, coord_cur, coord_init, n_leaves_batch, stream);
                if (m_mask_on_device) {
                    mask_resolved(n_leaves_batch, /*count=*/true);
                }
                const auto dw     = world_tree.get_direct_write();
                direct_write_head = dw.write_head;
                const core::EPDirectWriteCUDA<FoldTypeCUDA> target{
                    .leaves_batch =
                        cuda_utils::as_span(prune_ws.branched_leaves_d),
                    .leaves_stride = world_tree.get_leaves_stride(),
                    .leaves        = dw.leaves,
                    .folds         = dw.folds,
                    .scores        = dw.scores,
                    .write_head    = dw.write_head,
                };
                m_prune_funcs->shift_add_score(
                    world_tree.get_folds_span(),
                    cuda_utils::as_span(prune_ws.branched_indices_d),
                    cuda_utils::as_span(prune_ws.validation_mask_d),
                    ffa_fold_segment,
                    cuda_utils::as_span(prune_ws.branched_param_idx_d),
                    cuda_utils::as_span(prune_ws.branched_phase_shift_d),
                    cuda_utils::as_span(prune_ws.branched_folds_d),
                    cuda_utils::as_span(prune_ws.branched_scores_d),
                    cuda_utils::as_span(prune_ws.filtered_mask_d),
                    current_threshold, n_leaves_batch,
                    world_tree.get_physical_start_idx(),
                    world_tree.get_capacity(), scratch_ws, stream, &target);
                direct_pending = true;
                world_tree.consume_read(current_batch_size);
                continue;
            }

            m_prune_funcs->resolve(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.validation_mask_d),
                cuda_utils::as_span(prune_ws.branched_param_idx_d),
                cuda_utils::as_span(prune_ws.branched_phase_shift_d), coord_add,
                coord_cur, coord_init, n_leaves_batch, stream);
            if (m_mask_on_device) {
                mask_resolved(n_leaves_batch, /*count=*/true);
            }

            // Shift-add and score (one fused kernel for Fourier folds).
            // branched_indices_d are logical indices, converted to physical
            // indices in the kernel.
            SizeType n_valid      = 0;
            auto n_leaves_passing = m_prune_funcs->shift_add_score(
                world_tree.get_folds_span(),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d),
                ffa_fold_segment,
                cuda_utils::as_span(prune_ws.branched_param_idx_d),
                cuda_utils::as_span(prune_ws.branched_phase_shift_d),
                cuda::std::span<FoldTypeCUDA>{},
                cuda_utils::as_span(prune_ws.branched_scores_d),
                cuda_utils::as_span(prune_ws.filtered_mask_d),
                current_threshold, n_leaves_batch,
                world_tree.get_physical_start_idx(), world_tree.get_capacity(),
                scratch_ws, stream, nullptr, &batch_minmax, &n_valid);
            stats.n_leaves_phy += n_valid;

            accumulate_min_max(batch_minmax);

            if (harvest_active && n_leaves_passing > 0 &&
                batch_minmax.max >= harvest_threshold) {
                SizeType n_listed         = 0;
                const auto physical_start = world_tree.get_physical_start_idx();
                const auto tree_capacity  = world_tree.get_capacity();
                const auto n_h            = harvest_entries(
                    HarvestSource{.leaves = thrust::raw_pointer_cast(
                                      prune_ws.branched_leaves_d.data()),
                                  .scores = thrust::raw_pointer_cast(
                                      prune_ws.branched_scores_d.data()),
                                  .flags = thrust::raw_pointer_cast(
                                      prune_ws.filtered_mask_d.data()),
                                  .clear_valid = thrust::raw_pointer_cast(
                                      prune_ws.validation_mask_d.data()),
                                  .capacity = n_leaves_batch,
                                  .n        = n_leaves_batch},
                    harvest_threshold, coord_cur, seg_idx_cur,
                    [&](SizeType j, FoldTypeCUDA* dst) {
                        // The fused kernel kept no folds: recompute entry j.
                        m_prune_funcs->shift_add(
                            world_tree.get_folds_span(),
                            cuda_utils::as_span(prune_ws.branched_indices_d)
                                .subspan(j, 1),
                            cuda_utils::as_span(prune_ws.filtered_mask_d)
                                .subspan(j, 1),
                            ffa_fold_segment,
                            cuda_utils::as_span(prune_ws.branched_param_idx_d)
                                .subspan(j, 1),
                            cuda_utils::as_span(prune_ws.branched_phase_shift_d)
                                .subspan(j, 1),
                            cuda_utils::as_span(prune_ws.branched_folds_d), 1,
                            physical_start, tree_capacity, stream);
                        cuda_utils::check_cuda_call(
                            cudaMemcpyAsync(
                                dst,
                                thrust::raw_pointer_cast(
                                    prune_ws.branched_folds_d.data()),
                                prune_ws.folds_stride * sizeof(FoldTypeCUDA),
                                cudaMemcpyDeviceToDevice, stream),
                            "EP harvest fold copy failed");
                    },
                    &n_listed);
                stats.n_harvested += n_h;
                if (n_listed > 0) {
                    // The listed leaves are cleared; also drop every leaf of
                    // the batch in a cell the harvests masked (CPU step 4).
                    if (n_h > 0) {
                        mask_resolved(n_leaves_batch, /*count=*/false);
                    }
                    auto* const count_d =
                        thrust::raw_pointer_cast(m_counters_d.data()) + 1;
                    cuda_utils::check_cuda_call(
                        cudaMemsetAsync(count_d, 0, sizeof(unsigned long long),
                                        stream),
                        "EP harvest count reset failed");
                    kernel_and_count<<<rfi_grid(n_leaves_batch), kRfiBlock, 0,
                                       stream>>>(
                        thrust::raw_pointer_cast(
                            prune_ws.filtered_mask_d.data()),
                        thrust::raw_pointer_cast(
                            prune_ws.validation_mask_d.data()),
                        n_leaves_batch, count_d);
                    cuda_utils::check_last_cuda_error(
                        "EP harvest recount failed");
                    unsigned long long n_left = 0;
                    cuda_utils::check_cuda_call(
                        cudaMemcpyAsync(&n_left, count_d, sizeof(n_left),
                                        cudaMemcpyDeviceToHost, stream),
                        "EP harvest recount copy failed");
                    cuda_utils::check_cuda_call(
                        cudaStreamSynchronize(stream),
                        "EP harvest recount wait failed");
                    n_leaves_passing = static_cast<SizeType>(n_left);
                }
            }

            if (n_leaves_passing == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_passing, prune_ws.max_branched_leaves,
                "n_leaves_passing <= max_branched_leaves");

            // Transform kernel
            m_prune_funcs->transform(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.filtered_mask_d), coord_next,
                coord_cur, n_leaves_batch, stream);

            {
                scratch_ws.convert_mask_to_indices(
                    cuda_utils::as_span(prune_ws.filtered_mask_d),
                    cuda_utils::as_span(prune_ws.passing_indices_d),
                    n_leaves_batch, stream);
                current_threshold = world_tree.prepare_scored_batch(
                    cuda_utils::as_span(prune_ws.branched_scores_d),
                    cuda_utils::as_span(prune_ws.passing_indices_d),
                    n_leaves_passing, current_threshold,
                    m_cfg.get_overflow_keep_fraction(), stream);
                // Keep parents readable until every accepted child's fold
                // has been recomputed with the existing shift-add arithmetic.
                const auto stride = world_tree.get_leaves_stride();
                for (SizeType off = 0; off < n_leaves_batch;
                     off += m_batch_size) {
                    const auto count =
                        std::min(m_batch_size, n_leaves_batch - off);
                    const auto flags =
                        cuda_utils::as_span(prune_ws.filtered_mask_d)
                            .subspan(off, count);
                    const auto scores =
                        cuda_utils::as_span(prune_ws.branched_scores_d)
                            .subspan(off, count);
                    const auto valid =
                        cuda_utils::as_span(prune_ws.validation_mask_d)
                            .subspan(off, count);
                    const auto grid =
                        static_cast<unsigned>((count + 255) / 256);
                    kernel_cached_score_mask<<<grid, 256, 0, stream>>>(
                        scores.data(), valid.data(), flags.data(),
                        current_threshold, count);
                    cuda_utils::check_last_cuda_error(
                        "EP cached score mask failed");
                    scratch_ws.convert_mask_to_indices(
                        flags, cuda_utils::as_span(prune_ws.passing_indices_d),
                        count, stream);
                    uint32_t selected = 0;
                    cuda_utils::check_cuda_call(
                        cudaMemcpyAsync(&selected, scratch_ws.d_reduce_out,
                                        sizeof(selected),
                                        cudaMemcpyDeviceToHost, stream),
                        "EP recomputation count copy failed");
                    cuda_utils::check_cuda_call(
                        cudaStreamSynchronize(stream),
                        "EP recomputation count wait failed");
                    const auto passing = std::min<SizeType>(
                        selected, world_tree.get_space_left());
                    if (passing != 0) {
                        m_prune_funcs->shift_add(
                            world_tree.get_folds_span(),
                            cuda_utils::as_span(prune_ws.branched_indices_d)
                                .subspan(off, count),
                            flags, ffa_fold_segment,
                            cuda_utils::as_span(prune_ws.branched_param_idx_d)
                                .subspan(off, count),
                            cuda_utils::as_span(prune_ws.branched_phase_shift_d)
                                .subspan(off, count),
                            cuda_utils::as_span(prune_ws.branched_folds_d),
                            count, world_tree.get_physical_start_idx(),
                            world_tree.get_capacity(), stream);
                        current_threshold = world_tree.add_batch_scattered(
                            cuda_utils::as_span(prune_ws.branched_leaves_d)
                                .subspan(off * stride, count * stride),
                            cuda_utils::as_span(prune_ws.branched_folds_d),
                            scores,
                            cuda_utils::as_span(prune_ws.passing_indices_d),
                            current_threshold, passing,
                            m_cfg.get_overflow_keep_fraction(), stream);
                    }
                    if (passing < selected ||
                        world_tree.get_space_left() == 0) {
                        break;
                    }
                }
                world_tree.consume_read(current_batch_size);
                continue;
            }
        }
        collect_direct();
    }

}; // End PruneCUDAImpl implementation

} // End of anonymous namespace

// EPMultiPassCudaCore::Impl implementation
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class EPMultiPassCudaCore<FoldTypeCUDA>::Impl {
public:
    using HostFoldT = HostFoldType<FoldTypeCUDA>;

    Impl(search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         std::span<const SizeType> ascend_levels,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id)
        : m_cfg(cfg.has_explicit_nsegments() ? std::move(cfg)
                                             : cfg.with_ep_default_nsegments()),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_ascend_levels(ascend_levels.begin(), ascend_levels.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_device_id(device_id),
          m_execution_stream(device_id),
          m_ffa_plan(m_cfg),
          m_branching_pattern(m_ffa_plan.forecast_branching(m_poly_basis).mean),
          m_branch_max(detail::compute_branch_max(
              m_ffa_plan.forecast_branching(m_poly_basis).max_children)),
          m_workspace_storage(m_batch_size,
                              m_branch_max,
                              m_max_sugg,
                              m_ffa_plan.get_ncoords().back(),
                              m_cfg.get_nparams(),
                              std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                  ? m_cfg.get_nbins_f()
                                  : m_cfg.get_nbins(),
                              m_ffa_plan.get_nsegments_active(),
                              m_execution_stream.get()),
          m_workspace_ptr(&m_workspace_storage) {
        // Validate workspace
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const auto nsegments   = m_ffa_plan.get_nsegments_active();
        const auto nbins       = std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                     ? m_cfg.get_nbins_f()
                                     : m_cfg.get_nbins();
        const auto& ws         = get_workspace();
        ws.validate(m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), nbins, nsegments);
    }

    Impl(memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
         cudaStream_t execution_stream,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         std::span<const SizeType> ascend_levels,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id)
        : m_cfg(cfg.has_explicit_nsegments() ? std::move(cfg)
                                             : cfg.with_ep_default_nsegments()),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_ascend_levels(ascend_levels.begin(), ascend_levels.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_device_id(device_id),
          m_execution_stream(execution_stream),
          m_ffa_plan(m_cfg),
          m_workspace_storage(),
          m_workspace_ptr(&workspace) {
        // Create branching pattern and branch max
        const auto forecast = m_ffa_plan.forecast_branching(m_poly_basis);
        m_branching_pattern = forecast.mean;
        m_branch_max        = detail::compute_branch_max(forecast.max_children);

        // Validate workspaces
        const auto& ws         = get_workspace();
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const auto nsegments   = m_ffa_plan.get_nsegments_active();
        const SizeType nbins   = std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                     ? m_cfg.get_nbins_f()
                                     : m_cfg.get_nbins();
        ws.validate(m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), nbins, nsegments);
    }

    /// Shared-pipeline constructor: the sweep owns the workspaces, the device
    /// inputs and the fold buffer, and sized the EP workspace for a chunk
    /// group (@p pipeline's shape), of which this chunk is a member.
    Impl(const EPCudaSharedPipeline<FoldTypeCUDA>& pipeline,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         std::span<const SizeType> ascend_levels,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id,
         PruneRFIConfig rfi_config)
        : m_cfg(cfg.has_explicit_nsegments() ? std::move(cfg)
                                             : cfg.with_ep_default_nsegments()),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_ascend_levels(ascend_levels.begin(), ascend_levels.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_device_id(device_id),
          m_execution_stream(pipeline.stream),
          m_ffa_plan(m_cfg),
          m_workspace_storage(),
          m_workspace_ptr(pipeline.ep_workspace),
          m_pipeline(pipeline) {
        if (pipeline.ep_workspace == nullptr ||
            pipeline.ffa_workspace == nullptr ||
            pipeline.fft_manager == nullptr) {
            throw std::invalid_argument(
                "EPMultiPassCudaCore: the shared pipeline needs its "
                "workspaces and FFT manager");
        }
        const auto forecast = m_ffa_plan.forecast_branching(m_poly_basis);
        m_branching_pattern = forecast.mean;
        const auto branch_max_needed =
            detail::compute_branch_max(forecast.max_children);
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const auto nsegments   = m_ffa_plan.get_nsegments_active();
        // The group's workspace and the prune functors use the group's
        // branch_max: a stale plan whose groups are too small must not run.
        if (branch_max_needed > pipeline.ws_branch_max ||
            ncoords_ffa > pipeline.ws_ncoords ||
            nsegments > pipeline.ws_nsegments) {
            throw std::runtime_error(fmt::format(
                "EPMultiPassCudaCore: the chunk needs branch_max={}, "
                "ncoords={}, nsegments={} but the workspace was sized for "
                "{}, {}, {}. Re-plan (stale plan cache?).",
                branch_max_needed, ncoords_ffa, nsegments,
                pipeline.ws_branch_max, pipeline.ws_ncoords,
                pipeline.ws_nsegments));
        }
        m_branch_max         = pipeline.ws_branch_max;
        const SizeType nbins = std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                   ? m_cfg.get_nbins_f()
                                   : m_cfg.get_nbins();
        get_workspace().validate(m_batch_size, m_branch_max,
                                 pipeline.ws_max_sugg, pipeline.ws_ncoords,
                                 m_cfg.get_nparams(), nbins,
                                 pipeline.ws_nsegments);
        setup_rfi_control(std::move(rfi_config), nsegments);
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
        timing::SimpleTimer timer;
        timer.start();
        std::tie(ts_e, ts_v) = search::detail::ep_input_series(
            m_cfg, ts_e, ts_v, "EPMultiPass::execute");
        spdlog::debug("EPMultiPassCudaCore: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<thrust::device_vector<FoldTypeCUDA>,
                   plans::FFAPlan<HostFoldT>>
            result = compute_ffa_cuda_device<FoldTypeCUDA>(ts_e, ts_v, m_cfg,
                                                           m_device_id);
        // Take the fold out of the tuple: copying it would double the device
        // memory of the fold.
        const thrust::device_vector<FoldTypeCUDA> ffa_fold_d =
            std::move(std::get<0>(result));
        run_pruning(cuda_utils::as_span(ffa_fold_d), outdir, file_prefix,
                    timer);
    }

    /// Shared-pipeline execution: the FFA runs into the sweep's fold buffer
    /// from the sweep's device inputs, then the runs are pruned in turn.
    void execute_device(const std::filesystem::path& outdir,
                        std::string_view file_prefix) {
        timing::SimpleTimer timer;
        timer.start();
        spdlog::debug("EPMultiPassCudaCore: Initializing with FFA");
        const auto& pl = *m_pipeline;
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        {
            // Scoped: the FFA core's host state ends before the pruning (its
            // brute-fold tables live in the shared workspace; the memory
            // model counts them).
            const nvtx::Range range("ep: ffa");
            FFACudaCore<FoldTypeCUDA> ffa(*pl.ffa_workspace, *pl.fft_manager,
                                          m_cfg, m_device_id);
            const auto& plan = ffa.get_plan();
            error_check::check_greater_equal(
                pl.fold_d.size(), plan.get_buffer_size(),
                "EPMultiPassCudaCore: the shared fold buffer is smaller than "
                "the chunk's FFA buffer");
            ffa.execute(pl.ts_e_d, pl.ts_v_d,
                        pl.fold_d.first(plan.get_buffer_size()), pl.stream);
            cuda_utils::check_cuda_call(
                cudaStreamSynchronize(pl.stream),
                "EPMultiPassCudaCore: FFA synchronization failed");
        }
        run_pruning(cuda::std::span<const FoldTypeCUDA>(
                        pl.fold_d.data(), m_ffa_plan.get_fold_size()),
                    outdir, file_prefix, timer);
    }

private:
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    std::vector<SizeType> m_ascend_levels;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string m_poly_basis;
    int m_device_id;

    plans::FFAPlan<HostFoldT> m_ffa_plan;
    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};
    // RFI control of the chunk and its static mask on the FFA base grid.
    PruneRFIConfig m_rfi_config;
    search::GridMask m_mask_base;

    // Validate the RFI configuration and rasterise the static pulsar mask
    // onto the (accel, freq) grid that resolve() snaps to (as on CPU).
    void setup_rfi_control(PruneRFIConfig rfi_config, SizeType nsegments) {
        if (!rfi_config.is_active()) {
            return;
        }
        error_check::check(!rfi_config.impulsive_veto,
                           "EPMultiPassCudaCore: the impulsive veto is CPU "
                           "only; disable [ep.rfi] impulsive_veto or use the "
                           "CPU backend");
        rfi_config.validate(nsegments);
        m_rfi_config = rfi_config.expand_harvest(m_threshold_scheme);
        m_rfi_config.validate(nsegments);
        const auto n_params = m_cfg.get_nparams();
        error_check::check_greater_equal(
            n_params, SizeType{2},
            "EPMultiPassCudaCore: pruning requires at least 2 parameters");
        const auto& counts = m_ffa_plan.get_param_counts().back();
        const auto limits  = m_cfg.get_param_limits();
        m_mask_base =
            search::GridMask(limits[n_params - 2], counts[n_params - 2],
                             limits[n_params - 1], counts[n_params - 1]);
        m_mask_base.add_windows(m_rfi_config.pulsar_mask,
                                m_rfi_config.n_harmonics,
                                m_rfi_config.fractional_harmonics);
        spdlog::info(
            "EPMultiPassCudaCore: RFI control: {} mask window(s) ({} "
            "harmonics{}) -> {} of {} grid cells masked; harvesting {}",
            m_rfi_config.pulsar_mask.size(), m_rfi_config.n_harmonics,
            m_rfi_config.fractional_harmonics ? ", fractional" : "",
            m_mask_base.count(), m_mask_base.get_n_cells(),
            m_rfi_config.has_harvest() ? "on" : "off");
    }

    // EP workspace ownership
    ExecutionStream m_execution_stream;
    memory::EPWorkspaceCUDA<FoldTypeCUDA> m_workspace_storage;
    // The observer pointer that always points to the active workspace.
    memory::EPWorkspaceCUDA<FoldTypeCUDA>* m_workspace_ptr{nullptr};
    // Set only on the shared-pipeline constructor.
    std::optional<EPCudaSharedPipeline<FoldTypeCUDA>> m_pipeline;

    [[nodiscard]] memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    /// Prunes the selected runs of @p ffa_fold and writes the result file.
    void run_pruning(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                     const std::filesystem::path& outdir,
                     std::string_view file_prefix,
                     timing::SimpleTimer& timer) {
        const nvtx::Range range("ep: pruning");
        // Setup output files and directory
        const auto nsegments = m_ffa_plan.get_nsegments_active();
        const std::string filebase =
            fmt::format("{}_pruning_nstages_{}", file_prefix, nsegments);
        const auto log_file =
            (outdir / fmt::format("{}_log.txt", filebase)).lexically_normal();
        const auto result_file =
            (outdir / fmt::format("{}_results.h5", filebase))
                .lexically_normal();

        // Create output directory
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
        if (!std::filesystem::exists(outdir)) {
            throw std::runtime_error(fmt::format(
                "EPMultiPassCudaCore::execute: Failed to create output "
                "directory '{}': {}",
                outdir.string(), ec.message()));
        }

        // Determine ref_segs to process
        auto ref_segs_to_process = utils::determine_ref_segs(
            m_ffa_plan.get_nsegments_ref_pool(m_n_runs, m_ref_segs.has_value()),
            m_n_runs, m_ref_segs);
        spdlog::debug("Starting Pruning for {} runs, on CUDA device {}",
                      ref_segs_to_process.size(), m_device_id);

        // Initialize log file
        std::ofstream log(log_file);
        log << "Pruning log\n";
        log.close();

        // Write metadata to result file
        const auto sink =
            m_pipeline
                ? m_pipeline->result_sink
                : std::function<void(pipelines::detail::EPResultPayload)>{};
        if (!sink) {
            auto writer = search::PruneResultWriter(
                result_file, search::PruneResultWriter::Mode::kWrite);
            writer.write_metadata(m_cfg.get_param_names(), nsegments,
                                  m_max_sugg, m_threshold_scheme);
        }
        auto& ws   = get_workspace();
        auto prune = PruneCUDAImpl<FoldTypeCUDA>(
            ws, m_cfg, m_threshold_scheme, m_max_sugg, m_batch_size,
            m_branch_max, m_poly_basis, m_device_id, m_execution_stream.get(),
            sink, &m_rfi_config, &m_mask_base);
        for (const auto ref_seg : ref_segs_to_process) {
            loki::interrupt::check(m_execution_stream.get());
            prune.execute(ffa_fold, ref_seg, m_ascend_levels, outdir, log_file,
                          result_file,
                          /*task_id=*/0);
        }
        const auto ep_time = timer.stop();
        // Write final runtime to result file
        if (!sink) {
            auto writer_final = search::PruneResultWriter(
                result_file, search::PruneResultWriter::Mode::kAppend);
            writer_final.write_runtime(ep_time);
        }
        spdlog::info("Pruning complete in {:.2f} s. Results saved to {}",
                     ep_time, result_file.string());
    }

}; // End EPMultiPassCudaCore::Impl implementation

// --- Definitions for EPMultiPassCudaCore ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>::EPMultiPassCudaCore(
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id)
    : m_impl(std::make_unique<Impl>(std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    ascend_levels,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>::EPMultiPassCudaCore(
    memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
    cudaStream_t execution_stream,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id)
    : m_impl(std::make_unique<Impl>(workspace,
                                    execution_stream,
                                    std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    ascend_levels,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>::EPMultiPassCudaCore(
    const EPCudaSharedPipeline<FoldTypeCUDA>& pipeline,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id,
    PruneRFIConfig rfi_config)
    : m_impl(std::make_unique<Impl>(pipeline,
                                    std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    ascend_levels,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id,
                                    std::move(rfi_config))) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>::~EPMultiPassCudaCore() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>::EPMultiPassCudaCore(
    EPMultiPassCudaCore<FoldTypeCUDA>&& other) noexcept = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCudaCore<FoldTypeCUDA>& EPMultiPassCudaCore<FoldTypeCUDA>::operator=(
    EPMultiPassCudaCore<FoldTypeCUDA>&& other) noexcept = default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void EPMultiPassCudaCore<FoldTypeCUDA>::execute(
    std::span<const float> ts_e,
    std::span<const float> ts_v,
    const std::filesystem::path& outdir,
    std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void EPMultiPassCudaCore<FoldTypeCUDA>::execute_device(
    const std::filesystem::path& outdir, std::string_view file_prefix) {
    m_impl->execute_device(outdir, file_prefix);
}

// Explicit instantiation
template class EPMultiPassCudaCore<float>;
template class EPMultiPassCudaCore<ComplexTypeCUDA>;

template <SupportedFoldType FoldType>
class EPMultiPassCudaEngine : public detail::EPMultiPassEngine<FoldType> {
    using FoldTypeCUDA = CudaFoldType<FoldType>;

public:
    /// Runs on the caller's device workspace, on a stream owned by this
    /// engine (as the owning constructor does). The workspace handle was
    /// allocated on the legacy default stream, which orders before it.
    EPMultiPassCudaEngine(memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
                          search::PulsarSearchConfig cfg,
                          std::span<const float> threshold_scheme,
                          std::optional<SizeType> n_runs,
                          std::optional<std::vector<SizeType>> ref_segs,
                          std::span<const SizeType> ascend_levels,
                          SizeType max_sugg,
                          SizeType batch_size,
                          std::string_view poly_basis,
                          int device_id)
        : m_stream(std::in_place, device_id),
          m_impl(workspace,
                 m_stream->get(),
                 std::move(cfg),
                 threshold_scheme,
                 n_runs,
                 std::move(ref_segs),
                 ascend_levels,
                 max_sugg,
                 batch_size,
                 poly_basis,
                 device_id) {}

    EPMultiPassCudaEngine(search::PulsarSearchConfig cfg,
                          std::span<const float> threshold_scheme,
                          std::optional<SizeType> n_runs,
                          std::optional<std::vector<SizeType>> ref_segs,
                          std::span<const SizeType> ascend_levels,
                          SizeType max_sugg,
                          SizeType batch_size,
                          std::string_view poly_basis,
                          int device_id)
        : m_impl(std::move(cfg),
                 threshold_scheme,
                 n_runs,
                 std::move(ref_segs),
                 ascend_levels,
                 max_sugg,
                 batch_size,
                 poly_basis,
                 device_id) {}

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        m_impl.execute(ts_e, ts_v, outdir, file_prefix);
    }

private:
    // Set only on a shared workspace. Declared first so it outlives m_impl.
    std::optional<ExecutionStream> m_stream;
    typename EPMultiPassCudaCore<FoldTypeCUDA>::Impl m_impl;
};

namespace detail {

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id) {
    return std::make_unique<EPMultiPassCudaEngine<FoldType>>(
        std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
        ascend_levels, max_sugg, batch_size, poly_basis, device_id);
}

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(memory::EPWorkspace<FoldType>& workspace,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id) {
    return std::make_unique<EPMultiPassCudaEngine<FoldType>>(
        memory::detail::cuda_workspace(workspace, "EPMultiPass"),
        std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
        ascend_levels, max_sugg, batch_size, poly_basis, device_id);
}

template std::unique_ptr<EPMultiPassEngine<float>>
make_ep_gpu<float>(memory::EPWorkspace<float>&,
                   search::PulsarSearchConfig,
                   std::span<const float>,
                   std::optional<SizeType>,
                   std::optional<std::vector<SizeType>>,
                   std::span<const SizeType>,
                   SizeType,
                   SizeType,
                   std::string_view,
                   int);

template std::unique_ptr<EPMultiPassEngine<ComplexType>>
make_ep_gpu<ComplexType>(memory::EPWorkspace<ComplexType>&,
                         search::PulsarSearchConfig,
                         std::span<const float>,
                         std::optional<SizeType>,
                         std::optional<std::vector<SizeType>>,
                         std::span<const SizeType>,
                         SizeType,
                         SizeType,
                         std::string_view,
                         int);

template std::unique_ptr<EPMultiPassEngine<float>>
make_ep_gpu<float>(search::PulsarSearchConfig,
                   std::span<const float>,
                   std::optional<SizeType>,
                   std::optional<std::vector<SizeType>>,
                   std::span<const SizeType>,
                   SizeType,
                   SizeType,
                   std::string_view,
                   int);

template std::unique_ptr<EPMultiPassEngine<ComplexType>>
make_ep_gpu<ComplexType>(search::PulsarSearchConfig,
                         std::span<const float>,
                         std::optional<SizeType>,
                         std::optional<std::vector<SizeType>>,
                         std::span<const SizeType>,
                         SizeType,
                         SizeType,
                         std::string_view,
                         int);

} // namespace detail

} // namespace loki::algorithms
