#include "loki/algorithms/ffa.hpp"

#include <memory>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/plans_cuda.cuh"
#include "loki/timing.hpp"

namespace loki::algorithms {

namespace {

// OPTIMIZED: One thread per smallest work unit, optimized for memory coalescing
__global__ void kernel_ffa_iter(const float* __restrict__ fold_in,
                                float* __restrict__ fold_out,
                                const plans::FFACoordDPtrs coords,
                                int ncoords_cur,
                                int ncoords_prev,
                                int nsegments,
                                int nbins) {

    // 1D thread mapping with optimal work distribution
    const auto tid = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto total_work = ncoords_cur * nsegments * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, ibin) - OPTIMIZED ORDER for coalescing
    const int ibin   = tid % nbins; // Fastest varying (best for coalescing)
    const int temp   = tid / nbins;
    const int iseg   = temp % nsegments;
    const int icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const int coord_tail     = coords.i_tail[icoord];
    const int coord_head     = coords.i_head[icoord];
    const int shift_tail_raw = __double2int_rn(coords.shift_tail[icoord]);
    const int shift_head_raw = __double2int_rn(coords.shift_head[icoord]);
    const int shift_tail     = (shift_tail_raw % nbins + nbins) % nbins;
    const int shift_head     = (shift_head_raw % nbins + nbins) % nbins;

    const int idx_tail =
        (ibin < shift_tail) ? (ibin + nbins - shift_tail) : (ibin - shift_tail);
    const int idx_head =
        (ibin < shift_head) ? (ibin + nbins - shift_head) : (ibin - shift_head);

    // Calculate offsets
    const int tail_offset =
        ((iseg * 2) * ncoords_prev * 2 * nbins) + (coord_tail * 2 * nbins);
    const int head_offset =
        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) + (coord_head * 2 * nbins);
    const int out_offset =
        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + idx_tail] + fold_in[head_offset + idx_head];
    fold_out[out_offset + ibin + nbins] =
        fold_in[tail_offset + idx_tail + nbins] +
        fold_in[head_offset + idx_head + nbins];
}

} // namespace

class FFACUDA::Impl {
public:
    Impl(search::PulsarSearchConfig cfg, int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id) {
        cuda_utils::set_device(m_device_id);
        // Allocate memory for the FFA buffers
        m_fold_in_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        m_fold_out_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFoldCUDA>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_device_id);

        plans::transfer_ffa_plan_to_device(m_ffa_plan, m_ffa_plan_d);
        cuda_utils::check_last_cuda_error(
            "FFACUDA::Impl::transfer_ffa_plan_to_device failed");
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan& get_plan() const { return m_ffa_plan; }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   std::span<float> fold) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());

        // Resize buffers only if needed
        if (m_ts_e_d.size() != ts_e.size()) {
            m_ts_e_d.resize(ts_e.size());
            m_ts_v_d.resize(ts_v.size());
        }
        if (m_fold_output_d.size() != fold.size()) {
            m_fold_output_d.resize(fold.size());
        }
        // Copy input data to device
        cudaStream_t stream = nullptr;
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()), ts_e.data(),
                        ts_e.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()), ts_v.data(),
                        ts_v.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);

        // Execute FFA on device using persistent buffers
        execute_d(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_e_d.data()), m_ts_e_d.size()),
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_v_d.data()), m_ts_v_d.size()),
            cuda::std::span<float>(
                thrust::raw_pointer_cast(m_fold_output_d.data()),
                m_fold_output_d.size()),
            stream);

        // Copy result back to host
        thrust::copy(m_fold_output_d.begin(), m_fold_output_d.end(),
                     fold.begin());
        spdlog::debug("FFACUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<float> fold_d,
                   cudaStream_t stream) {
        check_inputs(ts_e_d.size(), ts_v_d.size(), fold_d.size());
        execute_device(ts_e_d, ts_v_d, fold_d, stream);
        spdlog::debug("FFACUDA::Impl: Device execution complete on stream");
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    plans::FFAPlanD m_ffa_plan_d;
    int m_device_id;
    std::unique_ptr<algorithms::BruteFoldCUDA> m_the_bf;

    // Buffers for the FFA plan
    thrust::device_vector<float> m_fold_in_d;
    thrust::device_vector<float> m_fold_out_d;

    // Add persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_output_d;

    void check_inputs(SizeType ts_e_size,
                      SizeType ts_v_size,
                      SizeType fold_size) const {
        error_check::check_equal(
            ts_e_size, m_cfg.get_nsamps(),
            "FFACUDA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_size, ts_e_size,
            "FFACUDA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold_size, m_ffa_plan.get_fold_size(),
            "FFACUDA::Impl::execute: fold must have size fold_size");
    }

    void initialize_device(cuda::std::span<const float> ts_e_d,
                           cuda::std::span<const float> ts_v_d,
                           cudaStream_t stream) {
        ScopeTimer timer("FFACUDA::Impl::initialize_device");
        m_the_bf->execute(
            ts_e_d, ts_v_d,
            cuda::std::span(thrust::raw_pointer_cast(m_fold_in_d.data()),
                            m_the_bf->get_fold_size()),
            stream);
    }

    void execute_device(cuda::std::span<const float> ts_e_d,
                        cuda::std::span<const float> ts_v_d,
                        cuda::std::span<float> fold_d,
                        cudaStream_t stream) {
        ScopeTimer timer("FFACUDA::Impl::execute_device");
        initialize_device(ts_e_d, ts_v_d, stream);

        // Use raw pointers for swapping buffers
        float* fold_in_ptr     = thrust::raw_pointer_cast(m_fold_in_d.data());
        float* fold_out_ptr    = thrust::raw_pointer_cast(m_fold_out_d.data());
        float* fold_result_ptr = thrust::raw_pointer_cast(fold_d.data());

        const auto levels = m_cfg.get_niters_ffa() + 1;

        auto coords_cur = m_ffa_plan_d.coordinates.get_raw_ptrs();
        cuda_utils::check_last_cuda_error("thrust::raw_pointer_cast failed");
        coords_cur.update_offsets(m_ffa_plan_d.ncoords[0]);

        // FFA iterations (levels 1 to levels)
        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const auto nsegments    = m_ffa_plan_d.nsegments[i_level];
            const auto nbins        = m_ffa_plan_d.nbins[i_level];
            const auto ncoords_cur  = m_ffa_plan_d.ncoords[i_level];
            const auto ncoords_prev = m_ffa_plan_d.ncoords[i_level - 1];

            const int total_work = ncoords_cur * nsegments * nbins;
            const int block_size = (total_work < 65536) ? 256 : 512;
            const int grid_size  = (total_work + block_size - 1) / block_size;

            const dim3 block_dim(block_size);
            const dim3 grid_dim(grid_size);

            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

            // Determine output buffer: final iteration writes to output buffer
            const bool is_last     = i_level == levels - 1;
            float* current_out_ptr = is_last ? fold_result_ptr : fold_out_ptr;

            kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
                fold_in_ptr, current_out_ptr, coords_cur, ncoords_cur,
                ncoords_prev, nsegments, nbins);
            cuda_utils::check_last_cuda_error("kernel_ffa_iter launch failed");

            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                coords_cur.update_offsets(ncoords_cur);
                std::swap(fold_in_ptr, fold_out_ptr);
            }
        }

        spdlog::debug("FFACUDA::Impl: Iterations submitted to stream.");
    }

}; // End FFACUDA::Impl definition

FFACUDA::FFACUDA(const search::PulsarSearchConfig& cfg, int device_id)
    : m_impl(std::make_unique<Impl>(cfg, device_id)) {}

FFACUDA::~FFACUDA()                                   = default;
FFACUDA::FFACUDA(FFACUDA&& other) noexcept            = default;
FFACUDA& FFACUDA::operator=(FFACUDA&& other) noexcept = default;

const plans::FFAPlan& FFACUDA::get_plan() const noexcept {
    return m_impl->get_plan();
}

void FFACUDA::execute(std::span<const float> ts_e,
                      std::span<const float> ts_v,
                      std::span<float> fold) {
    m_impl->execute_h(ts_e, ts_v, fold);
}

void FFACUDA::execute(cuda::std::span<const float> ts_e,
                      cuda::std::span<const float> ts_v,
                      cuda::std::span<float> fold,
                      cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

std::vector<float> compute_ffa_cuda(std::span<const float> ts_e,
                                    std::span<const float> ts_v,
                                    const search::PulsarSearchConfig& cfg,
                                    int device_id) {
    FFACUDA ffa(cfg, device_id);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms