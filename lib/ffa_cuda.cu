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
#include "loki/cuda_utils.cuh"
#include "loki/plans_cuda.cuh"

namespace {

// CUDA kernel for FFA iteration processing multiple coordinates
__global__ void kernel_ffa_iter(const float* __restrict__ fold_in,
                                float* __restrict__ fold_out,
                                const loki::plans::FFACoordDPtrs coords,
                                int ncoords_cur,
                                int ncoords_prev,
                                int nsegments,
                                int nbins) {

    const int icoord = blockIdx.y;
    const int iseg   = blockIdx.z;
    const int j      = blockIdx.x * blockDim.x + threadIdx.x;

    if (icoord >= ncoords_cur || iseg >= nsegments || j >= nbins) {
        return;
    }

    // Calculate offsets (same logic as CPU version)
    const int tail_offset = ((iseg * 2) * ncoords_prev * 2 * nbins) +
                            (coords.i_tail[icoord] * 2 * nbins);
    const int head_offset = ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                            (coords.i_head[icoord] * 2 * nbins);
    const int out_offset =
        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

    // Perform shift_add operation inline
    const int shift_tail = coords.shift_tail[icoord] % nbins;
    const int shift_head = coords.shift_head[icoord] % nbins;
    const int idx_tail =
        (j < shift_tail) ? (j + nbins - shift_tail) : (j - shift_tail);
    const int idx_head =
        (j < shift_head) ? (j + nbins - shift_head) : (j - shift_head);

    // Process both e and v components
    fold_out[out_offset + j] =
        fold_in[tail_offset + idx_tail] + fold_in[head_offset + idx_head];
    fold_out[out_offset + j + nbins] = fold_in[tail_offset + idx_tail + nbins] +
                                       fold_in[head_offset + idx_head + nbins];
}

} // namespace

namespace loki::algorithms {

class FFACUDA::Impl {
public:
    Impl(search::PulsarSearchConfig cfg, int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id) {
        cuda_utils::set_device(m_device_id);
        m_fold_in_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        m_fold_out_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
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

        // Copy input data to device
        cudaStream_t stream = nullptr;
        thrust::device_vector<float> ts_e_d(ts_e.begin(), ts_e.end());
        thrust::device_vector<float> ts_v_d(ts_v.begin(), ts_v.end());
        thrust::device_vector<float> fold_d(fold.size(), 0.0F);

        // Execute FFA on device
        execute_d(cuda::std::span<const float>(
                      thrust::raw_pointer_cast(ts_e_d.data()), ts_e_d.size()),
                  cuda::std::span<const float>(
                      thrust::raw_pointer_cast(ts_v_d.data()), ts_v_d.size()),
                  cuda::std::span<float>(
                      thrust::raw_pointer_cast(fold_d.data()), fold_d.size()),
                  stream);

        // Copy result back to host
        thrust::copy(fold_d.begin(), fold_d.end(), fold.begin());
        spdlog::debug("FFACUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e,
                   cuda::std::span<const float> ts_v,
                   cuda::std::span<float> fold,
                   cudaStream_t stream) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());

        // Initialize with BruteFoldCUDA
        initialize_device(ts_e, ts_v, stream);

        // Ping-pong between buffers for iterative FFA levels
        float* fold_in_ptr  = thrust::raw_pointer_cast(m_fold_in_d.data());
        float* fold_out_ptr = thrust::raw_pointer_cast(m_fold_out_d.data());
        const auto levels   = m_cfg.get_niters_ffa() + 1;

        // FFA iterations (levels 1 to levels-2)
        for (loki::SizeType i_level = 1; i_level < levels - 1; ++i_level) {
            execute_iter_device(fold_in_ptr, fold_out_ptr, i_level, stream);
            std::swap(fold_in_ptr, fold_out_ptr);
        }

        // Last iteration writes directly to output
        if (levels > 1) {
            execute_iter_device(fold_in_ptr, fold.data(),
                                m_cfg.get_niters_ffa(), stream);
        } else {
            // Single level case - just copy
            cudaMemcpyAsync(fold.data(), fold_in_ptr,
                            fold.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        spdlog::debug("FFACUDA::Impl: Device execution complete on stream");
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    plans::FFAPlanD m_ffa_plan_d;
    int m_device_id;

    // Buffers for the FFA plan
    thrust::device_vector<float> m_fold_in_d;
    thrust::device_vector<float> m_fold_out_d;

    void check_inputs(loki::SizeType ts_e_size,
                      loki::SizeType ts_v_size,
                      loki::SizeType fold_size) const {
        if (ts_e_size != m_cfg.get_nsamps()) {
            throw std::runtime_error(std::format(
                "FFACUDA::Impl: ts must have size nsamps. Expected {}, got {}",
                m_cfg.get_nsamps(), ts_e_size));
        }
        if (ts_v_size != ts_e_size) {
            throw std::runtime_error(
                std::format("FFACUDA::Impl: ts variance must have size nsamps. "
                            "Expected {}, got {}",
                            ts_e_size, ts_v_size));
        }
        if (fold_size != m_ffa_plan.get_fold_size()) {
            throw std::runtime_error(
                std::format("FFACUDA::Impl: Output array has wrong size. "
                            "Expected {}, got {}",
                            m_ffa_plan.get_fold_size(), fold_size));
        }
    }

    void initialize_device(cuda::std::span<const float> ts_e,
                           cuda::std::span<const float> ts_v,
                           cudaStream_t stream) {
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        algorithms::BruteFoldCUDA bf(freqs_arr, m_ffa_plan.segment_lens[0],
                                     m_cfg.get_nbins(), m_cfg.get_nsamps(),
                                     m_cfg.get_tsamp(), t_ref, m_device_id);

        bf.execute(ts_e, ts_v,
                   cuda::std::span(thrust::raw_pointer_cast(m_fold_in_d.data()),
                                   bf.get_fold_size()),
                   stream);
    }

    void execute_iter_device(const float* __restrict__ fold_in,
                             float* __restrict__ fold_out,
                             loki::SizeType i_level,
                             cudaStream_t stream) {

        const auto nsegments =
            static_cast<int>(m_ffa_plan.fold_shapes[i_level][0]);
        const auto nbins =
            static_cast<int>(m_ffa_plan.fold_shapes[i_level].back());
        const auto ncoords_cur =
            static_cast<int>(m_ffa_plan.coordinates[i_level].size());
        const auto ncoords_prev =
            static_cast<int>(m_ffa_plan.coordinates[i_level - 1].size());

        // Get device coordinate pointers for this level
        auto coords_ptrs = m_ffa_plan_d.coordinates.get_raw_ptrs();

        // Calculate offset for this level in the flattened coordinate arrays
        // The coordinates are flattened sequentially: level 1, level 2, ...,
        // level i
        int coord_offset = 0;
        for (loki::SizeType level = 1; level < i_level; ++level) {
            coord_offset +=
                static_cast<int>(m_ffa_plan.coordinates[level].size());
        }
        coords_ptrs.update_offsets(coord_offset);

        // Launch 3D kernel: (nbins, coordinates, segments)
        const int threads_per_block = 256;
        const dim3 block_dim(threads_per_block);
        const dim3 grid_dim((nbins + threads_per_block - 1) / threads_per_block,
                            ncoords_cur, nsegments);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
            fold_in, fold_out, coords_ptrs, ncoords_cur, ncoords_prev,
            nsegments, nbins);
        cuda_utils::check_last_cuda_error("kernel_ffa_iter launch failed");
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