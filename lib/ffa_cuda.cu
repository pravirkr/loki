#include "loki/algorithms/ffa.hpp"

#include <memory>

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
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/plans_cuda.cuh"
#include "loki/timing.hpp"
#include "loki/utils/fft.hpp"

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

// OPTIMIZED: One thread per smallest work unit, optimized for memory coalescing
__global__ void
kernel_ffa_complex_iter(const ComplexTypeCUDA* __restrict__ fold_in,
                        ComplexTypeCUDA* __restrict__ fold_out,
                        const plans::FFACoordDPtrs coords,
                        int ncoords_cur,
                        int ncoords_prev,
                        int nsegments,
                        int nbins_f,
                        int nbins) {

    // 1D thread mapping with optimal work distribution
    const auto tid = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto total_work = ncoords_cur * nsegments * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, k) - OPTIMIZED ORDER for coalescing
    const int k      = tid % nbins_f; // Frequency bin (fastest varying)
    const int temp   = tid / nbins_f;
    const int iseg   = temp % nsegments;
    const int icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const auto coord_tail = coords.i_tail[icoord];
    const auto coord_head = coords.i_head[icoord];
    const auto shift_tail = coords.shift_tail[icoord];
    const auto shift_head = coords.shift_head[icoord];

    // Precompute phase factors: exp(-2πi * k * shift / nbins)
    const auto phase_factor_tail =
        static_cast<float>(-2.0F * M_PI * k * shift_tail / nbins);
    const auto phase_factor_head =
        static_cast<float>(-2.0F * M_PI * k * shift_head / nbins);
    // Fast sincos computation
    float cos_tail, sin_tail, cos_head, sin_head;
    __sincosf(phase_factor_tail, &sin_tail, &cos_tail);
    __sincosf(phase_factor_head, &sin_head, &cos_head);

    // Calculate memory offsets for e and v components
    const int tail_offset_e =
        ((iseg * 2) * ncoords_prev * 2 * nbins_f) + (coord_tail * 2 * nbins_f);
    const int tail_offset_v = tail_offset_e + nbins_f;
    const int head_offset_e = ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
                              (coord_head * 2 * nbins_f);
    const int head_offset_v = head_offset_e + nbins_f;

    const int out_offset_e =
        (iseg * ncoords_cur * 2 * nbins_f) + (icoord * 2 * nbins_f);
    const int out_offset_v = out_offset_e + nbins_f;

    // Load complex values for both e and v components
    const ComplexTypeCUDA data_tail_e = fold_in[tail_offset_e + k];
    const ComplexTypeCUDA data_tail_v = fold_in[tail_offset_v + k];
    const ComplexTypeCUDA data_head_e = fold_in[head_offset_e + k];
    const ComplexTypeCUDA data_head_v = fold_in[head_offset_v + k];

    // OPTIMIZED complex multiplication using fmaf
    // tail_shifted_e = data_tail_e * exp(-2πi * k * shift_tail / nbins)
    const float real_tail_e =
        fmaf(data_tail_e.real(), cos_tail, -data_tail_e.imag() * sin_tail);
    const float imag_tail_e =
        fmaf(data_tail_e.real(), sin_tail, data_tail_e.imag() * cos_tail);
    const float real_head_e =
        fmaf(data_head_e.real(), cos_head, -data_head_e.imag() * sin_head);
    const float imag_head_e =
        fmaf(data_head_e.real(), sin_head, data_head_e.imag() * cos_head);
    const float real_tail_v =
        fmaf(data_tail_v.real(), cos_tail, -data_tail_v.imag() * sin_tail);
    const float imag_tail_v =
        fmaf(data_tail_v.real(), sin_tail, data_tail_v.imag() * cos_tail);
    const float real_head_v =
        fmaf(data_head_v.real(), cos_head, -data_head_v.imag() * sin_head);
    const float imag_head_v =
        fmaf(data_head_v.real(), sin_head, data_head_v.imag() * cos_head);
    // Complex addition and store results
    fold_out[out_offset_e + k] =
        ComplexTypeCUDA(real_tail_e + real_head_e, imag_tail_e + imag_head_e);

    fold_out[out_offset_v + k] =
        ComplexTypeCUDA(real_tail_v + real_head_v, imag_tail_v + imag_head_v);
}

} // namespace

class FFACUDA::Impl {
public:
    Impl(search::PulsarSearchConfig cfg, int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id),
          m_use_single_buffer(m_ffa_plan.get_fold_size() >=
                              m_ffa_plan.get_buffer_size()) {
        cuda_utils::set_device(m_device_id);
        // Allocate memory for the FFA buffers
        m_fold_in_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        if (!m_use_single_buffer) {
            m_fold_out_d.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        }
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
        if (m_fold_d.size() != fold.size()) {
            m_fold_d.resize(fold.size());
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
            cuda::std::span<float>(thrust::raw_pointer_cast(m_fold_d.data()),
                                   m_fold_d.size()),
            stream);

        // Copy result back to host
        thrust::copy(m_fold_d.begin(), m_fold_d.end(), fold.begin());
        spdlog::debug("FFACUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<float> fold_d,
                   cudaStream_t stream) {
        check_inputs(ts_e_d.size(), ts_v_d.size(), fold_d.size());
        if (m_use_single_buffer) {
            execute_single_buffer_device(ts_e_d, ts_v_d, fold_d, stream);
        } else {
            execute_double_buffer_device(ts_e_d, ts_v_d, fold_d, stream);
        }
        spdlog::debug("FFACUDA::Impl: Device execution complete on stream");
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    plans::FFAPlanD m_ffa_plan_d;
    int m_device_id;
    bool m_use_single_buffer;
    std::unique_ptr<algorithms::BruteFoldCUDA> m_the_bf;

    // Buffers for the FFA plan
    thrust::device_vector<float> m_fold_in_d;
    thrust::device_vector<float> m_fold_out_d;

    // Add persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_d;

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
                           float* init_buffer_d,
                           cudaStream_t stream) {
        ScopeTimer timer("FFACUDA::Impl::initialize_device");
        m_the_bf->execute(
            ts_e_d, ts_v_d,
            cuda::std::span(init_buffer_d, m_the_bf->get_fold_size()), stream);
    }

    void execute_double_buffer_device(cuda::std::span<const float> ts_e_d,
                                      cuda::std::span<const float> ts_v_d,
                                      cuda::std::span<float> fold_d,
                                      cudaStream_t stream) {
        ScopeTimer timer("FFACUDA::Impl::execute_device");
        initialize_device(ts_e_d, ts_v_d,
                          thrust::raw_pointer_cast(m_fold_in_d.data()), stream);

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

    void execute_single_buffer_device(cuda::std::span<const float> ts_e_d,
                                      cuda::std::span<const float> ts_v_d,
                                      cuda::std::span<float> fold_d,
                                      cudaStream_t stream) {
        ScopeTimer timer("FFACUDA::Impl::execute_device");
        const auto levels = m_cfg.get_niters_ffa() + 1;

        // Use m_fold_in and output fold buffer for ping-pong
        float* fold_internal_ptr = thrust::raw_pointer_cast(m_fold_in_d.data());
        float* fold_result_ptr   = thrust::raw_pointer_cast(fold_d.data());

        // Calculate the number of internal iterations (excluding final write)
        const SizeType internal_iters = levels - 2;
        // Determine starting configuration to ensure final result lands in fold
        bool odd_swaps = (internal_iters % 2) == 1;
        float *current_in_ptr, *current_out_ptr;
        if (odd_swaps) {
            // Initialize in fold, will end up in fold after odd swaps
            initialize_device(ts_e_d, ts_v_d, fold_result_ptr, stream);
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        } else {
            // Initialize in internal, will end up in fold after even swaps
            initialize_device(ts_e_d, ts_v_d, fold_internal_ptr, stream);
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        }

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
            const bool is_last = i_level == levels - 1;
            kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
                current_in_ptr, current_out_ptr, coords_cur, ncoords_cur,
                ncoords_prev, nsegments, nbins);
            cuda_utils::check_last_cuda_error("kernel_ffa_iter launch failed");

            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                coords_cur.update_offsets(ncoords_cur);
                std::swap(current_in_ptr, current_out_ptr);
            }
        }

        spdlog::debug("FFACUDA::Impl: Iterations submitted to stream.");
    }

}; // End FFACUDA::Impl definition

class FFACOMPLEXCUDA::Impl {
public:
    Impl(search::PulsarSearchConfig cfg, int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id),
          m_use_single_buffer(m_ffa_plan.get_fold_size_complex() >=
                              m_ffa_plan.get_buffer_size_complex()) {
        cuda_utils::set_device(m_device_id);
        // Allocate memory for the FFA buffers
        m_fold_in_d.resize(m_ffa_plan.get_buffer_size_complex(),
                           ComplexTypeCUDA(0.0F, 0.0F));
        if (!m_use_single_buffer) {
            m_fold_out_d.resize(m_ffa_plan.get_buffer_size_complex(),
                                ComplexTypeCUDA(0.0F, 0.0F));
        }
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
        if (m_fold_d.size() != fold.size()) {
            m_fold_d.resize(fold.size());
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
            cuda::std::span<float>(thrust::raw_pointer_cast(m_fold_d.data()),
                                   m_fold_d.size()),
            stream);

        // Copy result back to host
        thrust::copy(m_fold_d.begin(), m_fold_d.end(), fold.begin());
        spdlog::debug("FFACUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<float> fold_d,
                   cudaStream_t stream) {
        check_inputs(ts_e_d.size(), ts_v_d.size(), fold_d.size());
        auto fold_complex = cuda::std::span<ComplexTypeCUDA>(
            reinterpret_cast<ComplexTypeCUDA*>(fold_d.data()),
            m_ffa_plan.get_fold_size_complex());
        if (m_use_single_buffer) {
            execute_single_buffer_device(ts_e_d, ts_v_d, fold_complex, stream);
        } else {
            execute_double_buffer_device(ts_e_d, ts_v_d, fold_complex, stream);
        }
        // IRFFT in-place
        const auto nfft = m_ffa_plan.get_fold_size() / m_cfg.get_nbins();
        utils::irfft_batch_inplace_cuda(fold_complex, static_cast<int>(nfft),
                                        static_cast<int>(m_cfg.get_nbins()),
                                        stream);
        spdlog::debug("FFACUDA::Impl: Device execution complete on stream");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<ComplexTypeCUDA> fold_complex_d,
                   cudaStream_t stream) {
        check_inputs_complex(ts_e_d.size(), ts_v_d.size(),
                             fold_complex_d.size());
        if (m_use_single_buffer) {
            execute_single_buffer_device(ts_e_d, ts_v_d, fold_complex_d,
                                         stream);
        } else {
            execute_double_buffer_device(ts_e_d, ts_v_d, fold_complex_d,
                                         stream);
        }
        spdlog::debug("FFACUDA::Impl: Device execution complete on stream");
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan m_ffa_plan;
    plans::FFAPlanD m_ffa_plan_d;
    int m_device_id;
    bool m_use_single_buffer;
    std::unique_ptr<algorithms::BruteFoldCUDA> m_the_bf;

    // Buffers for the FFA plan
    thrust::device_vector<ComplexTypeCUDA> m_fold_in_d;
    thrust::device_vector<ComplexTypeCUDA> m_fold_out_d;

    // Add persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_d;

    void check_inputs(SizeType ts_e_size,
                      SizeType ts_v_size,
                      SizeType fold_size) const {
        error_check::check_equal(
            ts_e_size, m_cfg.get_nsamps(),
            "FFACOMPLEXCUDA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_size, ts_e_size,
            "FFACOMPLEXCUDA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold_size, 2 * m_ffa_plan.get_fold_size_complex(),
            "FFACOMPLEXCUDA::Impl::execute: fold must have size 2 * "
            "fold_size_complex");
    }

    void check_inputs_complex(SizeType ts_e_size,
                              SizeType ts_v_size,
                              SizeType fold_complex_size) const {
        error_check::check_equal(
            ts_e_size, m_cfg.get_nsamps(),
            "FFACOMPLEXCUDA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_size, ts_e_size,
            "FFACOMPLEXCUDA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(fold_complex_size,
                                 m_ffa_plan.get_fold_size_complex(),
                                 "FFACOMPLEXCUDA::Impl::execute: fold must "
                                 "have size fold_size_complex");
    }

    void initialize_device(cuda::std::span<const float> ts_e_d,
                           cuda::std::span<const float> ts_v_d,
                           ComplexTypeCUDA* init_buffer_d,
                           cudaStream_t stream) {
        const auto required_floats = m_ffa_plan.get_brute_fold_size_complex();

        // Cast complex buffer to float for brute folding
        auto real_view = cuda::std::span<float>(
            reinterpret_cast<float*>(init_buffer_d), required_floats);
        m_the_bf->execute_stride(ts_e_d, ts_v_d, real_view, stream);

        // In-place RFFT
        const auto nfft         = m_the_bf->get_fold_size() / m_cfg.get_nbins();
        const auto complex_size = nfft * ((m_cfg.get_nbins() / 2) + 1);
        utils::rfft_batch_inplace_cuda(
            cuda::std::span<ComplexTypeCUDA>(init_buffer_d, complex_size),
            static_cast<int>(nfft), static_cast<int>(m_cfg.get_nbins()),
            stream);
    }

    void execute_double_buffer_device(
        cuda::std::span<const float> ts_e_d,
        cuda::std::span<const float> ts_v_d,
        cuda::std::span<ComplexTypeCUDA> fold_complex_d,
        cudaStream_t stream) {
        ScopeTimer timer("FFACOMPLEXCUDA::Impl::execute_device");
        initialize_device(ts_e_d, ts_v_d,
                          thrust::raw_pointer_cast(m_fold_in_d.data()), stream);

        ComplexTypeCUDA* fold_in_ptr =
            thrust::raw_pointer_cast(m_fold_in_d.data());
        ComplexTypeCUDA* fold_out_ptr =
            thrust::raw_pointer_cast(m_fold_out_d.data());
        ComplexTypeCUDA* fold_result_ptr =
            thrust::raw_pointer_cast(fold_complex_d.data());

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
            const auto nbins_f      = (nbins / 2) + 1;

            const int total_work = ncoords_cur * nsegments * nbins_f;
            const int block_size = (total_work < 65536) ? 256 : 512;
            const int grid_size  = (total_work + block_size - 1) / block_size;

            const dim3 block_dim(block_size);
            const dim3 grid_dim(grid_size);

            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

            // Determine output buffer: final iteration writes to output buffer
            const bool is_last = i_level == levels - 1;
            ComplexTypeCUDA* current_out_ptr =
                is_last ? fold_result_ptr : fold_out_ptr;

            kernel_ffa_complex_iter<<<grid_dim, block_dim, 0, stream>>>(
                fold_in_ptr, current_out_ptr, coords_cur, ncoords_cur,
                ncoords_prev, nsegments, nbins_f, nbins);
            cuda_utils::check_last_cuda_error(
                "kernel_ffa_complex_iter launch failed");

            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                coords_cur.update_offsets(ncoords_cur);
                std::swap(fold_in_ptr, fold_out_ptr);
            }
        }

        spdlog::debug("FFACUDA::Impl: Iterations submitted to stream.");
    }

    void execute_single_buffer_device(
        cuda::std::span<const float> ts_e_d,
        cuda::std::span<const float> ts_v_d,
        cuda::std::span<ComplexTypeCUDA> fold_d_complex,
        cudaStream_t stream) {
        ScopeTimer timer("FFACOMPLEXCUDA::Impl::execute_device");
        const auto levels = m_cfg.get_niters_ffa() + 1;

        // Use m_fold_in and output fold buffer for ping-pong
        ComplexTypeCUDA* fold_internal_ptr =
            thrust::raw_pointer_cast(m_fold_in_d.data());
        ComplexTypeCUDA* fold_result_ptr =
            thrust::raw_pointer_cast(fold_d_complex.data());

        // Calculate the number of internal iterations (excluding final write)
        const SizeType internal_iters = levels - 2;

        // Determine starting configuration to ensure final result lands in fold
        bool odd_swaps = (internal_iters % 2) == 1;
        ComplexTypeCUDA *current_in_ptr, *current_out_ptr;
        if (odd_swaps) {
            // Initialize in fold, will end up in fold after odd swaps
            initialize_device(ts_e_d, ts_v_d, fold_result_ptr, stream);
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        } else {
            // Initialize in internal, will end up in fold after even swaps
            initialize_device(ts_e_d, ts_v_d, fold_internal_ptr, stream);
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        }

        auto coords_cur = m_ffa_plan_d.coordinates.get_raw_ptrs();
        cuda_utils::check_last_cuda_error("thrust::raw_pointer_cast failed");
        coords_cur.update_offsets(m_ffa_plan_d.ncoords[0]);

        // FFA iterations (levels 1 to levels)
        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const auto nsegments    = m_ffa_plan_d.nsegments[i_level];
            const auto nbins        = m_ffa_plan_d.nbins[i_level];
            const auto ncoords_cur  = m_ffa_plan_d.ncoords[i_level];
            const auto ncoords_prev = m_ffa_plan_d.ncoords[i_level - 1];
            const auto nbins_f      = (nbins / 2) + 1;

            const int total_work = ncoords_cur * nsegments * nbins_f;
            const int block_size = (total_work < 65536) ? 256 : 512;
            const int grid_size  = (total_work + block_size - 1) / block_size;

            const dim3 block_dim(block_size);
            const dim3 grid_dim(grid_size);

            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

            // Determine output buffer: final iteration writes to output buffer
            const bool is_last = i_level == levels - 1;
            kernel_ffa_complex_iter<<<grid_dim, block_dim, 0, stream>>>(
                current_in_ptr, current_out_ptr, coords_cur, ncoords_cur,
                ncoords_prev, nsegments, nbins_f, nbins);
            cuda_utils::check_last_cuda_error(
                "kernel_ffa_complex_iter launch failed");

            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                coords_cur.update_offsets(ncoords_cur);
                std::swap(current_in_ptr, current_out_ptr);
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

FFACOMPLEXCUDA::FFACOMPLEXCUDA(const search::PulsarSearchConfig& cfg,
                               int device_id)
    : m_impl(std::make_unique<Impl>(cfg, device_id)) {}

FFACOMPLEXCUDA::~FFACOMPLEXCUDA()                               = default;
FFACOMPLEXCUDA::FFACOMPLEXCUDA(FFACOMPLEXCUDA&& other) noexcept = default;
FFACOMPLEXCUDA&
FFACOMPLEXCUDA::operator=(FFACOMPLEXCUDA&& other) noexcept = default;

const plans::FFAPlan& FFACOMPLEXCUDA::get_plan() const noexcept {
    return m_impl->get_plan();
}

void FFACOMPLEXCUDA::execute(std::span<const float> ts_e,
                             std::span<const float> ts_v,
                             std::span<float> fold) {
    m_impl->execute_h(ts_e, ts_v, fold);
}

void FFACOMPLEXCUDA::execute(cuda::std::span<const float> ts_e,
                             cuda::std::span<const float> ts_v,
                             cuda::std::span<float> fold,
                             cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

void FFACOMPLEXCUDA::execute(cuda::std::span<const float> ts_e,
                             cuda::std::span<const float> ts_v,
                             cuda::std::span<ComplexTypeCUDA> fold,
                             cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_cuda(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const search::PulsarSearchConfig& cfg,
                 int device_id) {
    FFACUDA ffa(cfg, device_id);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return {std::move(fold), ffa_plan};
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_complex_cuda(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const search::PulsarSearchConfig& cfg,
                         int device_id) {
    FFACOMPLEXCUDA ffa(cfg, device_id);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(2 * ffa_plan.get_fold_size_complex(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    // RESIZE to actual result size
    fold.resize(ffa_plan.get_fold_size());
    return {std::move(fold), ffa_plan};
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_scores_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::PulsarSearchConfig& cfg,
                        int device_id) {
    const auto [fold, ffa_plan] =
        cfg.get_use_fft_shifts()
            ? compute_ffa_complex_cuda(ts_e, ts_v, cfg, device_id)
            : compute_ffa_cuda(ts_e, ts_v, cfg, device_id);
    const auto nsegments    = ffa_plan.nsegments.back();
    const auto n_param_sets = ffa_plan.ncoords.back();
    error_check::check_equal(
        nsegments, 1U, "compute_ffa_scores: nsegments must be 1 for scores");
    const auto& score_widths = cfg.get_score_widths();
    const auto nscores       = n_param_sets * score_widths.size();
    std::vector<float> scores(nscores);
    detection::snr_boxcar_3d_cuda(fold, n_param_sets, score_widths, scores,
                                  1.0F, device_id);
    return {std::move(scores), ffa_plan};
}

} // namespace loki::algorithms