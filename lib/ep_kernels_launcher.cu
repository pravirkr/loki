#include "loki/ep_kernels_launcher.hpp"

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/ep_kernels_taylor.cuh"
#include "loki/ep_kernels_utils.cuh"

namespace loki::ep_kernels {

// Phase 1a: Classify leaves into branching vs non-branching
__global__ void
compute_shift_bins_kernel(const double* __restrict__ leaves_tree,
                          cuda::std::pair<double, double> coord_cur,
                          double eta,
                          SizeType nbins,
                          SizeType n_leaves,
                          const double* __restrict__ param_limits_d2,
                          const double* __restrict__ param_limits_d1,
                          uint8_t* __restrict__ branch_flags,
                          SizeType* __restrict__ branch_counts) {

    constexpr double kCval           = 299792458.0;
    constexpr double kEps            = 1e-12;
    constexpr SizeType kLeavesStride = 8;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaves)
        return;

    // Load leaf data
    const double* leaf  = &leaves_tree[tid * kLeavesStride];
    const double d2_err = leaf[1];
    const double d1_err = leaf[3];
    const double f0     = leaf[6];

    // Compute shift bins
    const double dt      = coord_cur.second;
    const double dt2     = dt * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double dphi    = eta / static_cast<double>(nbins);
    const double dfactor = kCval / f0;

    // New step sizes
    const double d2_step = dphi * dfactor * 4.0 * inv_dt2;
    const double d1_step = dphi * dfactor * 1.0 * inv_dt;

    // Shift bins
    const double shift_d2 = (d2_err - d2_step) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 = (d1_err - d1_step) * dt * nbins / (1.0 * dfactor);

    const double eta_threshold = eta - kEps;
    const bool needs_d2_branch = shift_d2 >= eta_threshold;
    const bool needs_d1_branch = shift_d1 >= eta_threshold;
    const bool needs_branching = needs_d2_branch || needs_d1_branch;

    branch_flags[tid] = needs_branching ? 1 : 0;

    if (needs_branching) {
        // Compute branch counts
        const double d2_val = leaf[0];
        const double d1_val = leaf[2];

        int n_d2 = 1;
        if (needs_d2_branch) {
            n_d2 = detail::compute_branch_count_device(d2_val, d2_err, d2_step,
                                                       param_limits_d2[0],
                                                       param_limits_d2[1], 16);
        }

        int n_d1 = 1;
        if (needs_d1_branch) {
            const double d1_min = (1.0 - param_limits_d1[1] / f0) * kCval;
            const double d1_max = (1.0 - param_limits_d1[0] / f0) * kCval;
            n_d1 = detail::compute_branch_count_device(d1_val, d1_err, d1_step,
                                                       d1_min, d1_max, 16);
        }

        branch_counts[tid] = n_d2 * n_d1;
    } else {
        branch_counts[tid] = 1;
    }
}

// Phase 2a: Copy non-branching leaves
__global__ void copy_leaves_kernel(const double* __restrict__ leaves_tree,
                                   const uint8_t* __restrict__ branch_flags,
                                   const SizeType* __restrict__ copy_offsets,
                                   SizeType n_leaves,
                                   double* __restrict__ branched_leaves,
                                   SizeType* __restrict__ branched_indices) {

    constexpr SizeType kLeavesStride = 8;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaves)
        return;
    if (branch_flags[tid] == 1)
        return; // Skip branching leaves

    const SizeType out_idx = copy_offsets[tid];
    const double* src      = &leaves_tree[tid * kLeavesStride];
    double* dst            = &branched_leaves[out_idx * kLeavesStride];

// Unrolled copy (compiler optimizes to vector loads/stores)
#pragma unroll
    for (int i = 0; i < kLeavesStride; ++i) {
        dst[i] = src[i];
    }

    branched_indices[out_idx] = tid;
}

// Helper: Compact branching leaf IDs
__global__ void
compact_branching_ids_kernel(const uint8_t* __restrict__ branch_flags,
                             SizeType n_leaves,
                             SizeType* __restrict__ branching_leaf_ids,
                             SizeType* __restrict__ n_branching) {

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= static_cast<int>(n_leaves))
        return;
    if (branch_flags[tid] == 0)
        return;

    const SizeType idx      = atomicAdd(n_branching, 1);
    branching_leaf_ids[idx] = static_cast<SizeType>(tid);
}

// Phase 2b: Branch leaves that need branching
__global__ void
branch_leaves_kernel(const double* __restrict__ leaves_tree,
                     const SizeType* __restrict__ branching_leaf_ids,
                     const SizeType* __restrict__ branch_offsets,
                     cuda::std::pair<double, double> coord_cur,
                     double eta,
                     SizeType nbins,
                     SizeType n_branching_leaves,
                     SizeType branch_max,
                     const double* __restrict__ param_limits_d2,
                     const double* __restrict__ param_limits_d1,
                     double* __restrict__ branched_leaves,
                     SizeType* __restrict__ branched_indices,
                     SizeType n_copy_offset) {

    constexpr double kCval           = 299792458.0;
    constexpr SizeType kLeavesStride = 8;

    __shared__ double s_d2_vals[16];
    __shared__ double s_d1_vals[16];
    __shared__ double s_d2_err, s_d1_err;
    __shared__ double s_d0_val, s_d0_err, s_f0, s_basis;
    __shared__ int s_n_d2, s_n_d1;
    __shared__ SizeType s_out_base;
    __shared__ SizeType s_leaf_idx;

    const int bid        = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    if (bid >= n_branching_leaves)
        return;

    // Load leaf index
    if (tid == 0) {
        s_leaf_idx = branching_leaf_ids[bid];
    }
    __syncthreads();

    const double* leaf = &leaves_tree[s_leaf_idx * kLeavesStride];

    // Thread 0: Branch both parameters
    if (tid == 0) {
        const double d2_val      = leaf[0];
        const double d2_err_orig = leaf[1];
        const double d1_val      = leaf[2];
        const double d1_err_orig = leaf[3];
        const double f0          = leaf[6];

        const double dt      = coord_cur.second;
        const double dt2     = dt * dt;
        const double inv_dt  = 1.0 / dt;
        const double inv_dt2 = inv_dt * inv_dt;
        const double dphi    = eta / static_cast<double>(nbins);
        const double dfactor = kCval / f0;

        // Compute new step sizes
        const double d2_step = dphi * dfactor * 4.0 * inv_dt2;
        const double d1_step = dphi * dfactor * 1.0 * inv_dt;

        // Branch d2
        s_n_d2 = detail::branch_param_padded_device(
            s_d2_vals, d2_val, d2_err_orig, d2_step, param_limits_d2[0],
            param_limits_d2[1], static_cast<int>(branch_max));
        s_d2_err = d2_err_orig / static_cast<double>(s_n_d2);

        // Branch d1
        const double d1_min = (1.0 - param_limits_d1[1] / f0) * kCval;
        const double d1_max = (1.0 - param_limits_d1[0] / f0) * kCval;
        s_n_d1              = detail::branch_param_padded_device(
            s_d1_vals, d1_val, d1_err_orig, d1_step, d1_min, d1_max,
            static_cast<int>(branch_max));
        s_d1_err = d1_err_orig / static_cast<double>(s_n_d1);

        // Copy non-branching parameters
        s_d0_val = leaf[4];
        s_d0_err = leaf[5];
        s_f0     = leaf[6];
        s_basis  = leaf[7];

        // Compute output base
        s_out_base = n_copy_offset + branch_offsets[s_leaf_idx];
    }
    __syncthreads();

    // All threads: Cartesian product
    const int total = s_n_d2 * s_n_d1;
    for (int idx = tid; idx < total; idx += block_size) {
        const int i = idx / s_n_d1;
        const int j = idx % s_n_d1;

        const SizeType out_offset       = (s_out_base + idx) * kLeavesStride;
        branched_leaves[out_offset + 0] = s_d2_vals[i];
        branched_leaves[out_offset + 1] = s_d2_err;
        branched_leaves[out_offset + 2] = s_d1_vals[j];
        branched_leaves[out_offset + 3] = s_d1_err;
        branched_leaves[out_offset + 4] = s_d0_val;
        branched_leaves[out_offset + 5] = s_d0_err;
        branched_leaves[out_offset + 6] = s_f0;
        branched_leaves[out_offset + 7] = s_basis;

        branched_indices[s_out_base + idx] = s_leaf_idx;
    }
}

void launch_branch_and_validate(const double* __restrict__ leaves_tree,
                                cuda::std::pair<double, double> coord_cur,
                                cuda::std::pair<double, double> coord_prev,
                                const double* __restrict__ branched_leaves,
                                const SizeType* __restrict__ branched_indices,
                                SizeType n_leaves,
                                SizeType& n_leaves_after_branching,
                                SizeType& n_leaves_after_validation,
                                double* __restrict__ scratch_params,
                                double* __restrict__ scratch_dparams,
                                SizeType* __restrict__ scratch_counts,
                                SizeType n_params,
                                std::string_view poly_basis) {

    const dim3 block(512);
    const dim3 grid((n_leaves + block.x - 1) / block.x);

    const PolyBasisType basis = parse_basis(poly_basis);

    // Generic lambda to launch the kernel for a known N
    auto launch_for_nparams = [&](auto nparams_const) {
        constexpr int kNparams = nparams_const.value;
        branch_and_validate_kernel<kNparams, PolyBasisType::kTaylor>
            <<<grid, block>>>(leaves_tree, coord_cur, coord_prev,
                              branched_leaves, branched_indices, n_leaves,
                              n_leaves_after_branching,
                              n_leaves_after_validation, scratch_params,
                              scratch_dparams, scratch_counts);
    };

    // Clean Dispatcher
    if (basis == PolyBasisType::kTaylor) {
        switch (n_params) {
        case 2:
            launch_for_nparams(cuda::std::integral_constant<int, 2>{});
            break;
        // case 3:
        //     launch_for_nparams(cuda::std::integral_constant<int, 3>{});
        //     break;
        // case 4:
        //     launch_for_nparams(cuda::std::integral_constant<int, 4>{});
        //     break;
        // case 5:
        //     launch_for_nparams(cuda::std::integral_constant<int, 5>{});
        //     break;
        default:
            throw std::invalid_argument("Taylor NPARAMS not implemented");
        }
    } else {
        throw std::invalid_argument("Basis not implemented");
    }
}

void launch_resolve(const double* __restrict__ branched_leaves,
                    SizeType n_leaves,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    SizeType* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    SizeType nbins,
                    SizeType n_params,
                    std::string_view poly_basis) {

    const dim3 block(512);
    const dim3 grid((n_leaves + block.x - 1) / block.x);
    // Calculate Shared Memory Strategy
    const SizeType shmem_bytes = (n_accel + n_freq) * sizeof(float);
    const SizeType max_shared  = cuda_utils::get_max_shared_memory();

    const bool use_smem = (shmem_bytes <= max_shared);
    if (use_smem) {
        cuda_utils::check_kernel_launch_params(grid, block, shmem_bytes);
    }

    const PolyBasisType basis = parse_basis(poly_basis);

    // Generic lambda to launch the kernel for a known N
    auto launch_for_nparams = [&](auto nparams_const) {
        constexpr int kNparams = nparams_const.value;
        if (use_smem) {
            resolve_kernel<kNparams, PolyBasisType::kTaylor, true>
                <<<grid, block, shmem_bytes>>>(
                    branched_leaves, n_leaves, coord_add, coord_cur, coord_init,
                    accel_grid, n_accel, freq_grid, n_freq, param_idx,
                    phase_shift, nbins);
        } else {
            resolve_kernel<kNparams, PolyBasisType::kTaylor, false>
                <<<grid, block, 0>>>(branched_leaves, n_leaves, coord_add,
                                     coord_cur, coord_init, accel_grid, n_accel,
                                     freq_grid, n_freq, param_idx, phase_shift,
                                     nbins);
        }
    };

    // Clean Dispatcher
    if (basis == PolyBasisType::kTaylor) {
        switch (n_params) {
        case 2:
            launch_for_nparams(cuda::std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_for_nparams(cuda::std::integral_constant<int, 3>{});
            break;
        case 4:
            launch_for_nparams(cuda::std::integral_constant<int, 4>{});
            break;
        default:
            throw std::invalid_argument("Taylor NPARAMS not implemented");
        }
    } else {
        throw std::invalid_argument("Basis not implemented");
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void launch_shift_add(const FoldTypeCUDA* __restrict__ tree_folds,
                      const SizeType* __restrict__ tree_indices,
                      const FoldTypeCUDA* __restrict__ ffa_fold_segment,
                      const SizeType* __restrict__ param_idx,
                      const float* __restrict__ phase_shift,
                      FoldTypeCUDA* __restrict__ out_folds,
                      SizeType n_leaves,
                      SizeType nbins) {
    if (std::is_same_v<FoldTypeCUDA, float>) {
        const auto total_work = n_leaves * nbins;
        const auto block_size = (total_work < 65536) ? 256 : 512;
        const auto grid_size  = (total_work + block_size - 1) / block_size;
        const dim3 block_dim(block_size);
        const dim3 grid_dim(grid_size);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

        shift_add_kernel<<<grid_dim, block_dim>>>(
            tree_folds, tree_indices, ffa_fold_segment, param_idx, phase_shift,
            out_folds, n_leaves, nbins);
    } else {
        const auto nbins_f    = (nbins / 2) + 1;
        const auto total_work = n_leaves * nbins_f;
        const auto block_size = (total_work < 65536) ? 256 : 512;
        const auto grid_size  = (total_work + block_size - 1) / block_size;
        const dim3 block_dim(block_size);
        const dim3 grid_dim(grid_size);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        shift_add_complex_kernel<<<grid_dim, block_dim>>>(
            tree_folds, tree_indices, ffa_fold_segment, param_idx, phase_shift,
            out_folds, n_leaves, nbins, nbins_f);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType launch_score_and_filter(const FoldTypeCUDA* __restrict__ tree_folds,
                                 float* __restrict__ tree_scores,
                                 SizeType* __restrict__ tree_indices,
                                 const SizeType* __restrict__ widths,
                                 SizeType nwidths,
                                 float threshold,
                                 SizeType n_leaves,
                                 SizeType nbins,
                                 cudaStream_t stream) {
    SizeType n_leaves_passing = 0;
    uint32_t* d_n_leaves_passing;
    cudaMallocAsync(&d_n_leaves_passing, sizeof(uint32_t), stream);
    cudaMemsetAsync(d_n_leaves_passing, 0, sizeof(uint32_t), stream);

    if (std::is_same_v<FoldTypeCUDA, float>) {
        constexpr uint32_t kWarpKernelBlockThreads = 128;
        constexpr uint32_t kLeavesPerBlock = kWarpKernelBlockThreads / 32;

        const SizeType warp_kernel_shmem =
            kLeavesPerBlock * nbins * sizeof(float);
        const dim3 block_dim(kWarpKernelBlockThreads);
        const dim3 grid_dim((n_leaves + kLeavesPerBlock - 1) / kLeavesPerBlock);

        cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                               warp_kernel_shmem);

        score_and_filter_kernel<kWarpKernelBlockThreads>
            <<<grid_dim, block_dim, warp_kernel_shmem, stream>>>(
                tree_folds, tree_scores, tree_indices, widths, nwidths,
                threshold, n_leaves, nbins, d_n_leaves_passing);
    } else {
        throw std::invalid_argument("FoldType not implemented");
    }
    cudaMemcpyAsync(&n_leaves_passing, d_n_leaves_passing, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_n_leaves_passing, stream);
    cudaStreamSynchronize(stream);
    return n_leaves_passing;
}

void launch_transform(const double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur,
                      bool use_conservative_tile,
                      SizeType n_params,
                      std::string_view poly_basis) {

    const dim3 block(512);
    const dim3 grid((n_leaves + block.x - 1) / block.x);

    const PolyBasisType basis = parse_basis(poly_basis);

    // Generic lambda to launch the kernel for a known N
    auto launch_for_nparams = [&](auto nparams_const) {
        constexpr int kNparams = nparams_const.value;
        if (use_conservative_tile) {
            transform_kernel<kNparams, PolyBasisType::kTaylor, true>
                <<<grid, block>>>(leaves, n_leaves, coord_next, coord_cur);
        } else {
            transform_kernel<kNparams, PolyBasisType::kTaylor, false>
                <<<grid, block>>>(leaves, n_leaves, coord_next, coord_cur);
        }
    };

    // Clean Dispatcher
    if (basis == PolyBasisType::kTaylor) {
        switch (n_params) {
        case 2:
            launch_for_nparams(cuda::std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_for_nparams(cuda::std::integral_constant<int, 3>{});
            break;
        case 4:
            launch_for_nparams(cuda::std::integral_constant<int, 4>{});
            break;
        default:
            throw std::invalid_argument("Taylor NPARAMS not implemented");
        }
    } else {
        throw std::invalid_argument("Basis not implemented");
    }
}

// --- Explicit template instantiations ---
template void launch_shift_add<float>(const float* __restrict__,
                                      const SizeType* __restrict__,
                                      const float* __restrict__,
                                      const SizeType* __restrict__,
                                      const float* __restrict__,
                                      float* __restrict__,
                                      SizeType,
                                      SizeType);
template void
launch_shift_add<ComplexTypeCUDA>(const ComplexTypeCUDA* __restrict__,
                                  const SizeType* __restrict__,
                                  const ComplexTypeCUDA* __restrict__,
                                  const SizeType* __restrict__,
                                  const float* __restrict__,
                                  ComplexTypeCUDA* __restrict__,
                                  SizeType,
                                  SizeType);
template void launch_score_and_filter<float>(const float* __restrict__,
                                             float* __restrict__,
                                             const SizeType* __restrict__,
                                             SizeType,
                                             SizeType);
template void
launch_score_and_filter<ComplexTypeCUDA>(const ComplexTypeCUDA* __restrict__,
                                         float* __restrict__,
                                         const SizeType* __restrict__,
                                         SizeType,
                                         SizeType);
} // namespace loki::ep_kernels