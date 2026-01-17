#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"

#include <cstddef>
#include <cub/cub.cuh>
#include <cub/version.cuh>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::detection {

namespace {

// Optimized 2D max kernel using warp strategy (Assigns one warp per profile)
template <int BlockThreads, bool Is3D, bool FindMax>
__global__ void snr_boxcar_kernel_warp(const float* __restrict__ arr,
                                       int nprofiles,
                                       int nbins,
                                       const SizeType* __restrict__ widths,
                                       int nwidths,
                                       float* __restrict__ out,
                                       float stdnoise = 1.0F) {
    // Kernel Configuration & Indexing
    constexpr int kWarpSize         = 32;
    constexpr int kProfilesPerBlock = BlockThreads / kWarpSize;
    const int warp_id               = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id               = static_cast<int>(threadIdx.x) % kWarpSize;
    const int profile_idx = (blockIdx.x * kProfilesPerBlock) + warp_id;

    if (profile_idx >= nprofiles) {
        return;
    }

    // Dynamic Shared Memory
    extern __shared__ float s_mem_block[]; // NOLINT
    float* s_warp_data = &s_mem_block[static_cast<IndexType>(warp_id * nbins)];

    using WarpScan   = cub::WarpScan<float, kWarpSize>;
    using WarpReduce = cub::WarpReduce<float, kWarpSize>;
    __shared__ typename WarpScan::TempStorage temp_scan[kProfilesPerBlock];
    __shared__ typename WarpReduce::TempStorage temp_reduce[kProfilesPerBlock];

    // Load data from global to shared memory
    for (int j = lane_id; j < nbins; j += kWarpSize) {
        if constexpr (Is3D) {
            const int idx_e = (profile_idx * 2 * nbins) + j;
            const int idx_v = (profile_idx * 2 * nbins) + j + nbins;
            s_warp_data[j]  = arr[idx_e] / sqrtf(arr[idx_v]);
        } else {
            const int idx  = (profile_idx * nbins) + j;
            s_warp_data[j] = arr[idx];
        }
    }
    __syncthreads();

    // Perform warp-level complicated inclusive prefix sum
    float running_sum    = 0.0F;
    const int num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int idx   = (chunk * kWarpSize) + lane_id;
        float val = (idx < nbins) ? s_warp_data[idx] : 0.0F;
        WarpScan(temp_scan[warp_id]).InclusiveSum(val, val);
        if (idx < nbins) {
            s_warp_data[idx] = val + running_sum;
        }
        float chunk_sum = __shfl_sync(0xFFFFFFFF, val, kWarpSize - 1);
        if (lane_id == 0) {
            running_sum += chunk_sum;
        }
        running_sum = __shfl_sync(0xFFFFFFFF, running_sum, 0);
    }
    __syncthreads();

    // Find max SNR across all widths
    const float total_sum = s_warp_data[nbins - 1];
    float thread_max_snr  = -CUDART_INF_F;

    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = -CUDART_INF_F;
        for (int j = lane_id; j < nbins; j += kWarpSize) {
            float sum_before_start = (j > 0) ? s_warp_data[j - 1] : 0.0F;
            float current_sum;
            const int end_idx = j + w - 1;
            if (end_idx < nbins) {
                // Normal case: sum from j to j+w-1
                current_sum = s_warp_data[end_idx] - sum_before_start;
            } else {
                // Circular case: sum wraps around
                current_sum = (total_sum - sum_before_start) +
                              s_warp_data[end_idx % nbins];
            }
            thread_max_diff = fmaxf(thread_max_diff, current_sum);
        }
        float max_diff = WarpReduce(temp_reduce[warp_id])
                             .Reduce(thread_max_diff, CubMaxOp<float>());

        if (lane_id == 0) {
            float snr;
            if constexpr (Is3D) {
                snr = ((h + b) * max_diff) - (b * total_sum);
            } else {
                snr = (((h + b) * max_diff) - (b * total_sum)) / stdnoise;
            }

            if constexpr (FindMax) {
                thread_max_snr = fmaxf(thread_max_snr, snr);
            } else {
                out[(profile_idx * nwidths) + iw] = snr;
            }
        }
    }

    // Final reduction to get max SNR across all widths for this warp
    if constexpr (FindMax) {
        __shared__
            typename WarpReduce::TempStorage temp_final[kProfilesPerBlock];
        float final_max_snr = WarpReduce(temp_final[warp_id])
                                  .Reduce(thread_max_snr, CubMaxOp<float>());
        if (lane_id == 0) {
            out[profile_idx] = final_max_snr;
        }
    }
}

// Unified launch function template
template <bool Is3D, bool FindMax>
void launch_snr_boxcar_kernel(cuda::std::span<const float> arr,
                              SizeType nprofiles,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> out,
                              float stdnoise,
                              int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);

    const auto nbins =
        Is3D ? arr.size() / (2 * nprofiles) : arr.size() / nprofiles;
    const auto nwidths = widths.size();
    if constexpr (FindMax) {
        error_check::check_equal(
            out.size(), nprofiles,
            "launch_snr_boxcar_kernel: out size does not match");
    } else {
        error_check::check_equal(
            out.size(), nprofiles * nwidths,
            "launch_snr_boxcar_kernel: out size does not match");
    }

    constexpr int kWarpKernelBlockThreads = 128;
    constexpr int kProfilesPerBlock       = kWarpKernelBlockThreads / 32;
    const size_t warp_kernel_shmem = kProfilesPerBlock * nbins * sizeof(float);
    const dim3 block_dim(kWarpKernelBlockThreads);
    const dim3 grid_dim((nprofiles + kProfilesPerBlock - 1) /
                        kProfilesPerBlock);

    cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                           warp_kernel_shmem);

    snr_boxcar_kernel_warp<kWarpKernelBlockThreads, Is3D, FindMax>
        <<<grid_dim, block_dim, warp_kernel_shmem>>>(
            arr.data(), static_cast<int>(nprofiles), static_cast<int>(nbins),
            widths.data(), static_cast<int>(nwidths), out.data(), stdnoise);

    cuda_utils::check_last_cuda_error("snr_boxcar_kernel_warp launch failed");
}

// Unified host wrapper template
template <bool Is3D, bool FindMax>
void snr_boxcar_cuda_impl(std::span<const float> arr,
                          SizeType nprofiles,
                          std::span<const SizeType> widths,
                          std::span<float> out,
                          float stdnoise,
                          int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    thrust::device_vector<float> d_arr(arr.begin(), arr.end());
    thrust::device_vector<SizeType> d_widths(widths.begin(), widths.end());
    thrust::device_vector<float> d_out(out.size());

    launch_snr_boxcar_kernel<Is3D, FindMax>(
        cuda::std::span<const float>(thrust::raw_pointer_cast(d_arr.data()),
                                     d_arr.size()),
        nprofiles,
        cuda::std::span<const SizeType>(
            thrust::raw_pointer_cast(d_widths.data()), d_widths.size()),
        cuda::std::span<float>(thrust::raw_pointer_cast(d_out.data()),
                               d_out.size()),
        stdnoise, device_id);

    thrust::copy(d_out.begin(), d_out.end(), out.begin());
}

} // namespace

void snr_boxcar_2d_max_cuda(std::span<const float> arr,
                            SizeType nprofiles,
                            std::span<const SizeType> widths,
                            std::span<float> out,
                            float stdnoise,
                            int device_id) {
    snr_boxcar_cuda_impl<false, true>(arr, nprofiles, widths, out, stdnoise,
                                      device_id);
}

void snr_boxcar_2d_max_cuda_d(cuda::std::span<const float> arr,
                              SizeType nprofiles,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> out,
                              float stdnoise,
                              int device_id) {
    launch_snr_boxcar_kernel<false, true>(arr, nprofiles, widths, out, stdnoise,
                                          device_id);
}

void snr_boxcar_3d_cuda(std::span<const float> arr,
                        SizeType nprofiles,
                        std::span<const SizeType> widths,
                        std::span<float> out,
                        int device_id) {
    snr_boxcar_cuda_impl<true, false>(arr, nprofiles, widths, out, 1.0F,
                                      device_id);
}

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> arr,
                          SizeType nprofiles,
                          cuda::std::span<const SizeType> widths,
                          cuda::std::span<float> out,
                          int device_id) {
    launch_snr_boxcar_kernel<true, false>(arr, nprofiles, widths, out, 1.0F,
                                          device_id);
}

void snr_boxcar_3d_max_cuda(std::span<const float> arr,
                            SizeType nprofiles,
                            std::span<const SizeType> widths,
                            std::span<float> out,
                            int device_id) {
    snr_boxcar_cuda_impl<true, true>(arr, nprofiles, widths, out, 1.0F,
                                     device_id);
}

void snr_boxcar_3d_max_cuda_d(cuda::std::span<const float> arr,
                              SizeType nprofiles,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> out,
                              int device_id) {
    launch_snr_boxcar_kernel<true, true>(arr, nprofiles, widths, out, 1.0F,
                                         device_id);
}

} // namespace loki::detection