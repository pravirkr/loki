#include "loki/detection/score.hpp"

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::detection {

namespace {

enum class OutputMode : uint8_t {
    kMax          = 0, // Max SNR for each profile
    kMaxAndFilter = 1, // Max SNR for each profile passing the threshold and the
                       // index in unfiltered scores
    kPerWidth          = 2, // SNR for each width for each profile
    kPerWidthAndFilter = 3, // SNR for each width for each profile passing the
                            // threshold and the index in unfiltered scores
};

// Optimized kernel using warp strategy (Assigns one warp per profile)
template <uint32_t BlockThreads, bool Is3D, OutputMode Mode>
__global__ void kernel_snr_boxcar_warp(const float* __restrict__ folds,
                                       uint32_t nprofiles,
                                       uint32_t nbins,
                                       const uint32_t* __restrict__ widths,
                                       uint32_t nwidths,
                                       float* __restrict__ scores,
                                       uint32_t* __restrict__ indices_filtered,
                                       uint32_t* nprofiles_passing,
                                       float threshold = 0.0F,
                                       float stdnoise  = 1.0F) {
    // Kernel Configuration & Indexing
    constexpr uint32_t kWarpSize         = 32;
    constexpr uint32_t kProfilesPerBlock = BlockThreads / kWarpSize;

    const uint32_t warp_id     = threadIdx.x / kWarpSize;
    const uint32_t lane_id     = threadIdx.x % kWarpSize;
    const uint32_t profile_idx = (blockIdx.x * kProfilesPerBlock) + warp_id;

    if (profile_idx >= nprofiles) {
        return;
    }

    // Dynamic Shared Memory
    extern __shared__ float s_mem_block[];
    float* s_warp_data = &s_mem_block[static_cast<SizeType>(warp_id * nbins)];

    using WarpScan   = cub::WarpScan<float, kWarpSize>;
    using WarpReduce = cub::WarpReduce<float, kWarpSize>;
    __shared__ typename WarpScan::TempStorage temp_scan[kProfilesPerBlock];
    __shared__ typename WarpReduce::TempStorage temp_reduce[kProfilesPerBlock];

    // Load data from global to shared memory
    for (uint32_t j = lane_id; j < nbins; j += kWarpSize) {
        if constexpr (Is3D) {
            const uint32_t idx_e = (profile_idx * 2 * nbins) + j;
            const uint32_t idx_v = (profile_idx * 2 * nbins) + j + nbins;
            s_warp_data[j]       = folds[idx_e] / sqrtf(folds[idx_v]);
        } else {
            const uint32_t idx = (profile_idx * nbins) + j;
            s_warp_data[j]     = folds[idx];
        }
    }

    // stdnoise is only used for 2D
    const float inv_stdnoise = Is3D ? 1.0F : (1.0F / stdnoise);

    // Perform warp-level complicated inclusive prefix sum
    float running_sum         = 0.0F;
    const uint32_t num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t idx = (chunk * kWarpSize) + lane_id;
        float val          = (idx < nbins) ? s_warp_data[idx] : 0.0F;
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

    // Find max SNR across all widths
    const float total_sum = s_warp_data[nbins - 1];
    float thread_max_snr  = -cuda::std::numeric_limits<float>::infinity();

    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const uint32_t w = widths[iw];
        const float h    = sqrtf(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = -cuda::std::numeric_limits<float>::infinity();
        for (uint32_t j = lane_id; j < nbins; j += kWarpSize) {
            const float sum_before_start = (j > 0) ? s_warp_data[j - 1] : 0.0F;
            const uint32_t end_idx       = j + w - 1;
            float current_sum;
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
        const float max_diff = WarpReduce(temp_reduce[warp_id])
                                   .Reduce(thread_max_diff, CubMaxOp<float>());

        if (lane_id == 0) {
            const float snr_base = ((h + b) * max_diff) - (b * total_sum);
            const float snr      = snr_base * inv_stdnoise;
            if constexpr (Mode == OutputMode::kMax ||
                          Mode == OutputMode::kMaxAndFilter) {
                thread_max_snr = fmaxf(thread_max_snr, snr);
            } else {
                if constexpr (Mode == OutputMode::kPerWidthAndFilter) {
                    if (snr > threshold) {
                        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>
                            counter(*nprofiles_passing);
                        const uint32_t idx = counter.fetch_add(
                            1, cuda::std::memory_order_relaxed);
                        indices_filtered[idx] = (profile_idx * nwidths) + iw;
                        scores[idx]           = snr;
                    }
                } else {
                    scores[(profile_idx * nwidths) + iw] = snr;
                }
            }
        }
    }

    // Final reduction to get max SNR across all widths for this warp
    if constexpr (Mode == OutputMode::kMax ||
                  Mode == OutputMode::kMaxAndFilter) {
        __shared__
            typename WarpReduce::TempStorage temp_final[kProfilesPerBlock];
        float final_max_snr = WarpReduce(temp_final[warp_id])
                                  .Reduce(thread_max_snr, CubMaxOp<float>());

        if (lane_id == 0) {
            if constexpr (Mode == OutputMode::kMaxAndFilter) {
                if (final_max_snr > threshold) {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>
                        counter(*nprofiles_passing);
                    const uint32_t idx =
                        counter.fetch_add(1, cuda::std::memory_order_relaxed);
                    indices_filtered[idx] = profile_idx;
                    scores[idx]           = final_max_snr;
                }
            } else {
                scores[profile_idx] = final_max_snr;
            }
        }
    }
}

// Unified launch function template
template <bool Is3D, OutputMode Mode>
void snr_boxcar_cuda_impl_device(cuda::std::span<const float> folds,
                                 SizeType nprofiles,
                                 SizeType nbins,
                                 cuda::std::span<const SizeType> widths,
                                 cuda::std::span<float> scores,
                                 float stdnoise,
                                 cudaStream_t stream) {
    static_assert(Mode == OutputMode::kMax || Mode == OutputMode::kPerWidth,
                  "Filter Mode not allowed");
    const auto nwidths = widths.size();
    if constexpr (Mode == OutputMode::kMax) {
        error_check::check_equal(
            scores.size(), nprofiles,
            "snr_boxcar_cuda_impl_device: out size does not match");
    } else {
        error_check::check_equal(
            scores.size(), nprofiles * nwidths,
            "snr_boxcar_cuda_impl_device: out size does not match");
    }

    constexpr SizeType kWarpSize         = 32;
    constexpr SizeType kThreadsPerBlock  = 128;
    constexpr SizeType kProfilesPerBlock = kThreadsPerBlock / kWarpSize;
    const SizeType blocks_per_grid =
        (nprofiles + kProfilesPerBlock - 1) / kProfilesPerBlock;
    const SizeType shmem_size = kProfilesPerBlock * nbins * sizeof(float);
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

    kernel_snr_boxcar_warp<kThreadsPerBlock, Is3D, Mode>
        <<<grid_dim, block_dim, shmem_size, stream>>>(
            folds.data(), nprofiles, nbins, widths.data(), nwidths,
            scores.data(), nullptr, nullptr, 0.0F, stdnoise);
    cuda_utils::check_last_cuda_error(
        "snr_boxcar_cuda_impl_device launch failed");
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "snr_boxcar_cuda_impl_device synchronization failed");
}

// Unified host wrapper template
template <bool Is3D, OutputMode Mode>
void snr_boxcar_cuda_impl(std::span<const float> folds,
                          SizeType nprofiles,
                          SizeType nbins,
                          std::span<const SizeType> widths,
                          std::span<float> scores,
                          float stdnoise,
                          int device_id) {
    static_assert(Mode == OutputMode::kMax || Mode == OutputMode::kPerWidth,
                  "Filter Mode not allowed");
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    thrust::device_vector<float> folds_d(folds.begin(), folds.end());
    thrust::device_vector<SizeType> widths_d(widths.begin(), widths.end());
    thrust::device_vector<float> scores_d(scores.size());

    cudaStream_t stream = nullptr;
    snr_boxcar_cuda_impl_device<Is3D, Mode>(
        cuda_utils::as_span(folds_d), nprofiles, nbins,
        cuda_utils::as_span(widths_d), cuda_utils::as_span(scores_d), stdnoise,
        stream);

    thrust::copy(scores_d.begin(), scores_d.end(), scores.begin());
}

} // namespace

void snr_boxcar_2d_max_cuda(std::span<const float> folds,
                            std::span<const SizeType> widths,
                            std::span<float> scores,
                            SizeType nprofiles,
                            SizeType nbins,
                            float stdnoise,
                            int device_id) {
    snr_boxcar_cuda_impl<false, OutputMode::kMax>(
        folds, nprofiles, nbins, widths, scores, stdnoise, device_id);
}

void snr_boxcar_2d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              float stdnoise,
                              cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<false, OutputMode::kMax>(
        folds, nprofiles, nbins, widths, scores, stdnoise, stream);
}

void snr_boxcar_3d_cuda(std::span<const float> folds,
                        std::span<const SizeType> widths,
                        std::span<float> scores,
                        SizeType nprofiles,
                        SizeType nbins,
                        int device_id) {
    snr_boxcar_cuda_impl<true, OutputMode::kPerWidth>(
        folds, nprofiles, nbins, widths, scores, 1.0F, device_id);
}

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> folds,
                          cuda::std::span<const SizeType> widths,
                          cuda::std::span<float> scores,
                          SizeType nprofiles,
                          SizeType nbins,
                          cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<true, OutputMode::kPerWidth>(
        folds, nprofiles, nbins, widths, scores, 1.0F, stream);
}

void snr_boxcar_3d_max_cuda(std::span<const float> folds,
                            std::span<const SizeType> widths,
                            std::span<float> scores,
                            SizeType nprofiles,
                            SizeType nbins,
                            int device_id) {
    snr_boxcar_cuda_impl<true, OutputMode::kMax>(
        folds, nprofiles, nbins, widths, scores, 1.0F, device_id);
}

void snr_boxcar_3d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const SizeType> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<true, OutputMode::kMax>(
        folds, nprofiles, nbins, widths, scores, 1.0F, stream);
}

SizeType score_and_filter_cuda_d(cuda::std::span<const float> folds,
                                 cuda::std::span<const SizeType> widths,
                                 cuda::std::span<float> scores,
                                 cuda::std::span<SizeType> indices_filtered,
                                 float threshold,
                                 SizeType nprofiles,
                                 SizeType nbins,
                                 cudaStream_t stream) {
    SizeType nprofiles_passing = 0;
    uint32_t* d_nprofiles_passing;
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_nprofiles_passing, sizeof(uint32_t), stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_nprofiles_passing, 0, sizeof(uint32_t), stream),
        "cudaMemsetAsync failed");

    constexpr SizeType kWarpSize         = 32;
    constexpr SizeType kThreadsPerBlock  = 128;
    constexpr SizeType kProfilesPerBlock = kThreadsPerBlock / kWarpSize;
    const SizeType blocks_per_grid =
        (nprofiles + kProfilesPerBlock - 1) / kProfilesPerBlock;
    const SizeType shmem_size = kProfilesPerBlock * nbins * sizeof(float);
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

    kernel_snr_boxcar_warp<kThreadsPerBlock, true,
                           OutputMode::kPerWidthAndFilter>
        <<<grid_dim, block_dim, shmem_size, stream>>>(
            folds.data(), nprofiles, nbins, widths.data(), widths.size(),
            scores.data(), indices_filtered.data(), d_nprofiles_passing,
            threshold, 1.0F);
    cuda_utils::check_last_cuda_error(
        "score_and_filter_cuda_d kernel launch failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&nprofiles_passing, d_nprofiles_passing,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(cudaFreeAsync(d_nprofiles_passing, stream),
                                "cudaFreeAsync failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
    return nprofiles_passing;
}

SizeType score_and_filter_max_cuda_d(cuda::std::span<const float> folds,
                                     cuda::std::span<const SizeType> widths,
                                     cuda::std::span<float> scores,
                                     cuda::std::span<SizeType> indices_filtered,
                                     float threshold,
                                     SizeType nprofiles,
                                     SizeType nbins,
                                     cudaStream_t stream) {
    SizeType nprofiles_passing = 0;
    uint32_t* d_nprofiles_passing;
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_nprofiles_passing, sizeof(uint32_t), stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_nprofiles_passing, 0, sizeof(uint32_t), stream),
        "cudaMemsetAsync failed");

    constexpr SizeType kWarpSize         = 32;
    constexpr SizeType kThreadsPerBlock  = 128;
    constexpr SizeType kProfilesPerBlock = kThreadsPerBlock / kWarpSize;
    const SizeType blocks_per_grid =
        (nprofiles + kProfilesPerBlock - 1) / kProfilesPerBlock;
    const SizeType shmem_size = kProfilesPerBlock * nbins * sizeof(float);
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

    kernel_snr_boxcar_warp<kThreadsPerBlock, true, OutputMode::kMaxAndFilter>
        <<<grid_dim, block_dim, shmem_size, stream>>>(
            folds.data(), nprofiles, nbins, widths.data(), widths.size(),
            scores.data(), indices_filtered.data(), d_nprofiles_passing,
            threshold, 1.0F);
    cuda_utils::check_last_cuda_error(
        "score_and_filter_max_cuda_d kernel launch failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&nprofiles_passing, d_nprofiles_passing,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(cudaFreeAsync(d_nprofiles_passing, stream),
                                "cudaFreeAsync failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
    return nprofiles_passing;
}

} // namespace loki::detection