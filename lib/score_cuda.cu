#include "loki/detection/score.hpp"

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/climits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/cub_helpers.cuh"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/utils/workspace.hpp"

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

// Warp-level inclusive prefix sum, no shared memory (pure shuffles)
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    constexpr int kWarpSize          = 32;
    constexpr unsigned int kFullMask = 0xFFFFFFFF;
    const int lane_id                = threadIdx.x & (kWarpSize - 1);
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float tmp =
            __shfl_up_sync(kFullMask, val, static_cast<unsigned int>(offset));
        if (lane_id >= offset)
            val += tmp;
    }
    return val;
}

// Warp-level max reduction, no shared memory
__device__ __forceinline__ float warp_reduce_max(float val) {
    constexpr unsigned int kFullMask = 0xFFFFFFFF;

    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 16));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 8));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 4));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 2));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 1));

    return val;
}

// Optimized kernel using warp strategy (Assigns one warp per profile)
template <int BlockThreads, bool Is3D, OutputMode Mode>
__global__ void kernel_snr_boxcar_warp(const float* __restrict__ folds,
                                       int nprofiles,
                                       int nbins,
                                       const uint32_t* __restrict__ widths,
                                       int nwidths,
                                       float* __restrict__ scores,
                                       uint32_t* __restrict__ indices_filtered,
                                       uint32_t* nprofiles_passing,
                                       float threshold = 0.0F,
                                       float stdnoise  = 1.0F) {
    // Kernel Configuration & Indexing
    constexpr int kWarpSize         = 32;
    constexpr int kProfilesPerBlock = BlockThreads / kWarpSize;

    const int warp_id     = threadIdx.x / kWarpSize;
    const int lane_id     = threadIdx.x % kWarpSize;
    const int profile_idx = (blockIdx.x * kProfilesPerBlock) + warp_id;

    if (profile_idx >= nprofiles) {
        return;
    }

    // Dynamic Shared Memory [kWarpsPerBlock * nbins]
    extern __shared__ float s_psum[]; // NOLINT
    float* s_psum_warp = s_psum + (warp_id * nbins);

    const float* __restrict__ e_ptr =
        folds + (profile_idx * (Is3D ? 2 : 1) * nbins);
    const float* __restrict__ v_ptr = e_ptr + nbins; // only used in Is3D path

    // stdnoise is only used for 2D
    const float inv_stdnoise = Is3D ? 1.0F : (1.0F / stdnoise);

    // Perform warp-level complicated inclusive prefix sum
    float running_sum    = 0.0F;
    const int num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = (chunk * kWarpSize) + lane_id;
        // Zero-pad out-of-range lanes
        float val;
        if constexpr (Is3D) {
            val = (idx < nbins) ? e_ptr[idx] * rsqrtf(v_ptr[idx]) : 0.0F;
        } else {
            val = (idx < nbins) ? e_ptr[idx] : 0.0F;
        }
        // Warp-local inclusive scan
        val = warp_inclusive_scan(val);
        val += running_sum;
        if (idx < nbins) {
            s_psum_warp[idx] = val;
        }
        // Total sum up to this chunk
        running_sum = __shfl_sync(0xFFFFFFFF, val, kWarpSize - 1);
    }

    // Find max SNR across all widths
    const float total_sum = running_sum;
    float max_snr         = cuda::std::numeric_limits<float>::lowest();

    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = cuda::std::numeric_limits<float>::lowest();
        for (int j = lane_id; j < nbins; j += kWarpSize) {
            const int end_idx    = j + w - 1;
            const float psum_end = (j > 0) ? s_psum_warp[j - 1] : 0.0F;
            const float psum_start =
                (end_idx < nbins) ? s_psum_warp[end_idx]
                                  : total_sum + s_psum_warp[end_idx - nbins];
            thread_max_diff = fmaxf(thread_max_diff, psum_start - psum_end);
        }
        thread_max_diff = warp_reduce_max(thread_max_diff);

        if (lane_id == 0) {
            const float snr_base =
                ((h + b) * thread_max_diff) - (b * total_sum);
            const float snr = snr_base * inv_stdnoise;
            if constexpr (Mode == OutputMode::kMax ||
                          Mode == OutputMode::kMaxAndFilter) {
                max_snr = fmaxf(max_snr, snr);
            } else {
                if constexpr (Mode == OutputMode::kPerWidthAndFilter) {
                    if (snr >= threshold) {
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
        if (lane_id == 0) {
            if constexpr (Mode == OutputMode::kMaxAndFilter) {
                if (max_snr >= threshold) {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>
                        counter(*nprofiles_passing);
                    const uint32_t idx =
                        counter.fetch_add(1, cuda::std::memory_order_relaxed);
                    indices_filtered[idx] = profile_idx;
                    scores[idx]           = max_snr;
                }
            } else {
                scores[profile_idx] = max_snr;
            }
        }
    }
}

template <int MaxBins, int BlockThreads, bool Is3D, OutputMode Mode>
__launch_bounds__(256, 4) // Hint: Max 256 threads, min 4 blocks/SM
    __global__
    void kernel_snr_boxcar_thread(const float* __restrict__ folds,
                                  int nprofiles,
                                  int nbins,
                                  const uint32_t* __restrict__ widths,
                                  int nwidths,
                                  float* __restrict__ scores,
                                  uint32_t* __restrict__ indices_filtered,
                                  uint32_t* nprofiles_passing,
                                  float threshold = 0.0F,
                                  float stdnoise  = 1.0F) {
    const int profile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (profile_idx >= nprofiles) {
        return;
    }

    // Stack allocation matches the template size exactly.
    // For MAX_BINS=64, this is just 256 bytes (likely registers).
    // Guard at index 0: window at j uses psum[j+w] - psum[j]
    float psum[MaxBins + 1];
    psum[0] = 0.0F;

    const float* __restrict__ e_ptr =
        folds + (profile_idx * (Is3D ? 2 : 1) * nbins);
    const float* __restrict__ v_ptr = e_ptr + nbins; // only used in Is3D path

    const float inv_stdnoise = Is3D ? 1.0F : (1.0F / stdnoise);

    float running = 0.0F;
#pragma unroll 8
    for (int i = 0; i < nbins; ++i) {
        if constexpr (Is3D) {
            running += e_ptr[i] * rsqrtf(v_ptr[i]);
        } else {
            running += e_ptr[i];
        }
        psum[i + 1] = running;
    }
    const float total_sum = running;
    float max_snr         = cuda::std::numeric_limits<float>::lowest();

// Loop over widths
#pragma unroll 1
    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float max_diff = cuda::std::numeric_limits<float>::lowest();

        // Split the sliding window loop to eliminate branch in hot path
        // wrap_start is the first j where (j + w - 1) >= nbins
        const int wrap_start = nbins - w + 1;

        // Non-wrapping part: j + w - 1 < nbins
        for (int j = 0; j < wrap_start; ++j) {
            const float window_sum = psum[j + w] - psum[j];
            max_diff               = fmaxf(max_diff, window_sum);
        }
        // Wrapping part: j + w - 1 >= nbins (circular wrap)
        for (int j = wrap_start; j < nbins; ++j) {
            const float window_sum =
                (total_sum - psum[j]) + psum[j + w - nbins];
            max_diff = fmaxf(max_diff, window_sum);
        }
        const float snr_base = ((h + b) * max_diff) - (b * total_sum);
        const float snr      = snr_base * inv_stdnoise;
        if constexpr (Mode == OutputMode::kMax ||
                      Mode == OutputMode::kMaxAndFilter) {
            max_snr = fmaxf(max_snr, snr);
        } else {
            if constexpr (Mode == OutputMode::kPerWidthAndFilter) {
                if (snr >= threshold) {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>
                        counter(*nprofiles_passing);
                    const uint32_t idx =
                        counter.fetch_add(1, cuda::std::memory_order_relaxed);
                    indices_filtered[idx] =
                        static_cast<uint32_t>((profile_idx * nwidths) + iw);
                    scores[idx] = snr;
                }
            } else {
                scores[(profile_idx * nwidths) + iw] = snr;
            }
        }
    }
    if constexpr (Mode == OutputMode::kMax ||
                  Mode == OutputMode::kMaxAndFilter) {
        if constexpr (Mode == OutputMode::kMaxAndFilter) {
            if (max_snr >= threshold) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> counter(
                    *nprofiles_passing);
                const uint32_t idx =
                    counter.fetch_add(1, cuda::std::memory_order_relaxed);
                indices_filtered[idx] = profile_idx;
                scores[idx]           = max_snr;
            }
        } else {
            scores[profile_idx] = max_snr;
        }
    }
}

// Optimized kernel using warp strategy (Assigns one warp per profile)
template <int BlockThreads>
__global__ void
kernel_snr_boxcar_filter_warp(const float* __restrict__ folds,
                              int nprofiles,
                              int nbins,
                              const uint32_t* __restrict__ widths,
                              int nwidths,
                              float* __restrict__ scores,
                              const uint8_t* __restrict__ validation_mask,
                              uint8_t* __restrict__ filtered_mask,
                              float threshold) {
    // Kernel Configuration & Indexing
    constexpr int kWarpSize         = 32;
    constexpr int kProfilesPerBlock = BlockThreads / kWarpSize;

    const int warp_id     = threadIdx.x / kWarpSize;
    const int lane_id     = threadIdx.x % kWarpSize;
    const int profile_idx = (blockIdx.x * kProfilesPerBlock) + warp_id;

    if (profile_idx >= nprofiles) {
        return;
    }
    if (validation_mask[profile_idx] == 0) {
        filtered_mask[profile_idx] = 0;
        return;
    }

    // Dynamic Shared Memory [kWarpsPerBlock * nbins]
    extern __shared__ float s_psum[]; // NOLINT
    float* s_psum_warp = s_psum + (warp_id * nbins);

    const float* __restrict__ e_ptr = folds + (profile_idx * 2 * nbins);
    const float* __restrict__ v_ptr = e_ptr + nbins;

    // Perform warp-level complicated inclusive prefix sum
    float running_sum    = 0.0F;
    const int num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = (chunk * kWarpSize) + lane_id;
        // Zero-pad out-of-range lanes
        float val = (idx < nbins) ? e_ptr[idx] * rsqrtf(v_ptr[idx]) : 0.0F;
        // Warp-local inclusive scan
        val = warp_inclusive_scan(val);
        val += running_sum;
        if (idx < nbins) {
            s_psum_warp[idx] = val;
        }
        // Total sum up to this chunk
        running_sum = __shfl_sync(0xFFFFFFFF, val, kWarpSize - 1);
    }

    // Find max SNR across all widths
    const float total_sum = running_sum;
    float max_snr         = cuda::std::numeric_limits<float>::lowest();

    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = cuda::std::numeric_limits<float>::lowest();
        for (int j = lane_id; j < nbins; j += kWarpSize) {
            const int end_idx    = j + w - 1;
            const float psum_end = (j > 0) ? s_psum_warp[j - 1] : 0.0F;
            const float psum_start =
                (end_idx < nbins) ? s_psum_warp[end_idx]
                                  : total_sum + s_psum_warp[end_idx - nbins];
            thread_max_diff = fmaxf(thread_max_diff, psum_start - psum_end);
        }
        thread_max_diff = warp_reduce_max(thread_max_diff);

        if (lane_id == 0) {
            const float snr = ((h + b) * thread_max_diff) - (b * total_sum);
            max_snr         = fmaxf(max_snr, snr);
        }
    }

    // Final reduction to get max SNR across all widths for this warp
    if (lane_id == 0) {
        scores[profile_idx]        = max_snr;
        filtered_mask[profile_idx] = (max_snr >= threshold);
    }
}

template <int MaxBins>
__launch_bounds__(256, 4) // Hint: Max 256 threads, min 4 blocks/SM
    __global__ void kernel_snr_boxcar_filter_thread(
        const float* __restrict__ folds,
        int nprofiles,
        int nbins,
        const uint32_t* __restrict__ widths,
        int nwidths,
        float* __restrict__ scores,
        const uint8_t* __restrict__ validation_mask,
        uint8_t* __restrict__ filtered_mask,
        float threshold) {
    const int profile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (profile_idx >= nprofiles) {
        return;
    }
    if (validation_mask[profile_idx] == 0) {
        filtered_mask[profile_idx] = 0;
        return;
    }

    // Stack allocation matches the template size exactly.
    // For MAX_BINS=64, this is just 256 bytes (likely registers).
    // Guard at index 0: window at j uses psum[j+w] - psum[j]
    float psum[MaxBins + 1];
    psum[0] = 0.0F;

    const float* __restrict__ e_ptr = folds + (profile_idx * 2 * nbins);
    const float* __restrict__ v_ptr = e_ptr + nbins;

    float running = 0.0F;
#pragma unroll 8
    for (int i = 0; i < nbins; ++i) {
        running += e_ptr[i] * rsqrtf(v_ptr[i]);
        psum[i + 1] = running;
    }
    const float total_sum = running;
    float max_snr         = cuda::std::numeric_limits<float>::lowest();

// Loop over widths
#pragma unroll 1
    for (int iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float max_diff = cuda::std::numeric_limits<float>::lowest();

        // Split the sliding window loop to eliminate branch in hot path
        // wrap_start is the first j where (j + w - 1) >= nbins
        const int wrap_start = nbins - w + 1;

        // Non-wrapping part: j + w - 1 < nbins
        for (int j = 0; j < wrap_start; ++j) {
            const float window_sum = psum[j + w] - psum[j];
            max_diff               = fmaxf(max_diff, window_sum);
        }
        // Wrapping part: j + w - 1 >= nbins (circular wrap)
        for (int j = wrap_start; j < nbins; ++j) {
            const float window_sum =
                (total_sum - psum[j]) + psum[j + w - nbins];
            max_diff = fmaxf(max_diff, window_sum);
        }
        const float snr = ((h + b) * max_diff) - (b * total_sum);
        max_snr         = fmaxf(max_snr, snr);
    }
    scores[profile_idx] = max_snr;
    // Set validation mask for filtered profiles
    filtered_mask[profile_idx] = (max_snr >= threshold);
}

// Unified launch function template
template <bool Is3D, OutputMode Mode>
void snr_boxcar_cuda_impl_device(cuda::std::span<const float> folds,
                                 cuda::std::span<const uint32_t> widths,
                                 cuda::std::span<float> scores,
                                 SizeType nprofiles,
                                 SizeType nbins,
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
    // Dispatch mechanism: Use thread-based when nbins<=64 and nprofiles>=2^16
    constexpr SizeType kWarpSize                = 32;
    constexpr SizeType kThreadsPerBlock         = 256;
    constexpr SizeType kWarpsPerBlock           = kThreadsPerBlock / kWarpSize;
    constexpr SizeType kThreadRegimeMaxBins     = 64;
    constexpr SizeType kThreadRegimeMinProfiles = 1 << 16;

    const bool use_thread_regime = (nbins <= kThreadRegimeMaxBins) &&
                                   (nprofiles >= kThreadRegimeMinProfiles);

    if (use_thread_regime) {
        const SizeType blocks_per_grid =
            (nprofiles + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

        auto dispatch_thread_kernel = [&](auto... args) {
            if (nbins <= 32)
                kernel_snr_boxcar_thread<32, kThreadsPerBlock, Is3D, Mode>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else if (nbins <= 64)
                kernel_snr_boxcar_thread<64, kThreadsPerBlock, Is3D, Mode>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else
                throw std::runtime_error(
                    "thread regime: nbins exceeds limit of 64");
        };
        dispatch_thread_kernel(folds.data(), static_cast<int>(nprofiles),
                               static_cast<int>(nbins), widths.data(),
                               static_cast<int>(nwidths), scores.data(),
                               nullptr, nullptr, 0.0F, stdnoise);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_thread launch failed");

    } else {
        const SizeType blocks_per_grid =
            (nprofiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const SizeType shmem_size = kWarpsPerBlock * nbins * sizeof(float);
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

        kernel_snr_boxcar_warp<kThreadsPerBlock, Is3D, Mode>
            <<<grid_dim, block_dim, shmem_size, stream>>>(
                folds.data(), static_cast<int>(nprofiles),
                static_cast<int>(nbins), widths.data(),
                static_cast<int>(nwidths), scores.data(), nullptr, nullptr,
                0.0F, stdnoise);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_warp launch failed");
    }
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "snr_boxcar_cuda_impl_device synchronization failed");
}

// Unified host wrapper template
template <bool Is3D, OutputMode Mode>
void snr_boxcar_cuda_impl(std::span<const float> folds,
                          std::span<const SizeType> widths,
                          std::span<float> scores,
                          SizeType nprofiles,
                          SizeType nbins,
                          float stdnoise,
                          int device_id) {
    static_assert(Mode == OutputMode::kMax || Mode == OutputMode::kPerWidth,
                  "Filter Mode not allowed");
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    thrust::device_vector<float> folds_d(folds.begin(), folds.end());
    thrust::device_vector<uint32_t> widths_d(widths.begin(), widths.end());
    thrust::device_vector<float> scores_d(scores.size());

    cudaStream_t stream = nullptr;
    snr_boxcar_cuda_impl_device<Is3D, Mode>(
        cuda_utils::as_span(folds_d), cuda_utils::as_span(widths_d),
        cuda_utils::as_span(scores_d), nprofiles, nbins, stdnoise, stream);

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
        folds, widths, scores, nprofiles, nbins, stdnoise, device_id);
}

void snr_boxcar_2d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const uint32_t> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              float stdnoise,
                              cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<false, OutputMode::kMax>(
        folds, widths, scores, nprofiles, nbins, stdnoise, stream);
}

void snr_boxcar_3d_cuda(std::span<const float> folds,
                        std::span<const SizeType> widths,
                        std::span<float> scores,
                        SizeType nprofiles,
                        SizeType nbins,
                        int device_id) {
    snr_boxcar_cuda_impl<true, OutputMode::kPerWidth>(
        folds, widths, scores, nprofiles, nbins, 1.0F, device_id);
}

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> folds,
                          cuda::std::span<const uint32_t> widths,
                          cuda::std::span<float> scores,
                          SizeType nprofiles,
                          SizeType nbins,
                          cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<true, OutputMode::kPerWidth>(
        folds, widths, scores, nprofiles, nbins, 1.0F, stream);
}

void snr_boxcar_3d_max_cuda(std::span<const float> folds,
                            std::span<const SizeType> widths,
                            std::span<float> scores,
                            SizeType nprofiles,
                            SizeType nbins,
                            int device_id) {
    snr_boxcar_cuda_impl<true, OutputMode::kMax>(
        folds, widths, scores, nprofiles, nbins, 1.0F, device_id);
}

void snr_boxcar_3d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const uint32_t> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              cudaStream_t stream) {
    snr_boxcar_cuda_impl_device<true, OutputMode::kMax>(
        folds, widths, scores, nprofiles, nbins, 1.0F, stream);
}

SizeType score_and_filter_cuda_d(cuda::std::span<const float> folds,
                                 cuda::std::span<const uint32_t> widths,
                                 cuda::std::span<float> scores,
                                 cuda::std::span<uint32_t> indices_filtered,
                                 float threshold,
                                 SizeType nprofiles,
                                 SizeType nbins,
                                 cudaStream_t stream,
                                 memory::DeviceCounter& counter) {
    counter.reset(stream);

    // Dispatch mechanism: Use thread-based when nbins<=64 and nprofiles>=2^16
    constexpr SizeType kWarpSize                = 32;
    constexpr SizeType kThreadsPerBlock         = 256;
    constexpr SizeType kWarpsPerBlock           = kThreadsPerBlock / kWarpSize;
    constexpr SizeType kThreadRegimeMaxBins     = 64;
    constexpr SizeType kThreadRegimeMinProfiles = 1 << 16;

    const bool use_thread_regime = (nbins <= kThreadRegimeMaxBins) &&
                                   (nprofiles >= kThreadRegimeMinProfiles);

    if (use_thread_regime) {
        const SizeType blocks_per_grid =
            (nprofiles + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

        auto dispatch_thread_kernel = [&](auto... args) {
            if (nbins <= 32)
                kernel_snr_boxcar_thread<32, kThreadsPerBlock, true,
                                         OutputMode::kPerWidthAndFilter>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else if (nbins <= 64)
                kernel_snr_boxcar_thread<64, kThreadsPerBlock, true,
                                         OutputMode::kPerWidthAndFilter>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else
                throw std::runtime_error(
                    "thread regime: nbins exceeds limit of 64");
        };
        dispatch_thread_kernel(
            folds.data(), static_cast<int>(nprofiles), static_cast<int>(nbins),
            widths.data(), static_cast<int>(widths.size()), scores.data(),
            indices_filtered.data(), counter.d_ptr, threshold, 1.0F);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_thread launch failed");

    } else {
        const SizeType blocks_per_grid =
            (nprofiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const SizeType shmem_size = kWarpsPerBlock * nbins * sizeof(float);
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

        kernel_snr_boxcar_warp<kThreadsPerBlock, true,
                               OutputMode::kPerWidthAndFilter>
            <<<grid_dim, block_dim, shmem_size, stream>>>(
                folds.data(), static_cast<int>(nprofiles),
                static_cast<int>(nbins), widths.data(),
                static_cast<int>(widths.size()), scores.data(),
                indices_filtered.data(), counter.d_ptr, threshold, 1.0F);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_warp launch failed");
    }
    return counter.value_sync(stream);
}

SizeType
score_and_filter_max_cuda_d(cuda::std::span<const float> folds,
                            cuda::std::span<const uint32_t> widths,
                            cuda::std::span<float> scores,
                            cuda::std::span<const uint8_t> validation_mask,
                            cuda::std::span<uint8_t> filtered_mask,
                            float threshold,
                            SizeType nprofiles,
                            SizeType nbins,
                            memory::CUBScratchArena& scratch_ws,
                            cudaStream_t stream) {
    // Dispatch mechanism: Use thread-based when nbins<=64 and nprofiles>=2^16
    constexpr SizeType kWarpSize                = 32;
    constexpr SizeType kThreadsPerBlock         = 256;
    constexpr SizeType kWarpsPerBlock           = kThreadsPerBlock / kWarpSize;
    constexpr SizeType kThreadRegimeMaxBins     = 64;
    constexpr SizeType kThreadRegimeMinProfiles = 1 << 16;

    const bool use_thread_regime = (nbins <= kThreadRegimeMaxBins) &&
                                   (nprofiles >= kThreadRegimeMinProfiles);

    if (use_thread_regime) {
        const SizeType blocks_per_grid =
            (nprofiles + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

        auto dispatch_thread_kernel = [&](auto... args) {
            if (nbins <= 32)
                kernel_snr_boxcar_filter_thread<32>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else if (nbins <= 64)
                kernel_snr_boxcar_filter_thread<64>
                    <<<grid_dim, block_dim, 0, stream>>>(args...);
            else
                throw std::runtime_error(
                    "thread regime: nbins exceeds limit of 64");
        };
        dispatch_thread_kernel(
            folds.data(), static_cast<int>(nprofiles), static_cast<int>(nbins),
            widths.data(), static_cast<int>(widths.size()), scores.data(),
            validation_mask.data(), filtered_mask.data(), threshold);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_filter_thread launch failed");

    } else {
        const SizeType blocks_per_grid =
            (nprofiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const SizeType shmem_size = kWarpsPerBlock * nbins * sizeof(float);
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);

        kernel_snr_boxcar_filter_warp<kThreadsPerBlock>
            <<<grid_dim, block_dim, shmem_size, stream>>>(
                folds.data(), static_cast<int>(nprofiles),
                static_cast<int>(nbins), widths.data(),
                static_cast<int>(widths.size()), scores.data(),
                validation_mask.data(), filtered_mask.data(), threshold);
        cuda_utils::check_last_cuda_error(
            "kernel_snr_boxcar_filter_warp launch failed");
    }
    // Count number of passing profiles

    auto transform_it =
        thrust::make_transform_iterator(filtered_mask.data(), Uint8ToUint32{});
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Sum(scratch_ws.cub_temp_storage,
                               scratch_ws.cub_temp_bytes, transform_it,
                               scratch_ws.d_reduce_out, nprofiles, stream),
        "cub::DeviceReduce::Sum failed");

    // Copy result back
    uint32_t nprofiles_passing = 0;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&nprofiles_passing, scratch_ws.d_reduce_out,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    // You already sync elsewhere usually, but if needed:
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "stream sync failed");

    return nprofiles_passing;
}

} // namespace loki::detection