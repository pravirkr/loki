#include "loki/detection/score.hpp"

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::detection {

namespace {

#define CUDART_INF_F __int_as_float(0x7f800000)

__global__ void snr_boxcar_kernel_cub(const float* __restrict__ arr,
                                      int nprofiles,
                                      const SizeType* __restrict__ widths,
                                      int nwidths,
                                      float* __restrict__ out,
                                      int nbins,
                                      float stdnoise,
                                      int profiles_per_block) {
    // Typedef for CUB BlockScan for a 1D block of 256 threads.
    using BlockScan = cub::BlockScan<float, 256>;

    const int block_start = static_cast<int>(blockIdx.x) * profiles_per_block;
    const int block_end = std::min(block_start + profiles_per_block, nprofiles);
    const int tid       = static_cast<int>(threadIdx.x);
    const int stride    = static_cast<int>(blockDim.x);

    // Shared memory for CUB's internal storage.
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Shared memory as a staging area for our data.
    __shared__ float s_data[2048]; // Assumes nbins <= 2048

    for (int profile_idx = block_start; profile_idx < block_end;
         ++profile_idx) {
        // Step 1: Cooperatively load data from global to shared memory staging
        // area.
        for (int j = tid; j < nbins; j += stride) {
            const int idx_e = (profile_idx * 2 * nbins) + j;
            const int idx_v = (profile_idx * 2 * nbins) + j + nbins;
            s_data[j]       = arr[idx_e] / sqrtf(arr[idx_v]);
        }
        __syncthreads();

        // Step 2: The Correct CUB Scan Pattern
        // a) Each thread loads its value from shared memory into a register.
        float thread_val = 0.0F;
        if (tid < nbins) {
            thread_val = s_data[tid];
        }

        // b) Perform the inclusive scan on the per-thread register data.
        BlockScan(temp_storage).InclusiveSum(thread_val, thread_val);

        // c) Each thread writes its result back to shared memory.
        if (tid < nbins) {
            s_data[tid] = thread_val;
        }
        __syncthreads();
        const float total_sum = s_data[nbins - 1];

        for (int iw = tid; iw < nwidths; iw += stride) {
            const int w   = static_cast<int>(widths[iw]);
            const float h = sqrtf(static_cast<float>(nbins - w) /
                                  static_cast<float>(nbins * w));
            const float b =
                static_cast<float>(w) * h / static_cast<float>(nbins - w);
            float max_diff = -CUDART_INF_F;

            for (int j = 0; j < nbins; ++j) {
                float current_sum;
                int end_idx = j + w - 1;

                if (end_idx < nbins) {
                    float sum_at_end       = s_data[end_idx];
                    float sum_before_start = (j > 0) ? s_data[j - 1] : 0.0F;
                    current_sum            = sum_at_end - sum_before_start;
                } else {
                    float sum_tail =
                        total_sum - ((j > 0) ? s_data[j - 1] : 0.0F);
                    float sum_head = s_data[end_idx % nbins];
                    current_sum    = sum_tail + sum_head;
                }
                max_diff = fmaxf(max_diff, current_sum);
            }

            const float snr = ((h + b) * max_diff - b * total_sum) / stdnoise;
            out[(profile_idx * nwidths) + iw] = snr;
        }
        __syncthreads();
    }
}

} // namespace

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> arr,
                          SizeType nprofiles,
                          std::span<const SizeType> widths,
                          std::span<float> out,
                          float stdnoise,
                          int device_id) {
    cuda_utils::set_device(device_id);

    const auto nbins      = arr.size() / (2 * nprofiles);
    const auto ntemplates = widths.size();
    error_check::check_equal(out.size(), nprofiles * ntemplates,
                             "snr_boxcar_3d_cuda_d: out size does not match");

    thrust::device_vector<SizeType> d_widths(widths.begin(), widths.end());
    thrust::device_vector<float> d_out(out.size());

    const int profiles_per_block = 8;
    const int block_size         = 256;
    const int grid_size =
        (static_cast<int>(nprofiles) + profiles_per_block - 1) /
        profiles_per_block;
    const size_t shared_mem_size = sizeof(float) * 2048;
    const dim3 block_dim(block_size);
    const dim3 grid_dim(grid_size);

    snr_boxcar_kernel_cub<<<grid_dim, block_dim, shared_mem_size>>>(
        arr.data(), static_cast<int>(nprofiles), d_widths.data().get(),
        static_cast<int>(ntemplates), d_out.data().get(),
        static_cast<int>(nbins), stdnoise, profiles_per_block);

    cuda_utils::check_last_cuda_error(
        "snr_boxcar_kernel_cub_fixed launch failed");

    thrust::copy(d_out.begin(), d_out.end(), out.begin());
}

void snr_boxcar_3d_cuda(std::span<const float> arr,
                        SizeType nprofiles,
                        std::span<const SizeType> widths,
                        std::span<float> out,
                        float stdnoise,
                        int device_id) {
    cuda_utils::set_device(device_id);
    thrust::device_vector<float> d_arr(arr.begin(), arr.end());

    snr_boxcar_3d_cuda_d(
        cuda::std::span<const float>(thrust::raw_pointer_cast(d_arr.data()),
                                     d_arr.size()),
        nprofiles, widths, out, stdnoise, device_id);
}

} // namespace loki::detection