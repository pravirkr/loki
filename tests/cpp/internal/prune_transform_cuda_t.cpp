#ifdef LOKI_ENABLE_CUDA

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/dynamic_cuda.cuh"

using loki::Backend;
using loki::ParamLimit;
using loki::SizeType;
using loki::search::PulsarSearchConfig;

// The direct-write path transforms the passing leaves of a batch with no
// validation mask (an empty span: every leaf is valid). Every basis must read
// an empty mask as "all valid", or the kernel dereferences a null pointer.

namespace {

constexpr SizeType kLeaves = 300;

PulsarSearchConfig make_cfg(SizeType nparams) {
    std::vector<ParamLimit> limits(nparams,
                                   ParamLimit{.min = -1.0, .max = 1.0});
    limits.back() = ParamLimit{.min = 140.0, .max = 145.0};
    return {1U << 16U,
            64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*nsegments=*/8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 20U,
            /*prune_poly_order=*/nparams,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0,
            /*m_p_min=*/1.4,
            /*propagator_significance=*/2.0,
            /*validation_significance=*/5.0,
            /*use_conservative_tile=*/false,
            /*use_boxcar_kadane=*/false};
}

} // namespace

TEST_CASE("CUDA prune transform treats an empty mask as all valid",
          "[prune][cuda][transform]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard guard(0);
    const auto [basis, nparams] =
        GENERATE(std::pair<std::string, SizeType>{"taylor", 2},
                 std::pair<std::string, SizeType>{"taylor", 3},
                 std::pair<std::string, SizeType>{"taylor", 4},
                 std::pair<std::string, SizeType>{"taylor", 5},
                 std::pair<std::string, SizeType>{"chebyshev", 2},
                 std::pair<std::string, SizeType>{"chebyshev", 3},
                 std::pair<std::string, SizeType>{"chebyshev", 4});
    CAPTURE(basis, nparams);

    const std::vector<SizeType> grid(nparams, 4U);
    const std::vector<double> dparams(nparams, 0.1);
    const auto functs = loki::core::create_prune_dp_functs_cuda<float>(
        basis, grid, dparams, /*nseg_ffa=*/8, /*tseg_ffa=*/1.0,
        make_cfg(nparams), /*batch_size=*/kLeaves, /*branch_max=*/4,
        /*device_id=*/0);

    // (value, error) per parameter, then the two trailing slots.
    const SizeType stride = (nparams + 2) * 2;
    std::vector<double> leaves(kLeaves * stride);
    for (SizeType i = 0; i < kLeaves; ++i) {
        for (SizeType j = 0; j < stride; ++j) {
            const bool is_error = (j % 2) == 1;
            leaves[(i * stride) + j] =
                is_error ? 0.01 + (0.0001 * static_cast<double>(j))
                         : 0.1 + (0.001 * static_cast<double>((i + j) % 97));
        }
    }
    const std::pair<double, double> coord_next{2.0, 0.3};
    const std::pair<double, double> coord_cur{1.0, 0.2};
    const std::vector<std::uint8_t> ones(kLeaves, 1);
    const thrust::device_vector<std::uint8_t> mask_d(ones.begin(), ones.end());
    const auto stream = cudaStreamPerThread;

    const auto run = [&](bool with_mask) {
        thrust::device_vector<double> leaves_d(leaves.begin(), leaves.end());
        const cuda::std::span<const std::uint8_t> mask =
            with_mask ? cuda::std::span<const std::uint8_t>(
                            loki::cuda_utils::as_span(mask_d))
                      : cuda::std::span<const std::uint8_t>{};
        functs->transform(loki::cuda_utils::as_span(leaves_d), mask, coord_next,
                          coord_cur, kLeaves, stream);
        loki::cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                          "transform failed");
        std::vector<double> out(leaves.size());
        thrust::copy(leaves_d.begin(), leaves_d.end(), out.begin());
        return out;
    };
    const auto masked   = run(true);
    const auto unmasked = run(false);

    REQUIRE(masked.size() == unmasked.size());
    CHECK(std::memcmp(masked.data(), unmasked.data(),
                      masked.size() * sizeof(double)) == 0);
    // The transform moved the leaves: the comparison is not of two no-ops.
    CHECK(std::memcmp(masked.data(), leaves.data(),
                      leaves.size() * sizeof(double)) != 0);
}

#endif // LOKI_ENABLE_CUDA
