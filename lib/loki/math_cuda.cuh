#pragma once

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <curand.h>

#include "loki/cuda_utils.cuh"

namespace loki::math {

class CuRandRNG {
private:
    curandGenerator_t m_gen;

public:
    /**
     * @brief Constructs the cuRAND generator.
     * @param seed The seed for the pseudo-random number generator.
     */
    CuRandRNG(const unsigned long long seed = 1234) {
        // Use XORWOW for top performance, as benchmarked.
        cuda_utils::check_curand_call(
            curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_XORWOW));
        cuda_utils::check_curand_call(
            curandSetPseudoRandomGeneratorSeed(m_gen, seed));
        // Set a default ordering for best statistical properties.
        cuda_utils::check_curand_call(
            curandSetGeneratorOrdering(m_gen, CURAND_ORDERING_PSEUDO_BEST));
    }

    /**
     * @brief Destroys the cuRAND generator, releasing its resources.
     */
    ~CuRandRNG() {
        if (m_gen) {
            cuda_utils::check_curand_call(curandDestroyGenerator(m_gen));
        }
    }

    // Deleted copy/move constructors to prevent accidental copies.
    CuRandRNG(const CuRandRNG&)            = delete;
    CuRandRNG& operator=(const CuRandRNG&) = delete;
    CuRandRNG(CuRandRNG&&)                 = delete;
    CuRandRNG& operator=(CuRandRNG&&)      = delete;

    /**
     * @brief Sets the generator's offset to jump to a specific point in the
     * sequence.
     * @param offset The number of random numbers to skip.
     */
    void set_offset(unsigned long long offset) {
        cuda_utils::check_curand_call(curandSetGeneratorOffset(m_gen, offset));
    }

    /**
     * @brief Associates the generator with a specific CUDA stream for
     * asynchronous execution.
     * @param stream The CUDA stream to use.
     */
    void set_stream(cudaStream_t stream) {
        cuda_utils::check_curand_call(curandSetStream(m_gen, stream));
    }

    /**
     * @brief Generates normally distributed random numbers into a device memory
     * span.
     * @param range A span pointing to the device memory to fill.
     * @param mean The mean of the normal distribution.
     * @param stddev The standard deviation of the normal distribution.
     */
    void
    generate_range(cuda::std::span<float> range, float mean, float stddev) {
        cuda_utils::check_curand_call(curandGenerateNormal(
            m_gen, range.data(), range.size(), mean, stddev));
    }
};

} // namespace loki::math