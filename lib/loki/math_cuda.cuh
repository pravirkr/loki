#pragma once

#include <cstdint>
#include <cuda/std/cassert>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <curand.h>
#include <curanddx.hpp>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::math {

/**
 * @brief RAII wrapper for cuRAND host-side generator for bulk RNG operations.
 *
 * @note This is NOT suitable for per-thread RNG in kernels. Use DeviceRNG
 * instead.
 */
class CuRandRNG {
public:
    /**
     * @brief Constructs the cuRAND generator with optional stream.
     * @param seed The seed for the pseudo-random number generator.
     * @param stream Optional CUDA stream for asynchronous operations. If
     * nullptr, a new stream is created and owned by this instance.
     * @param rng_type The random number generator type (default: XORWOW for
     * performance).
     */
    explicit CuRandRNG(SizeType seed            = 1234,
                       cudaStream_t stream      = nullptr,
                       curandRngType_t rng_type = CURAND_RNG_PSEUDO_XORWOW)
        : m_stream(stream) {
        if (m_stream == nullptr) {
            cuda_utils::check_cuda_call(cudaStreamCreate(&m_stream));
            m_owns_stream = true;
        }
        cuda_utils::check_curand_call(curandCreateGenerator(&m_gen, rng_type));
        cuda_utils::check_curand_call(
            curandSetPseudoRandomGeneratorSeed(m_gen, seed));
        cuda_utils::check_curand_call(curandSetStream(m_gen, m_stream));
        // Set a default ordering for best statistical properties.
        cuda_utils::check_curand_call(
            curandSetGeneratorOrdering(m_gen, CURAND_ORDERING_PSEUDO_BEST));
    }

    /**
     * @brief Destroys the cuRAND generator, releasing its resources.
     */
    ~CuRandRNG() {
        if (m_gen != nullptr) {
            curandDestroyGenerator(m_gen);
        }
        if (m_owns_stream && m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
        }
    }

    CuRandRNG(const CuRandRNG&)            = delete;
    CuRandRNG& operator=(const CuRandRNG&) = delete;
    CuRandRNG(CuRandRNG&& other) noexcept
        : m_gen(other.m_gen),
          m_stream(other.m_stream),
          m_owns_stream(other.m_owns_stream) {
        other.m_gen         = nullptr;
        other.m_stream      = nullptr;
        other.m_owns_stream = false;
    }

    CuRandRNG& operator=(CuRandRNG&& other) noexcept {
        if (this != &other) {
            if (m_gen != nullptr) {
                curandDestroyGenerator(m_gen);
            }
            if (m_owns_stream && m_stream != nullptr) {
                cudaStreamDestroy(m_stream);
            }

            m_gen         = other.m_gen;
            m_stream      = other.m_stream;
            m_owns_stream = other.m_owns_stream;

            other.m_gen         = nullptr;
            other.m_stream      = nullptr;
            other.m_owns_stream = false;
        }
        return *this;
    }

    /**
     * @brief Sets the generator's offset for reproducibility.
     * @param offset The number of random numbers to skip (useful for parallel
     * streams).
     */
    void set_offset(SizeType offset) {
        cuda_utils::check_curand_call(curandSetGeneratorOffset(m_gen, offset));
    }

    /**
     * @brief Changes the associated CUDA stream
     * @param stream The new CUDA stream to use.
     */
    void set_stream(cudaStream_t stream) {
        assert(stream != nullptr && "Cannot set null stream");
        m_stream      = stream;
        m_owns_stream = false;
        cuda_utils::check_curand_call(curandSetStream(m_gen, stream));
    }

    /**
     * @brief Returns the associated CUDA stream.
     */
    [[nodiscard]] cudaStream_t get_stream() const noexcept { return m_stream; }

    /**
     * @brief Synchronizes the internal stream
     */
    void synchronize() const {
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream));
    }

    /**
     * @brief Generates uniformly distributed floats in [0, 1).
     * @param range Span pointing to device memory.
     */
    void generate_uniform(cuda::std::span<float> range) {
        cuda_utils::check_curand_call(
            curandGenerateUniform(m_gen, range.data(), range.size()));
    }

    /**
     * @brief Generates normally distributed floats.
     * @param range Span pointing to device memory (size must be even for
     * cuRAND).
     * @param mean The mean of the normal distribution.
     * @param stddev The standard deviation of the normal distribution.
     */
    void generate_normal(cuda::std::span<float> range,
                         float mean   = 0.0F,
                         float stddev = 1.0F) {
        assert((range.size() & 1U) == 0U &&
               "curandGenerateNormal requires even size");
        cuda_utils::check_curand_call(curandGenerateNormal(
            m_gen, range.data(), range.size(), mean, stddev));
    }

private:
    curandGenerator_t m_gen{nullptr};
    cudaStream_t m_stream{nullptr};
    bool m_owns_stream{false};
};

/**
 * @brief cuRANDDx generator configuration for high-performance per-thread RNG.
 *
 * @tparam Rounds Number of Philox rounds
 * @tparam SM Target SM architecture
 */
template <uint32_t Rounds = 10, uint32_t SM = CURANDDX_SM>
struct DeviceRNGConfig {
    using Generator = decltype(curanddx::Generator<curanddx::philox4_32>() +
                               curanddx::PhiloxRounds<Rounds>() +
                               curanddx::SM<SM>() + curanddx::Thread());

    // Aliases for common distributions
    using NormalFloat  = curanddx::normal<float, curanddx::box_muller>;
    using UniformFloat = curanddx::uniform<float>;
};

// Default configuration (Philox4_32_10)
using DefaultDeviceRNG = DeviceRNGConfig<10>;

} // namespace loki::math