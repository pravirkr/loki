#pragma once

#include <random>
#include <span>
#include <vector>

#include <omp.h>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval = 299792458.0;

// Return the next power of two greater than or equal to n
SizeType next_power_of_two(SizeType n) noexcept;

// return max(x[i] - y[i])
float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size);

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

// return index of nearest value in sorted array
SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val);

std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

class ThreadSafeRNG {
private:
    std::vector<std::mt19937> m_rngs;

public:
    explicit ThreadSafeRNG(unsigned int base_seed = std::random_device{}()) {
        const int max_threads = omp_get_max_threads();
        if (max_threads <= 0) {
            throw std::runtime_error("OpenMP: Invalid thread count");
        }
        m_rngs.reserve(max_threads);
        std::seed_seq seed_seq{base_seed};
        std::vector<std::uint32_t> seeds(max_threads);
        seed_seq.generate(seeds.begin(), seeds.end());
        for (int i = 0; i < max_threads; ++i) {
            // Each thread gets a unique seed by offsetting the base seed.
            m_rngs.emplace_back(seeds[i]);
        }
    }
    ThreadSafeRNG(const ThreadSafeRNG&)            = delete;
    ThreadSafeRNG& operator=(const ThreadSafeRNG&) = delete;
    ThreadSafeRNG(ThreadSafeRNG&&)                 = delete;
    ThreadSafeRNG& operator=(ThreadSafeRNG&&)      = delete;
    ~ThreadSafeRNG()                               = default;

    unsigned int operator()() {
        const int tid = omp_get_thread_num();
        assert(tid < static_cast<int>(m_rngs.size()));
        return m_rngs[tid]();
    }

    std::mt19937& get_engine_for_current_thread() {
        const int tid = omp_get_thread_num();
        if (tid < 0 || tid >= static_cast<int>(m_rngs.size())) {
            throw std::out_of_range("Invalid OpenMP thread id");
        }
        return m_rngs[tid];
    }

    template <class Distribution>
    typename Distribution::result_type generate(Distribution& dist) noexcept {
        return dist(get_engine_for_current_thread());
    }

    // Fills a range with random numbers using a provided distribution.
    template <class Distribution, typename T>
    void generate_range(Distribution& dist, std::span<T> range) {
        std::generate(range.begin(), range.end(),
                      [&]() { return dist(get_engine_for_current_thread()); });
    }
};

} // namespace loki::utils
