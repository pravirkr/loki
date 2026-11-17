#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/pipelines/ep_sweep_common.hpp"
#include "lib/search/cands.hpp"

// Shared by the seed-slice tests of both backends (ep_seed_slices_t.cpp and
// ep_seed_slices_cuda_t.cpp): a small EP search, and its final leaves and
// level stats in an order-free form. Defined in ep_seed_slices.cpp, which
// the host compiler builds.

namespace loki::test {

/// One final leaf: its parameter rows, scores and width.
struct SliceLeaf {
    std::vector<double> params;
    float score{};
    float score_ep{};
    std::uint32_t width{};
    /// Defined (defaulted) in ep_seed_slices.cpp, which the host compiler
    /// builds: no C++20 comparison in a header that nvcc reads.
    [[nodiscard]] bool operator==(const SliceLeaf& other) const;
};

/// One run of a result: leaves sorted (slices change their order), stats.
struct RunResults {
    std::string name;
    std::vector<SliceLeaf> leaves;
    std::vector<search::PruneStats> stats;
    SizeType n_harvested_total{0};
    /// The per-run reduction's counts, when the run was reduced:
    /// n_final_leaves, n_leaves_above_snr_min, n_cell_dropped, n_dominated,
    /// n_leaves_retained.
    std::vector<std::uint64_t> reduction;

    [[nodiscard]] SizeType n_leaves() const noexcept { return leaves.size(); }
    [[nodiscard]] SizeType total_overloads() const noexcept;
    [[nodiscard]] SizeType total_masked() const noexcept;
};

/// 2^18 samples in @p nsegments segments over 140-160 Hz, nbins 32, with
/// @p nparams parameters (2: acceleration, 3: jerk, 5: circular), wide
/// enough that the higher derivatives branch. Acceleration within
/// +-@p accel_max m/s^2 (the default gives one acceleration cell of seeds).
search::PulsarSearchConfig seed_slice_config(SizeType nparams,
                                             bool fourier,
                                             int nthreads,
                                             double accel_max   = 30.0,
                                             SizeType nsegments = 16);

/// Unit-variance noise and one bright pulsar near 143 Hz.
std::pair<std::vector<float>, std::vector<float>>
seed_slice_series(const search::PulsarSearchConfig& cfg);

/// A rising threshold per pruning level that keeps the pulsar and some
/// noise, and never fills the tree.
std::vector<float> seed_slice_thresholds(const search::PulsarSearchConfig& cfg);

/// Every run of the only `*_results.h5` file in @p dir (an EPMultiPass
/// output), in name order.
std::vector<RunResults> read_run_results(const std::filesystem::path& dir);

/// A run handed to the sweep's result sink.
RunResults to_run_results(const pipelines::detail::EPResultPayload& payload);

/// Checks that @p a and @p b hold the same runs with the same leaves and
/// level stats, exactly.
void check_same_runs(const std::vector<RunResults>& a,
                     const std::vector<RunResults>& b);

} // namespace loki::test
