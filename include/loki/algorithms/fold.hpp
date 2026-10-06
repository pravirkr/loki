#pragma once

#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms {

/**
 * @brief Brute-force folding algorithm for Pulsar Search
 *
 * @tparam FoldType The type of fold to use (float for time domain, ComplexType
 * for Fourier domain)
 */
template <SupportedFoldType FoldType> class BruteFold {
public:
    BruteFold(std::span<const double> freq_arr,
              SizeType segment_len,
              SizeType nbins,
              SizeType nsamps,
              double tsamp,
              double t_ref = 0.0,
              Exec exec    = {});

    ~BruteFold();
    BruteFold(BruteFold&&) noexcept;
    BruteFold& operator=(BruteFold&&) noexcept;
    BruteFold(const BruteFold&)            = delete;
    BruteFold& operator=(const BruteFold&) = delete;

    SizeType get_fold_size() const;
    /**
     * @brief Fold time series using brute-force method
     *
     * @param ts_e Time series signal
     * @param ts_v Time series variance
     * @param fold  Folded time series with shape [nsegments, nfreqs, 2, nbins]
     * (time domain) or [nsegments, nfreqs, 2, nbins_f] (Fourier domain)
     */
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold);

    /**
     * @brief Fold time series on device using brute-force method
     */
    void execute(DeviceSpan<const float> ts_e,
                 DeviceSpan<const float> ts_v,
                 DeviceSpan<FoldType> fold,
                 Stream stream = {});

    /**
     * @brief Brute fold fused with the first `nlevels` frequency-only FFA
     * merge levels (time domain only).
     *
     * Equivalent to execute() followed by `nlevels` frequency-only FFA merge
     * iterations (bit-exact), but each tile of 2^nlevels segments is folded
     * and merged in cache, so the intermediate levels never touch DRAM.
     *
     * @param ts_e Time series signal
     * @param ts_v Time series variance
     * @param fold_out FFA level-`nlevels` fold, shape
     * [nsegments >> nlevels, ncoords[nlevels], 2, nbins]
     * @param coords_levels coords_levels[j] are the level-j coordinates
     * (j = 1..nlevels; entry 0 is unused)
     * @param ncoords Profiles per segment at levels 0..nlevels
     * @param nlevels Number of merge levels to fuse (>= 1, with
     * nsegments divisible by 2^nlevels)
     */
    void execute_fused_freq(
        std::span<const float> ts_e,
        std::span<const float> ts_v,
        std::span<float> fold_out,
        std::span<const coord::FFACoordFreq* const> coords_levels,
        std::span<const SizeType> ncoords,
        SizeType nlevels)
        requires(std::is_same_v<FoldType, float>);

    /// Phase-run table used by the time-domain fold (empty for Fourier).
    [[nodiscard]] std::span<const coord::PhaseRun> runs() const
        requires(std::is_same_v<FoldType, float>);
    /// `nfreqs + 1` offsets into runs().
    [[nodiscard]] std::span<const SizeType> run_offsets() const
        requires(std::is_same_v<FoldType, float>);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using BruteFoldFloat   = BruteFold<float>;
using BruteFoldComplex = BruteFold<ComplexType>;

/* Convenience function to fold time series using brute-force method */
template <SupportedFoldType FoldType>
std::vector<FoldType> compute_brute_fold(std::span<const float> ts_e,
                                         std::span<const float> ts_v,
                                         std::span<const double> freq_arr,
                                         SizeType segment_len,
                                         SizeType nbins,
                                         double tsamp,
                                         double t_ref = 0.0,
                                         Exec exec    = {});

} // namespace loki::algorithms