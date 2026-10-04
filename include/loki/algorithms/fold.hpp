#pragma once

#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

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
              int nthreads = 1);
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
                                         int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA

/**
 * @brief Brute-force folding algorithm for Pulsar Search on Device
 *
 * @tparam FoldTypeCUDA The type of fold to use (float for time domain,
 * ComplexType for Fourier domain) on Host and ComplexTypeCUDA for Fourier
 * domain on Device
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA> class BruteFoldCUDA {
public:
    using HostFoldType   = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = typename FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    BruteFoldCUDA(std::span<const double> freq_arr,
                  SizeType segment_len,
                  SizeType nbins,
                  SizeType nsamps,
                  double tsamp,
                  double t_ref  = 0.0,
                  int device_id = 0);
    ~BruteFoldCUDA();
    BruteFoldCUDA(BruteFoldCUDA&&) noexcept;
    BruteFoldCUDA& operator=(BruteFoldCUDA&&) noexcept;
    BruteFoldCUDA(const BruteFoldCUDA&)            = delete;
    BruteFoldCUDA& operator=(const BruteFoldCUDA&) = delete;

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
                 std::span<HostFoldType> fold);
    // Device interface
    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<DeviceFoldType> fold,
                 cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using BruteFoldFloatCUDA   = BruteFoldCUDA<float>;
using BruteFoldComplexCUDA = BruteFoldCUDA<ComplexTypeCUDA>;

/* Convenience function to fold time series using brute-force method on Device
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::vector<typename FoldTypeTraits<FoldTypeCUDA>::HostType>
compute_brute_fold_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<const double> freq_arr,
                        SizeType segment_len,
                        SizeType nbins,
                        double tsamp,
                        double t_ref  = 0.0,
                        int device_id = 0);
#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms