#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

/**
 * @brief Fold time series using brute-force method
 *
 */
class BruteFold {
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
     */
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

class BruteFoldComplex {
public:
    BruteFoldComplex(std::span<const double> freq_arr,
                     SizeType segment_len,
                     SizeType nbins,
                     SizeType nsamps,
                     double tsamp,
                     double t_ref = 0.0,
                     int nthreads = 1);
    ~BruteFoldComplex();
    BruteFoldComplex(BruteFoldComplex&&) noexcept;
    BruteFoldComplex& operator=(BruteFoldComplex&&) noexcept;
    BruteFoldComplex(const BruteFoldComplex&)            = delete;
    BruteFoldComplex& operator=(const BruteFoldComplex&) = delete;

    SizeType get_fold_size() const;
    /**
     * @brief Fold time series using brute-force method
     *
     * @param ts_e Time series signal
     * @param ts_v Time series variance
     * @param fold  Folded time series with shape [nsegments, nfreqs, 2,
     * nbins_f]
     */
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/* Convenience function to fold time series using brute-force method */
std::vector<float> compute_brute_fold(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const double> freq_arr,
                                      SizeType segment_len,
                                      SizeType nbins,
                                      double tsamp,
                                      double t_ref = 0.0,
                                      int nthreads = 1);
std::vector<ComplexType>
compute_brute_fold_complex(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           std::span<const double> freq_arr,
                           SizeType segment_len,
                           SizeType nbins,
                           double tsamp,
                           double t_ref = 0.0,
                           int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA
/**
 * @brief Fold time series using brute-force method
 *
 */
class BruteFoldCUDA {
public:
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
     */
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);
    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<float> fold,
                 cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

std::vector<float> compute_brute_fold_cuda(std::span<const float> ts_e,
                                           std::span<const float> ts_v,
                                           std::span<const double> freq_arr,
                                           SizeType segment_len,
                                           SizeType nbins,
                                           double tsamp,
                                           double t_ref  = 0.0,
                                           int device_id = 0);
#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms