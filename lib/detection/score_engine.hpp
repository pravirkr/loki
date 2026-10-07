#pragma once

/**
 * @file score_engine.hpp
 * @brief Backend entry points for the boxcar S/N functions in
 * loki/detection/score.hpp. Internal.
 */

#include <cstdint>
#include <span>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::detection::detail {

// *_cpu are defined in lib/cpu/score_cpu.cpp, *_gpu in lib/cuda/score_cuda.cu.
// The std::span *_gpu overloads stage host data through the device.

void snr_boxcar_2d_cpu(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       float stdnoise,
                       int nthreads);

void snr_boxcar_2d_max_cpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           float stdnoise,
                           int nthreads);

void snr_boxcar_3d_cpu(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       int nthreads);

void snr_boxcar_3d_max_cpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           int nthreads);

void snr_boxcar_2d_max_gpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           float stdnoise,
                           int device_id);

void snr_boxcar_3d_gpu(std::span<const float> folds,
                       std::span<const SizeType> widths,
                       std::span<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       int device_id);

void snr_boxcar_3d_max_gpu(std::span<const float> folds,
                           std::span<const SizeType> widths,
                           std::span<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           int device_id);

// Device-memory overloads run on `device_id`, the device all views agree on
// (`common_device`), or on the current device when `device_id < 0`.
void snr_boxcar_2d_max_gpu(DeviceSpan<const float> folds,
                           DeviceSpan<const uint32_t> widths,
                           DeviceSpan<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           float stdnoise,
                           Stream stream,
                           int device_id);

void snr_boxcar_3d_gpu(DeviceSpan<const float> folds,
                       DeviceSpan<const uint32_t> widths,
                       DeviceSpan<float> scores,
                       SizeType nprofiles,
                       SizeType nbins,
                       Stream stream,
                       int device_id);

void snr_boxcar_3d_max_gpu(DeviceSpan<const float> folds,
                           DeviceSpan<const uint32_t> widths,
                           DeviceSpan<float> scores,
                           SizeType nprofiles,
                           SizeType nbins,
                           Stream stream,
                           int device_id);

} // namespace loki::detection::detail
