#pragma once

/**
 * @file preprocess_engine.hpp
 * @brief Backend entry points of loki::io::preprocess. Internal.
 */

#include <span>

#include "loki/io/preprocess.hpp"

namespace loki::io::detail {

/// CPU implementation, in lib/cpu/preprocess_cpu.cpp. Arguments are already
/// validated for size; @p raw may alias @p ts_e.
PreprocessReport preprocess_cpu(std::span<const float> raw,
                                double tsamp,
                                std::span<float> ts_e,
                                std::span<float> ts_v,
                                const PreprocessOptions& options,
                                int nthreads);

/// GPU implementation of the robust method, spectral zap included, in
/// lib/cuda/preprocess_cuda.cu (GPU builds only). The z-score method is
/// CPU-only; preprocess() runs it there.
PreprocessReport preprocess_gpu(std::span<const float> raw,
                                double tsamp,
                                std::span<float> ts_e,
                                std::span<float> ts_v,
                                const PreprocessOptions& options,
                                int device_id);

} // namespace loki::io::detail
