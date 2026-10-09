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

} // namespace loki::io::detail
