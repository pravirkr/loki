#pragma once

/**
 * @file nvtx.hpp
 * @brief NVTX ranges for profiling (nsys), behind the CMake option
 * LOKI_ENABLE_NVTX (compile definition LOKI_NVTX). Without it a Range is an
 * empty object and costs nothing. Host code only; works for every backend.
 *
 * @code
 * const nvtx::Range range("ffa sweep: chunk");
 * @endcode
 */

#ifdef LOKI_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace loki::nvtx {

/// Push a named range on construction and pop it on destruction. @p name
/// must outlive the range (a string literal).
class Range {
public:
#ifdef LOKI_NVTX
    explicit Range(const char* name) noexcept { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
#else
    explicit Range(const char* /*name*/) noexcept {}
    ~Range() = default;
#endif
    Range(const Range&)            = delete;
    Range& operator=(const Range&) = delete;
    Range(Range&&)                 = delete;
    Range& operator=(Range&&)      = delete;
};

} // namespace loki::nvtx
