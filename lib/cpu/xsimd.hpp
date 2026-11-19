#pragma once

/**
 * @file xsimd.hpp
 * @brief The one include of xsimd (CPU only). Internal.
 *
 * xsimd 14 compiles its constants (xsimd/arch/xsimd_constants.hpp) inside
 * `#pragma GCC push_options` / `optimize("signed-zeros")`, and both that
 * push and the closing `pop_options` sit under `#if defined(__FAST_MATH__)`.
 * GCC 16 undefines `__FAST_MATH__` inside the region, so the pop is skipped
 * and the rest of the translation unit compiles with -fsigned-zeros. GCC then
 * refuses to inline any function defined before the include (std::sqrt,
 * std::max, ...) into code after it: "optimization level attribute
 * mismatch". GCC 13 leaves the macro alone and pops.
 *
 * The region is still open exactly when `__FAST_MATH__` was defined before
 * the include and is not after it (a pop restores the macro), so close it
 * here then.
 */

#if defined(__FAST_MATH__) && defined(__GNUC__) && !defined(__clang__)
#define LOKI_XSIMD_FAST_MATH_BEFORE
#endif

#include <xsimd/xsimd.hpp>

#if defined(LOKI_XSIMD_FAST_MATH_BEFORE) && !defined(__FAST_MATH__)
#pragma GCC pop_options
#endif
#undef LOKI_XSIMD_FAST_MATH_BEFORE
