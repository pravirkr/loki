#=============================================================================
# LOKI — Dependency management (CPU + shared CPM deps)
#
# Design rules:
#   1. OpenMP and HDF5 are hard system dependencies — never fetched.
#   2. OpenMP and HDF5 are internal. For a SHARED loki they are linked PRIVATE;
#      for STATIC loki they are linked PUBLIC so consumers can resolve symbols.
#   3. fmt and spdlog are CPM-pinned and compiled once, into loki itself
#      (loki_fmt_spdlog objects), not header-only in every translation unit.
#      Loki's targets use the loki_fmt / loki_spdlog interfaces; consumers of
#      an installed loki need neither library.
#   4. CLI11, xsimd, Boost::math, BS_thread_pool, tomlplusplus, HighFive,
#      psrio and pocketfft are header-only internal dependencies ->
#      BUILD_INTERFACE only. pocketfft is included by lib/cpu/fft_cpu.cpp only.
#=============================================================================

# -----------------------------------------------------------------------
# 1. System dependencies — must be supplied by the user, never fetched
# -----------------------------------------------------------------------
message(STATUS "Searching for required system dependencies...")
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED) # For HighFive

# -----------------------------------------------------------------------
# 2. CPM-managed dependencies (pinned versions, bundled by default)
# -----------------------------------------------------------------------
if(LOKI_USE_SYSTEM_DEPS)
  message(STATUS "LOKI_USE_SYSTEM_DEPS=ON: compatible system packages may be used.")
else()
  message(STATUS "LOKI_USE_SYSTEM_DEPS=OFF: using pinned CPM dependencies.")
endif()



# fmt and spdlog — sources only. Header-only, every translation unit that
# includes them compiles their implementation again (about 1 s per unit with
# both); compiled once, they cost a few seconds per build. The objects go into
# loki (lib/CMakeLists.txt), so a static or shared loki carries the symbols.
# fmt — keep version in sync with spdlog's expected fmt release.
CPMAddPackage(
  NAME fmt
  VERSION 12.1.0
  URL https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz
  DOWNLOAD_ONLY YES
)
CPMAddPackage(
  NAME spdlog
  VERSION 1.17.0
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
  DOWNLOAD_ONLY YES
)

find_package(Threads REQUIRED)
add_library(loki_fmt INTERFACE)
add_library(loki_spdlog INTERFACE)
set(LOKI_FMT_SPDLOG_SOURCES "")
if(fmt_ADDED)
  target_include_directories(loki_fmt SYSTEM INTERFACE "${fmt_SOURCE_DIR}/include")
  list(APPEND LOKI_FMT_SPDLOG_SOURCES "${fmt_SOURCE_DIR}/src/format.cc"
       "${fmt_SOURCE_DIR}/src/os.cc"
  )
  # psrio links fmt::fmt-header-only when that target exists: give it this fmt,
  # so every translation unit uses the same (compiled) fmt.
  add_library(fmt::fmt-header-only ALIAS loki_fmt)
elseif(TARGET fmt::fmt-header-only)
  # LOKI_USE_SYSTEM_DEPS found an installed fmt: its header-only target.
  target_link_libraries(loki_fmt INTERFACE fmt::fmt-header-only)
else()
  message(FATAL_ERROR "fmt is not available after CPM setup.")
endif()
if(spdlog_ADDED)
  target_include_directories(loki_spdlog SYSTEM INTERFACE "${spdlog_SOURCE_DIR}/include")
  target_compile_definitions(loki_spdlog INTERFACE SPDLOG_COMPILED_LIB SPDLOG_FMT_EXTERNAL)
  target_link_libraries(loki_spdlog INTERFACE loki_fmt Threads::Threads)
  foreach(src IN ITEMS spdlog stdout_sinks color_sinks file_sinks async cfg)
    list(APPEND LOKI_FMT_SPDLOG_SOURCES "${spdlog_SOURCE_DIR}/src/${src}.cpp")
  endforeach()
elseif(TARGET spdlog::spdlog_header_only)
  target_link_libraries(loki_spdlog INTERFACE spdlog::spdlog_header_only)
else()
  message(FATAL_ERROR "spdlog is not available after CPM setup.")
endif()
if(LOKI_FMT_SPDLOG_SOURCES)
  add_library(loki_fmt_spdlog OBJECT ${LOKI_FMT_SPDLOG_SOURCES})
  target_link_libraries(loki_fmt_spdlog PRIVATE loki_fmt loki_spdlog)
  if(spdlog_ADDED)
    # As spdlog's own build: the unlocked fwrite where libc has it.
    include(CheckSymbolExists)
    check_symbol_exists(fwrite_unlocked "stdio.h" LOKI_HAVE_FWRITE_UNLOCKED)
    if(LOKI_HAVE_FWRITE_UNLOCKED)
      target_compile_definitions(loki_fmt_spdlog PRIVATE SPDLOG_FWRITE_UNLOCKED)
    endif()
  endif()
endif()

# psrio — header-only SIGPROC/PRESTO reader. After fmt: psrio links fmt::fmt-header-only
# and reuses loki's fmt (loki_fmt above). CPM adds the subdirectory so the
# psrio::psrio interface target (include path and C++20) is available.
CPMAddPackage(
  NAME psrio
  VERSION 0.4.0
  URL https://github.com/pravirkr/psrio/archive/refs/tags/v0.4.0.tar.gz
  OPTIONS "PSRIO_BUILD_TESTS OFF"
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(NOT TARGET psrio::psrio)
  message(FATAL_ERROR "psrio::psrio target not available after CPM setup.")
endif()

# -----------------------------------------------------------------------
# 3. Header-only CPM dependencies (build-time only)
# -----------------------------------------------------------------------

CPMAddPackage(
  NAME HighFive
  VERSION 3.3.0
  URL https://github.com/highfive-devs/highfive/archive/refs/tags/v3.3.0.tar.gz
  OPTIONS "HIGHFIVE_FIND_HDF5 ON"
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(NOT TARGET HighFive::HighFive)
  message(FATAL_ERROR "HighFive::HighFive target not available after CPM setup.")
endif()

# xsimd (SIMD library) - Header-only
CPMAddPackage(
  NAME xsimd
  VERSION 14.0.0
  URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/14.0.0.tar.gz
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(xsimd_ADDED)
  add_library(xsimd INTERFACE IMPORTED GLOBAL)
  target_include_directories(xsimd INTERFACE "${xsimd_SOURCE_DIR}/include")
elseif(NOT TARGET xsimd)
  message(FATAL_ERROR "xsimd is required but was not found or downloaded.")
endif()

# pocketfft (C++ branch, header-only, BSD-3-Clause): the host FFT for long
# transforms (lib/cpu/fft_cpu.cpp). Upstream is
# https://gitlab.mpcdf.mpg.de/mtr/pocketfft (branch cpp, commit c90e55b); the
# author's GitHub mirror serves the same commit as a plain archive. The branch
# has no releases, so VERSION is a placeholder: it must stay a plain number,
# which LOKI_USE_SYSTEM_DEPS=ON passes to find_package(pocketfft).
CPMAddPackage(
  NAME pocketfft
  VERSION 0.0.0
  URL https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz
  URL_HASH SHA256=2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(pocketfft_ADDED)
  add_library(pocketfft INTERFACE IMPORTED GLOBAL)
  target_include_directories(pocketfft SYSTEM INTERFACE "${pocketfft_SOURCE_DIR}")
  # No pocketfft thread pool (callers parallelise over rows with OpenMP); a
  # small plan cache for repeated long transforms.
  target_compile_definitions(
    pocketfft INTERFACE POCKETFFT_NO_MULTITHREADING POCKETFFT_CACHE_SIZE=16
  )
elseif(NOT TARGET pocketfft)
  message(FATAL_ERROR "pocketfft is required but was not found or downloaded.")
endif()

CPMAddPackage(
  NAME BS_thread_pool
  VERSION 5.0.0
  URL https://github.com/bshoshany/thread-pool/archive/refs/tags/v5.0.0.tar.gz
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(BS_thread_pool_ADDED)
  add_library(BS_thread_pool INTERFACE IMPORTED GLOBAL)
  target_include_directories(BS_thread_pool INTERFACE "${BS_thread_pool_SOURCE_DIR}/include")
elseif(NOT TARGET BS_thread_pool)
  message(FATAL_ERROR "BS_thread_pool is required but was not found or downloaded.")
endif()

CPMAddPackage(
  NAME tomlplusplus
  VERSION 3.4.0
  URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(tomlplusplus_ADDED)
  add_library(tomlplusplus INTERFACE IMPORTED GLOBAL)
  target_include_directories(tomlplusplus INTERFACE "${tomlplusplus_SOURCE_DIR}/include")
elseif(NOT TARGET tomlplusplus)
  message(FATAL_ERROR "tomlplusplus is required but was not found or downloaded.")
endif()

CPMAddPackage(
  NAME CLI11
  VERSION 2.7.0
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.7.0.tar.gz
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(CLI11_ADDED)
  add_library(CLI11::CLI11 INTERFACE IMPORTED GLOBAL)
  target_include_directories(CLI11::CLI11 INTERFACE "${CLI11_SOURCE_DIR}/include")
elseif(NOT TARGET CLI11::CLI11)
  message(FATAL_ERROR "CLI11 is required but was not found or downloaded.")
endif()

CPMAddPackage(
  NAME BoostMath
  VERSION 1.92.0
  URL https://github.com/boostorg/math/archive/refs/tags/boost-1.92.0.tar.gz
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)
if(BoostMath_ADDED)
  add_library(Boost::math INTERFACE IMPORTED GLOBAL)
  target_include_directories(Boost::math INTERFACE "${BoostMath_SOURCE_DIR}/include")
  target_compile_definitions(Boost::math INTERFACE BOOST_MATH_STANDALONE)
elseif(NOT TARGET Boost::math)
  message(FATAL_ERROR "Boost.Math is required but was not found or downloaded.")
endif()
