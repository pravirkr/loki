cmake_minimum_required(VERSION 3.18)

project(
  loki
  VERSION 0.0.3
  LANGUAGES CXX
  DESCRIPTION "A high-performance C++ Pulsar searching library"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler version requirements
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.2)
    message(
      FATAL_ERROR
        "GCC >= 13.2 required for complete C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(
      FATAL_ERROR
        "Clang >= 18.0 required for complete C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
else()
  message(
    WARNING
      "Compiler '${CMAKE_CXX_COMPILER_ID}' is not officially tested. Proceed at your own risk."
  )
endif()

# loki configuration options
option(LOKI_BUILD_DOCS "Build documentation" OFF)
option(LOKI_BUILD_PYTHON "Build Python bindings" ON)
option(LOKI_BUILD_TESTING "Build C++ tests" OFF)
option(LOKI_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(LOKI_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(LOKI_ENABLE_IPO "Enable interprocedural optimization (LTO) on Release builds" OFF)
option(LOKI_ENABLE_NATIVE_ARCH "Use -march=native in Release builds" ON)
option(LOKI_USE_SYSTEM_DEPS "Prefer compatible system packages over pinned bundled dependencies"
       OFF
)
# BUILD_SHARED_LIBS is a standard CMake option (ON = shared loki, OFF = static loki).
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Python extensions are installed as a single module file; embed loki statically
# instead of shipping a separate libloki.so plus loader search-path tweaks.
if(LOKI_BUILD_PYTHON AND BUILD_SHARED_LIBS)
  message(
    FATAL_ERROR
      "LOKI_BUILD_PYTHON=ON requires BUILD_SHARED_LIBS=OFF. "
      "Python wheels should statically link loki into libloki.cpython-*.so."
  )
endif()

# CUDA tristate: AUTO / ON / OFF
set(LOKI_CUDA
    "AUTO"
    CACHE STRING
          "CUDA build mode: AUTO (use GPU if toolchain found), ON (require GPU), OFF (CPU only)"
)
set_property(CACHE LOKI_CUDA PROPERTY STRINGS "AUTO" "ON" "OFF")
if(NOT LOKI_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "LOKI_CUDA must be AUTO, ON, or OFF. Got: '${LOKI_CUDA}'.")
endif()

# GPU architecture(s) for nvcc ("native" = detect from local GPU at build time).
set(LOKI_CUDA_ARCHITECTURES
    "native"
    CACHE STRING "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES (e.g. native, 80, 86;90)"
)

# Force stock cuRAND Philox device RNG on all GPU architectures (disables cuRANDDx).
set(LOKI_FORCE_CURAND_RNG
    OFF
    CACHE BOOL "Use stock cuRAND Philox device RNG on all GPUs instead of cuRANDDx on sm_70+"
)

# C++20 for the whole project.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release when no build type is specified (packaging-friendly).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build: Debug, Release, MinSizeRel, RelWithDebInfo." FORCE
  )
  set_property(
    CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

# Optional LTO for Release builds.
if(CMAKE_BUILD_TYPE MATCHES "Release" AND LOKI_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build.")
  else()
    message(FATAL_ERROR "LOKI_ENABLE_IPO=ON but IPO/LTO is not supported by this toolchain.")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)

# Internal compile-option interface (build tree only — not exported on install).
# Interface target: link any library/executable to this to inherit these flags.
# Usage: target_link_libraries(my_target PRIVATE ${LOKI_COMPILE_OPTIONS})
set(LOKI_COMPILE_OPTIONS ${PROJECT_NAME}_compile_options)
add_library(${LOKI_COMPILE_OPTIONS} INTERFACE)
target_compile_options(${LOKI_COMPILE_OPTIONS} INTERFACE -Wall -Wextra -Wpedantic)
target_compile_options(
  ${LOKI_COMPILE_OPTIONS}
  INTERFACE $<$<CONFIG:Debug>:-O0;-g> $<$<CONFIG:Release>:-O3;-DNDEBUG;-ffast-math>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${LOKI_ENABLE_NATIVE_ARCH}>>:-march=native>
            $<$<CONFIG:RelWithDebInfo>:-O2;-DNDEBUG;-g> $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
)

# CUDA toolchain detection (LOKI_CUDA user option; LOKI_ENABLE_CUDA is the C++ macro).
if(LOKI_CUDA STREQUAL "ON" OR LOKI_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  # nvcc usually lives outside the default PATH; seed CUDACXX before check_language.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(
      LOKI_NVCC_EXECUTABLE nvcc
      HINTS ENV
            CUDA_PATH
            ENV
            CUDA_HOME
            ENV
            CUDA_ROOT
            /usr/local/cuda
            /opt/cuda
      PATH_SUFFIXES bin
    )
    if(LOKI_NVCC_EXECUTABLE)
      set(ENV{CUDACXX} "${LOKI_NVCC_EXECUTABLE}")
    endif()
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  elseif(LOKI_CUDA STREQUAL "ON")
    message(
      FATAL_ERROR
        "LOKI_CUDA=ON but no CUDA compiler (nvcc) was found. Install CUDA >= 12.6 or use LOKI_CUDA=AUTO/OFF."
    )
  else()
    message(STATUS "CUDA compiler not found. Building CPU-only (LOKI_CUDA=AUTO).")
  endif()
elseif(LOKI_CUDA STREQUAL "OFF")
  message(STATUS "CUDA disabled by user (LOKI_CUDA=OFF). Building CPU-only.")
endif()

# When CUDA is explicitly required, verify a driver/GPU is present (not just nvcc).
if(CMAKE_CUDA_COMPILER AND LOKI_CUDA STREQUAL "ON")
  find_program(LOKI_NVIDIA_SMI nvidia-smi)
  if(NOT LOKI_NVIDIA_SMI)
    message(
      FATAL_ERROR
        "LOKI_CUDA=ON but nvidia-smi was not found. An NVIDIA driver and GPU are required."
    )
  endif()
  execute_process(
    COMMAND ${LOKI_NVIDIA_SMI} --query-gpu=name --format=csv,noheader
    OUTPUT_VARIABLE _loki_gpu_names
    RESULT_VARIABLE _loki_gpu_status
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  )
  if(NOT _loki_gpu_status EQUAL 0 OR _loki_gpu_names STREQUAL "")
    message(
      FATAL_ERROR "LOKI_CUDA=ON but nvidia-smi reports no usable GPU. Check driver installation."
    )
  endif()
  message(STATUS "LOKI_CUDA=ON: detected GPU(s): ${_loki_gpu_names}")
endif()

if(CMAKE_CUDA_COMPILER)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6.0)
    message(FATAL_ERROR "CUDA >= 12.6 required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")

  # Find CUDA Toolkit for proper include dirs and linking
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  set(LOKI_CUDA_OPTIONS ${PROJECT_NAME}_cuda_options)
  add_library(${LOKI_CUDA_OPTIONS} INTERFACE)
  target_compile_options(
    ${LOKI_CUDA_OPTIONS}
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
              -Wno-pedantic
              --expt-extended-lambda
              --expt-relaxed-constexpr
              -Xcompiler=-Wall,-Wextra
              $<$<CONFIG:Debug>:-G;-g;-O0>
              $<$<CONFIG:Release>:-O3;-use_fast_math;-DNDEBUG>
              $<$<CONFIG:RelWithDebInfo>:-O2;-g;-lineinfo;-DNDEBUG>
              $<$<CONFIG:MinSizeRel>:-O2;-DNDEBUG>
              >
  )
  if(LOKI_FORCE_CURAND_RNG)
    target_compile_definitions(${LOKI_CUDA_OPTIONS} INTERFACE LOKI_FORCE_CURAND_RNG)
  endif()

  include(${PROJECT_SOURCE_DIR}/cmake/LokiCUDAArch.cmake)
  loki_validate_cuda_architectures()

  set(CMAKE_CUDA_ARCHITECTURES "${LOKI_CUDA_ARCHITECTURES}")
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  # Disable response files for better IDE integration
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# CPM: bundled deps by default; LOKI_USE_SYSTEM_DEPS=ON prefers find_package first.
include(cmake/CPM.cmake)
set(CPM_USE_LOCAL_PACKAGES
    ${LOKI_USE_SYSTEM_DEPS}
    CACHE BOOL "Try to find packages locally before downloading" FORCE
)
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE
      $ENV{CPM_SOURCE_CACHE}
      CACHE PATH "Directory to cache downloaded CPM sources"
  )
endif()

# Target naming: library target is `loki`, exported alias is `loki::loki`.
set(LOKI_LIBRARY_NAME ${PROJECT_NAME})
set(LOKI_CMAKE_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

add_subdirectory(lib)
if(LOKI_BUILD_PYTHON)
  add_subdirectory(src)
endif()
if(LOKI_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
if(LOKI_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Install / export: find_package(loki) + target loki::loki
include(CMakePackageConfigHelpers)

set(LOKI_BUILT_WITH_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  set(LOKI_BUILT_WITH_CUDA ON)
endif()

set(LOKI_BUILT_AS_SHARED OFF)
if(BUILD_SHARED_LIBS)
  set(LOKI_BUILT_AS_SHARED ON)
endif()

install(
  TARGETS ${LOKI_LIBRARY_NAME}
  EXPORT lokiTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(CMAKE_CUDA_COMPILER)
  install(TARGETS ${LOKI_CUDA_OPTIONS} EXPORT lokiTargets)
endif()

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN ".DS_Store" EXCLUDE
)

install(
  EXPORT lokiTargets
  FILE lokiTargets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${LOKI_CMAKE_PACKAGE_DIR}
)

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/LokiCUDAConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lokiCUDAConfig.cmake @ONLY
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/LokiConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/lokiConfig.cmake
  INSTALL_DESTINATION ${LOKI_CMAKE_PACKAGE_DIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lokiConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/lokiConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/lokiConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/lokiCUDAConfig.cmake DESTINATION ${LOKI_CMAKE_PACKAGE_DIR}
)

include(FeatureSummary)
feature_summary(WHAT ALL)

if(CMAKE_CUDA_COMPILER)
  message(
    STATUS "loki build mode: CPU + CUDA (LOKI_CUDA=${LOKI_CUDA}, LOKI_ENABLE_CUDA macro defined)"
  )
else()
  message(STATUS "loki build mode: CPU-only (LOKI_CUDA=${LOKI_CUDA})")
endif()
