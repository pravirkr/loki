# Define the library
file(GLOB LOKI_LIBRARY_CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CXX_SOURCES})

# Add CUDA sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB LOKI_LIBRARY_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  list(APPEND LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CUDA_SOURCES})

  # Explicitly set CUDA files to be compiled with CUDA
  set_source_files_properties(${LOKI_LIBRARY_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(${LOKI_LIBRARY_NAME} ${LOKI_LIBRARY_ALL_SOURCES})
add_library(${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ALIAS ${LOKI_LIBRARY_NAME})

# System dependencies
message(STATUS "Searching for required system dependencies...")
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB FLOAT_OPENMP_LIB)

# CPM Dependencies
message(STATUS "Searching for CPM dependencies (will download if not found locally)...")

# spdlog (Header-only logging library)
CPMAddPackage(
  NAME spdlog
  VERSION 1.15.3
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
  OPTIONS "SPDLOG_INSTALL ON"
)

# xsimd (SIMD library) - Header-only
CPMAddPackage(
  NAME xsimd
  VERSION 13.2.0
  URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz
  DOWNLOAD_ONLY YES
)
if(xsimd_ADDED)
  add_library(xsimd INTERFACE IMPORTED GLOBAL)
  target_include_directories(xsimd INTERFACE ${xsimd_SOURCE_DIR}/include)
endif()

# Try to find header-only Boost for math+random(no COMPONENTS needed)
find_package(Boost 1.83.0 QUIET)
if(NOT Boost_FOUND)
  message(STATUS "Boost not found or too old, downloading Boost.Math via CPM...")
  CPMAddPackage(
    NAME BoostMath
    VERSION 1.83.0
    URL https://github.com/boostorg/math/archive/refs/tags/boost-1.83.0.tar.gz
    DOWNLOAD_ONLY YES
  )
  CPMAddPackage(
    NAME BoostRandom
    VERSION 1.83.0
    URL https://github.com/boostorg/random/archive/refs/tags/boost-1.83.0.tar.gz
    DOWNLOAD_ONLY YES
  )
  if(BoostMath_ADDED)
    add_library(Boost::math INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::math INTERFACE ${BoostMath_SOURCE_DIR}/include)
  endif()
  if(BoostRandom_ADDED)
    add_library(Boost::random INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::random INTERFACE ${BoostRandom_SOURCE_DIR}/include)
  endif()
else()
  # Use system headers
  message(STATUS "Using system Boost headers for math+random: ${Boost_INCLUDE_DIRS}")
  add_library(Boost::math INTERFACE IMPORTED GLOBAL)
  target_include_directories(Boost::math INTERFACE ${Boost_INCLUDE_DIRS})
  add_library(Boost::random INTERFACE IMPORTED GLOBAL)
  target_include_directories(Boost::random INTERFACE ${Boost_INCLUDE_DIRS})
endif()

# HighFive (HDF5 C++ interface) - Depends on HDF5 found earlier
CPMAddPackage(
  NAME HighFive
  VERSION 3.0.0
  URL https://github.com/highfive-devs/highfive/archive/refs/tags/v3.0.0.tar.gz
  OPTIONS "HIGHFIVE_FIND_HDF5 ON"
)

# Thread pool
CPMAddPackage(
  NAME BS_thread_pool
  VERSION 5.0.0
  URL https://github.com/bshoshany/thread-pool/archive/refs/tags/v5.0.0.tar.gz
  DOWNLOAD_ONLY YES EXCLUDE_FROM_ALL SYSTEM
)
if(BS_thread_pool_ADDED)
  add_library(BS_thread_pool INTERFACE IMPORTED GLOBAL)
  target_include_directories(BS_thread_pool INTERFACE ${BS_thread_pool_SOURCE_DIR}/include)
endif()

# CLI11
CPMAddPackage(
  NAME CLI11
  VERSION 2.5.0
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.5.0.tar.gz
  DOWNLOAD_ONLY YES EXCLUDE_FROM_ALL SYSTEM
)
if(CLI11_ADDED)
  add_library(CLI11::CLI11 INTERFACE IMPORTED GLOBAL)
  target_include_directories(CLI11::CLI11 INTERFACE ${CLI11_SOURCE_DIR}/include)
endif()

# toml++
CPMAddPackage(
  NAME tomlplusplus
  VERSION 3.4.0
  URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
  DOWNLOAD_ONLY YES EXCLUDE_FROM_ALL SYSTEM
)
if(tomlplusplus_ADDED)
  add_library(tomlplusplus INTERFACE IMPORTED GLOBAL)
  target_include_directories(tomlplusplus INTERFACE ${tomlplusplus_SOURCE_DIR}/include)
endif()

# --- NVIDIA MathDX Configuration ---
if(CMAKE_CUDA_COMPILER)
  set(MATHDX_VERSION "25.12.1")
  set(MATHDX_SUBDIR "25.12") # Redist folder uses YY.MM versioning

  find_package(MathDX_SM REQUIRED)
  find_package(mathdx QUIET COMPONENTS curanddx CONFIG)

  # Download components via CPM if not found
  if(NOT curanddx_FOUND)
    CPMAddPackage(
      NAME mathdx
      VERSION ${MATHDX_VERSION}
      URL "https://developer.nvidia.com/downloads/compute/cuRANDDx/redist/cuRANDDx/cuda12/nvidia-mathdx-${MATHDX_VERSION}-cuda12.tar.gz"
      DOWNLOAD_ONLY YES
    )
    if(mathdx_ADDED)
      set(MATHDX_INCLUDE_DIR "${mathdx_SOURCE_DIR}/nvidia/mathdx/${MATHDX_SUBDIR}/include")

      if(NOT curanddx_FOUND)
        add_library(mathdx::curanddx INTERFACE IMPORTED GLOBAL)
        target_include_directories(mathdx::curanddx INTERFACE "${MATHDX_INCLUDE_DIR}")
        target_compile_definitions(mathdx::curanddx INTERFACE CURANDDX_SM=${MATHDX_SM})
      endif()
    endif()
  endif()
endif()

# Code coverage setup
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_library(coverage_config INTERFACE)
    target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(coverage_config INTERFACE --coverage)
    message(STATUS "Code coverage enabled.")
  else()
    message(
      WARNING "Code coverage requested, but not supported for compiler ${CMAKE_CXX_COMPILER_ID}."
    )
    set(ENABLE_COVERAGE
        OFF
        CACHE BOOL "" FORCE
    )
    add_library(coverage_config INTERFACE)
  endif()
else()
  add_library(coverage_config INTERFACE)
endif()

# Set include directories
target_include_directories(
  ${LOKI_LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add CUDA include directories when available
if(CMAKE_CUDA_COMPILER)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${LOKI_LIBRARY_NAME} PUBLIC "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${LOKI_LIBRARY_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(
  ${LOKI_LIBRARY_NAME}
  PRIVATE ${LOKI_COMPILE_OPTIONS} coverage_config
  PUBLIC OpenMP::OpenMP_CXX
         HDF5::HDF5
         FFTW::Float
         FFTW::FloatOpenMP
         spdlog::spdlog_header_only
         xsimd
         Boost::math
         Boost::random
         HighFive::HighFive
         BS_thread_pool
         tomlplusplus
         CLI11::CLI11
)

# Add CUDA options and libraries when available
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(
    ${LOKI_LIBRARY_NAME}
    PRIVATE ${LOKI_CUDA_OPTIONS}
    PUBLIC CUDA::cudart CUDA::cufft
    PRIVATE CUDA::cuda_driver mathdx::curanddx
  )
endif()
