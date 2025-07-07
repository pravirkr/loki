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
CPMFindPackage(
  NAME spdlog
  VERSION 1.15.2
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)

# xsimd (SIMD library) - Header-only
CPMFindPackage(
  NAME xsimd
  VERSION 13.2.0
  URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz
)

# xtensor (Tensor library) - Header-only, depends on xsimd
CPMFindPackage(
  NAME xtensor
  VERSION 0.26.0
  URL https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.26.0.tar.gz
  OPTIONS "XTENSOR_USE_XSIMD ON" "XTENSOR_USE_OPENMP ON"
)

# Try to find header-only Boost for math(no COMPONENTS needed)
find_package(Boost 1.83.0 QUIET)
if(NOT Boost_FOUND)
  message(STATUS "Boost not found or too old, downloading Boost.Math via CPM...")
  CPMAddPackage(
    NAME BoostMath
    VERSION 1.83.0
    URL https://github.com/boostorg/math/archive/refs/tags/boost-1.83.0.tar.gz
    DOWNLOAD_ONLY YES
  )
  if(BoostMath_ADDED)
    add_library(Boost::math INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::math INTERFACE ${BoostMath_SOURCE_DIR}/include)
  endif()
else()
  # Use system headers
  message(STATUS "Using system Boost headers for math: ${Boost_INCLUDE_DIRS}")
  add_library(Boost::math INTERFACE IMPORTED GLOBAL)
  target_include_directories(Boost::math INTERFACE ${Boost_INCLUDE_DIRS})
endif()

# HighFive (HDF5 C++ interface) - Depends on HDF5 found earlier
CPMFindPackage(
  NAME HighFive
  VERSION 2.10.1
  URL https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.10.1.tar.gz
  OPTIONS "HIGHFIVE_FIND_HDF5 ON" "HIGHFIVE_USE_XTENSOR ON"
)

# indicators (Progress indicators) - Header-only (Use git main branch)
CPMFindPackage(
  NAME indicators
  VERSION 2.3
  GIT_TAG ac6c93ea2b1f97a220d10a0729a625b3f51e320b
  GITHUB_REPOSITORY p-ranav/indicators
  DOWNLOAD_ONLY YES
)
if(indicators_ADDED)
  add_library(indicators::indicators INTERFACE IMPORTED GLOBAL)
  target_include_directories(indicators::indicators INTERFACE ${indicators_SOURCE_DIR}/include)
endif()

# Thread pool
CPMAddPackage(
  NAME BS_thread_pool
  GITHUB_REPOSITORY bshoshany/thread-pool
  VERSION 5.0.0 EXCLUDE_FROM_ALL SYSTEM
)
if(BS_thread_pool_ADDED)
  add_library(BS_thread_pool INTERFACE IMPORTED GLOBAL)
  target_include_directories(BS_thread_pool INTERFACE ${BS_thread_pool_SOURCE_DIR}/include)
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
         xtensor
         Boost::math
         HighFive
         indicators::indicators
         BS_thread_pool
)

# Add CUDA options and libraries when available
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(
    ${LOKI_LIBRARY_NAME}
    PRIVATE ${LOKI_CUDA_OPTIONS}
    PUBLIC CUDA::cudart CUDA::cufft CUDA::curand
    PRIVATE CUDA::cuda_driver
  )
endif()
