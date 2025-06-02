# Define the library
file(GLOB LOKI_CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(LOKI_ALL_SOURCES ${LOKI_CXX_SOURCES})

# Add CUDA sources if available
if(CMAKE_CUDA_COMPILER)
  file(GLOB LOKI_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  # Set CUDA-specific compile options for these sources
  set_source_files_properties(${LOKI_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
  list(APPEND LOKI_ALL_SOURCES ${LOKI_CUDA_SOURCES})
endif()

add_library(${LOKI_LIBRARY_NAME} ${LOKI_ALL_SOURCES})
# Alias for easier usage downstream if needed
add_library(${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ALIAS ${LOKI_LIBRARY_NAME})

# Set include directories
target_include_directories(
  ${LOKI_LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# System dependencies
message(STATUS "Searching for required system dependencies...")
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)
find_package(FFTW REQUIRED)

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

# Boost (Using specific components, header-only for math)
find_package(Boost 1.87.0 QUIET COMPONENTS math)
if(NOT Boost_FOUND OR NOT Boost_math_FOUND)
  message(STATUS "Local Boost::math not found or too old, downloading via CPM...")
  CPMAddPackage(
    NAME Boost
    VERSION 1.87.0
    URL https://github.com/boostorg/math/archive/refs/tags/boost-1.87.0.tar.gz
    DOWNLOAD_ONLY YES
  )
  if(Boost_ADDED)
    add_library(Boost::math INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::math INTERFACE ${Boost_SOURCE_DIR}/include)
    set(Boost_FOUND TRUE)
  endif()
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

# Link libraries
target_link_libraries(
  ${LOKI_LIBRARY_NAME}
  PRIVATE ${LOKI_COMPILE_OPTIONS}
  PUBLIC OpenMP::OpenMP_CXX
         HDF5::HDF5
         ${FFTW_FLOAT_LIB}
         spdlog::spdlog_header_only
         xtensor
         Boost::math
         HighFive
         indicators::indicators
         coverage_config
)
