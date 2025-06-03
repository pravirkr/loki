cmake_minimum_required(VERSION 3.24)

project(
  loki
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "A high-performance C++ Pulsar searching library"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Modern CMake policies
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Compiler version requirements
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
    message(
      FATAL_ERROR
        "GCC 13.0 or higher is required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(
      FATAL_ERROR
        "Clang 18.0 or higher is required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
endif()

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_IPO "Enable Interprocedural Optimization (Link-Time Optimization)" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Required for shared libraries

# Set default build type
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

# Interprocedural Optimization (IPO/LTO)
if(CMAKE_BUILD_TYPE MATCHES "Release" AND ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build.")
  else()
    message(WARNING "IPO/LTO requested but not supported.")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# Project structure and paths
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs) # Defines standard installation paths

# Compiler flags (Interface target to hold global compile options)
set(LOKI_COMPILE_OPTIONS ${PROJECT_NAME}_COMPILE_OPTIONS)
add_library(${LOKI_COMPILE_OPTIONS} INTERFACE)
target_compile_options(${LOKI_COMPILE_OPTIONS} INTERFACE -Wall -Wextra -Wpedantic)
target_compile_options(
  ${LOKI_COMPILE_OPTIONS}
  INTERFACE $<$<CONFIG:Debug>:-g> $<$<CONFIG:Release>:-O3;-DNDEBUG>
            $<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>
            $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-march=native;-ffast-math>
)

# Find CUDA and set up the project
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6.0)
    message(FATAL_ERROR "CUDA 12.6 or higher is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")

  # Find CUDA Toolkit for proper include dirs and linking
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # Define a target for CUDA compilation options
  set(LOKI_CUDA_OPTIONS ${PROJECT_NAME}_CUDA_OPTIONS)
  add_library(${LOKI_CUDA_OPTIONS} INTERFACE)
  target_compile_options(
    ${LOKI_CUDA_OPTIONS}
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
              -Wno-pedantic
              --expt-extended-lambda
              --expt-relaxed-constexpr
              -Xcompiler=-Wall,-Wextra
              $<$<CONFIG:Debug>:-G;-g>
              $<$<CONFIG:Release>:-O3;-use_fast_math>
              >
  )
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  # Disable response files for better IDE integration
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
  add_compile_definitions(LOKI_ENABLE_CUDA)
else()
  message(STATUS "CUDA compiler not found. GPU code will not be built.")
  remove_definitions(LOKI_ENABLE_CUDA)
endif()

# CPM.cmake Setup
include(cmake/CPM.cmake)
set(CPM_USE_LOCAL_PACKAGES
    ON
    CACHE BOOL "Try to find packages locally before downloading" FORCE
)
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE
      $ENV{CPM_SOURCE_CACHE}
      CACHE PATH "Directory to cache downloaded CPM sources"
  )
endif()

# Define library names and Add subdirectories
set(LOKI_LIBRARY_NAME ${PROJECT_NAME})
set(LOKI_PYTHON_LIBRARY_NAME libloki)
add_subdirectory(lib)
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Install targets
install(
  TARGETS ${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS} coverage_config
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Also install CUDA options if available
if(CMAKE_CUDA_COMPILER)
  install(
    TARGETS ${LOKI_CUDA_OPTIONS}
    EXPORT ${PROJECT_NAME}Targets
  )
endif()

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install export file
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)
