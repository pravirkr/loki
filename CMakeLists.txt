cmake_minimum_required(VERSION 3.22)

project(
  loki
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "A high-performance C++ Pulsar searching library"
)

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

# Configuration options
option(${PROJECT_NAME}_BUILD_DOCS "Build documentation" OFF)
option(${PROJECT_NAME}_BUILD_PYTHON "Build Python bindings" ON)
option(${PROJECT_NAME}_BUILD_TESTING "Build tests" ON)
option(${PROJECT_NAME}_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(${PROJECT_NAME}_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(${PROJECT_NAME}_OPTIMIZE_NATIVE "Enable native CPU optimizations (-march=native)" ON)
option(${PROJECT_NAME}_FAST_MATH "Enable fast math optimizations (-ffast-math)" ON)
option(${PROJECT_NAME}_ENABLE_IPO "Enable Interprocedural Optimization (Link-Time Optimization)"
       OFF
)

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
if(${PROJECT_NAME}_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "Interprocedural Optimization (IPO/LTO) enabled.")
  else()
    message(
      WARNING "Interprocedural Optimization (IPO/LTO) requested but not supported by the toolchain."
    )
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options($<$<CONFIG:Debug>:-g>)
add_compile_options($<$<CONFIG:Release,RelWithDebInfo>:-O3;-DNDEBUG>) # Optimizations
add_compile_options($<$<CONFIG:MinSizeRel>:-Os;-DNDEBUG>) # Size optimizations
if(${PROJECT_NAME}_OPTIMIZE_NATIVE)
  add_compile_options($<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-march=native>)
  message(STATUS "Native CPU optimizations enabled (-march=native) for C++ Release builds.")
endif()
if(${PROJECT_NAME}_FAST_MATH)
  add_compile_options($<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>)
  message(STATUS "Fast math enabled (-ffast-math) for C++ Release builds.")
endif()

# Project structure and paths
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs) # Defines standard installation paths

# Find CUDA and set up the project
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0.0)
    message(FATAL_ERROR "CUDA 12.0 or higher is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
  endif()

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -c -Xcompiler=-Wall,-Wextra --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
  # Optional performance flags (CUDA)
  if(${PROJECT_NAME}_OPTIMIZE_NATIVE)
    # Add native arch for the host compiler part of CUDA compilation
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=-march=native")
    message(
      STATUS "Native CPU optimizations enabled (-Xcompiler=-march=native) for CUDA Release builds."
    )
  endif()
  if(${PROJECT_NAME}_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use-fast-math")
    message(STATUS "Fast math enabled (--use-fast-math) for CUDA Release builds.")
  endif()
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

else()
  message(STATUS "CUDA compiler not found or language check failed. GPU code will not be built.")
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

# Code coverage setup
if(${PROJECT_NAME}_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_library(coverage_config INTERFACE)
    target_compile_options(coverage_config INTERFACE --coverage)
    target_link_options(coverage_config INTERFACE --coverage)
    message(STATUS "Code coverage enabled.")
  else()
    message(
      WARNING "Code coverage requested, but not supported for compiler ${CMAKE_CXX_COMPILER_ID}."
    )
    set(${PROJECT_NAME}_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
    add_library(coverage_config INTERFACE)
  endif()
else()
  add_library(coverage_config INTERFACE)
endif()

# Add subdirectories
set(LOKI_LIBRARY_NAME ${PROJECT_NAME})
add_subdirectory(lib)
if(${PROJECT_NAME}_BUILD_PYTHON)
  add_subdirectory(src)
endif()
if(${PROJECT_NAME}_BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
if(LOKI_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Installation
install(
  TARGETS ${LOKI_LIBRARY_NAME} coverage_config
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Feature summary
include(FeatureSummary)
feature_summary(WHAT ALL)
