cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0148 OLD)

project(
  loki
  VERSION 0.0.1
  LANGUAGES CXX
)

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CODE_COVERAGE "Enable coverage reporting" OFF)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -ffast-math -fno-relaxed-template-template-args")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Initialize some default paths
include(GNUInstallDirs)
include(cmake/CPM.cmake)

# Code Coverage Configuration
add_library(coverage_config INTERFACE)
if(CODE_COVERAGE)
  target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(coverage_config INTERFACE --coverage)
endif()

# Build the C++ library
set(LIBRARY_NAME loki)
add_subdirectory(lib)
# Build the Python bindings
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()
# Build tests
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
# Build benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# Install targets and configuration
install(
  TARGETS loki
  EXPORT loki_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)