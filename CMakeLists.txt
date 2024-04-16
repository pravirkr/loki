cmake_minimum_required(VERSION 3.15)
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
option(ENABLE_FAST_MATH "Enable fast math flags" ON)
option(ENABLE_CLANG_TIDY "Run Clang Tidy to get static analysis" OFF)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -fopenmp")

if(ENABLE_FAST_MATH)
  set(FAST_MATH_FLAGS_SAFE
      "-fassociative-math -fno-rounding-math -fno-signed-zeros -fno-trapping-math"
  )
  set(FAST_MATH_FLAGS_UNSAFE "-ffinite-math-only -fno-math-errno")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} ${FAST_MATH_FLAGS_SAFE} ${FAST_MATH_FLAGS_UNSAFE}"
  )
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Initialize some default paths
include(GNUInstallDirs)
include(CTest)
include(cmake/CPM.cmake)

# Find dependencies
CPMFindPackage(
  NAME spdlog
  VERSION 1.10.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)
if(BUILD_TESTING)

endif()
find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
# Build the c++ library
add_subdirectory(lib)
# Build the c++ tests
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
# Build the python bindings
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()

# Install targets and configuration
install(
  TARGETS loki
  EXPORT loki_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT loki_config
  NAMESPACE loki::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loki
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
