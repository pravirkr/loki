cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 OLD)

project(loki VERSION 0.0.1 LANGUAGES CXX)

# Configuration options
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTING "Build tests" OFF)
option(ENABLE_CLANG_TIDY "Run Clang Tidy to get static analysis" OFF)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Initialize some default paths
include(GNUInstallDirs)
include(CTest)
include(cmake/CPM.cmake)

# Find dependencies
CPMFindPackage(
  NAME spdlog
  VERSION 1.13.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)
if(BUILD_TESTING)
  CPMFindPackage(
    NAME Catch2
    VERSION 3.5.3
    GITHUB_REPOSITORY catchorg/Catch2 
  )
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()
find_package(FFTW REQUIRED)

# Build the c++ library
add_subdirectory(lib)

# Build the c++ tests
if(BUILD_TESTING)
  include(Catch)
  add_subdirectory(tests/cpp)
endif()

# Build the python bindings
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()

target_include_directories(loki PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(loki PRIVATE spdlog::spdlog_header_only ${FFTW_FLOAT_LIB})

# Add an alias target for use if this project is included as a subproject in another project
add_library(loki::loki ALIAS loki)

# Install targets and configuration
install(
  TARGETS loki
  EXPORT loki_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT loki_config NAMESPACE loki:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loki)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
