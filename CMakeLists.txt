cmake_minimum_required(VERSION 3.15..3.26)

# Set a name and a version number for your project:
project(
  loki
  VERSION 0.0.1
  LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)
include(cmake/CPM.cmake)

CPMAddPackage(
  NAME spdlog
  VERSION 1.13.0
  GIT_REPOSITORY https://github.com/gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
  )

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configuration options
option(BUILD_PYTHON "Enable building of Python bindings" ON)

# Build the c++ library
add_subdirectory(lib)

# Build the python bindings
if(BUILD_PYTHON)
  add_subdirectory(src)
endif()

target_link_libraries(loki PRIVATE spdlog::spdlog_header_only)

# Add an alias target for use if this project is included as a subproject in
# another project
add_library(loki::loki ALIAS loki)

# Install targets and configuration
install(
  TARGETS loki
  EXPORT loki_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  EXPORT loki_config
  NAMESPACE loki::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loki)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
