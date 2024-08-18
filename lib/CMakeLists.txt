# Find dependencies
CPMFindPackage(
  NAME spdlog
  VERSION 1.13.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)

CPMFindPackage(
  NAME Eigen3
  VERSION 3.4.0
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_ONLY YES
)

if(Eigen3_ADDED)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE ${Eigen3_SOURCE_DIR})
endif()

CPMFindPackage(
  NAME Boost
  VERSION 1.84
  URL https://github.com/boostorg/math/archive/refs/tags/boost-1.84.0.tar.gz
  DOWNLOAD_ONLY YES
)

if(Boost_ADDED)
  add_library(Boost::boost INTERFACE IMPORTED)
  target_include_directories(Boost::boost INTERFACE ${Boost_SOURCE_DIR}/include)
endif()

CPMFindPackage(
  NAME HighFive
  VERSION 2.10.0
  URL https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.10.0.tar.gz
  DOWNLOAD_ONLY YES
  OPTIONS "HIGHFIVE_FIND_HDF5 OFF"
)

if(HighFive_ADDED)
  add_library(HighFive INTERFACE IMPORTED)
  find_package(HDF5 REQUIRED)
  target_link_libraries(HighFive INTERFACE HDF5::HDF5)
  target_include_directories(HighFive INTERFACE ${HighFive_SOURCE_DIR}/include)
endif()

CPMFindPackage(
  NAME indicators
  VERSION 2.3
  URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
  DOWNLOAD_ONLY YES
)

if(indicators_ADDED)
  add_library(indicators::indicators INTERFACE IMPORTED)
  target_include_directories(indicators::indicators INTERFACE ${indicators_SOURCE_DIR}/include)
endif()

find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB LIBRARY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${LIBRARY_NAME} ${LIBRARY_SOURCES})

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW_INCLUDE_DIRS}
)

target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC coverage_config
         spdlog::spdlog_header_only
         Eigen3::Eigen
         Boost::boost
         HighFive
         indicators::indicators
         ${FFTW_FLOAT_LIB}
         OpenMP::OpenMP_CXX
)
