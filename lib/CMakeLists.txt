# Collect CPU sources (*.cpp) via GLOB; CUDA *.cu files added when nvcc is available.
file(GLOB LOKI_LIBRARY_CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(LOKI_LIBRARY_CXX_SOURCES STREQUAL "")
  message(FATAL_ERROR "No C++ sources (*.cpp) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()
set(LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CXX_SOURCES})

if(CMAKE_CUDA_COMPILER)
  file(GLOB LOKI_LIBRARY_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  if(LOKI_LIBRARY_CUDA_SOURCES STREQUAL "")
    message(
      FATAL_ERROR
        "LOKI_CUDA enabled but no CUDA sources (*.cu) found in ${CMAKE_CURRENT_SOURCE_DIR}."
    )
  endif()
  list(APPEND LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CUDA_SOURCES})
  set_source_files_properties(${LOKI_LIBRARY_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(${LOKI_LIBRARY_NAME} ${LOKI_LIBRARY_ALL_SOURCES})
add_library(${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ALIAS ${LOKI_LIBRARY_NAME})

target_compile_features(${LOKI_LIBRARY_NAME} PUBLIC cxx_std_20)

# C++ code uses #ifdef LOKI_ENABLE_CUDA for optional GPU API (preprocessor macro, not a CMake option).
if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(${LOKI_LIBRARY_NAME} PUBLIC LOKI_ENABLE_CUDA)
endif()

# Dependencies
include(${PROJECT_SOURCE_DIR}/cmake/LokiDependencies.cmake)
if(CMAKE_CUDA_COMPILER)
  include(${PROJECT_SOURCE_DIR}/cmake/LokiCUDADependencies.cmake)
endif()

if(LOKI_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${LOKI_LIBRARY_NAME} PRIVATE -O0 -g --coverage)
    target_link_options(${LOKI_LIBRARY_NAME} PRIVATE --coverage)
    message(STATUS "Code coverage enabled.")
  else()
    message(
      FATAL_ERROR "LOKI_ENABLE_COVERAGE=ON is not supported for compiler ${CMAKE_CXX_COMPILER_ID}."
    )
  endif()
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

# Header-only internal deps: BUILD_INTERFACE only (never exported to consumers).
target_link_libraries(
  ${LOKI_LIBRARY_NAME}
  PRIVATE $<BUILD_INTERFACE:${LOKI_COMPILE_OPTIONS}>
          $<BUILD_INTERFACE:CLI11::CLI11>
          $<BUILD_INTERFACE:xsimd>
          $<BUILD_INTERFACE:Boost::math>
          $<BUILD_INTERFACE:BS_thread_pool>
          $<BUILD_INTERFACE:tomlplusplus>
          $<BUILD_INTERFACE:HighFive::HighFive>
          $<BUILD_INTERFACE:fmt::fmt-header-only>
          $<BUILD_INTERFACE:spdlog::spdlog_header_only>
)

# System/runtime deps: PRIVATE for shared loki, PUBLIC for static loki.
if(BUILD_SHARED_LIBS)
  target_link_libraries(${LOKI_LIBRARY_NAME} PRIVATE OpenMP::OpenMP_CXX HDF5::HDF5 FFTW::Float)
else()
  target_link_libraries(
    ${LOKI_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX HDF5::HDF5 $<LINK_ONLY:FFTW::Float>
  )
endif()

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(
    ${LOKI_LIBRARY_NAME}
    PRIVATE ${LOKI_CUDA_OPTIONS} CUDA::cuda_driver CUDA::curand
            $<$<BOOL:${LOKI_NEEDS_CURANDDX}>:mathdx::curanddx>
    PUBLIC CUDA::cudart CUDA::cufft
  )
endif()
