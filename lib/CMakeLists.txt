# Layout (see docs/architecture.md):
#   lib/<domain>/*.cpp   facades and backend-neutral code, mirroring include/loki/<domain>/
#   lib/cpu/*_cpu.cpp    CPU engines and kernels
#   lib/cuda/*_cuda.cu   GPU engines and kernels, compiled only when a GPU backend is enabled
# Every *.cpp under lib/ is host code; *.cu files are only picked up from lib/cuda/.
file(GLOB_RECURSE LOKI_LIBRARY_CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(LOKI_LIBRARY_CXX_SOURCES STREQUAL "")
  message(FATAL_ERROR "No C++ sources (*.cpp) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()
set(LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CXX_SOURCES})

if(CMAKE_CUDA_COMPILER)
  file(
    GLOB
    LOKI_LIBRARY_CUDA_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu
  )
  if(LOKI_LIBRARY_CUDA_SOURCES STREQUAL "")
    message(
      FATAL_ERROR
        "LOKI_CUDA enabled but no CUDA sources (*.cu) found in ${CMAKE_CURRENT_SOURCE_DIR}/cuda."
    )
  endif()
  list(APPEND LOKI_LIBRARY_ALL_SOURCES ${LOKI_LIBRARY_CUDA_SOURCES})
  set_source_files_properties(${LOKI_LIBRARY_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(${LOKI_LIBRARY_NAME} ${LOKI_LIBRARY_ALL_SOURCES})
add_library(${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ALIAS ${LOKI_LIBRARY_NAME})

target_compile_features(${LOKI_LIBRARY_NAME} PUBLIC cxx_std_20)

# The backend macros are PRIVATE: they only select which engines the facades
# can build. Installed headers are identical in every build; callers ask
# loki::available_backends() at run time. LOKI_ENABLE_GPU marks "a GPU backend
# is compiled in"; LOKI_ENABLE_CUDA names which one.
set(LOKI_BACKEND_DEFINITIONS "")
if(CMAKE_CUDA_COMPILER)
  list(APPEND LOKI_BACKEND_DEFINITIONS LOKI_ENABLE_CUDA LOKI_ENABLE_GPU)
endif()
set(LOKI_BACKEND_DEFINITIONS ${LOKI_BACKEND_DEFINITIONS} PARENT_SCOPE)
target_compile_definitions(${LOKI_LIBRARY_NAME} PRIVATE ${LOKI_BACKEND_DEFINITIONS})

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

# CUDA headers are used by lib/cuda/ only; public headers never include them.
if(CMAKE_CUDA_COMPILER)
  set(LOKI_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    list(PREPEND LOKI_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(${LOKI_LIBRARY_NAME} PRIVATE ${LOKI_CUDA_INCLUDE_DIRS})
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
          $<BUILD_INTERFACE:psrio::psrio>
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
  )
  # Runtime libraries: PRIVATE for shared loki, link-only PUBLIC for static loki.
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${LOKI_LIBRARY_NAME} PRIVATE CUDA::cudart CUDA::cufft)
  else()
    target_link_libraries(
      ${LOKI_LIBRARY_NAME} PUBLIC $<LINK_ONLY:CUDA::cudart> $<LINK_ONLY:CUDA::cufft>
    )
  endif()
endif()

# White-box access for in-tree tests and benchmarks: private headers (lib/),
# the backend macros, and the CUDA headers. Never installed or exported.
add_library(${LOKI_LIBRARY_NAME}_internal INTERFACE)
add_library(${PROJECT_NAME}::internal ALIAS ${LOKI_LIBRARY_NAME}_internal)
target_include_directories(${LOKI_LIBRARY_NAME}_internal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(
  ${LOKI_LIBRARY_NAME}_internal INTERFACE ${LOKI_BACKEND_DEFINITIONS}
)
if(CMAKE_CUDA_COMPILER)
  target_include_directories(
    ${LOKI_LIBRARY_NAME}_internal SYSTEM INTERFACE ${LOKI_CUDA_INCLUDE_DIRS}
  )
endif()
