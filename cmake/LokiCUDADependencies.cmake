# CUDA-only dependencies (cuRANDDx / MathDX). Included only when nvcc is active.

if(NOT LOKI_NEEDS_CURANDDX)
  message(
    STATUS "Skipping MathDX/cuRANDDx download (no sm_70+ targets or LOKI_FORCE_CURAND_RNG=ON)."
  )
  return()
endif()

# MathDX - NVIDIA MathDX library
set(MATHDX_VERSION "25.12.1")
set(MATHDX_SUBDIR "25.12") # Redist folder uses YY.MM versioning

find_package(mathdx QUIET COMPONENTS curanddx CONFIG)

# Download components via CPM if not found
if(NOT curanddx_FOUND)
  string(
    CONCAT MATHDX_REDIST_URL
    "https://developer.nvidia.com/downloads/compute/cuRANDDx/redist/cuRANDDx/cuda12/"
    "nvidia-mathdx-${MATHDX_VERSION}-cuda12.tar.gz"
  )
  CPMAddPackage(
    NAME mathdx
    VERSION ${MATHDX_VERSION}
    URL "${MATHDX_REDIST_URL}"
    DOWNLOAD_ONLY YES
  )
  if(mathdx_ADDED)
    set(MATHDX_INCLUDE_DIR "${mathdx_SOURCE_DIR}/nvidia/mathdx/${MATHDX_SUBDIR}/include")
    if(NOT TARGET mathdx::curanddx)
      add_library(mathdx::curanddx INTERFACE IMPORTED GLOBAL)
      target_include_directories(mathdx::curanddx INTERFACE "${MATHDX_INCLUDE_DIR}")
    endif()
  endif()
endif()

if(NOT TARGET mathdx::curanddx)
  if(LOKI_CUDA STREQUAL "ON")
    message(FATAL_ERROR "LOKI_CUDA=ON but mathdx::curanddx could not be found or downloaded. "
                        "Check network access and CUDA >= 12.2."
    )
  else()
    message(FATAL_ERROR "CUDA compiler was enabled but mathdx::curanddx is unavailable. "
                        "Set LOKI_CUDA=OFF for a CPU-only build."
    )
  endif()
endif()

# cuRANDDx's SM<> value follows each architecture slice of the build
# (lib/cuda/device_rng.cuh). LOKI_CURANDDX_SM pins one value for every slice.
set(LOKI_CURANDDX_SM
    ""
    CACHE STRING "Pin the cuRANDDx SM<> value (e.g. 890) for every slice; empty: per slice"
)
if(LOKI_CURANDDX_SM)
  if(LOKI_CURANDDX_SM LESS 700)
    message(FATAL_ERROR "LOKI_CURANDDX_SM=${LOKI_CURANDDX_SM}: cuRANDDx needs 700 or higher.")
  endif()
  target_compile_definitions(mathdx::curanddx INTERFACE LOKI_CURANDDX_SM=${LOKI_CURANDDX_SM})
  message(STATUS "cuRANDDx SM<> pinned to ${LOKI_CURANDDX_SM} (LOKI_CURANDDX_SM).")
endif()
