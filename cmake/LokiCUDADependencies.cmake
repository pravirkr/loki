# CUDA-only dependencies (cuRANDDx / MathDX). Included only when nvcc is active.

# MathDX - NVIDIA MathDX library
set(MATHDX_VERSION "25.12.1")
set(MATHDX_SUBDIR "25.12") # Redist folder uses YY.MM versioning

find_package(MathDX_SM REQUIRED)
find_package(mathdx QUIET COMPONENTS curanddx CONFIG)

# Download components via CPM if not found
if(NOT curanddx_FOUND)
  CPMAddPackage(
    NAME mathdx
    VERSION ${MATHDX_VERSION}
    URL "https://developer.nvidia.com/downloads/compute/cuRANDDx/redist/cuRANDDx/cuda12/nvidia-mathdx-${MATHDX_VERSION}-cuda12.tar.gz"
    DOWNLOAD_ONLY YES
  )
  if(mathdx_ADDED)
    set(MATHDX_INCLUDE_DIR "${mathdx_SOURCE_DIR}/nvidia/mathdx/${MATHDX_SUBDIR}/include")
    if(NOT TARGET mathdx::curanddx)
      add_library(mathdx::curanddx INTERFACE IMPORTED GLOBAL)
      target_include_directories(mathdx::curanddx INTERFACE "${MATHDX_INCLUDE_DIR}")
      target_compile_definitions(mathdx::curanddx INTERFACE CURANDDX_SM=${MATHDX_SM})
    endif()
  endif()
endif()

if(NOT TARGET mathdx::curanddx)
  if(LOKI_CUDA STREQUAL "ON")
    message(FATAL_ERROR "LOKI_CUDA=ON but mathdx::curanddx could not be found or downloaded. "
                        "Check network access and CUDA >= 12.6."
    )
  else()
    message(FATAL_ERROR "CUDA compiler was enabled but mathdx::curanddx is unavailable. "
                        "Set LOKI_CUDA=OFF for a CPU-only build."
    )
  endif()
endif()
