#--------------------------------------------------
# FindMathDX_SM.cmake
# Select cuRANDDx SM template value (>= 700) from resolved CUDA architectures.
#--------------------------------------------------

# Prevent multiple inclusion
if(MathDX_SM_FIND_INCLUDED)
  return()
endif()
set(MathDX_SM_FIND_INCLUDED TRUE)

include(${CMAKE_CURRENT_LIST_DIR}/LokiCUDAArch.cmake)

# 1. User override.
if(MATHDX_SM)
  if(MATHDX_SM LESS 700)
    message(WARNING "MATHDX_SM=${MATHDX_SM} is below cuRANDDx minimum (700). Clamping to 700.")
    set(MATHDX_SM 700)
  endif()
else()
  # 2. Highest resolved architecture from loki_validate_cuda_architectures().
  if(DEFINED LOKI_MAX_CUDA_ARCH AND LOKI_MAX_CUDA_ARCH GREATER 0)
    if(LOKI_MAX_CUDA_ARCH LESS 70)
      set(MATHDX_SM 700)
    else()
      math(EXPR MATHDX_SM "${LOKI_MAX_CUDA_ARCH} * 10")
    endif()
  else()
    # 3. Fallback: first explicit LOKI_CUDA_ARCHITECTURES entry.
    if(DEFINED LOKI_CUDA_ARCHITECTURES
       AND NOT LOKI_CUDA_ARCHITECTURES STREQUAL "native"
       AND NOT LOKI_CUDA_ARCHITECTURES MATCHES "^(all-major|all)$"
    )
      string(REPLACE ";" "," _raw "${LOKI_CUDA_ARCHITECTURES}")
      string(REGEX MATCH "[0-9]+" _first "${_raw}")
      if(_first)
        if(_first LESS 70)
          set(MATHDX_SM 700)
        else()
          math(EXPR MATHDX_SM "${_first} * 10")
        endif()
      endif()
    endif()

    # 4. Query local GPU via nvidia-smi.
    if(NOT MATHDX_SM)
      _loki_query_native_gpu_arch(_native_arch)
      if(_native_arch)
        if(_native_arch LESS 70)
          set(MATHDX_SM 700)
        else()
          math(EXPR MATHDX_SM "${_native_arch} * 10")
        endif()
      endif()
    endif()
  endif()

  # 5. Final fallback when CUDA was explicitly requested.
  if(NOT MATHDX_SM)
    if(LOKI_CUDA STREQUAL "ON")
      message(
        FATAL_ERROR
          "LOKI_CUDA=ON but GPU compute capability could not be detected. "
          "Ensure nvidia-smi works, or set -DLOKI_CUDA_ARCHITECTURES=80 (or -DMATHDX_SM=800)."
      )
    else()
      set(MATHDX_SM "800")
      message(
        WARNING
          "MathDX: GPU architecture not detected (LOKI_CUDA=AUTO). Defaulting MATHDX_SM=${MATHDX_SM}."
      )
    endif()
  endif()
endif()

set(MATHDX_SM
    ${MATHDX_SM}
    CACHE STRING "MathDx SM architecture value for cuRANDDx (>= 700)"
)
set(MathDX_SM_FOUND TRUE)
