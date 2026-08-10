#--------------------------------------------------
# FindMathDX_SM.cmake
# Auto-detect GPU compute capability for mathdx libraries
#--------------------------------------------------

# Prevent multiple inclusion
if(MathDX_SM_FIND_INCLUDED)
  return()
endif()
set(MathDX_SM_FIND_INCLUDED TRUE)

# 1. User override or CMAKE_CUDA_ARCHITECTURES (first entry wins for MathDX).
# Take the first architecture if multiple are specified (MathDX templates need a single SM)
if(NOT MATHDX_SM
   AND DEFINED LOKI_CUDA_ARCHITECTURES
   AND NOT LOKI_CUDA_ARCHITECTURES STREQUAL "native"
)
  list(GET LOKI_CUDA_ARCHITECTURES 0 _target_arch)
  string(REGEX MATCH "[0-9]+" _arch_digits "${_target_arch}")
  if(_arch_digits)
    set(MATHDX_SM "${_arch_digits}0")
  endif()
elseif(
  NOT MATHDX_SM
  AND DEFINED CMAKE_CUDA_ARCHITECTURES
  AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _target_arch)
  string(REGEX MATCH "[0-9]+" _arch_digits "${_target_arch}")
  if(_arch_digits)
    set(MATHDX_SM "${_arch_digits}0")
  endif()
endif()

# 2. Query the local GPU via nvidia-smi.
if(NOT MATHDX_SM)
  find_program(_LOKI_NVSMI_EXECUTABLE nvidia-smi)
  if(_LOKI_NVSMI_EXECUTABLE)
    execute_process(
      COMMAND ${_LOKI_NVSMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _smi_output
      RESULT_VARIABLE _smi_status
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(_smi_status EQUAL 0 AND _smi_output MATCHES "^([0-9]+)\\.([0-9]+)")
      math(EXPR MATHDX_SM "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2} * 10")
    endif()
  endif()
endif()

# 3. Fail or fall back depending on whether CUDA was explicitly requested.
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

set(MATHDX_SM
    ${MATHDX_SM}
    CACHE STRING "MathDx SM architecture value for cuRANDDx"
)
set(MathDX_SM_FOUND TRUE)
