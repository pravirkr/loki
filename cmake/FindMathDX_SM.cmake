#--------------------------------------------------
# FindMathDX_SM.cmake
# Auto-detect GPU compute capability for mathdx libraries
#--------------------------------------------------

# Prevent multiple inclusion
if(MathDX_SM_FIND_INCLUDED)
  return()
endif()
set(MathDX_SM_FIND_INCLUDED TRUE)

# 1. Prioritize CMAKE_CUDA_ARCHITECTURES (if set by user or project)
if(NOT DEFINED MATHDX_SM AND DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Take the first architecture if multiple are specified (MathDX templates need a single SM)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _target_arch)
  # Remove 'sm_' or 'compute_' prefix if present
  string(REGEX MATCH "[0-9]+" _arch_digits "${_target_arch}")
  set(MATHDX_SM "${_arch_digits}0")
endif()

# 2. Fallback to nvidia-smi if still not found
if(NOT MATHDX_SM)
  find_program(_NVSMI_EXECUTABLE nvidia-smi)
  if(_NVSMI_EXECUTABLE)
    execute_process(
      COMMAND ${_NVSMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _smi_output
      RESULT_VARIABLE _smi_status
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(_smi_status EQUAL 0 AND _smi_output MATCHES "^([0-9]+)\\.([0-9]+)")
      math(EXPR MATHDX_SM "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2} * 10")
    endif()
  endif()
endif()

# 3. Final Fallback/Error
if(NOT MATHDX_SM)
  message(WARNING "MathDX_SM: Could not detect GPU architecture. Defaulting to 800. "
                  "Set -DCMAKE_CUDA_ARCHITECTURES=80 or -DMATHDX_SM=800 manually."
  )
  set(MATHDX_SM "800")
endif()

# Export variables for the components
set(MATHDX_SM
    ${MATHDX_SM}
    CACHE STRING "MathDx SM architecture value"
)
set(MathDX_SM_FOUND TRUE)
