#--------------------------------------------------
# FindCURANDDX_SM.cmake
# Auto-detect (or allow override of) the GPU compute capability
# for use with curanddx::SM<> in your project.
# Usage in your top-level CMakeLists.txt:
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#   find_package(CURANDDX_SM REQUIRED)
#   target_compile_definitions(your_target PRIVATE CURANDDX_SM=${CURANDDX_SM})
#--------------------------------------------------

# Prevent multiple inclusion
if(CURANDDX_SM_FIND_INCLUDED)
  return()
endif()
set(CURANDDX_SM_FIND_INCLUDED TRUE)

# 1) Use user-provided override if set
if(DEFINED CURANDDX_SM)
  set(_cc_val ${CURANDDX_SM})
else()
  # 2) Locate nvidia-smi
  find_program(_NVSMI_EXECUTABLE nvidia-smi)
  if(_NVSMI_EXECUTABLE)
    # 3) Query compute capability
    execute_process(
      COMMAND ${_NVSMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
      RESULT_VARIABLE _smi_status
      OUTPUT_VARIABLE _smi_output
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(_smi_status STREQUAL "0" AND _smi_output)
      # Handle multiple GPUs: take first line
      string(REPLACE "\n" ";" _smi_lines "${_smi_output}")
      list(GET _smi_lines 0 _smi_cc)
      # Expect format "8.6"
      if(_smi_cc MATCHES "^[0-9]+\\.[0-9]+$")
        string(REPLACE "." ";" _cc_parts ${_smi_cc})
        list(GET _cc_parts 0 _cc_major)
        list(GET _cc_parts 1 _cc_minor)
        # Compute integer, e.g. 8*100 + 6*10 = 860
        math(EXPR _cc_val "${_cc_major} * 100 + ${_cc_minor} * 10")
      endif()
    endif()
  endif()

  if(NOT _cc_val)
    message(WARNING "FindCURANDDX_SM: could not detect compute capability via nvidia-smi; "
                    "please set -DCURANDDX_SM=<e.g. 800> manually."
    )
    set(_cc_val "")
  endif()
endif()

# 4) Populate cache and find_package variables
if(_cc_val)
  set(CURANDDX_SM
      ${_cc_val}
      CACHE STRING "Compute capability for cuRANDDx SM<>, override with -DCURANDDX_SM="
  )
  set(CURANDDX_SM_FOUND TRUE)
else()
  set(CURANDDX_SM_FOUND FALSE)
endif()

# Export FIND result back to parent scope
if(CURANDDX_SM_FOUND)
  set(CURANDDX_SM_FOUND
      TRUE
      PARENT_SCOPE
  )
endif()
