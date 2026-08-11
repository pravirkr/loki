#--------------------------------------------------
# LokiCUDAArch.cmake
# Resolve LOKI_CUDA_ARCHITECTURES for validation and cuRANDDx policy.
#--------------------------------------------------

if(LokiCUDAArch_INCLUDED)
  return()
endif()
set(LokiCUDAArch_INCLUDED TRUE)

# Minimum supported compute capability (Maxwell sm_50+).
set(LOKI_MIN_CUDA_ARCH 50)

function(_loki_arch_string_to_number arch_str out_var)
  string(STRIP "${arch_str}" _arch_str)
  if(_arch_str MATCHES "^[0-9]+$")
    set(${out_var}
        "${_arch_str}"
        PARENT_SCOPE
    )
  else()
    set(${out_var}
        ""
        PARENT_SCOPE
    )
  endif()
endfunction()

function(_loki_query_native_gpu_arch out_var)
  set(_arch "")
  find_program(_LOKI_NVSMI_EXECUTABLE nvidia-smi)
  if(_LOKI_NVSMI_EXECUTABLE)
    execute_process(
      COMMAND ${_LOKI_NVSMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _smi_output
      RESULT_VARIABLE _smi_status
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(_smi_status EQUAL 0 AND _smi_output MATCHES "^([0-9]+)\\.([0-9]+)")
      math(EXPR _arch "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
    endif()
  endif()
  set(${out_var}
      "${_arch}"
      PARENT_SCOPE
  )
endfunction()

function(loki_resolve_cuda_arch_numbers out_list_var)
  set(_resolved "")
  set(_raw "${LOKI_CUDA_ARCHITECTURES}")

  if(_raw STREQUAL "native")
    _loki_query_native_gpu_arch(_native_arch)
    if(_native_arch)
      list(APPEND _resolved "${_native_arch}")
    endif()
  elseif(_raw MATCHES "^(all-major|all)$")
    # Fat-binary presets include Volta+ slices; assume cuRANDDx may be used.
    list(APPEND _resolved "70")
  else()
    string(REPLACE ";" "," _raw "${_raw}")
    string(REPLACE " " "," _raw "${_raw}")
    string(REPLACE "," ";" _list "${_raw}")
    foreach(_entry IN LISTS _list)
      string(STRIP "${_entry}" _entry)
      if(_entry STREQUAL "")
        continue()
      endif()
      _loki_arch_string_to_number("${_entry}" _num)
      if(_num)
        list(APPEND _resolved "${_num}")
      endif()
    endforeach()
  endif()

  set(${out_list_var}
      "${_resolved}"
      PARENT_SCOPE
  )
endfunction()

function(loki_validate_cuda_architectures)
  loki_resolve_cuda_arch_numbers(_arch_numbers)

  if(_arch_numbers STREQUAL "")
    if(LOKI_CUDA_ARCHITECTURES STREQUAL "native")
      message(WARNING "Could not resolve native GPU compute capability (nvidia-smi unavailable). "
                      "Architecture policy checks skipped; cuRANDDx may be fetched conservatively."
      )
      set(LOKI_NEEDS_CURANDDX
          TRUE
          CACHE INTERNAL "Whether MathDX/cuRANDDx is required" FORCE
      )
      set(LOKI_MAX_CUDA_ARCH
          800
          CACHE INTERNAL "Highest resolved CUDA arch number" FORCE
      )
      return()
    endif()
    message(FATAL_ERROR "Could not parse LOKI_CUDA_ARCHITECTURES='${LOKI_CUDA_ARCHITECTURES}'. "
                        "Use native, all-major, or numeric values such as 61 or 61;80."
    )
  endif()

  set(_min_arch 9999)
  set(_max_arch 0)
  foreach(_arch IN LISTS _arch_numbers)
    math(EXPR _arch_int "${_arch}")
    if(_arch_int LESS _min_arch)
      set(_min_arch "${_arch_int}")
    endif()
    if(_arch_int GREATER _max_arch)
      set(_max_arch "${_arch_int}")
    endif()
    if(_arch_int LESS LOKI_MIN_CUDA_ARCH)
      message(
        FATAL_ERROR
          "LOKI_CUDA_ARCHITECTURES includes sm_${_arch_int}, but loki requires sm_${LOKI_MIN_CUDA_ARCH} or higher."
      )
    endif()
  endforeach()

  if(_max_arch LESS 70 OR LOKI_FORCE_CURAND_RNG)
    set(_needs_curanddx FALSE)
    message(
      STATUS
        "Device RNG: stock cuRAND Philox (sm_${_min_arch}-sm_${_max_arch} targets; cuRANDDx not required)."
    )
  else()
    set(_needs_curanddx TRUE)
    message(
      STATUS
        "Device RNG: cuRANDDx on sm_70+, cuRAND Philox below (targets sm_${_min_arch}-sm_${_max_arch})."
    )
  endif()

  set(LOKI_RESOLVED_CUDA_ARCHS
      "${_arch_numbers}"
      CACHE INTERNAL "Resolved numeric CUDA arch list" FORCE
  )
  set(LOKI_NEEDS_CURANDDX
      "${_needs_curanddx}"
      CACHE INTERNAL "Whether MathDX/cuRANDDx is required" FORCE
  )
  set(LOKI_MAX_CUDA_ARCH
      "${_max_arch}"
      CACHE INTERNAL "Highest resolved CUDA arch number" FORCE
  )
endfunction()
