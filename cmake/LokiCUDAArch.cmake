#--------------------------------------------------
# LokiCUDAArch.cmake
# Resolve LOKI_CUDA_ARCHITECTURES for validation and cuRANDDx policy.
#--------------------------------------------------

if(LOKI_CUDA_ARCH_INCLUDED)
  return()
endif()
set(LOKI_CUDA_ARCH_INCLUDED TRUE)

# Minimum supported compute capability (Maxwell sm_50+).
set(LOKI_MIN_CUDA_ARCH 50)

#
# Parse a numeric CUDA architecture token (e.g. "80") into out_var.
#
function(_loki_arch_string_to_number arch_str out_var)
  string(STRIP "${arch_str}" stripped)
  if(stripped MATCHES "^[0-9]+$")
    set(${out_var}
        "${stripped}"
        PARENT_SCOPE
    )
  else()
    set(${out_var}
        ""
        PARENT_SCOPE
    )
  endif()
endfunction()

#
# Query the local GPU compute capability via nvidia-smi into out_var (numeric, e.g. 80).
#
function(_loki_query_native_gpu_arch out_var)
  set(arch_value "")
  find_program(loki_nvsmi_executable nvidia-smi)
  if(loki_nvsmi_executable)
    execute_process(
      COMMAND ${loki_nvsmi_executable} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE smi_output
      RESULT_VARIABLE smi_status
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(smi_status EQUAL 0 AND smi_output MATCHES "^([0-9]+)\\.([0-9]+)")
      math(EXPR arch_value "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
    endif()
  endif()
  set(${out_var}
      "${arch_value}"
      PARENT_SCOPE
  )
endfunction()

#
# Resolve LOKI_CUDA_ARCHITECTURES into a list of numeric arch values in out_list_var.
#
function(loki_resolve_cuda_arch_numbers out_list_var)
  set(resolved "")
  set(raw_archs "${LOKI_CUDA_ARCHITECTURES}")

  if(raw_archs STREQUAL "native")
    _loki_query_native_gpu_arch(native_arch)
    if(native_arch)
      list(APPEND resolved "${native_arch}")
    endif()
  elseif(raw_archs MATCHES "^(all-major|all)$")
    # Fat-binary presets include Volta+ slices; assume cuRANDDx may be used.
    list(APPEND resolved "70")
  else()
    string(REPLACE ";" "," raw_archs "${raw_archs}")
    string(REPLACE " " "," raw_archs "${raw_archs}")
    string(REPLACE "," ";" arch_list "${raw_archs}")
    foreach(entry IN LISTS arch_list)
      string(STRIP "${entry}" entry)
      if(entry STREQUAL "")
        continue()
      endif()
      _loki_arch_string_to_number("${entry}" arch_num)
      if(arch_num)
        list(APPEND resolved "${arch_num}")
      endif()
    endforeach()
  endif()

  set(${out_list_var}
      "${resolved}"
      PARENT_SCOPE
  )
endfunction()

#
# Validate LOKI_CUDA_ARCHITECTURES and set LOKI_NEEDS_CURANDDX / LOKI_MAX_CUDA_ARCH.
#
function(loki_validate_cuda_architectures)
  loki_resolve_cuda_arch_numbers(arch_numbers)

  if(arch_numbers STREQUAL "")
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

  set(min_arch 9999)
  set(max_arch 0)
  foreach(arch_entry IN LISTS arch_numbers)
    math(EXPR arch_int "${arch_entry}")
    if(arch_int LESS min_arch)
      set(min_arch "${arch_int}")
    endif()
    if(arch_int GREATER max_arch)
      set(max_arch "${arch_int}")
    endif()
    if(arch_int LESS LOKI_MIN_CUDA_ARCH)
      message(FATAL_ERROR "LOKI_CUDA_ARCHITECTURES includes sm_${arch_int}, but loki requires "
                          "sm_${LOKI_MIN_CUDA_ARCH} or higher."
      )
    endif()
  endforeach()

  if(max_arch LESS 70 OR LOKI_FORCE_CURAND_RNG)
    set(needs_curanddx FALSE)
    message(STATUS "Device RNG: stock cuRAND Philox (sm_${min_arch}-sm_${max_arch} targets; "
                   "cuRANDDx not required)."
    )
  else()
    set(needs_curanddx TRUE)
    message(STATUS "Device RNG: cuRANDDx on sm_70+, cuRAND Philox below "
                   "(targets sm_${min_arch}-sm_${max_arch})."
    )
  endif()

  set(LOKI_RESOLVED_CUDA_ARCHS
      "${arch_numbers}"
      CACHE INTERNAL "Resolved numeric CUDA arch list" FORCE
  )
  set(LOKI_NEEDS_CURANDDX
      "${needs_curanddx}"
      CACHE INTERNAL "Whether MathDX/cuRANDDx is required" FORCE
  )
  set(LOKI_MAX_CUDA_ARCH
      "${max_arch}"
      CACHE INTERNAL "Highest resolved CUDA arch number" FORCE
  )
endfunction()
