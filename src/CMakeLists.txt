set(PYBIND11_FINDPYTHON ON)
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)
find_package(pybind11 CONFIG REQUIRED)

# CPU Python extension module (libloki).
# loki is built as a static archive when LOKI_BUILD_PYTHON=ON; symbols are
# embedded in the extension module at link time.
set(LOKI_PYMODULE_NAME lib${LOKI_LIBRARY_NAME})
pybind11_add_module(${LOKI_PYMODULE_NAME} loki_python.cpp)
target_include_directories(
  ${LOKI_PYMODULE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(
  ${LOKI_PYMODULE_NAME} PUBLIC ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
)
install(TARGETS ${LOKI_PYMODULE_NAME} DESTINATION ${LOKI_LIBRARY_NAME})

# GPU Python extension module (libculoki) — only when the CUDA toolchain is active.
if(CMAKE_CUDA_COMPILER)
  set(LOKI_PYGPUMODULE_NAME libcu${LOKI_LIBRARY_NAME})
  pybind11_add_module(${LOKI_PYGPUMODULE_NAME} loki_python_cuda.cpp)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    target_include_directories(${LOKI_PYGPUMODULE_NAME} PUBLIC "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  endif()
  target_include_directories(
    ${LOKI_PYGPUMODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
                                    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(
    ${LOKI_PYGPUMODULE_NAME}
    PUBLIC ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
    PRIVATE ${LOKI_CUDA_OPTIONS}
  )
  install(TARGETS ${LOKI_PYGPUMODULE_NAME} DESTINATION ${LOKI_LIBRARY_NAME})
elseif(LOKI_CUDA STREQUAL "ON")
  message(
    FATAL_ERROR
      "LOKI_CUDA=ON with LOKI_BUILD_PYTHON=ON requires the GPU Python module (libculoki), "
      "but CUDA was not configured successfully."
  )
endif()
