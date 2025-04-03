set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(LOKI_PYMODULE_NAME lib${LOKI_LIBRARY_NAME})
pybind11_add_module(${LOKI_PYMODULE_NAME} loki_python.cpp)

# Include directories needed for the binding code
target_include_directories(
  ${LOKI_PYMODULE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
)

# Link against the main library
target_link_libraries(${LOKI_PYMODULE_NAME} PRIVATE ${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS})

# Install the Python module
install(TARGETS ${LOKI_PYMODULE_NAME} DESTINATION ${LOKI_LIBRARY_NAME})
