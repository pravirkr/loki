set(PYBIND11_FINDPYTHON ON)
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)
find_package(pybind11 CONFIG REQUIRED)

# Python extension module (libloki).
# loki is built as a static archive when LOKI_BUILD_PYTHON=ON; symbols are
# embedded in the extension module at link time.
set(LOKI_PYMODULE_NAME lib${LOKI_LIBRARY_NAME})
# One module for every backend: the backend is chosen at run time (backend=),
# so the bindings never include a GPU runtime header. Each submodule is bound
# in bindings/bind_<submodule>.cpp, using the public headers only.
file(GLOB LOKI_PYBIND_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp)
pybind11_add_module(${LOKI_PYMODULE_NAME} loki_python.cpp ${LOKI_PYBIND_SOURCES})
target_include_directories(${LOKI_PYMODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  ${LOKI_PYMODULE_NAME} PUBLIC ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
)
install(TARGETS ${LOKI_PYMODULE_NAME} DESTINATION ${LOKI_LIBRARY_NAME})
