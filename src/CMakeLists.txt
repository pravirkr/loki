set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(PYMODULE_NAME libloki)
pybind11_add_module(${PYMODULE_NAME} loki_python.cpp)
target_include_directories(${PYMODULE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PYMODULE_NAME} PUBLIC ${LIBRARY_NAME})
install(TARGETS ${PYMODULE_NAME} DESTINATION ${LIBRARY_NAME})
