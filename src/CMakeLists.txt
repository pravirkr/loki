find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(libloki loki_python.cpp)
target_link_libraries(libloki PUBLIC loki)
install(TARGETS libloki DESTINATION loki)
