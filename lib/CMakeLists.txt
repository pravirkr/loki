file(GLOB LOKI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(loki ${LOKI_SOURCES})
target_include_directories(loki PRIVATE ${FFTW_INCLUDE_DIRS})
target_include_directories(
  loki PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(loki PRIVATE spdlog::spdlog_header_only ${FFTW_FLOAT_LIB})
# Check if OpenMP was found
if(OpenMP_CXX_FOUND)
  target_link_libraries(loki PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(loki PRIVATE USE_OPENMP)
endif()
