# Google Benchmark (CPM).
CPMAddPackage(
  NAME benchmark
  VERSION 1.9.2
  GITHUB_REPOSITORY google/benchmark
  OPTIONS "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_FORCE_WERROR OFF" "BENCHMARK_ENABLE_TESTING OFF"
)
if(NOT TARGET benchmark::benchmark_main)
  message(
    FATAL_ERROR
      "google/benchmark is required for LOKI_BUILD_BENCHMARKS=ON but was not found or downloaded."
  )
endif()

set(LOKI_BENCH_TARGET ${PROJECT_NAME}_benchmarks)
file(GLOB LOKI_BENCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(LOKI_BENCH_SOURCES STREQUAL "")
  message(FATAL_ERROR "No benchmark sources (*.cpp) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()

add_executable(${LOKI_BENCH_TARGET} ${LOKI_BENCH_SOURCES})
target_include_directories(
  ${LOKI_BENCH_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(
  ${LOKI_BENCH_TARGET} PRIVATE ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
                               benchmark::benchmark_main
)
