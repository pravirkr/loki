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
# Clang 23 treats __COUNTER__ in benchmark 1.9.2 headers as an error under
# -pedantic-errors. The warning is unused on older Clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  foreach(bench_target benchmark benchmark_main)
    if(TARGET ${bench_target})
      target_compile_options(${bench_target} PRIVATE -Wno-c2y-extensions)
    endif()
  endforeach()
endif()

set(LOKI_BENCH_TARGET ${PROJECT_NAME}_benchmarks)
file(GLOB LOKI_BENCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(LOKI_BENCH_SOURCES STREQUAL "")
  message(FATAL_ERROR "No benchmark sources (*.cpp) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()

add_executable(${LOKI_BENCH_TARGET} ${LOKI_BENCH_SOURCES})
target_link_libraries(
  ${LOKI_BENCH_TARGET} PRIVATE ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${PROJECT_NAME}::internal
                               ${LOKI_COMPILE_OPTIONS}
                               benchmark::benchmark_main
)

if(NOT TARGET loki_app)
  message(
    FATAL_ERROR
      "loki_app is required for LOKI_BUILD_BENCHMARKS=ON. Enable LOKI_BUILD_APPLICATIONS."
  )
endif()
add_dependencies(${LOKI_BENCH_TARGET} loki_app)
target_compile_definitions(
  ${LOKI_BENCH_TARGET} PRIVATE LOKI_APP_PATH="$<TARGET_FILE:loki_app>"
)
# Keep this after LOKI_COMPILE_OPTIONS so -Wpedantic does not re-enable the warning.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_library(loki_bench_no_c2y INTERFACE)
  target_compile_options(loki_bench_no_c2y INTERFACE -Wno-c2y-extensions)
  target_link_libraries(${LOKI_BENCH_TARGET} PRIVATE loki_bench_no_c2y)
endif()
