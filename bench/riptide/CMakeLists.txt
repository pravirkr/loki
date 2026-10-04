cmake_minimum_required(VERSION 3.18)

# Standalone-capable so the bench can be configured without the rest of loki:
#   cmake -S bench/riptide -B build-riptide -DLOKI_RIPTIDE_DIR=/path/to/riptide
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(riptide_stage_bench LANGUAGES CXX)
endif()

if(NOT LOKI_RIPTIDE_DIR)
  message(FATAL_ERROR "Set -DLOKI_RIPTIDE_DIR to the riptide source tree.")
endif()
if(NOT EXISTS "${LOKI_RIPTIDE_DIR}/src/riptide/cpp/periodogram.hpp")
  message(
    FATAL_ERROR
      "LOKI_RIPTIDE_DIR='${LOKI_RIPTIDE_DIR}' does not look like a riptide checkout."
  )
endif()

add_executable(riptide_stage_bench riptide_stage_bench.cpp)
target_include_directories(
  riptide_stage_bench PRIVATE "${LOKI_RIPTIDE_DIR}/src/riptide/cpp"
)
target_compile_features(riptide_stage_bench PRIVATE cxx_std_17)
# Match riptide's own setup.py flags. Do not inherit loki's -Wpedantic /
# -pedantic-errors: riptide uses C99 VLAs.
target_compile_options(
  riptide_stage_bench
  PRIVATE -O3
          -DNDEBUG
          -march=native
          -fassociative-math
          -fno-math-errno
          -ffinite-math-only
          -fno-rounding-math
          -fno-signed-zeros
          -fno-trapping-math
          -Wno-vla
          -Wno-vla-extension
          -Wno-unused-parameter
          -Wno-error
)
