# Catch2 test framework (CPM).
CPMFindPackage(
  NAME Catch2
  VERSION 3.16.0
  GITHUB_REPOSITORY catchorg/Catch2
  OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
)
if(NOT Catch2_FOUND AND NOT TARGET Catch2::Catch2WithMain)
  message(
    FATAL_ERROR "Catch2 is required for LOKI_BUILD_TESTING=ON but was not found or downloaded."
  )
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(LOKI_TEST_TARGET ${LOKI_LIBRARY_NAME}_tests)
file(GLOB LOKI_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(LOKI_TEST_SOURCES STREQUAL "")
  message(FATAL_ERROR "No C++ test sources (*.cpp) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()

add_executable(${LOKI_TEST_TARGET} ${LOKI_TEST_SOURCES})
# fft_t.cpp uses Thrust device vectors. nvcc must compile that translation
# unit; a host-only compile fails to build loki_tests when CUDA is on.
if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_t.cpp PROPERTIES LANGUAGE CUDA
  )
endif()

# White-box tests may include private headers via loki::internal.
target_link_libraries(
  ${LOKI_TEST_TARGET} PRIVATE ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${PROJECT_NAME}::internal
                              ${LOKI_COMPILE_OPTIONS}
                              Catch2::Catch2WithMain
                              Boost::math
                              OpenMP::OpenMP_CXX
                              HighFive::HighFive
                              $<BUILD_INTERFACE:tomlplusplus>
)
# fft_t.cpp compiles as CUDA: give it the library's nvcc options. Linked after
# LOKI_COMPILE_OPTIONS so -Wno-pedantic overrides -Wpedantic for nvcc output.
if(CMAKE_CUDA_COMPILER AND TARGET ${PROJECT_NAME}_cuda_options)
  target_link_libraries(${LOKI_TEST_TARGET} PRIVATE ${PROJECT_NAME}_cuda_options)
endif()

catch_discover_tests(${LOKI_TEST_TARGET})

if(LOKI_BUILD_APPLICATIONS AND TARGET loki_app)
  target_compile_definitions(
    ${LOKI_TEST_TARGET} PRIVATE LOKI_APP_PATH="$<TARGET_FILE:loki_app>"
  )
  set(LOKI_FFA_CLI_CONFIG ${CMAKE_CURRENT_BINARY_DIR}/loki_cli_dry_run.toml)
  file(WRITE ${LOKI_FFA_CLI_CONFIG}
       "[search]\nf_min = 1.0\nf_max = 10.0\nnbins = 64\n[performance]\nnthreads = 1\nmax_process_memory_gb = 4.0\n"
  )
  add_test(
    NAME loki_search_ffa_dry_run
    COMMAND $<TARGET_FILE:loki_app> search ffa --config ${LOKI_FFA_CLI_CONFIG}
            --dry-run
  )
  add_test(
    NAME loki_search_ffa_bad_config
    COMMAND $<TARGET_FILE:loki_app> search ffa --config
            ${CMAKE_CURRENT_BINARY_DIR}/missing_ffa.toml --dry-run
  )
  set_tests_properties(loki_search_ffa_bad_config PROPERTIES WILL_FAIL TRUE)
endif()
