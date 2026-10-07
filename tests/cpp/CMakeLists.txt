# Catch2: prefer system >= 3.16 (Ubuntu 24.04 catch2 is 3.4.0 and breaks CTest on SKIP()).
# Pin 3.16.0 via CPM when no suitable system package exists.
find_package(Catch2 3.16 CONFIG QUIET)
if(Catch2_FOUND)
  message(STATUS "Using system Catch2 ${Catch2_VERSION}")
else()
  CPMFindPackage(
    NAME Catch2
    VERSION 3.16.0
    GITHUB_REPOSITORY catchorg/Catch2
    OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
  )
endif()
if(NOT TARGET Catch2::Catch2WithMain)
  message(
    FATAL_ERROR "Catch2 is required for LOKI_BUILD_TESTING=ON but was not found or downloaded."
  )
endif()

if(Catch2_DIR)
  list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()
if(Catch2_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()

include(CTest)
include(Catch)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Two test executables (docs/architecture.md, rule 6):
# - api/:      black-box tests. Link loki::loki only, exactly like a downstream
#              user, so a private "lib/..." include fails to compile.
# - internal/: white-box tests of private code the public API cannot reach.
#              Link loki::internal (lib/ include root, backend macros, CUDA
#              headers).
set(LOKI_API_TEST_TARGET ${LOKI_LIBRARY_NAME}_api_tests)
set(LOKI_INTERNAL_TEST_TARGET ${LOKI_LIBRARY_NAME}_internal_tests)

foreach(kind api internal)
  file(GLOB _sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${kind}/*.cpp")
  if(_sources STREQUAL "")
    message(FATAL_ERROR "No C++ test sources found in ${CMAKE_CURRENT_SOURCE_DIR}/${kind}.")
  endif()
  set(_target ${LOKI_LIBRARY_NAME}_${kind}_tests)
  add_executable(${_target} ${_sources})
  target_link_libraries(
    ${_target} PRIVATE ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
                       Catch2::Catch2WithMain OpenMP::OpenMP_CXX HighFive::HighFive
  )
  catch_discover_tests(${_target})
endforeach()

target_link_libraries(
  ${LOKI_INTERNAL_TEST_TARGET} PRIVATE ${PROJECT_NAME}::internal Boost::math
                                       $<BUILD_INTERFACE:tomlplusplus>
)

# internal/fft_t.cpp uses Thrust device vectors, so nvcc must compile it, with
# the library's nvcc options (linked after LOKI_COMPILE_OPTIONS so
# -Wno-pedantic overrides -Wpedantic for nvcc output).
if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/fft_t.cpp PROPERTIES LANGUAGE CUDA
  )
  if(TARGET ${PROJECT_NAME}_cuda_options)
    target_link_libraries(${LOKI_INTERNAL_TEST_TARGET} PRIVATE ${PROJECT_NAME}_cuda_options)
  endif()
endif()

if(LOKI_BUILD_APPLICATIONS AND TARGET loki_app)
  target_compile_definitions(
    ${LOKI_API_TEST_TARGET} PRIVATE LOKI_APP_PATH="$<TARGET_FILE:loki_app>"
  )
  set(LOKI_FFA_CLI_CONFIG ${CMAKE_CURRENT_BINARY_DIR}/loki_cli_dry_run.toml)
  string(
    JOIN
    ""
    LOKI_FFA_CLI_CONFIG_CONTENT
    "[search]\n"
    "f_min = 1.0\n"
    "f_max = 10.0\n"
    "nbins = 64\n"
    "[performance]\n"
    "nthreads = 1\n"
    "max_process_memory_gb = 4.0\n"
  )
  file(WRITE ${LOKI_FFA_CLI_CONFIG} "${LOKI_FFA_CLI_CONFIG_CONTENT}")
  add_test(NAME loki_search_ffa_dry_run COMMAND $<TARGET_FILE:loki_app> search ffa --config
                                                ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  add_test(NAME loki_search_ffa_bad_config
           COMMAND $<TARGET_FILE:loki_app> search ffa --config
                   ${CMAKE_CURRENT_BINARY_DIR}/missing_ffa.toml --dry-run
  )
  set_tests_properties(loki_search_ffa_bad_config PROPERTIES WILL_FAIL TRUE)
endif()
