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

# These internal tests use Thrust device vectors, so nvcc must compile them, with the
# library's nvcc options (linked after LOKI_COMPILE_OPTIONS so -Wno-pedantic overrides
# -Wpedantic for nvcc output).
if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/fft_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ep_memory_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ffa_fused_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/cub_scratch_cuda_t.cpp PROPERTIES LANGUAGE CUDA
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

  # EP search: config generation, a plan-only run that writes the plan cache, a
  # dry run that loads that cache, and two rejections (a missing file and an
  # RFI table, which the EP search does not accept).
  set(LOKI_EP_CLI_DIR ${CMAKE_CURRENT_BINARY_DIR}/loki_cli_ep)
  file(MAKE_DIRECTORY ${LOKI_EP_CLI_DIR})
  string(
    JOIN
    ""
    LOKI_EP_CLI_CONFIG_CONTENT
    "[input]\n"
    "nsamps = 32768\n"
    "tsamp = 0.000064\n"
    "[search]\n"
    "f_min = 140.0\n"
    "f_max = 142.0\n"
    "acc_min = -10.0\n"
    "acc_max = 10.0\n"
    "nbins = 32\n"
    "[ep]\n"
    "n_runs = 1\n"
    "prune_poly_order = 2\n"
    "[performance]\n"
    "nthreads = 1\n"
    "max_process_memory_gb = 4.0\n"
    "bseg_ffa = 8192\n"
    "[output]\n"
    "outdir = \"${LOKI_EP_CLI_DIR}/out\"\n"
    "prefix = \"cli\"\n"
    "plan_cache = \"${LOKI_EP_CLI_DIR}/out/cli_ep_plan.h5\"\n"
  )
  set(LOKI_EP_CLI_CONFIG ${LOKI_EP_CLI_DIR}/ep.toml)
  file(WRITE ${LOKI_EP_CLI_CONFIG} "${LOKI_EP_CLI_CONFIG_CONTENT}")
  file(WRITE ${LOKI_EP_CLI_DIR}/rfi.toml "${LOKI_EP_CLI_CONFIG_CONTENT}[rfi]\npulsar_mask = []\n")
  add_test(NAME loki_search_ep_generate_config
           COMMAND $<TARGET_FILE:loki_app> search ep --generate-config
                   ${LOKI_EP_CLI_DIR}/generated.toml
  )
  # The plan-only run writes the cache from scratch: a cache left by an earlier
  # build (another plan version) would be rejected, so remove it first.
  add_test(NAME loki_search_ep_clean_cache
           COMMAND ${CMAKE_COMMAND} -E rm -f ${LOKI_EP_CLI_DIR}/out/cli_ep_plan.h5
  )
  set_tests_properties(loki_search_ep_clean_cache PROPERTIES FIXTURES_SETUP loki_ep_cli_cache)
  add_test(NAME loki_search_ep_plan_only
           COMMAND $<TARGET_FILE:loki_app> search ep --config ${LOKI_EP_CLI_CONFIG} --plan-only
  )
  set_tests_properties(loki_search_ep_plan_only PROPERTIES FIXTURES_REQUIRED loki_ep_cli_cache)
  add_test(NAME loki_search_ep_dry_run
           COMMAND $<TARGET_FILE:loki_app> search ep --config ${LOKI_EP_CLI_CONFIG} --dry-run
  )
  # The dry run loads the plan cache that the plan-only run wrote.
  set_tests_properties(loki_search_ep_dry_run PROPERTIES DEPENDS loki_search_ep_plan_only)
  add_test(NAME loki_search_ep_bad_config
           COMMAND $<TARGET_FILE:loki_app> search ep --config
                   ${LOKI_EP_CLI_DIR}/missing.toml --plan-only
  )
  set_tests_properties(loki_search_ep_bad_config PROPERTIES WILL_FAIL TRUE)
  add_test(NAME loki_search_ep_rejects_rfi
           COMMAND $<TARGET_FILE:loki_app> search ep --config ${LOKI_EP_CLI_DIR}/rfi.toml
                   --plan-only
  )
  set_tests_properties(loki_search_ep_rejects_rfi PROPERTIES WILL_FAIL TRUE)
endif()
