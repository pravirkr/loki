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
    ${_target} PRIVATE loki_fmt loki_spdlog ${PROJECT_NAME}::${LOKI_LIBRARY_NAME}
                       ${LOKI_COMPILE_OPTIONS} Catch2::Catch2WithMain OpenMP::OpenMP_CXX
                       HighFive::HighFive
  )
  # Catch2 tags become CTest labels: [slow] marks the tests the sanitizer preset
  # skips (ctest -LE slow). Two tags run alone (RUN_SERIAL), whatever ctest -j:
  # [bigmem] tests peak at several GB (a CI runner has 16 GB), and the
  # [ep_memory] tests measure free device memory, which other GPU tests would
  # allocate meanwhile.
  catch_discover_tests(${_target} TEST_SPEC "~[bigmem]~[ep_memory]" ADD_TAGS_AS_LABELS)
  catch_discover_tests(
    ${_target} TEST_SPEC "[bigmem],[ep_memory]" ADD_TAGS_AS_LABELS PROPERTIES RUN_SERIAL TRUE
  )
endforeach()

target_link_libraries(
  ${LOKI_INTERNAL_TEST_TARGET}
  PRIVATE ${PROJECT_NAME}::internal Boost::math $<BUILD_INTERFACE:tomlplusplus>
          $<BUILD_INTERFACE:xsimd> pocketfft
)

# These internal tests use Thrust device vectors, so nvcc must compile them, with the
# library's nvcc options (linked after LOKI_COMPILE_OPTIONS so -Wno-pedantic overrides
# -Wpedantic for nvcc output).
if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/fft_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ep_memory_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ep_score_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ffa_fused_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ffa_final_score_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/kadane_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/ffa_resolve_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/fourier_phasor_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/cub_scratch_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/world_tree_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/branch_capacity_cuda_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/circular_mask_cuda_t.cpp PROPERTIES LANGUAGE CUDA
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

  # simulate -c: a missing file or an unknown key is reported, not a crash.
  add_test(NAME loki_simulate_bad_config
           COMMAND $<TARGET_FILE:loki_app> simulate --config
                   ${CMAKE_CURRENT_BINARY_DIR}/missing_simulate.toml
  )
  set_tests_properties(loki_simulate_bad_config PROPERTIES WILL_FAIL TRUE)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/loki_simulate_typo.toml "[signal]\nperoid = 0.5\n")
  add_test(NAME loki_simulate_unknown_key
           COMMAND $<TARGET_FILE:loki_app> simulate --config
                   ${CMAKE_CURRENT_BINARY_DIR}/loki_simulate_typo.toml
  )
  set_tests_properties(
    loki_simulate_unknown_key PROPERTIES PASS_REGULAR_EXPRESSION "signal\\.peroid: unknown key"
  )

  # Global output flags: --version, -q (no info lines on the console), a log
  # file that keeps the info lines -qq hides on the console, and -q with -v
  # rejected.
  add_test(NAME loki_cli_version COMMAND $<TARGET_FILE:loki_app> --version)
  set(LOKI_CLI_VERSION_RE "^loki [0-9]+\\.[0-9]+\\.[0-9]+ \\(backends: cpu")
  set_tests_properties(loki_cli_version PROPERTIES PASS_REGULAR_EXPRESSION ${LOKI_CLI_VERSION_RE})
  add_test(NAME loki_cli_quiet COMMAND $<TARGET_FILE:loki_app> search ffa --config
                                       ${LOKI_FFA_CLI_CONFIG} --dry-run -q
  )
  set_tests_properties(loki_cli_quiet PROPERTIES FAIL_REGULAR_EXPRESSION "\\[info\\]")
  set(LOKI_CLI_LOG_FILE ${CMAKE_CURRENT_BINARY_DIR}/loki_cli_log_file.log)
  add_test(NAME loki_cli_log_file_clean COMMAND ${CMAKE_COMMAND} -E rm -f ${LOKI_CLI_LOG_FILE})
  set_tests_properties(loki_cli_log_file_clean PROPERTIES FIXTURES_SETUP loki_cli_log_file)
  add_test(NAME loki_cli_log_file
           COMMAND $<TARGET_FILE:loki_app> -qq --log-file ${LOKI_CLI_LOG_FILE} search
                   ffa --config ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  set_tests_properties(
    loki_cli_log_file PROPERTIES FIXTURES_REQUIRED loki_cli_log_file FIXTURES_SETUP
                                 loki_cli_log_file_written FAIL_REGULAR_EXPRESSION "\\[info\\]"
  )
  add_test(NAME loki_cli_log_file_content COMMAND ${CMAKE_COMMAND} -E cat ${LOKI_CLI_LOG_FILE})
  set_tests_properties(
    loki_cli_log_file_content PROPERTIES FIXTURES_REQUIRED loki_cli_log_file_written
                                         PASS_REGULAR_EXPRESSION "\\[info\\] Dry run complete"
  )
  # Colour: none when the output is not a terminal (ctest captures it, as a
  # batch job's log file would); CLICOLOR_FORCE forces it, NO_COLOR wins.
  string(ASCII 27 LOKI_ESC)
  add_test(NAME loki_cli_plain_when_redirected
           COMMAND $<TARGET_FILE:loki_app> search ffa --config ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  set_tests_properties(
    loki_cli_plain_when_redirected PROPERTIES FAIL_REGULAR_EXPRESSION ${LOKI_ESC}
  )
  add_test(NAME loki_cli_color_always COMMAND $<TARGET_FILE:loki_app> search ffa --config
                                              ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  set_tests_properties(
    loki_cli_color_always PROPERTIES ENVIRONMENT "CLICOLOR_FORCE=1" PASS_REGULAR_EXPRESSION
                                     "${LOKI_ESC}\\[[0-9;]*mFFA plan:"
  )
  add_test(NAME loki_cli_no_color_wins COMMAND $<TARGET_FILE:loki_app> search ffa --config
                                               ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  set_tests_properties(
    loki_cli_no_color_wins PROPERTIES ENVIRONMENT "CLICOLOR_FORCE=1;NO_COLOR=1"
                                      FAIL_REGULAR_EXPRESSION ${LOKI_ESC}
  )
  # --help: option groups as titled lists, plain when redirected, coloured
  # when forced.
  add_test(NAME loki_cli_help_plain COMMAND $<TARGET_FILE:loki_app> --help)
  set_tests_properties(
    loki_cli_help_plain PROPERTIES PASS_REGULAR_EXPRESSION "\nOutput & Logging:\n  -v,"
                                   FAIL_REGULAR_EXPRESSION "${LOKI_ESC}|Option Group|Excludes"
  )
  add_test(NAME loki_cli_help_color COMMAND $<TARGET_FILE:loki_app> search ffa --help)
  set_tests_properties(
    loki_cli_help_color PROPERTIES ENVIRONMENT "CLICOLOR_FORCE=1" PASS_REGULAR_EXPRESSION
                                   "${LOKI_ESC}\\[[0-9;]*m--config${LOKI_ESC}"
  )
  add_test(NAME loki_cli_quiet_verbose_conflict COMMAND $<TARGET_FILE:loki_app> -q -v search ffa
                                                        --config ${LOKI_FFA_CLI_CONFIG} --dry-run
  )
  set_tests_properties(loki_cli_quiet_verbose_conflict PROPERTIES WILL_FAIL TRUE)

  # EP search: config generation, a plan-only run that writes the plan cache, a
  # dry run that loads that cache, and two rejections (a missing file and an
  # RFI table, which the EP search does not accept). Eight segments give this
  # seeded fixture an eligible path at min_pd=0.1; four relied on fallback.
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
    "min_pd = 0.1\n"
    "prune_poly_order = 2\n"
    "nsegments = 8\n"
    "[performance]\n"
    "nthreads = 1\n"
    "max_process_memory_gb = 4.0\n"
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

  # inspect: simulate a 100 Hz pulsar, search it with the FFA (unclustered,
  # so -n has rows to cut), then summarise the result as text, JSON and with
  # --profile (search files store no profile); plain text when redirected,
  # colour with CLICOLOR_FORCE. Also the flat FFA file of --no-candidates and
  # an EP result, each found by the layout it reports.
  set(LOKI_INSPECT_DIR ${CMAKE_CURRENT_BINARY_DIR}/loki_cli_inspect)
  set(LOKI_INSPECT_RESULT ${LOKI_INSPECT_DIR}/out/inspect_ffa_results.h5)
  file(MAKE_DIRECTORY ${LOKI_INSPECT_DIR})
  string(
    JOIN
    ""
    LOKI_INSPECT_CONFIG_CONTENT
    "[input]\n"
    "timeseries = \"${LOKI_INSPECT_DIR}/sim.tim\"\n"
    "[preprocessing]\n"
    "preprocess = false\n"
    "[search]\n"
    "f_min = 99.0\n"
    "f_max = 101.0\n"
    "nbins = 64\n"
    "[performance]\n"
    "nthreads = 1\n"
    "max_process_memory_gb = 4.0\n"
    "[output]\n"
    "outdir = \"${LOKI_INSPECT_DIR}/out\"\n"
    "prefix = \"inspect\"\n"
    "[candidates]\n"
    "cluster = false\n"
  )
  file(WRITE ${LOKI_INSPECT_DIR}/ffa.toml "${LOKI_INSPECT_CONFIG_CONTENT}")
  add_test(NAME loki_inspect_clean COMMAND ${CMAKE_COMMAND} -E rm -rf ${LOKI_INSPECT_DIR}/out
                                           ${LOKI_INSPECT_DIR}/sim.tim ${LOKI_INSPECT_DIR}/ep.tim
  )
  set_tests_properties(loki_inspect_clean PROPERTIES FIXTURES_SETUP loki_inspect_clean)
  add_test(NAME loki_inspect_simulate
           COMMAND $<TARGET_FILE:loki_app> simulate --period 0.01 --dt 0.0001 --nsamps 65536
                   --snr 30 --seed 1 -o ${LOKI_INSPECT_DIR}/sim.tim
  )
  set_tests_properties(
    loki_inspect_simulate PROPERTIES FIXTURES_REQUIRED loki_inspect_clean FIXTURES_SETUP
                                     loki_inspect_sim
  )
  add_test(NAME loki_inspect_search COMMAND $<TARGET_FILE:loki_app> search ffa --config
                                            ${LOKI_INSPECT_DIR}/ffa.toml
  )
  set_tests_properties(
    loki_inspect_search PROPERTIES FIXTURES_REQUIRED loki_inspect_sim FIXTURES_SETUP
                                   loki_inspect_result
  )
  add_test(NAME loki_inspect_text COMMAND $<TARGET_FILE:loki_app> inspect ${LOKI_INSPECT_RESULT}
                                          -n 3
  )
  string(
    JOIN
    ""
    LOKI_INSPECT_TEXT_RE
    "=== Run metadata ===.*Search +ffa \\(candidate schema 1\\.0\\)"
    ".*Command +[^\n]*loki[^\n]* search ffa --config [^\n]*ffa\\.toml\n.*nsamps +65536"
    ".*Range freq +\\[99\\.0+, 101\\.0+\\] Hz.*=== Top candidates \\(3 of [0-9]+\\) ==="
    ".*freq \\(Hz\\) +S/N.*\n +3 +(99|100|101)\\.[0-9]+ +[0-9.]+ .*region 0\n\n"
    "=== Configuration \\(config_effective\\) ===.*f_min = 99\\.0"
  )
  set_tests_properties(
    loki_inspect_text
    PROPERTIES FIXTURES_REQUIRED loki_inspect_result PASS_REGULAR_EXPRESSION
               "${LOKI_INSPECT_TEXT_RE}" FAIL_REGULAR_EXPRESSION "${LOKI_ESC};\n +4 "
  )
  # Only the JSON object reaches stdout: no log line, no colour.
  add_test(NAME loki_inspect_json COMMAND $<TARGET_FILE:loki_app> inspect ${LOKI_INSPECT_RESULT}
                                          --json -n 2
  )
  string(
    JOIN
    ""
    LOKI_INSPECT_JSON_RE
    "^{\n  \"file\": \"[^\n]*inspect_ffa_results\\.h5\",\n  \"layout\": \"common\",\n"
    ".*\"nsamps\": 65536,.*\"loki_version\": \"[0-9][^\"]*\",\n  \"command_line\": "
    "\"[^\n]*search ffa"
    ".*\"config_effective\": \"\\[input\\]\\\\n.*\"config_toml\": \"\\[input\\]\\\\n"
    ".*\"rank\": 2, [^\n]*\"params\": {\"freq\": [0-9.e+-]+}[^\n]*\n  \\]\n}\n$"
  )
  set_tests_properties(
    loki_inspect_json
    PROPERTIES FIXTURES_REQUIRED loki_inspect_result PASS_REGULAR_EXPRESSION
               "${LOKI_INSPECT_JSON_RE}" FAIL_REGULAR_EXPRESSION
               "\\[info\\];${LOKI_ESC};\"rank\": 3;\"profile\""
  )
  add_test(NAME loki_inspect_profile COMMAND $<TARGET_FILE:loki_app> inspect
                                             ${LOKI_INSPECT_RESULT} --profile
  )
  set(LOKI_INSPECT_PROFILE_RE
      "=== Folded profile \\(top candidate\\) ===\n  This file stores no folded profiles\\."
  )
  set_tests_properties(
    loki_inspect_profile PROPERTIES FIXTURES_REQUIRED loki_inspect_result PASS_REGULAR_EXPRESSION
                                    "${LOKI_INSPECT_PROFILE_RE}"
  )
  add_test(NAME loki_inspect_json_profile COMMAND $<TARGET_FILE:loki_app> inspect
                                                  ${LOKI_INSPECT_RESULT} --json --profile
  )
  set_tests_properties(
    loki_inspect_json_profile PROPERTIES FIXTURES_REQUIRED loki_inspect_result
                                         PASS_REGULAR_EXPRESSION "\n  \"profile\": null\n}\n$"
  )
  add_test(NAME loki_inspect_color_always COMMAND $<TARGET_FILE:loki_app> inspect
                                                  ${LOKI_INSPECT_RESULT}
  )
  set(LOKI_INSPECT_COLOR_RE "${LOKI_ESC}\\[[0-9;]*m── Run metadata")
  set_tests_properties(
    loki_inspect_color_always PROPERTIES FIXTURES_REQUIRED loki_inspect_result ENVIRONMENT
                                         "CLICOLOR_FORCE=1" PASS_REGULAR_EXPRESSION
                                         "${LOKI_INSPECT_COLOR_RE}"
  )
  add_test(NAME loki_inspect_flat_search
           COMMAND $<TARGET_FILE:loki_app> search ffa --config ${LOKI_INSPECT_DIR}/ffa.toml
                   --no-candidates --prefix flat
  )
  set_tests_properties(
    loki_inspect_flat_search PROPERTIES FIXTURES_REQUIRED loki_inspect_sim FIXTURES_SETUP
                                        loki_inspect_flat
  )
  add_test(NAME loki_inspect_flat COMMAND $<TARGET_FILE:loki_app> inspect --json
                                          ${LOKI_INSPECT_DIR}/out/flat_ffa_results.h5
  )
  set_tests_properties(
    loki_inspect_flat
    PROPERTIES FIXTURES_REQUIRED loki_inspect_flat PASS_REGULAR_EXPRESSION
               "\"layout\": \"legacy_ffa\",.*\"writer_version\": \"[0-9][^\"]*\",.*\"rank\": 10,"
  )
  # EP: a 141 Hz pulsar and the EP CLI fixture's search, with one run.
  string(REPLACE "outdir = \"${LOKI_EP_CLI_DIR}/out\"" "outdir = \"${LOKI_INSPECT_DIR}/out\""
                 LOKI_INSPECT_EP_CONFIG_CONTENT "${LOKI_EP_CLI_CONFIG_CONTENT}"
  )
  string(REPLACE "plan_cache = \"${LOKI_EP_CLI_DIR}/out/cli_ep_plan.h5\"\n" ""
                 LOKI_INSPECT_EP_CONFIG_CONTENT "${LOKI_INSPECT_EP_CONFIG_CONTENT}"
  )
  file(WRITE ${LOKI_INSPECT_DIR}/ep.toml
       "${LOKI_INSPECT_EP_CONFIG_CONTENT}[preprocessing]\npreprocess = false\n"
  )
  add_test(NAME loki_inspect_ep_simulate
           COMMAND $<TARGET_FILE:loki_app> simulate --period 0.0070921985815602835 --dt 0.000064
                   --nsamps 32768 --snr 30 --seed 1 -o ${LOKI_INSPECT_DIR}/ep.tim
  )
  set_tests_properties(
    loki_inspect_ep_simulate PROPERTIES FIXTURES_REQUIRED loki_inspect_clean FIXTURES_SETUP
                                        loki_inspect_ep_sim
  )
  add_test(NAME loki_inspect_ep_search
           COMMAND $<TARGET_FILE:loki_app> search ep --config ${LOKI_INSPECT_DIR}/ep.toml --input
                   ${LOKI_INSPECT_DIR}/ep.tim --no-plan-cache
  )
  set_tests_properties(
    loki_inspect_ep_search PROPERTIES FIXTURES_REQUIRED loki_inspect_ep_sim FIXTURES_SETUP
                                      loki_inspect_ep
  )
  add_test(NAME loki_inspect_ep COMMAND $<TARGET_FILE:loki_app> inspect
                                        ${LOKI_INSPECT_DIR}/out/cli_ep_results.h5
  )
  string(
    JOIN
    ""
    LOKI_INSPECT_EP_RE
    "Search +ep \\(candidate schema 1\\.0\\).*accel \\(m/s\\^2\\) +freq \\(Hz\\) +S/N.*\n"
    " +1 +[-0-9.e]+ +14[012]\\.[0-9]+ +[0-9.]+ .*chunk 0 run [0-9_]+\n"
  )
  set_tests_properties(
    loki_inspect_ep PROPERTIES FIXTURES_REQUIRED loki_inspect_ep PASS_REGULAR_EXPRESSION
                               "${LOKI_INSPECT_EP_RE}"
  )
  add_test(NAME loki_inspect_missing COMMAND $<TARGET_FILE:loki_app> inspect
                                             ${LOKI_INSPECT_DIR}/missing.h5
  )
  set_tests_properties(loki_inspect_missing PROPERTIES WILL_FAIL TRUE)
  add_test(NAME loki_inspect_not_hdf5 COMMAND $<TARGET_FILE:loki_app> inspect
                                              ${LOKI_INSPECT_DIR}/ffa.toml
  )
  set_tests_properties(
    loki_inspect_not_hdf5 PROPERTIES PASS_REGULAR_EXPRESSION "not an HDF5 file"
  )
endif()
