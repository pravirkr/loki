CPMFindPackage(
  NAME Catch2
  VERSION 3.5.0
  GITHUB_REPOSITORY catchorg/Catch2
)
list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PUBLIC loki Catch2::Catch2WithMain)

include(Catch)
include(CTest)
# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
