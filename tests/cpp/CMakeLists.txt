CPMFindPackage(
  NAME Catch2
  VERSION 3.8.0
  GITHUB_REPOSITORY catchorg/Catch2
  OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
)

list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

# CTest configuration module (needed for catch_discover_tests)
include(CTest)
# Catch2 CMake integration module (provides catch_discover_tests)
include(Catch)

# Define the test executable target
set(LOKI_TEST_TARGET ${LOKI_LIBRARY_NAME}_tests)
file(GLOB LOKI_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${LOKI_TEST_TARGET} ${LOKI_TEST_SOURCES})

target_include_directories(
  ${LOKI_TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(
  ${LOKI_TEST_TARGET} PRIVATE ${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS} Catch2::Catch2WithMain
)

catch_discover_tests(${LOKI_TEST_TARGET})
