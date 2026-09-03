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

target_include_directories(
  ${LOKI_TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(
  ${LOKI_TEST_TARGET} PRIVATE ${PROJECT_NAME}::${LOKI_LIBRARY_NAME} ${LOKI_COMPILE_OPTIONS}
                              Catch2::Catch2WithMain
                              Boost::math
                              OpenMP::OpenMP_CXX
)

catch_discover_tests(${LOKI_TEST_TARGET})
