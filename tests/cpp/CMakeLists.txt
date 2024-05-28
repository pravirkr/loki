CPMFindPackage(
  NAME Catch2
  VERSION 3.5.0
  GITHUB_REPOSITORY catchorg/Catch2
)
list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

set(TARGET_TESTS tests)
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_TESTS} ${TEST_SOURCES})
target_include_directories(${TARGET_TESTS} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${TARGET_TESTS} PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(${TARGET_TESTS} PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)

include(Catch)
include(CTest)
catch_discover_tests(${TARGET_TESTS})
