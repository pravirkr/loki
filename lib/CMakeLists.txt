# Find dependencies
CPMFindPackage(
  NAME spdlog
  VERSION 1.13.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL ON"
)
find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB LIBRARY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${LIBRARY_NAME} ${LIBRARY_SOURCES})

target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW_INCLUDE_DIRS})
target_link_libraries(
  ${LIBRARY_NAME} PUBLIC coverage_config spdlog::spdlog_header_only ${FFTW_FLOAT_LIB}
                         OpenMP::OpenMP_CXX
)
