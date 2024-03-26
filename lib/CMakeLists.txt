add_library(loki loki.cpp)
target_include_directories(
  loki PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
