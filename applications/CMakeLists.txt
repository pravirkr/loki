file(GLOB LOKI_APPLICATION_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LOKI_APPLICATIONS)
foreach(source_file IN LISTS LOKI_APPLICATION_SOURCES)
  get_filename_component(EXECUTABLE_NAME ${source_file} NAME_WE)
  # The CMake target name is suffixed to avoid colliding with the "loki"
  # library target (or any other application/test target); the installed
  # binary itself is still named after the source file via OUTPUT_NAME.
  set(TARGET_NAME "${EXECUTABLE_NAME}_app")
  add_executable(${TARGET_NAME} ${source_file})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXECUTABLE_NAME})
  target_link_libraries(
    ${TARGET_NAME}
    PRIVATE ${LOKI_LIBRARY_NAME}
            CLI11::CLI11
            ${LOKI_COMPILE_OPTIONS}
            $<BUILD_INTERFACE:fmt::fmt-header-only>
            $<BUILD_INTERFACE:spdlog::spdlog_header_only>
            $<BUILD_INTERFACE:tomlplusplus>
            $<BUILD_INTERFACE:HighFive::HighFive>
  )
  list(APPEND LOKI_APPLICATIONS ${TARGET_NAME})
endforeach()

install(
  TARGETS ${LOKI_APPLICATIONS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
