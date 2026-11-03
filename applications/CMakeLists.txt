file(GLOB LOKI_APPLICATION_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LOKI_APPLICATIONS)
foreach(source_file IN LISTS LOKI_APPLICATION_SOURCES)
  get_filename_component(EXECUTABLE_NAME ${source_file} NAME_WE)
  # The CMake target name is suffixed to avoid colliding with the "loki"
  # library target (or any other application/test target); the installed
  # binary itself is still named after the source file via OUTPUT_NAME.
  set(TARGET_NAME "${EXECUTABLE_NAME}_app")
  add_executable(${TARGET_NAME} ${source_file})
  # The installed binary keeps the directories of the libraries it links
  # (libomp, HDF5) on its run path, as the build tree does.
  set_target_properties(
    ${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXECUTABLE_NAME} INSTALL_RPATH_USE_LINK_PATH ON
  )
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
  target_compile_definitions(${TARGET_NAME} PRIVATE LOKI_VERSION="${PROJECT_VERSION}")
  list(APPEND LOKI_APPLICATIONS ${TARGET_NAME})
endforeach()

# Under pip, the wheel's scripts directory: the installer puts it on PATH (the
# environment's bin/). A plain bin/ would land inside site-packages.
if(SKBUILD)
  set(LOKI_APPLICATION_DESTINATION ${SKBUILD_SCRIPTS_DIR})
else()
  set(LOKI_APPLICATION_DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS ${LOKI_APPLICATIONS} RUNTIME DESTINATION ${LOKI_APPLICATION_DESTINATION})
