project(comm-study LANGUAGES CXX)

file(GLOB CPP_SOURCES "*.cpp")

foreach(source_file ${CPP_SOURCES})
  # get the file name without an extension
  get_filename_component(exec_name ${source_file} NAME_WE)

  # add an executable with the same name as the source file
  add_executable(${exec_name} ${_EXCLUDE} ${source_file})
  set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX
                                                        LINKER_LANGUAGE CXX)
  target_include_directories(
    ${exec_name}
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${MDSPAN_INCLUDE_DIR})

  target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS} stdexec hpcpp-core)

  set_target_properties(
    ${exec_name}
    PROPERTIES CXX_STANDARD ${CXX_STANDARD}
               CXX_EXTENSIONS ${CMAKE_GNU_EXTENSIONS}
               INSTALL_RPATH_USE_LINK_PATH ON)

  # installation
  install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()
