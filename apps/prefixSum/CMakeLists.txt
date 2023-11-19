project(prefixSum LANGUAGES CXX)

file(GLOB CPP_SOURCES "*.cpp")

# add -cudalib=cublas if -stdpar=gpu
if (STDPAR STREQUAL "gpu")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

foreach(source_file ${CPP_SOURCES})
  if(NOT STDPAR STREQUAL "gpu")
    if("${source_file}" MATCHES ".*stdpar.*gpu.*" OR "${source_file}"
                                                    MATCHES ".*gpu.*stdpar.*")
      message(STATUS "Skipping ${source_file} as stdpar=${STDPAR}")
      continue()
    endif()
  endif()

  # get the file name without an extension
  get_filename_component(exec_name ${source_file} NAME_WE)

  # add an executable with the same name as the source file
  add_executable(${exec_name} ${_EXCLUDE} ${source_file})

  # add dependency on argparse
  add_dependencies(${exec_name} argparse)

  set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX
                                                        LINKER_LANGUAGE CXX)
  target_include_directories(
    ${exec_name}
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  # uncomment only if using nvc++/23.1 - no need if nvc++/23.7
  # target_link_directories(${exec_name} PRIVATE /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/lib64)

  target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS} stdexec fmt )

  set_target_properties(
    ${exec_name}
    PROPERTIES CXX_STANDARD ${CXX_STANDARD}
               CXX_EXTENSIONS NO
               INSTALL_RPATH_USE_LINK_PATH ON)

  # installation
  install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()
