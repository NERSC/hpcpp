project(mdspan-stdpar LANGUAGES CXX)

add_executable(mdspan-stdpar ${_EXCLUDE}
                             ${CMAKE_CURRENT_LIST_DIR}/mdspan-stdpar.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/mdspan-stdpar.cpp
                            PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet) add_dependencies(mdspan-stdpar
# magic_enum argparse)

# include core/include and generated files DO NOT include
# ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due
# to CUDA compiler.
target_include_directories(
  mdspan-stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${MDSPAN_INCLUDE_DIR})

target_link_libraries(mdspan-stdpar PUBLIC ${MPI_LIBS} stdexec hpcpp-core)

set_target_properties(
  mdspan-stdpar
  PROPERTIES CXX_STANDARD ${CXX_STANDARD}
             CXX_EXTENSIONS NO
             INSTALL_RPATH_USE_LINK_PATH ON)

# installation
install(TARGETS mdspan-stdpar DESTINATION ${CMAKE_INSTALL_BINDIR})
