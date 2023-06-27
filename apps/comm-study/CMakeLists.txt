project(comm-study LANGUAGES CXX)

add_executable(comm-study ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/comm-study.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/comm-study.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet)
# add_dependencies(comm-study magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(comm-study PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${MDSPAN_INCLUDE_DIR})

target_link_libraries(comm-study PUBLIC ${MPI_LIBS})

set_target_properties(comm-study
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS comm-study DESTINATION ${CMAKE_INSTALL_BINDIR})