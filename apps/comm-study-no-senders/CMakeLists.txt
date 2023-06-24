project(comm-study-no-senders LANGUAGES CXX)

add_executable(comm-study-no-senders ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/comm-study-no-senders.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/comm-study-no-senders.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet)
# add_dependencies(comm-study-no-senders magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(comm-study-no-senders PRIVATE ${CMAKE_BINARY_DIR} ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(comm-study-no-senders PUBLIC ${MPI_LIBS})

set_target_properties(comm-study-no-senders
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS comm-study-no-senders DESTINATION ${CMAKE_INSTALL_BINDIR})