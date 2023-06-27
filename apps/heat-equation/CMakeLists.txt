project(heat-equation LANGUAGES CXX)

add_executable(heat-equation ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/heat-equation.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/heat-equation.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet)
# add_dependencies(heat-equation magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(heat-equation PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(heat-equation PUBLIC ${MPI_LIBS})

set_target_properties(heat-equation
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS heat-equation DESTINATION ${CMAKE_INSTALL_BINDIR})