project(1d_stencil_stdpar LANGUAGES CXX)

FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_executable(1d_stencil_stdpar ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/1d_stencil_1_stdpar.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/1d_stencil_stdpar.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

target_include_directories(1d_stencil_stdpar PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(1d_stencil_stdpar PUBLIC ${MPI_LIBS} ${Boost_LIBRARIES})

set_target_properties(1d_stencil_stdpar
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS 1d_stencil_stdpar DESTINATION ${CMAKE_INSTALL_BINDIR})