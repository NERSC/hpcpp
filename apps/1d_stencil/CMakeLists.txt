project(1d_stencil_stdpar LANGUAGES CXX)

file(GLOB STDPAR_SOURCES 
    "1d_stencil_2_stdpar.cpp"
    "1d_stencil_3_partition.cpp")

foreach(source_file ${STDPAR_SOURCES})
    # get the file name without an extension
    get_filename_component(exec_name ${source_file} NAME_WE)

    # add an executable with the same name as the source file
    add_executable(${exec_name} ${_EXCLUDE} ${source_file})
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)
    target_include_directories(${exec_name} PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

    target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS} ) 

    set_target_properties(${exec_name}
        PROPERTIES
            CXX_STANDARD ${CXX_STANDARD}
            CXX_EXTENSIONS NO
            INSTALL_RPATH_USE_LINK_PATH ON
    )

    # installation
    install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

add_executable(1d_stencil_2_snd 1d_stencil_2_snd.cpp)
target_link_libraries(1d_stencil_2_snd stdexec)
target_include_directories(1d_stencil_2_snd PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(stencil_stdpar stencil_stdpar.cpp)
target_link_libraries(stencil_stdpar stdexec)
target_include_directories(stencil_stdpar PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(stencil_stdpar_snd stencil_stdpar_snd.cpp)
target_link_libraries(stencil_stdpar_snd stdexec)
target_include_directories(stencil_stdpar_snd PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd 1d_stencil_3_partition_snd.cpp)
target_link_libraries(1d_stencil_3_partition_snd stdexec)
target_include_directories(1d_stencil_3_partition_snd PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd_gpu_single 1d_stencil_3_partition_snd_gpu_single.cpp)
target_link_libraries(1d_stencil_3_partition_snd_gpu_single stdexec)
target_include_directories(1d_stencil_3_partition_snd_gpu_single PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd_gpu_multi 1d_stencil_3_partition_snd_gpu_multi.cpp)
target_link_libraries(1d_stencil_3_partition_snd_gpu_multi stdexec)
target_include_directories(1d_stencil_3_partition_snd_gpu_multi PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})