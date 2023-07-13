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

# 1d_stencil_2_snd example needs some headers from stdexec library 
# where these headers are not shipped by nvc++. The stdexec library
# is recommended to built with CPM CMake project management tool. 
# TODO: move CPM to external project; consider use cmake?
set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main # This will always pull the latest code from the `main` branch. You may also use a specific release version or tag
  OPTIONS 
    "STDEXEC_BUILD_EXAMPLES OFF" 
    "STDEXEC_BUILD_TESTS OFF" 
    "STDEXEC_ENABLE_IO_URING_TESTS OFF"
    "BUILD_TESTING OFF"
)

add_executable(1d_stencil_2_snd 1d_stencil_2_snd.cpp)
target_link_libraries(1d_stencil_2_snd stdexec)
target_include_directories(1d_stencil_2_snd PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd 1d_stencil_3_partition_snd.cpp)
target_link_libraries(1d_stencil_3_partition_snd stdexec)
target_include_directories(1d_stencil_3_partition_snd PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd_gpu_single 1d_stencil_3_partition_snd_gpu_single.cpp)
target_link_libraries(1d_stencil_3_partition_snd_gpu_single stdexec)
target_include_directories(1d_stencil_3_partition_snd_gpu_single PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})

add_executable(1d_stencil_3_partition_snd_gpu_multi 1d_stencil_3_partition_snd_gpu_multi.cpp)
target_link_libraries(1d_stencil_3_partition_snd_gpu_multi stdexec)
target_include_directories(1d_stencil_3_partition_snd_gpu_multi PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR})