project(choleskey_stdpar LANGUAGES CXX)

add_executable(choleskey_serial choleskey_serial.cpp)
target_include_directories(
  choleskey_serial
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
target_link_libraries(choleskey_serial hpcpp-core)

add_executable(choleskey_stdpar choleskey_stdpar.cpp)
target_link_libraries(choleskey_stdpar stdexec hpcpp-core)
target_include_directories(
  choleskey_stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

# TODO: remove this example
# add_executable(choleskey_stdpar_snd choleskey_stdpar_snd.cpp)
# target_link_libraries(choleskey_stdpar_snd stdexec)
# target_include_directories(
#   choleskey_stdpar_snd
#   PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
#           ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

#tiled cholesky
add_executable(cholesky_tiled cholesky_tiled.cpp)
target_link_libraries(cholesky_tiled PRIVATE hpcpp-core blas lapack -fortranlibs gfortran)
target_include_directories(
  cholesky_tiled
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR} ${OPENBLAS_DIR}/include/openblas ${OPENBLAS_DIR}/lib64/cmake/OpenBLAS)

#tiled cholesky omp
add_executable(cholesky_tiled_omp matrixutil.hpp cholesky_tiled_omp.cpp )
target_link_libraries(cholesky_tiled_omp PRIVATE hpcpp-core blas lapack -fortranlibs gfortran)
target_include_directories(
  cholesky_tiled_omp
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR} ${OPENBLAS_DIR}/include/openblas ${OPENBLAS_DIR}/lib64/cmake/OpenBLAS)
