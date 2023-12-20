project(cholesky_stdpar LANGUAGES CXX)

add_executable(cholesky_serial cholesky_serial.cpp)
target_include_directories(
  cholesky_serial
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
target_link_libraries(cholesky_serial hpcpp-core)

add_executable(cholesky_stdpar cholesky_stdpar.cpp)
target_link_libraries(cholesky_stdpar stdexec hpcpp-core)
target_include_directories(
  cholesky_stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

# TODO: remove this example add_executable(cholesky_stdpar_snd
# cholesky_stdpar_snd.cpp) target_link_libraries(cholesky_stdpar_snd stdexec)
# target_include_directories( cholesky_stdpar_snd PRIVATE ${CMAKE_BINARY_DIR}
# ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR}
# ${MDSPAN_INCLUDE_DIR})
