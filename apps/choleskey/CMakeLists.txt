project(cholesky_stdpar LANGUAGES CXX)

add_executable(cholesky_serial cholesky_serial.cpp)
target_include_directories(
  cholesky_serial
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
target_link_libraries(cholesky_serial hpcpp-core)

add_executable(choleskey_stdpar choleskey_stdpar.cpp)
target_link_libraries(choleskey_stdpar stdexec hpcpp-core)
target_include_directories(
  choleskey_stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

# TODO: remove this example add_executable(choleskey_stdpar_snd
# choleskey_stdpar_snd.cpp) target_link_libraries(choleskey_stdpar_snd stdexec)
# target_include_directories( choleskey_stdpar_snd PRIVATE ${CMAKE_BINARY_DIR}
# ${CMAKE_CURRENT_LIST_DIR}/../../include ${ARGPARSE_INCLUDE_DIR}
# ${MDSPAN_INCLUDE_DIR})
