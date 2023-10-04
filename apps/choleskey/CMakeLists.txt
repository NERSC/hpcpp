project(choleskey_stdpar LANGUAGES CXX)

add_executable(choleskey_serial choleskey_serial.cpp)
target_include_directories(
  choleskey_serial
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(choleskey_stdpar choleskey_stdpar.cpp)
target_link_libraries(choleskey_stdpar stdexec)
target_include_directories(
  choleskey_stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(choleskey_stdpar_snd choleskey_stdpar_snd.cpp)
target_link_libraries(choleskey_stdpar_snd stdexec)
target_include_directories(
  choleskey_stdpar_snd
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
