project(1d_stencil_stdpar LANGUAGES CXX)

add_executable(stencil_serial stencil_serial.cpp)
target_include_directories(
  stencil_serial
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(stencil_stdpar stencil_stdpar.cpp)
target_link_libraries(stencil_stdpar stdexec)
target_include_directories(
  stencil_stdpar
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

if("${STDPAR}" STREQUAL "gpu")
  # TODO, fix cmake
  add_executable(stencil_stdexec stencil_stdexec.cpp)
  target_link_libraries(stencil_stdexec stdexec)
  target_include_directories(
    stencil_stdexec
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  add_executable(stencil_cuda stencil_cuda.cpp)
  target_include_directories(
    stencil_cuda
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
else()
  message(STATUS "Skipping stencil_stdexec.cpp as stdpar=${STDPAR}")
endif()
