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

add_executable(stencil_stdpar_snd stencil_stdpar_snd.cpp)
target_link_libraries(stencil_stdpar_snd stdexec)
target_include_directories(
  stencil_stdpar_snd
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

add_executable(stencil_stdpar_snd_iter stencil_stdpar_snd_iter.cpp)
target_link_libraries(stencil_stdpar_snd_iter stdexec)
target_include_directories(
  stencil_stdpar_snd_iter
  PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
          ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

if("${STDPAR}" STREQUAL "gpu")
  add_executable(stencil_snd_gpu_s stencil_snd_gpu_s.cpp)
  target_link_libraries(stencil_snd_gpu_s stdexec)
  target_include_directories(
    stencil_snd_gpu_s
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  add_executable(stencil_snd_gpu_m stencil_snd_gpu_m.cpp)
  target_link_libraries(stencil_snd_gpu_m stdexec)
  target_include_directories(
    stencil_snd_gpu_m
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  add_executable(stencil_cuda stencil_cuda.cpp)
  target_include_directories(
    stencil_cuda
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})
else()
  message(STATUS "Skipping stencil_snd_gpu_s.cpp as stdpar=${STDPAR}")
  message(STATUS "Skipping stencil_snd_gpu_m.cpp as stdpar=${STDPAR}")
endif()
