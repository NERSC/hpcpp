# ##############################################################################
# Add colors
# ##############################################################################
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
endif()

# ##############################################################################
# CMake settings
# ##############################################################################

# cmake min required
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(hpcpp LANGUAGES CXX CUDA)

# in source build warning
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  set(MSG "")
  message(
    STATUS "Warning! Building from the source directory is not recommended")
  message(
    STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
  message(STATUS "and build from a separate directory")
  message(WARNING "In-source build")
endif()

# set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/external/")

# Set a default build type if none was specified
set(HPCPP_BUILD_TYPE "RelWithDebInfo")

# set the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "${BoldCyan}Setting build type to '${HPCPP_BUILD_TYPE}' as none was specified.${ColourReset}"
  )
  set(CMAKE_BUILD_TYPE
      "${HPCPP_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# ##############################################################################
# GCC version check
# ##############################################################################
set(GCC_EXPECTED_VERSION 11.2)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_EXPECTED_VERSION)
  message(
    FATAL_ERROR
      "GCC: hpcpp requires GCC v${GCC_EXPECTED_VERSION} or higher to build but found v${CMAKE_CXX_COMPILER_VERSION}"
  )
endif()

# ##############################################################################
# CXX standard
# ##############################################################################
set(CXX_STANDARD_REQUIRED ON)

# required minimum CXX standard
set(CMAKE_CXX_STANDARD_REQUIRED 23)
set(CMAKE_GNU_EXTENSIONS ON)

if(NOT CXX_STANDARD OR (CXX_STANDARD LESS ${CMAKE_CXX_STANDARD_REQUIRED}))
  set(CXX_STANDARD ${CMAKE_CXX_STANDARD_REQUIRED})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CXX_STANDARD}")
  message(STATUS "Setting CXX_STANDARD to ${CMAKE_CXX_STANDARD_REQUIRED}")
endif()

# ##############################################################################
# Setup STDEXEC
# ##############################################################################

# this is a hack should be automatically detected from the CMAKE_PREFIX_PATH
# instead of manual
set(CPM_DOWNLOAD_VERSION 0.35.6)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION
      "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION
      "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION
      "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(
    DOWNLOAD
    https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

cpmaddpackage(
  NAME
  stdexec
  GITHUB_REPOSITORY
  NVIDIA/stdexec
  GIT_TAG
  main
  OPTIONS
  "STDEXEC_ENABLE_CUDA ON"
  "STDEXEC_BUILD_EXAMPLES OFF"
  "STDEXEC_BUILD_TESTS OFF"
  "STDEXEC_ENABLE_IO_URING_TESTS OFF"
  "BUILD_TESTING OFF")

cpmaddpackage(NAME mdspan GITHUB_REPOSITORY kokkos/mdspan GIT_TAG stable)

cpmaddpackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG master)

cpmaddpackage(NAME mdspan_formatter GITHUB_REPOSITORY weilewei/mdspan_formatter
              GIT_TAG main)

cpmaddpackage(NAME argparse GITHUB_REPOSITORY mhaseeb123/argparse GIT_TAG
              master)

add_library(hpcpp-core INTERFACE)

# Link external libraries
target_link_libraries(hpcpp-core INTERFACE mdspan fmt mdspan_formatter argparse)

# stdpar and openmp
set(STDPAR_TYPE "gpu")
set(OMP_TYPE "multicore")

# set the build type
if(NOT STDPAR)
  message(
    STATUS
      "${BoldCyan}Setting -stdpar=${STDPAR_TYPE} as none was specified.${ColourReset}"
  )
  set(STDPAR
      "${STDPAR_TYPE}"
      CACHE STRING "Choose the stdpar accelerator." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE STDPAR PROPERTY STRINGS "gpu" "multicore")
endif()

# set the omp offload type
if(NOT OMP)
  message(
    STATUS
      "${BoldCyan}Setting -mp=${OMP_TYPE} as none was specified.${ColourReset}")
  set(OMP
      "${OMP_TYPE}"
      CACHE STRING "Choose the OpenMP accelerator." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE OMP PROPERTY STRINGS "multicore" "gpu")
endif()

# need to add appropriate flags for stdexec
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdpar=${STDPAR} -mp=${OMP}")

# add -cudalib=cublas if -stdpar=gpu
if(STDPAR STREQUAL "gpu")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -UUSE_GPU")
endif()

# ##############################################################################
# Add sub-directories
# ##############################################################################

# ----------------------------------------------------------------------------------------#
# apps
# ----------------------------------------------------------------------------------------#

message(STATUS "Adding hpcpp apps...")
add_subdirectory(apps)
