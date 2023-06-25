##########################################################################################
#       CMake settings
##########################################################################################

# cmake min required
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(nvstdpar)
project(nvstdpar LANGUAGES CXX CUDA)

# in source build warning
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

# set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/external/")

# Set a default build type if none was specified
set(NVSTDPAR_BUILD_TYPE "RelWithDebInfo")

# set the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${NVSTDPAR_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${NVSTDPAR_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

##########################################################################################
#       CMake settings
##########################################################################################
option(USE_MDSPAN "Enable MDSPAN support" ON)

##########################################################################################
#       GCC version check
##########################################################################################
set(GCC_EXPECTED_VERSION 11.2)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_EXPECTED_VERSION)
    message(FATAL_ERROR "GCC: GCB requires GCC v${GCC_EXPECTED_VERSION} or higher to build but found v${CMAKE_CXX_COMPILER_VERSION}")
endif()

##########################################################################################
#       CXX standard
##########################################################################################
set(CXX_STANDARD_REQUIRED ON)

# required minimum CXX standard
set(CMAKE_CXX_STANDARD_REQUIRED 20)

if(NOT CXX_STANDARD OR (CXX_STANDARD LESS ${CMAKE_CXX_STANDARD_REQUIRED}))
    set(CXX_STANDARD ${CMAKE_CXX_STANDARD_REQUIRED})
    message(STATUS "Setting CXX_STANDARD to ${CMAKE_CXX_STANDARD_REQUIRED}")
endif()

##########################################################################################
#       Setup STDEXEC
##########################################################################################

# this is a hack should be automatically detected from the CMAKE_PREFIX_PATH instead of manual
set(STDEXEC_PATH /global/homes/m/mhaseeb/repos/stdexec/install/)

include_directories(${STDEXEC_PATH}/include)
link_directories(${STD_EXEC_PATH}/lib64)

# need to add appropriate flags for stdexec
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --experimental-stdpar -stdpar=gpu --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/bin/ -pthread")

##########################################################################################
#       Add sub-directories
##########################################################################################

#----------------------------------------------------------------------------------------#
#   externals
#----------------------------------------------------------------------------------------#

if (USE_MDSPAN)
    message(STATUS "Adding externals...")
    add_subdirectory(externals)
endif()

#----------------------------------------------------------------------------------------#
#   apps
#----------------------------------------------------------------------------------------#

message(STATUS "Adding nvstdpar apps...")
add_subdirectory(apps)