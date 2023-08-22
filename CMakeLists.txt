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
set(NVSTDPAR_BUILD_TYPE "Release")

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
    "STDEXEC_ENABLE_CUDA ON"
    "STDEXEC_BUILD_EXAMPLES OFF" 
    "STDEXEC_BUILD_TESTS OFF" 
    "STDEXEC_ENABLE_IO_URING_TESTS OFF"
    "BUILD_TESTING OFF"
)

# need to add appropriate flags for stdexec
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdpar=multicore --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/bin/ -pthread")

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