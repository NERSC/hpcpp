# ----------------------------------------------------------------------------------------#
# Add comm-study
# ----------------------------------------------------------------------------------------#

message(STATUS "Adding comm-study...")
add_subdirectory(comm-study)

# ----------------------------------------------------------------------------------------#
# Add heat equation demo
# ----------------------------------------------------------------------------------------#

message(STATUS "Adding heat-equation...")
add_subdirectory(heat-equation)

# ----------------------------------------------------------------------------------------#
# Add MDSPAN + nvstdpar demo
# ----------------------------------------------------------------------------------------#

message(STATUS "Adding mdspan-stdpar...")
add_subdirectory(mdspan-stdpar)

message(STATUS "Adding 1d_stencil_stdpar...")
add_subdirectory(1d_stencil)

# ----------------------------------------------------------------------------------------#
# Add choleskey demo
# ----------------------------------------------------------------------------------------#
message(STATUS "Adding choleskey example...")
add_subdirectory(choleskey)
