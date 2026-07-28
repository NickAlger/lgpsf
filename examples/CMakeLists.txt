# SPDX-License-Identifier: MIT
add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE lgpsf::lgpsf)

# The C++ half of the example set. Figures come from ellipsoid_tree's plot2d,
# which lgpsf already depends on, so there is nothing extra to install.
add_executable(operator_fit_frog operator_fit_frog.cpp)
target_link_libraries(operator_fit_frog PRIVATE lgpsf::lgpsf)

foreach(_example hello_world operator_fit_frog)
    target_compile_options(${_example} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
endforeach()
