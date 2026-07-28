# SPDX-License-Identifier: MIT
add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE lgpsf::lgpsf)

# The C++ half of the example set. Figures come from ellipsoid_tree's plot2d,
# which lgpsf already depends on, so there is nothing extra to install.
add_executable(operator_fit_frog operator_fit_frog.cpp)
target_link_libraries(operator_fit_frog PRIVATE lgpsf::lgpsf)

add_executable(fit_one_psf fit_one_psf.cpp)
target_link_libraries(fit_one_psf PRIVATE lgpsf::lgpsf)

add_executable(preconditioner preconditioner.cpp)
target_link_libraries(preconditioner PRIVATE lgpsf::lgpsf)

foreach(_example hello_world operator_fit_frog fit_one_psf preconditioner)
    target_compile_options(${_example} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
endforeach()
