# SPDX-License-Identifier: MIT
add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE lgpsf::lgpsf)
target_compile_options(hello_world PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
