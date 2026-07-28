# SPDX-License-Identifier: MIT
# Measurements about the library. Not built by default -- these are studies,
# not tests, and some take minutes. Enable with -DLGPSF_BUILD_EXPERIMENTS=ON.
add_executable(mesh_scalability mesh_scalability.cpp)
target_link_libraries(mesh_scalability PRIVATE lgpsf::lgpsf)
target_include_directories(mesh_scalability PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
target_compile_options(mesh_scalability PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
