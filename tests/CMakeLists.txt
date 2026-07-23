# SPDX-License-Identifier: MIT
add_executable(lgpsf_tests
    test_main.cpp)
target_link_libraries(lgpsf_tests PRIVATE lgpsf::lgpsf)
target_include_directories(lgpsf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_compile_options(lgpsf_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
add_test(NAME lgpsf_tests COMMAND lgpsf_tests)
