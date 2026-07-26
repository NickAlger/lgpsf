# SPDX-License-Identifier: MIT
add_executable(lgpsf_tests
    test_main.cpp
    test_harmonic_polynomials.cpp
    test_lg_functions.cpp
    test_ellipsoid_transform.cpp
    test_lg_ellipsoid_feature.cpp
    test_whitening.cpp
    test_levenberg_marquardt.cpp
    test_varpro.cpp
    test_init_dictionary.cpp)
target_link_libraries(lgpsf_tests PRIVATE lgpsf::lgpsf)
target_include_directories(lgpsf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_compile_options(lgpsf_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)

# Eigen dominates the per-TU compile cost and this machine has been
# OOM-crashed by parallel Eigen builds; see the COMPILE MEMORY SAFETY section
# of docs/cpp-port-plan.md. pch.hpp explains why doctest stays out of it.
target_precompile_headers(lgpsf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pch.hpp)

add_test(NAME lgpsf_tests COMMAND lgpsf_tests)
