# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.16)

# Version is single-sourced from the public header (see include/lgpsf/lgpsf.hpp).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/lgpsf/lgpsf.hpp" _lgpsf_hpp)
string(REGEX MATCH "LGPSF_VERSION_MAJOR ([0-9]+)" _ "${_lgpsf_hpp}")
set(_lgpsf_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "LGPSF_VERSION_MINOR ([0-9]+)" _ "${_lgpsf_hpp}")
set(_lgpsf_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "LGPSF_VERSION_PATCH ([0-9]+)" _ "${_lgpsf_hpp}")
set(_lgpsf_patch "${CMAKE_MATCH_1}")
project(lgpsf
    VERSION ${_lgpsf_major}.${_lgpsf_minor}.${_lgpsf_patch}
    LANGUAGES CXX)

add_library(lgpsf INTERFACE)
add_library(lgpsf::lgpsf ALIAS lgpsf)
target_compile_features(lgpsf INTERFACE cxx_std_17)
target_include_directories(lgpsf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ellipsoid_tree (which brings Eigen and Threads): use an installed copy if available,
# otherwise download a pinned release. For local development against a
# checkout, configure with -DFETCHCONTENT_SOURCE_DIR_ELLIPSOID_TREE=/path/to/ellipsoid_tree.
set(_lgpsf_ellipsoid_tree_vendored FALSE)
find_package(ellipsoid_tree 0.2 QUIET CONFIG)
if(NOT TARGET ellipsoid_tree::ellipsoid_tree)
    set(_lgpsf_ellipsoid_tree_vendored TRUE)
    option(LGPSF_FETCH_ELLIPSOID_TREE "Download ellipsoid_tree if no installed copy is found" ON)
    if(NOT LGPSF_FETCH_ELLIPSOID_TREE)
        message(FATAL_ERROR
            "lgpsf requires ellipsoid_tree. Install it (https://github.com/NickAlger/ellipsoid_tree) "
            "or enable LGPSF_FETCH_ELLIPSOID_TREE.")
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    # Single-sourced so the URL and the status message cannot drift apart.
    set(_ellipsoid_tree_pinned_version 0.2.0)
    FetchContent_Declare(ellipsoid_tree
        URL https://github.com/NickAlger/ellipsoid_tree/archive/refs/tags/v${_ellipsoid_tree_pinned_version}.tar.gz
        URL_HASH SHA256=24e66d161a85a9a1f98cb3838b0bb14edeba98d7c7d1dfb98678bdd61852ad66)
    FetchContent_MakeAvailable(ellipsoid_tree)
    message(STATUS "lgpsf: using downloaded ellipsoid_tree v${_ellipsoid_tree_pinned_version} (${ellipsoid_tree_SOURCE_DIR})")
endif()
target_link_libraries(lgpsf INTERFACE ellipsoid_tree::ellipsoid_tree)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(LGPSF_BUILD_TESTS "Build lgpsf tests" ON)
    option(LGPSF_BUILD_EXAMPLES "Build lgpsf examples" ON)
else()
    option(LGPSF_BUILD_TESTS "Build lgpsf tests" OFF)
    option(LGPSF_BUILD_EXAMPLES "Build lgpsf examples" OFF)
endif()

if(LGPSF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LGPSF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(LGPSF_BUILD_PYTHON "Build the lgpsf Python bindings (pybind11 via FetchContent)" OFF)
if(LGPSF_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

# ------------------------- Installation / CMake package export -------------------------
# Default ON when built standalone against an *installed* ellipsoid_tree; OFF as a
# subproject, OFF inside pip builds (scikit-build-core would otherwise install
# the headers into the wheel), and OFF when ellipsoid_tree came from the FetchContent
# fallback — the exported lgpsfConfig.cmake does find_dependency(ellipsoid_tree),
# which only works if consumers can find an installed ellipsoid_tree.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT SKBUILD AND NOT _lgpsf_ellipsoid_tree_vendored)
    set(LGPSF_INSTALL_DEFAULT ON)
else()
    set(LGPSF_INSTALL_DEFAULT OFF)
endif()
option(LGPSF_INSTALL
    "Generate install and CMake package export rules" ${LGPSF_INSTALL_DEFAULT})

if(LGPSF_INSTALL AND _lgpsf_ellipsoid_tree_vendored)
    message(FATAL_ERROR
        "LGPSF_INSTALL requires an installed ellipsoid_tree (found via find_package); the "
        "FetchContent fallback cannot be exported. Install ellipsoid_tree first, then reconfigure.")
endif()

if(LGPSF_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS lgpsf EXPORT lgpsfTargets)
    install(EXPORT lgpsfTargets
        FILE lgpsfTargets.cmake
        NAMESPACE lgpsf::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lgpsfConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/lgpsfConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/lgpsfConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/lgpsfConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/lgpsfConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf)
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
