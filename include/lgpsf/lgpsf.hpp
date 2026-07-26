#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Umbrella header — includes the whole lgpsf public API.
///
/// lgpsf: Laguerre-Gaussian point spread function interpolation with VarPro
/// ellipsoid fitting. Header-only C++17; depends on Eigen and ellipsoid_tree.
/// https://github.com/NickAlger/lgpsf

// Single source of truth for the version. CMakeLists.txt parses these macros to
// set the project version; LGPSF_VERSION is the composed "MAJOR.MINOR.PATCH" string.
#define LGPSF_VERSION_MAJOR 0
#define LGPSF_VERSION_MINOR 1
#define LGPSF_VERSION_PATCH 0
#define LGPSF_STRINGIZE_IMPL(x) #x
#define LGPSF_STRINGIZE(x)      LGPSF_STRINGIZE_IMPL(x)
#define LGPSF_VERSION                        \
    LGPSF_STRINGIZE(LGPSF_VERSION_MAJOR) "." \
    LGPSF_STRINGIZE(LGPSF_VERSION_MINOR) "." \
    LGPSF_STRINGIZE(LGPSF_VERSION_PATCH)

#include <ellipsoid_tree/ellipsoid_tree.hpp>

#include "lgpsf/harmonic_polynomials.hpp"
#include "lgpsf/lg_functions.hpp"
// Further public headers land here as the port proceeds (docs/cpp-port-plan.md).
