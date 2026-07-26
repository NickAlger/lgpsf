#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Precompiled header for the test executable.
///
/// Eigen dominates the compile cost of every lgpsf translation unit
/// (measured on ellipsoid_tree: ~1.5 s / ~180 MB per TU without a PCH vs
/// ~0.2 s / ~125 MB with), and this machine has been OOM-crashed by
/// parallel Eigen builds -- see the COMPILE MEMORY SAFETY section of
/// docs/cpp-port-plan.md before building anything here.
///
/// doctest is deliberately NOT in this header: test_main.cpp defines
/// DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN before including it, and a
/// force-included PCH would land ahead of that define. Revisit with
/// SKIP_PRECOMPILE_HEADERS on that one TU if doctest ever shows up as a
/// meaningful share of the build.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
