// SPDX-License-Identifier: MIT
//
// Minimal smoke test for the VS Code build/debug setup: links against lgpsf
// (and, transitively, ellipsoid_tree + Eigen), does a trivial point-in-ellipsoid
// check, and prints the result. Set a breakpoint on the `inside` line below and
// hit Debug in the CMake Tools status bar to verify the whole toolchain works.

#include <iostream>

#include "lgpsf/lgpsf.hpp"

int main()
{
    std::cout << "lgpsf v" << LGPSF_VERSION << " hello world\n";

    ellipsoid_tree::Ellipsoid E;
    E.mu = Eigen::Vector2d(0.0, 0.0);
    E.Sigma = Eigen::Matrix2d::Identity() * 2.0;

    Eigen::Vector2d p(1.0, 1.0);
    bool inside = ellipsoid_tree::intersects(p, E, /*tau=*/1.0);

    std::cout << "point (" << p.x() << ", " << p.y() << ") inside ellipsoid: "
              << std::boolalpha << inside << "\n";

    return 0;
}
