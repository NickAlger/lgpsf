# SPDX-License-Identifier: MIT
"""Choosing which modes to add, and in what order.

The mode ladder is a search: the fitter tries progressively larger mode sets
and keeps the best-scoring one. A `ModePolicy` decides what "progressively
larger" means, and the choice matters because the counting rule caps how many
modes a probe budget can afford -- so the ORDER decides which modes you get to
spend that budget on.

The built-ins:

    ShellLadder([0, 1, 2, ...])  every mode up to each oscillator level.
                                 Complete, but a shell grows fast.
    WedgeLadder(max_level, ell_max)
                                 level-ordered but angularly capped: only
                                 |ell| <= ell_max. Far cheaper per rung, and
                                 the operator layer's DEFAULT at (10, 2).
    RadialFirstLadder(...)       radial refinement before angular.
    ExplicitLadder([[...], ...]) mode sets you choose.
    FixedSet([...])              one set, no ladder.

There is no universally best order -- it depends on the probe budget and on
what the target actually looks like. That is why this is a policy and not a
constant. This example runs them head to head at several budgets.

    python examples/mode_policies.py
"""
import numpy as np

import lgpsf
from frog_kernel import build_problem

TARGET = (0.55, 0.35)
BUDGETS = [20, 45, 110]


def policies():
    return {
        "ShellLadder(0..6)": lgpsf.ShellLadder([0, 1, 2, 3, 4, 5, 6]),
        "WedgeLadder(10, 2)": lgpsf.WedgeLadder(10, 2),
        "WedgeLadder(10, 4)": lgpsf.WedgeLadder(10, 4),
        "RadialFirstLadder()": lgpsf.RadialFirstLadder(10, 2, 2),
    }


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]
    rho = int(np.argmin(np.linalg.norm(x - np.array(TARGET)[:, None], axis=0)))
    mu0, sigma0 = x[:, rho], problem["sigma"][rho]
    truth = problem["H"][rho, :]

    rng = np.random.default_rng(0)
    print(f"{'policy':>22}" + "".join(f"{'k=' + str(k):>22}" for k in BUDGETS))
    print(f"{'':>22}" + "".join(f"{'score / modes':>22}" for _ in BUDGETS))

    best = {}
    for name, policy in policies().items():
        cells = []
        for k in BUDGETS:
            z = rng.normal(size=(k, problem["count"]))
            y = z @ truth

            config = lgpsf.ProbeFitConfig()
            config.mode_policy = policy
            config.target_score = None      # no early exit: compare like for like

            result = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                           sigma0=sigma0, target_mass=mass[rho])
            cells.append(f"{result.score:.4f} / {result.model.num_modes}")
            best.setdefault(k, []).append((result.score, name))
        print(f"{name:>22}" + "".join(f"{c:>22}" for c in cells))

    print()
    for k in BUDGETS:
        score, name = min(best[k])
        print(f"  best at k={k:<4} {name}  ({score:.4f})")

    print("\nNo policy wins everywhere, and note that the operator-layer "
          "DEFAULT --\nWedgeLadder(10, 2) -- is not the winner on this target "
          "at any budget.\nThat is the lesson, not a bug: the frog kernel's "
          "modulation is strongly\nANGULAR, so capping |ell| <= 2 throws away "
          "exactly the modes it needs.\nRelaxing the cap to 4 is competitive, "
          "and complete shells win outright.\n\nThe default was chosen against "
          "a glaciology Hessian whose rows are much\nmore nearly elliptical, "
          "where the wedge reaches the same accuracy for a\nsixth of the modes "
          "(docs/validation.md). If your target has angular\nstructure, say so "
          "with the policy -- that is what the axis is for.")


if __name__ == "__main__":
    main()
