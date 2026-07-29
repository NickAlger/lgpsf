# SPDX-License-Identifier: MIT
"""Fit ONE point-spread function from random probes.

`fit_operator` does a whole operator; underneath it, `fit_from_probes` does a
single target. Reach for it when you want one PSF, when you are diagnosing a
row the operator fit found hard, or when your target is not an operator row at
all -- the row layer only assumes a function you can take inner products with.

**The data model is the thing to understand here.** You never hand the fitter
function values. You hand it:

    z   (k, K)   probe fields
    y   (k,)     the inner products <z_i, phi>, one scalar per probe

and it recovers `phi`. For an operator row, `z` is the probes and `y` is the
row of `H @ V` -- which is why fitting an operator costs matvecs and not
entries. Here we take one row of the frog operator and fit only that.

The other required inputs are structural, and the library never guesses them:
`x` (where the columns are), `m2_diag` (their quadrature weights), `mu0` (where
the PSF is centered) and `sigma0` (its approximate shape -- the best guess a
physicist would supply).

    python examples/fit_one_psf.py
"""
import numpy as np

import lgpsf
from frog_kernel import build_problem

TARGET = (0.55, 0.35)
BUDGETS = [10, 20, 45, 110]


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]

    # The row we are going to fit, and where we think it lives.
    rho = int(np.argmin(np.linalg.norm(x - np.array(TARGET)[:, None], axis=0)))
    mu0 = x[:, rho]
    sigma0 = problem["sigma"][rho]
    truth = problem["H"][rho, :]

    print(f"target row {rho} at ({mu0[0]:.3f}, {mu0[1]:.3f})")
    print(f"{'k':>5}  {'modes':>6}  {'held-out':>9}  {'rel. L2 vs truth':>17}  stop")

    rng = np.random.default_rng(0)
    for k in BUDGETS:
        z = rng.normal(size=(k, problem["count"]))     # (k, K) probe fields
        y = z @ truth                                  # (k,)   inner products

        config = lgpsf.ProbeFitConfig()
        config.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5, 6])
        config.target_score = None                     # sweep the whole ladder

        result = lgpsf.fit_from_probes(x, mass, z, y, mu0,
                                       config=config,
                                       guesses=[lgpsf.InitialGuess(sigma0)],
                                       target_mass=mass[rho])

        # `result.model` is an LGExpansion -- a standalone, evaluable model.
        # eval_expansion gives the smooth CONTINUUM kernel, so multiply by the
        # masses to compare against a discrete operator row.
        predicted = mass[rho] * lgpsf.eval_expansion(result.model, x) * mass
        error = np.linalg.norm(predicted - truth) / np.linalg.norm(truth)

        print(f"{k:>5}  {result.model.num_modes:>6}  {result.score:>9.4f}  "
              f"{error:>17.4f}  {lgpsf.StopReason(result.stop_reason).name}")

    # More probes must not make the model worse.
    assert error < 0.1, "110 probes should fit this row well"
    print("\nThe held-out score is what the library selects on -- it is "
          "measured\nwithout the truth, which is all you have on a real "
          "problem. The last\ncolumn compares against the truth we happen to "
          "know, as a check on it.")


if __name__ == "__main__":
    main()
