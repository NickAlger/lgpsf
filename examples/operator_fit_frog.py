# SPDX-License-Identifier: MIT
"""Fit a whole operator from random matvecs, and watch the error fall with k.

This is the end-to-end example: build a known dense operator, hand `fit_operator`
nothing but random probes and their responses, and compare. It is also the
library's integration test -- everything from the harmonic table to the sparse
assembly runs, on a problem that is entirely self-contained.

The target is the rotating frog kernel from `frog_kernel.py`, which the other
examples share. The fitter never sees its matrix; it sees `V` (random probes)
and `HV = H @ V`.

`operator_fit_frog.cpp` is this same example in C++, against the same problem.

Run:

    python examples/operator_fit_frog.py            # figures + convergence table
    python examples/operator_fit_frog.py --quick    # small, no figures

Needs the `lgpsf` bindings importable (build with `-DLGPSF_BUILD_PYTHON=ON`)
and scipy for the sparse assembly. matplotlib is optional -- without it the
convergence table still prints and the figures are skipped.
"""
import argparse

import numpy as np

import lgpsf
from frog_kernel import build_problem, probes


def fit_at(problem, num_probes, tau_window=3.0, max_level=8, seed=None):
    """Fit the operator from `num_probes` random matvecs and assemble it."""
    V, HV = probes(problem, num_probes, seed)

    config = lgpsf.OperatorFitConfig()
    config.tau_window = tau_window
    config.spike = False          # the kernel is mesh-resolved; no spike needed
    config.row.mode_policy = lgpsf.ShellLadder(list(range(max_level + 1)))
    config.row.target_score = None

    fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"],
                             V, HV, problem["sigma"], config=config)
    approx = np.asarray(lgpsf.assemble_sparse(fit.model, np.inf).todense())
    return fit, approx


def relative_frobenius(approx, H):
    return np.linalg.norm(approx - H, "fro") / np.linalg.norm(H, "fro")


# ---------------------------------------------------------------------------
# The counting rule sets a floor on k. With the center pinned in 2-D the row
# fit spends P = 3 parameters, and a mode set of size m is admissible only when
# k >= 2 (m + P). So k = 5 cannot fit even ONE mode, k = 10 affords one, k = 20
# affords six. That is why the small budgets below start at 10 rather than 5.
# ---------------------------------------------------------------------------

# `max_level = 8` above leaves 45 modes reachable, so at the top of this range
# the ladder is not the binding constraint -- the probe budget is, which is what
# the curve is meant to show.
CONVERGENCE_K = [10, 14, 20, 30, 45, 70, 110]
IMPULSE_K = [10, 20, 45]
IMPULSE_TARGETS = [(0.30, 0.30), (0.55, 0.35), (0.70, 0.65)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small grid, two budgets, no figures")
    parser.add_argument("--grid", type=int, default=40)
    args = parser.parse_args()

    grid = 20 if args.quick else args.grid
    budgets = [10, 45] if args.quick else CONVERGENCE_K

    print(f"building the frog operator on a {grid} x {grid} grid "
          f"({grid**2} dofs) ...", flush=True)
    problem = build_problem(grid=grid)

    print(f"\n{'k':>5}  {'rel. Frobenius':>15}  {'modes/row':>10}  {'rows fit':>9}")
    errors, keep = [], {}
    for k in budgets:
        fit, approx = fit_at(problem, k)
        error = relative_frobenius(approx, problem["H"])
        errors.append(error)
        if k in IMPULSE_K:
            keep[k] = approx          # reused below rather than refitted
        modeled = [rho for rho in range(problem["count"])
                   if fit.model.has_model(rho)]
        modes = np.mean([len(fit.model.row_modes(rho)) for rho in modeled])
        shipped = int((fit.diagnostics.status == int(lgpsf.RowStatus.Fit)).sum())
        print(f"{k:>5}  {error:>15.4f}  {modes:>10.1f}  {shipped:>9}", flush=True)

    assert errors[-1] < errors[0], "more probes should not make it worse"
    if args.quick:
        print("\nquick check passed")
        return

    try:
        import matplotlib
    except ImportError:
        print("\n(matplotlib not installed -- numbers above are the whole "
              "result; install it for the figures)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(budgets, errors, "o-")
    ax.set_xlabel("number of random matvecs, k")
    ax.set_ylabel(r"$\|\tilde{H} - H\|_F / \|H\|_F$")
    ax.set_title("Frog operator: accuracy against probe budget")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/operator_fit_frog_convergence.png", dpi=130)
    print("\nwrote examples/operator_fit_frog_convergence.png")

    # A ROW of H is the point-spread function at that target -- one model,
    # evaluated over every source. It is what the fitter represents, so it is
    # what the eye should judge.
    fig, axes = plt.subplots(len(IMPULSE_TARGETS), len(IMPULSE_K) + 1,
                             figsize=(3.1 * (len(IMPULSE_K) + 1),
                                      2.9 * len(IMPULSE_TARGETS)))
    for row, target in enumerate(IMPULSE_TARGETS):
        i = int(np.argmin(np.linalg.norm(
            problem["x"] - np.array(target)[:, None], axis=0)))
        truth = problem["H"][i, :].reshape(grid, grid)
        limit = np.abs(truth).max()
        for col, (title, field) in enumerate(
                [("true", truth)]
                + [(f"k = {k}", keep[k][i, :].reshape(grid, grid))
                   for k in IMPULSE_K]):
            ax = axes[row, col]
            ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                      cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"target {target}")
    fig.suptitle("Point-spread functions (rows of H): truth, and fits at "
                 "three probe budgets")
    fig.tight_layout()
    fig.savefig("examples/operator_fit_frog_impulses.png", dpi=130)
    print("wrote examples/operator_fit_frog_impulses.png")


if __name__ == "__main__":
    main()
