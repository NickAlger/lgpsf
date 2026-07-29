# SPDX-License-Identifier: MIT
"""How tight does the Levenberg-Marquardt tolerance need to be?

Each candidate in the per-row search is a nonlinear least-squares fit, and a
solver author's instinct is to run it to `ftol = xtol = gtol = 1e-8` -- which is
what lgpsf 0.1.0 shipped. But nothing downstream reads the fit to eight digits:
candidates are SELECTED on a held-out cross-validation score, and on a
well-fitted row that score sits around 1e-1. The outer loop may be polishing
digits the selection rule cannot see.

That is the question here -- not whether the tolerance should be a user option
(it always was, `config.row.varpro.ftol` and friends), but what its default
should be. **This experiment is why the default is now 1e-4**; the tables below
measure against 1e-8 as the reference, so they read as "what loosening buys".

Three things are measured, because they answer different halves of it.

  Which test binds.  ftol, xtol and gtol stop on different quantities:
      relative cost reduction, relative step size, gradient orthogonality.
      Loosening one that never fires is free and meaningless. Swept one at a
      time before they are swept together.

  Iterations, at the row layer.  The tolerance controls LM iterations directly;
      everything else is downstream of that. The row layer is the only place
      the count is exposed (`ProbeFitResult.candidates[i].num_iterations`).

  Error and wall time, at the operator layer.  What a user actually gets.

One confound has to be controlled for, and it is larger than the effect under
test. Both of the mode ladder's stopping rules -- `target_score` and
`mode_patience` -- read the cross-validation score. A tolerance change that
moves a score in the fifth decimal can therefore flip whether a level counted
as an improvement, and ship a row with a different NUMBER OF MODES. Modes are
worth far more than solver digits, so an error column that moves with the
tolerance is more likely reporting that than reporting solver quality. The
`fixed_ladder` block pins the ladder to full depth to tell the two apart.

The failure mode to watch for is a CLIFF. A tolerance that is merely loose
should cost accuracy gradually. If the error is flat and then jumps, the loop
is stopping before it reaches the basin, and the last safe value is not the one
just before the jump -- it is further back than that.

    python experiments/lm_tolerance.py            # the full sweep, ~13 min
    python experiments/lm_tolerance.py --quick    # the row-layer blocks only

PROVISIONAL, on the same terms as `fitting_defaults.py`: one smooth synthetic
two-dimensional kernel.
"""
import argparse
import time

import numpy as np

import lgpsf
from frog_kernel import build_problem

GRID = 24
PROBES = 45
MAX_LEVEL = 6
THREADS = 4          # leave the machine usable while this runs

TOLERANCES = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]

# The tight end of the sweep, and the value every block measures against. It is
# what the library shipped in 0.1.0, so the tables read as "what loosening from
# a solver author's default buys". Held explicitly rather than inherited -- see
# `row_config`.
REFERENCE_TOL = 1e-8

# The rung families follow the LIBRARY defaults (sigma0 + num_rungs circles,
# no window-shape family), so the candidate counts here are what a user gets.


def interior_rows(grid=GRID, stride=4, margin=4):
    """A regular subgrid of rows, away from the boundary.

    The frog kernel is multiplied by a bump that vanishes on the boundary of
    the unit square, so rows near the edge are nearly zero and their fits are
    dominated by noise. Sampling those would measure the wrong thing.
    """
    index = np.arange(margin, grid - margin + 1, stride)
    return np.array([i * grid + j for i in index for j in index])


def row_config(num_rungs, tol, max_evaluations=100, which=("ftol", "xtol", "gtol"),
               fixed_ladder=False):
    """A row-layer config with the named tolerances set to `tol`.

    `which` is what makes the first block possible: the tolerances not named
    are held at the REFERENCE value, so a change in iteration count is
    attributable to the one that moved.

    The reference is written out rather than inherited. It used to be the
    library default, and when that default moved to 1e-4 the first block
    silently became three copies of the same configuration -- "ftol only" sets
    ftol to a value the other two already had. Pinning it keeps the block
    meaningful whatever the default is.

    `fixed_ladder` forces every row to the top of the mode ladder. Both of the
    ladder's stopping rules -- `target_score` and `mode_patience` -- read the
    cross-validation score, so a tolerance change that nudges a score by a hair
    can flip whether a level counted as an improvement and change how many
    modes ship. With the ladder pinned, the candidate count is identical across
    tolerances by construction, and the error column is the solver alone.
    """
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder(list(range(MAX_LEVEL + 1)))
    config.num_rungs = num_rungs
    config.varpro.max_evaluations = max_evaluations
    config.varpro.ftol = REFERENCE_TOL
    config.varpro.xtol = REFERENCE_TOL
    config.varpro.gtol = REFERENCE_TOL
    for name in which:
        setattr(config.varpro, name, tol)
    if fixed_ladder:
        config.target_score = None
        config.mode_patience = MAX_LEVEL + 2
    return config


def fit_rows(problem, rows, V, config):
    """Fit each row on its own. The LM iteration count lives only here.

    Returns (iterations/row, candidates/row, mean score, mean rel. error,
    mean modes, seconds).
    """
    x, mass, H = problem["x"], problem["mass"], problem["H"]
    iterations = candidates = 0
    scores, errors, modes = [], [], []

    started = time.perf_counter()
    for rho in rows:
        truth = H[rho, :]
        result = lgpsf.fit_from_probes(x, mass, V, V @ truth, x[:, rho],
                                       config=config, sigma0=problem["sigma"][rho],
                                       target_mass=mass[rho])
        iterations += sum(c.num_iterations for c in result.candidates)
        candidates += len(result.candidates)
        scores.append(result.score)
        modes.append(result.model.num_modes)

        # The unrestricted parametric model against the true row. The operator
        # layer would clip this to a window; here there is none, so the two
        # error columns in this file are not comparable to each other.
        predicted = mass[rho] * lgpsf.eval_expansion(result.model, x) * mass
        errors.append(np.linalg.norm(predicted - truth) / np.linalg.norm(truth))
    elapsed = time.perf_counter() - started

    n = len(rows)
    return (iterations / n, candidates / n, float(np.mean(scores)),
            float(np.mean(errors)), float(np.mean(modes)), elapsed)


def fit_whole(problem, sigma, num_rungs, tol, max_evaluations=100,
              target_score=0.05):
    """One whole-operator fit. Returns (rel. error, mean score, modes, seconds)."""
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.num_threads = THREADS
    config.row.mode_policy = lgpsf.ShellLadder(list(range(MAX_LEVEL + 1)))
    config.row.target_score = target_score
    config.row.num_rungs = num_rungs
    config.row.varpro.max_evaluations = max_evaluations
    config.row.varpro.ftol = tol
    config.row.varpro.xtol = tol
    config.row.varpro.gtol = tol

    # Twice, keeping the faster: the first fit of a process pays warm-up costs
    # larger than several of the effects here. Same reasoning as
    # `fitting_defaults.py`, which see.
    elapsed = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        result = lgpsf.fit_operator(problem["x"], problem["mass"],
                                    problem["mass"], problem["V"], problem["HV"],
                                    sigma, config=config)
        elapsed = min(elapsed, time.perf_counter() - started)

    approx = np.asarray(lgpsf.assemble_sparse(result.model, np.inf).todense())
    error = np.linalg.norm(approx - problem["H"]) / np.linalg.norm(problem["H"])
    score = result.diagnostics.score
    model = result.model
    modes = np.mean([len(model.mode_sets[i]) for i in model.mode_set_id if i >= 0])
    return (error, float(np.nanmean(score[np.isfinite(score)])), float(modes),
            elapsed)


def heading(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def row_block(problem, rows, V, title, settings):
    heading(title)
    print(f"{'setting':<30} {'iters/row':>10} {'cands':>7} {'it/cand':>8} "
          f"{'score':>10} {'rel err':>10} {'modes':>6} {'sec':>7}")
    baseline = None
    for name, config in settings.items():
        iters, cands, score, error, modes, seconds = fit_rows(problem, rows, V,
                                                              config)
        if baseline is None:
            baseline = (iters, error, seconds)
            marker = ""
        else:
            base_iters, base_error, base_seconds = baseline
            marker = (f"  ({base_iters / iters:.2f}x fewer, "
                      f"{error / base_error:.2f}x err, "
                      f"{base_seconds / seconds:.2f}x faster)")
        print(f"{name:<30} {iters:>10.1f} {cands:>7.1f} {iters / cands:>8.2f} "
              f"{score:>10.6f} {error:>10.6f} {modes:>6.1f} {seconds:>7.1f}"
              f"{marker}", flush=True)


def whole_block(problem, sigma, title, settings):
    heading(title)
    print(f"{'setting':<30} {'rel err':>10} {'score':>10} {'modes':>6} {'sec':>7}")
    baseline = None
    for name, kwargs in settings.items():
        error, score, modes, seconds = fit_whole(problem, sigma, **kwargs)
        if baseline is None:
            baseline = (error, seconds)
            marker = ""
        else:
            base_error, base_seconds = baseline
            marker = (f"  ({error / base_error:.2f}x err, "
                      f"{base_seconds / seconds:.2f}x faster)")
        print(f"{name:<30} {error:>10.6f} {score:>10.6f} {modes:>6.1f} "
              f"{seconds:>7.1f}{marker}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="the row-layer blocks only")
    args = parser.parse_args()

    problem = build_problem(grid=GRID)
    rng = np.random.default_rng(0)
    problem["V"] = rng.normal(size=(PROBES, problem["count"]))
    problem["HV"] = problem["V"] @ problem["H"].T

    rows = interior_rows()
    print(f"grid {GRID} ({problem['count']} rows), {PROBES} probes, shells to "
          f"level {MAX_LEVEL}, {THREADS} threads")
    print(f"row-layer blocks fit {len(rows)} interior rows individually; "
          f"operator-layer\nblocks fit all {problem['count']} and report the "
          f"faster of two runs")

    # 1. Three stopping tests, three quantities. Which one is actually firing?
    row_block(problem, rows, problem["V"],
              "which stopping test binds (row layer, num_rungs = 6)", {
                  "all at the reference": row_config(6, REFERENCE_TOL),
                  "ftol = 1e-4 only": row_config(6, 1e-4, which=("ftol",)),
                  "xtol = 1e-4 only": row_config(6, 1e-4, which=("xtol",)),
                  "gtol = 1e-4 only": row_config(6, 1e-4, which=("gtol",)),
                  "all three at 1e-4": row_config(6, 1e-4),
              })

    # 2. The sweep proper, at both rung counts -- a looser solve from a worse
    #    start may not reach the same optimum, so the two knobs may interact.
    for num_rungs in (6, 3):
        row_block(problem, rows, problem["V"],
                  f"tolerance (row layer, num_rungs = {num_rungs})",
                  {f"ftol = xtol = gtol = {t:g}": row_config(num_rungs, t)
                   for t in TOLERANCES})

    # 3. The same sweep with the ladder pinned to full depth, which is the only
    #    way to read the error column as a statement about the SOLVER. Above,
    #    the mode count moves with the tolerance, and modes are worth far more
    #    than solver digits.
    row_block(problem, rows, problem["V"],
              "tolerance at fixed ladder depth (row layer, num_rungs = 6)",
              {f"ftol = xtol = gtol = {t:g}":
                   row_config(6, t, fixed_ladder=True) for t in TOLERANCES})

    # 4. And the evaluation cap, which is the other way to stop early -- but
    #    stops on effort rather than on any property of the fit.
    row_block(problem, rows, problem["V"],
              "max_evaluations (row layer, num_rungs = 6, tol = reference)",
              {f"max_evaluations = {m}": row_config(6, REFERENCE_TOL, max_evaluations=m)
               for m in (100, 50, 25)})

    if args.quick:
        return

    # 5. The same sweep on whole operators, at the shipping target_score, which
    #    is where a looser fit can pay twice: worse rows may fail to certify
    #    and climb further up the ladder, spending back what was saved.
    for num_rungs in (6, 3):
        whole_block(problem, problem["sigma"],
                    f"tolerance (whole operator, num_rungs = {num_rungs})",
                    {f"ftol = xtol = gtol = {t:g}":
                         dict(num_rungs=num_rungs, tol=t) for t in TOLERANCES})

    # 6. Control: with the early exit disabled the ladder depth is fixed, so
    #    any difference from block 5 is the early exit reacting to the
    #    tolerance rather than the tolerance itself.
    whole_block(problem, problem["sigma"],
                "tolerance (whole operator, num_rungs = 6, no early exit)",
                {f"ftol = xtol = gtol = {t:g}":
                     dict(num_rungs=6, tol=t, target_score=None)
                 for t in TOLERANCES})

    # 7. A prior with no shape information, to check the recommendation is not
    #    an artifact of starting next to the answer.
    isotropic = np.stack([np.trace(s) / 2.0 * np.eye(2)
                          for s in problem["sigma"]])
    whole_block(problem, isotropic,
                "tolerance (whole operator, num_rungs = 6, isotropic prior)",
                {f"ftol = xtol = gtol = {t:g}": dict(num_rungs=6, tol=t)
                 for t in TOLERANCES})

    # 8. The evaluation cap on whole operators, at both ends of the tolerance
    #    range: a loose tolerance may already be stopping first.
    for tol in (REFERENCE_TOL, 1e-3):
        whole_block(problem, problem["sigma"],
                    f"max_evaluations (whole operator, num_rungs = 6, "
                    f"tol = {tol:g})",
                    {f"max_evaluations = {m}":
                         dict(num_rungs=6, tol=tol, max_evaluations=m)
                     for m in (100, 50, 25)})


if __name__ == "__main__":
    main()
