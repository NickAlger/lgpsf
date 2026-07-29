# SPDX-License-Identifier: MIT
"""What is the initial-guess ladder actually buying?

**This experiment is why the defaults changed.** lgpsf 0.1.0 shipped a
deliberately conservative per-row search: at every rung of the mode ladder it
tried `1 + num_rungs` window-shaped starts and `num_rungs` circular ones, plus a
warm start -- 14 nonlinear fits per rung, roughly 70 per row over a five-rung
ladder. It is now `sigma0` + 3 circles + a warm start, about 4.4x fewer. The
sweep below is what that decision rested on; the tables measure against the old
configuration, which is why every block sets its knobs explicitly rather than
inheriting them.

Two knobs are under test.

  `circle_rungs_above_aspect`  Circle rungs exist to rescue a prior that
      misleads: they sweep the SCALE axis at a neutral shape. When the prior is
      already nearly round they look like duplicates of what `sigma0` and the
      window rungs cover. Setting this above 1 skips them for near-isotropic
      priors. (The frog says they cost nothing. A real Hessian later said they
      are worth 2x when the prior's scale is wrong -- see the caveat below,
      which anticipated exactly that. The default stayed at 1.0.)

  `num_rungs`  How many log-spaced scales the ladder tries. Both families draw
      on the same scales, so this is one count, not one per family.

The axis that should decide both is PRIOR QUALITY, so the sweep runs over
deliberately damaged priors as well as correct ones. A setting that only works
when `sigma0` is right is not a default -- it is a trap for whoever supplies a
mediocre prior, which is everyone on a real problem.

    python experiments/fitting_defaults.py            # the full sweep
    python experiments/fitting_defaults.py --quick    # fewer conditions

PROVISIONAL, and it stayed provisional for good reason. The frog kernel is
smooth, synthetic and two-dimensional, and it damages the prior's SHAPE while
leaving its scale about right -- which is why it could not see the one effect
that turned out to matter.
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

ANISOTROPIC = np.array([0.04, 0.0025])     # 4:1 rather than the default 2:1


def damaged_priors(problem):
    """The a-priori field, correct and variously wrong.

    Each is a plausible failure of a real prior: a length scale estimated from
    the wrong quantity, an orientation taken from the wrong field, or no shape
    information at all.
    """
    sigma = problem["sigma"]
    rotate = np.array([[0.0, -1.0], [1.0, 0.0]])
    isotropic = np.stack([np.trace(s) / 2.0 * np.eye(2) for s in sigma])
    return {
        "correct": sigma,
        "rotated 90 deg": np.stack([rotate @ s @ rotate.T for s in sigma]),
        "4x too wide": 4.0 * sigma,
        "4x too narrow": 0.25 * sigma,
        "isotropic (no shape)": isotropic,
    }


def fit(problem, sigma, num_rungs, circle_aspect, window_rungs):
    """One whole-operator fit. Returns (relative error, mean score, seconds)."""
    V, HV = problem["V"], problem["HV"]

    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.num_threads = THREADS
    config.row.mode_policy = lgpsf.ShellLadder(list(range(MAX_LEVEL + 1)))
    config.row.target_score = None          # sweep the ladder; compare like for like
    config.row.num_rungs = num_rungs
    config.row.circle_rungs_above_aspect = circle_aspect
    config.row.window_shape_rungs = window_rungs

    # Twice, keeping the faster. The first fit of a process pays page faults
    # and allocator warm-up that have nothing to do with the setting under
    # test -- measured at 18.6 s against 11.6 s for the SAME configuration
    # before this was added, which is larger than most of the effects here.
    elapsed = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        result = lgpsf.fit_operator(problem["x"], problem["mass"],
                                    problem["mass"], V, HV, sigma, config=config)
        elapsed = min(elapsed, time.perf_counter() - started)

    approx = np.asarray(lgpsf.assemble_sparse(result.model, np.inf).todense())
    error = (np.linalg.norm(approx - problem["H"])
             / np.linalg.norm(problem["H"]))
    score = result.diagnostics.score
    return error, float(np.nanmean(score[np.isfinite(score)])), elapsed


def prior_aspect(sigma):
    """Median 1-sigma aspect ratio of a prior field -- what the knob tests."""
    values = np.linalg.eigvalsh(sigma)
    return float(np.median(np.sqrt(values[:, 1] / np.maximum(values[:, 0], 1e-300))))


def sweep(problem, label, conditions, settings):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"{'prior':<22} {'aspect':>7} {'setting':<26} "
          f"{'rel err':>9} {'score':>8} {'sec':>7}")
    baseline = {}
    for name, sigma in conditions.items():
        aspect = prior_aspect(sigma)
        for setting_name, kwargs in settings.items():
            error, score, seconds = fit(problem, sigma, **kwargs)
            if setting_name == next(iter(settings)):
                baseline[name] = (error, seconds)
                marker = ""
            else:
                base_error, base_seconds = baseline[name]
                marker = (f"  ({error / base_error:.2f}x err, "
                          f"{base_seconds / seconds:.2f}x faster)")
            print(f"{name:<22} {aspect:>6.1f}: {setting_name:<26} "
                  f"{error:>9.4f} {score:>8.4f} {seconds:>7.1f}{marker}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="the correct and isotropic priors only")
    args = parser.parse_args()

    print(f"grid {GRID} ({GRID**2} rows), {PROBES} probes, shells to level "
          f"{MAX_LEVEL}, {THREADS} threads")
    print("each configuration is fitted twice and the faster run reported; "
          "timings\nare still wall clock on a shared machine, so read ratios "
          "rather than seconds")

    for tag, sigma0_diag in (("2:1 kernel", None), ("4:1 kernel", ANISOTROPIC)):
        problem = build_problem(grid=GRID) if sigma0_diag is None else \
            build_problem(grid=GRID, sigma0_diag=sigma0_diag)
        rng = np.random.default_rng(0)
        problem["V"] = rng.normal(size=(PROBES, problem["count"]))
        problem["HV"] = problem["V"] @ problem["H"].T

        conditions = damaged_priors(problem)
        if args.quick:
            conditions = {k: conditions[k] for k in
                          ("correct", "isotropic (no shape)")}

        # 1. Do the circle rungs earn their keep?
        sweep(problem, f"{tag}: circle rungs, at num_rungs = 6", conditions, {
            "circles always (kappa=1)":
                dict(num_rungs=6, circle_aspect=1.0, window_rungs=True),
            "circles above 3:1 (kappa=3)":
                dict(num_rungs=6, circle_aspect=3.0, window_rungs=True),
            "circles never (kappa=inf)":
                dict(num_rungs=6, circle_aspect=float("inf"), window_rungs=True),
        })

        # 2. How many rungs are load-bearing?
        sweep(problem, f"{tag}: number of rungs, circles always", conditions, {
            f"num_rungs = {n}": dict(num_rungs=n, circle_aspect=1.0,
                                     window_rungs=True)
            for n in (6, 4, 3, 2)
        })

        # 3. And the window-shape family, which costs the other half.
        sweep(problem, f"{tag}: window-shape rungs, at num_rungs = 4",
              conditions, {
                  "window rungs on":
                      dict(num_rungs=4, circle_aspect=1.0, window_rungs=True),
                  "window rungs off":
                      dict(num_rungs=4, circle_aspect=1.0, window_rungs=False),
              })


if __name__ == "__main__":
    main()
