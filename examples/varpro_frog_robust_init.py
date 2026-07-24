"""Initial-guess robustness experiments on an anisotropy-hardened frog.

Background, from varpro_frog_fit.py at the frog's standard 2:1
anisotropy: small circular inits rescue the probe-starved 3-mode fit,
any radius works for the comfortable 21-mode fit, the level-0 Gaussian
pilot never runs away but its optimum is a saddle w.r.t. level-1
enrichment and its basin is poor. Separately, research-repo experience
(ellipsoid_psf_pig) says too-SMALL radii can fail on other problems, so
the standing remedy there is multi-start over radius. The open worry is
anisotropy: a circle carries no orientation information at all, so a
strongly anisotropic target attacks the circle guess where it is
weakest -- the fit must develop both aspect ratio and orientation from
scratch.

This script sharpens the frog's aspect ratio (ASPECT:1 along the same
81-degree rotation; the fit grid is refined to keep the thin axis and
its modulation resolved) and runs:

 1. BREAK -- circle-radius sweeps on the anisotropic problem, for the
    hard case (20 probes / 3 modes) and the comfortable case (60 probes
    / 21 modes), with the a-priori true-shape init as the reference row.
 2. FIX A (pilot shape) -- fit the single level-0 Gaussian mode from a
    small circle (the pilot has never been observed to run away), keep
    its fitted SHAPE, and multi-start the target fit from scaled copies
    of that ellipsoid, theta_pilot with L -> t L. Scaling avoids
    restarting exactly at the pilot optimum (a stationary point of the
    enriched problem -- see varpro_frog_fit.py's module docstring) and
    probes the scale axis the pilot gets wrong (it over-inflates to
    cover the modulation lobes).
 3. FIX B (oriented dictionary) -- multi-start from a small dictionary
    of oriented ellipses (ANGLES x SHAPES at a moderate 4:1 aspect),
    attacking the circle family's specific blind spot: orientation.
 4. FIX C (fixed-mu stage + release) -- fit with mu pinned (the
    library-wide mu0 switch; in the real pipeline mu is the known node
    location), which removes the two center-drift degrees of freedom
    from the search, then release: warm-start the free-mu fit from the
    fixed-mu result via ellipsoid_transform.release_mu (the free-mu
    theta is [mu, fixed-mu theta] by construction). Run both with mu
    pinned at the exact center and pinned at the offset (a-priori-error)
    center, to separate "fewer dof" from "correct center" effects.
 5. The head-to-head summary: best-of each strategy, plus the union
    portfolio.

Findings at aspect 8:1 (the printed tables + the sweep figures):

  - BREAK confirmed, and worse than expected. Hard case: the good basin
    narrows to a sliver around r = 0.05 (misfit 0.255); larger radii
    land in distinct bad minima (0.646, or divergence at 0.15) and
    SMALLER radii fail too (0.478) -- reproducing the research-repo
    experience that too-small inits can also break. Comfortable case:
    only r <= 0.035 reaches 0.029. Most striking: the exact a-priori
    moment-ellipsoid shape lands in local minima on BOTH targets
    (0.340 / 0.239). High anisotropy makes the reduced landscape
    genuinely multimodal; no single init family is safe.
  - The Gaussian pilot degrades as well: its own 1-mode landscape is
    multimodal (several pilot inits wander to junk solutions with mu
    far outside the window), and a single positive Gaussian cannot
    lock onto a strongly modulated 8:1 target. Pilot-shape multi-start
    fixes the comfortable case but not the hard one.
  - The oriented dictionary works where circles cannot: the entries
    oriented near the true 81 degrees find the best known solution on
    both targets; misoriented entries fail. Orientation coverage was
    the missing ingredient, not radius coverage.
  - Fixed-mu stage + release collapses the outcome variance: no
    runaways anywhere, and inits coalesce into a few basins instead of
    the free-mu zoo. On the comfortable target it is the strongest
    single strategy observed -- with mu pinned at the OFFSET center,
    every tested radius converged to the same pin-limited solution
    (0.250) and every release recovered the global best (0.029); pinned
    at the exact center, 4/6 radii reached 0.029 directly (vs 2/6
    free-mu). On the probe-starved 3-mode target the shape landscape
    stays multimodal even with mu pinned (most radii coalesce at a
    mediocre 0.62 minimum), but the smallest radius + release reached
    the global best 0.255, and every release improved on (never hurt)
    its fixed stage. The release warm start has no enrichment saddle:
    the fixed-stage optimum is not mu-stationary, so the freed
    components carry gradient immediately.
  - Conclusion: the robust recipe is a PORTFOLIO multi-start -- a
    small, deliberately diverse init dictionary, keeping the best final
    cost. When mu is known a priori (the real pipeline), the fixed-mu
    stage + release with circles across scales is the natural primary
    strategy; oriented ellipses across angles cover the remaining shape
    multimodality; the a-priori and pilot shapes are cheap extra
    members. Every failure observed here is covered by some portfolio
    member, and members are cheap: all fits reuse the same probe data,
    so the expensive currency (matvecs) is spent once.

Needs matplotlib, so run with the `tttt` conda env rather than `t3toolbox`:
    /home/nick/miniconda3/envs/tttt/bin/python examples/varpro_frog_robust_init.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))

from ellipsoid_transform import release_mu
from varpro_frog_fit import (
    MU_OFFSET,
    X0,
    build_problem,
    fit_with_history,
    run_init_sweep,
    theta_circle,
    theta_oriented,
    theta_true_shape,
)

ASPECT = 8.0
SIGMA_ANISO = np.array([0.01, 0.01 / ASPECT**2])   # 1-sigma axes 0.1 x 0.0125
FIT_GRID_ANISO = 81        # refine so the thin axis / its modulation resolve
TITLE_NOTE = f" -- aspect {ASPECT:g}:1"

CIRCLE_RADII = [0.15, 0.10, 0.07, 0.05, 0.035, 0.025]
PILOT_SCALES = [0.3, 0.5, 0.7, 1.0]
PILOT_INIT_RADIUS = 0.05
DICT_ANGLES = [0, 45, 90, 135]          # true orientation is 81 degrees
DICT_SHAPES = [(0.07, 0.0175), (0.12, 0.03)]   # (a, b), both 4:1 aspect

TARGETS = [
    ("p020_L1", 20, 1),    # hard: 3 modes, probe-starved
    ("p060_L5", 60, 5),    # comfortable: 21 modes
]


def scaled_theta(theta, t):
    """theta with the ellipsoid scaled, L -> t L: log-Cholesky diagonals
    shift by log t, the off-diagonal entry scales by t; mu unchanged."""
    th = np.asarray(theta, dtype=float).copy()
    th[2] += math.log(t)
    th[3] += math.log(t)
    th[4] *= t
    return th


def main():
    circle_results = {}
    print("################ BREAK: circle inits vs anisotropy ################")
    for tag, num_probes, max_level in TARGETS:
        inits = [(f"circle r={r:g}", theta_circle(r)) for r in CIRCLE_RADII]
        inits.append(("true shape", theta_true_shape(sigma0_diag=SIGMA_ANISO)))
        circle_results[tag] = run_init_sweep(
            f"aniso_{tag}", num_probes, max_level, inits,
            sigma0_diag=SIGMA_ANISO, fit_grid=FIT_GRID_ANISO,
            title_note=TITLE_NOTE)
        print()

    pilot_results = {}
    print("################ FIX: pilot-shape multi-start ################")
    for tag, num_probes, max_level in TARGETS:
        prob_pilot = build_problem(num_probes, 0,
                                   sigma0_diag=SIGMA_ANISO,
                                   fit_grid=FIT_GRID_ANISO)
        pilot, _, pilot_misfit = fit_with_history(
            prob_pilot, theta_circle(PILOT_INIT_RADIUS))
        print(f"[{tag}] level-0 pilot from circle r={PILOT_INIT_RADIUS:g}: "
              f"success={pilot.success}, {pilot.n_iterations} iters, "
              f"misfit {pilot_misfit:.3f}, theta {np.round(pilot.theta, 3)}")

        inits = [(f"pilot shape x{t:g}", scaled_theta(pilot.theta, t))
                 for t in PILOT_SCALES]
        pilot_results[tag] = run_init_sweep(
            f"aniso_pilot_{tag}", num_probes, max_level, inits,
            sigma0_diag=SIGMA_ANISO, fit_grid=FIT_GRID_ANISO,
            title_note=TITLE_NOTE + ", pilot-shape inits")
        print()

    dict_results = {}
    print("################ FIX B: oriented-ellipse dictionary ################")
    for tag, num_probes, max_level in TARGETS:
        inits = [(f"a={a:g} ang={ang}", theta_oriented(a, b, ang))
                 for (a, b) in DICT_SHAPES for ang in DICT_ANGLES]
        dict_results[tag] = run_init_sweep(
            f"aniso_dict_{tag}", num_probes, max_level, inits,
            sigma0_diag=SIGMA_ANISO, fit_grid=FIT_GRID_ANISO,
            title_note=TITLE_NOTE + ", oriented-dictionary inits")
        print()

    fixedmu_results = {}
    print("################ FIX C: fixed-mu stage + release ################")
    for tag, num_probes, max_level in TARGETS:
        prob_free = build_problem(num_probes, max_level,
                                  sigma0_diag=SIGMA_ANISO,
                                  fit_grid=FIT_GRID_ANISO)
        # theta_circle(r)[2:] drops the mu entries: the fixed-mu encoding
        # is the free-mu encoding's tail, by construction
        inits_fixed = [(f"circle r={r:g}", theta_circle(r)[2:])
                       for r in CIRCLE_RADII]

        # (i) mu pinned at the exact center (the pipeline's knowledge)
        rows_exact = run_init_sweep(
            f"aniso_fixedmu_{tag}", num_probes, max_level, inits_fixed,
            sigma0_diag=SIGMA_ANISO, fit_grid=FIT_GRID_ANISO, mu0=X0,
            title_note=TITLE_NOTE + ", mu fixed at exact center")
        released = []
        for label, success, iters, misfit, theta_fixed in rows_exact:
            res_r, _, mis_r = fit_with_history(
                prob_free, release_mu(theta_fixed, X0))
            released.append((f"{label} +rel", res_r.success,
                             res_r.n_iterations, mis_r, res_r.theta))
            print(f"    release {label:16s} fixed {misfit:.3f} -> "
                  f"free {mis_r:.3f}{'' if res_r.success else '  (no conv.)'}")
        fixedmu_results[tag] = released

        # (ii) mu pinned at the offset center (a-priori center error baked
        # in): no figure, console only -- how much does the wrong pin cost,
        # and does the release recover it?
        prob_fixed_off = build_problem(num_probes, max_level,
                                       sigma0_diag=SIGMA_ANISO,
                                       fit_grid=FIT_GRID_ANISO,
                                       mu0=X0 + MU_OFFSET)
        print("  -- mu pinned at the offset center --")
        for label, theta0 in inits_fixed:
            res_f, _, mis_f = fit_with_history(prob_fixed_off, theta0)
            res_r, _, mis_r = fit_with_history(
                prob_free, release_mu(res_f.theta, X0 + MU_OFFSET))
            print(f"    {label:16s} fixed@offset {mis_f:.3f}"
                  f"{'' if res_f.success else ' (no conv.)'} -> "
                  f"release {mis_r:.3f}"
                  f"{'' if res_r.success else ' (no conv.)'}")
        print()

    print("################ head to head ################")
    for tag, _, _ in TARGETS:
        strategies = [
            ("circle multi-start", circle_results[tag]),
            ("pilot-shape multi-start", pilot_results[tag]),
            ("oriented dictionary", dict_results[tag]),
            ("fixed-mu@exact + release", fixedmu_results[tag]),
            ("union portfolio", circle_results[tag] + pilot_results[tag]
             + dict_results[tag] + fixedmu_results[tag]),
        ]
        for name, rows in strategies:
            best = min(rows, key=lambda r: r[3])
            print(f"[{tag}] {name:26s} best: {best[0]:20s} "
                  f"misfit {best[3]:.3f}{'' if best[1] else '  (no conv.)'}")
        print()


if __name__ == "__main__":
    main()
