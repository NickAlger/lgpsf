# SPDX-License-Identifier: MIT
"""Reading a fit: which rows worked, which did not, and how you would know.

A whole-operator fit is thousands of independent small nonlinear problems.
Some will be harder than others, and on a real problem you have no truth to
check against -- so the fit ships its own evidence. `OperatorFit` is a pair:

    fit.model         the operator (see deploying_a_fit.py)
    fit.diagnostics   per-row provenance, read by nothing in evaluation

The fields, and what each answers:

    status          RowStatus.Fit | FallbackBaseline | Failed | GatedOut
    score           the winning candidate's held-out score
    baseline_score  the score of the A-PRIORI model -- an LG fit at the sigma
                    you supplied, pinned, with no search. The searched fit
                    ships only if it beats this, so a fit is never worse than
                    the prior you handed in.
    stop_reason     why the ladder stopped on that row
    failures        {row: message} for rows that threw

Plus two views of the model itself: `qc_map` scores every row against probes
it never saw, and `spike_measure` reports how much of each row went into the
discrete spike rather than the smooth part.

    python examples/operator_diagnostics.py
"""
import numpy as np

import lgpsf
from frog_kernel import build_problem, probes


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]

    # A deliberately thin budget, so some rows genuinely struggle.
    V, HV = probes(problem, 14)

    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3])
    fit = lgpsf.fit_operator(x, mass, mass, V, HV, problem["sigma"],
                             config=config)
    model, diag = fit.model, fit.diagnostics

    print("structural check:", lgpsf.validate_operator(model) or "clean")

    # ---- 1. who shipped what --------------------------------------------
    print("\nstatus")
    for status in lgpsf.RowStatus.__members__.values():
        count = int((diag.status == int(status)).sum())
        if count:
            print(f"  {status.name:<18} {count:>5}")
    if diag.failures:
        first = next(iter(diag.failures.items()))
        print(f"  first failure: row {first[0]}: {first[1][:70]}")

    print("\nstop_reason")
    for reason in lgpsf.RowStop.__members__.values():
        count = int((diag.stop_reason == int(reason)).sum())
        if count:
            print(f"  {reason.name:<18} {count:>5}")

    # ---- 2. the baseline guard ------------------------------------------
    # Every row is also fitted the cheap a-priori way and scored on the same
    # folds. The searched fit ships only if it is strictly better.
    modelled = np.array(lgpsf.model_rows(model))
    searched = diag.score[modelled]
    prior = diag.baseline_score[modelled]
    better = np.isfinite(searched) & np.isfinite(prior) & (searched < prior)
    print(f"\nsearch beat the a-priori model on {better.sum()} of "
          f"{len(modelled)} rows")
    print(f"  median searched score {np.nanmedian(searched):.4f}, "
          f"median a-priori {np.nanmedian(prior):.4f}")
    print("  Where it did not, the a-priori model shipped -- which is why a "
          "fit can\n  never be worse than the prior you supplied.")

    # ---- 3. held-out QC over the whole field -----------------------------
    # Probes the fit never saw. This is the honest field-scale quality map,
    # and it needs no truth.
    V_qc, HV_qc = probes(problem, 30, seed=999)
    qc = lgpsf.qc_map(model, V_qc, HV_qc)
    finite = qc[np.isfinite(qc)]
    print(f"\nqc_map over {len(finite)} modelled rows: "
          f"median {np.median(finite):.4f}, "
          f"90th pct {np.percentile(finite, 90):.4f}, "
          f"worst {finite.max():.4f}")
    print(f"  NaN on {int(np.isnan(qc).sum())} rows -- those carry no model.")

    worst = int(np.nanargmax(np.where(np.isfinite(qc), qc, np.nan)))
    print(f"  worst row {worst} at ({model.mu[worst][0]:.2f}, "
          f"{model.mu[worst][1]:.2f}), {len(model.row_modes(worst))} modes, "
          f"score {diag.score[worst]:.4f}")

    # Where the hard rows are is worth asking of any fit, and the answer is
    # often not the one you would guess. Here the worst rows are INTERIOR, not
    # at the boundary: the frog kernel's bump is largest in the middle, so
    # those rows carry the most modulation structure -- and at k = 14 probes
    # the counting rule affords only k/2 - P = 4 modes to represent it. The
    # near-boundary rows are almost plain Gaussians and fit easily.
    edge = np.minimum.reduce([model.mu[:, 0], model.mu[:, 1],
                              1 - model.mu[:, 0], 1 - model.mu[:, 1]])
    order = np.argsort(np.where(np.isfinite(qc), qc, -np.inf))
    print(f"\n  mean distance to the boundary: "
          f"{edge[order[-20:]].mean():.3f} for the 20 worst rows, "
          f"{edge[order[:20]].mean():.3f} for the 20 best "
          "-- the hard rows are\n  INTERIOR, where there is the most structure "
          "to represent and the probe\n  budget caps the modes that could "
          "represent it.")

    # ---- 4. how much went into the spike ---------------------------------
    spike = lgpsf.spike_measure(model)
    print(f"\nspike_measure: max |m1 * s| = {np.nanmax(np.abs(spike)):.2e} "
          f"(spike disabled here, so this is zero)")

    # ---- 5. the geometry the fit recovered -------------------------------
    mu, Sigma = lgpsf.ellipsoid_field(model)
    axes = np.sqrt(np.linalg.eigvalsh(Sigma[modelled]))
    print(f"\nellipsoid_field: {len(modelled)} ellipsoids, "
          f"median 1-sigma axes {np.median(axes[:, 0]):.4f} x "
          f"{np.median(axes[:, 1]):.4f}")
    print("  This is exactly what ellipsoid_tree consumes, so the fit doubles "
          "as a\n  geometric description of the operator's locality.")


if __name__ == "__main__":
    main()
