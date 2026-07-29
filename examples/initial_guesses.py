# SPDX-License-Identifier: MIT
"""Telling the fitter where to start.

Each row is fitted by a MULTI-START: several trial ellipsoids, each run to
convergence, best held-out score wins. By default the trials are `num_rungs`
circles at log-spaced scales, and that default is deliberately hard to beat --
it sweeps the one axis a prior is most often wrong about, its SCALE.

But where to seed a nonlinear search is problem-specific, so you can hand the
trials in yourself:

    lgpsf.fit_from_probes(..., guesses=[lgpsf.InitialGuess(sigma)])

Each guess is `(sigma, mu=None, label="")`. `mu` defaults to the `default_mu`
you passed; under `MuPolicy.Pinned` it is where that trial's center STAYS.
`config.num_rungs = 0` says "only my guesses" -- otherwise the default circles
are appended after yours.

**Where this pays.** On an easy row it does not: every start finds the same
optimum and you are choosing between identical answers. It pays on rows whose
landscape is genuinely multimodal, and this example builds one deliberately --
8:1 anisotropy, a SIGNED modulation (`a_mod = 2`, so the row has negative
lobes), and a released center, which is the setting where the prototype's
initialization sweeps found the reduced landscape worst behaved.

Even there the honest result is CONDITIONAL, which is why the table below runs
four different rows: a good guess is never worse, ties on most rows, and
occasionally wins by 2-3x. Watch the mode count when it wins -- the default
often ships FEWER modes, because no circular start scored well enough to
justify climbing further up the ladder. That is the failure a good guess
prevents: not a slightly worse fit, but a search that gave up early.

    python examples/initial_guesses.py
"""
import numpy as np

import lgpsf
from frog_kernel import frog_row, frog_covariance

ASPECT = 8.0                                   # 1-sigma axes 0.100 x 0.0125
SIGMA_DIAG = np.array([0.01, 0.01 / ASPECT**2])
A_MOD = 2.0                                    # > 1, so the row goes negative
GRID = 81                                      # resolves the thin axis
RADIUS = 0.35                                  # the batch each row is fitted on
PROBES = 60

TARGETS = [(0.55, 0.35), (0.40, 0.60), (0.62, 0.48), (0.35, 0.45)]


def build_row(target):
    """One row of the hardened frog, plus the shape a physicist would supply."""
    axis = (np.arange(GRID) + 0.5) / GRID
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x_all = np.vstack([mesh[0].ravel(), mesh[1].ravel()])
    mass_all = np.full(x_all.shape[1], (1.0 / GRID) ** 2)

    where = np.array(target)
    mu0 = x_all[:, int(np.argmin(np.linalg.norm(x_all - where[:, None], axis=0)))]
    keep = np.linalg.norm(x_all - mu0[:, None], axis=0) <= RADIUS
    x, mass = x_all[:, keep], mass_all[keep]
    rho = int(np.argmin(np.linalg.norm(x - mu0[:, None], axis=0)))

    truth = mass[rho] * frog_row(mu0, x, SIGMA_DIAG, A_MOD) * mass
    return x, mass, rho, mu0, truth, frog_covariance(mu0, SIGMA_DIAG)


def fit(x, mass, rho, mu0, truth, z, y, num_rungs, guesses):
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5])
    config.target_score = None      # compare dictionaries, not early exits
    config.mu = lgpsf.MuPolicy.Free  # the hard setting; see the module docstring
    config.num_rungs = num_rungs

    result = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                   guesses=guesses, target_mass=mass[rho])
    predicted = mass[rho] * lgpsf.eval_expansion(result.model, x) * mass
    error = np.linalg.norm(predicted - truth) / np.linalg.norm(truth)
    return error, len(result.candidates), result.model.num_modes


def main():
    print(f"{ASPECT:g}:1 frog with a signed modulation (a_mod = {A_MOD:g}), "
          f"{PROBES} probes, center released\n")
    print(f"{'row':<14} {'3 circles (default)':>26} {'your sigma only':>26}"
          f" {'ratio':>7}")

    rng = np.random.default_rng(0)
    ratios = []
    for target in TARGETS:
        x, mass, rho, mu0, truth, sigma = build_row(target)
        z = rng.normal(size=(PROBES, x.shape[1]))
        y = z @ truth                       # all the fitter ever sees

        # The default: no guesses, so the fit supplies num_rungs circles.
        base, base_n, base_m = fit(x, mass, rho, mu0, truth, z, y, 3, [])

        # One guess, and nothing else. `num_rungs = 0` drops the circles.
        mine, mine_n, mine_m = fit(x, mass, rho, mu0, truth, z, y, 0,
                                   [lgpsf.InitialGuess(sigma, label="prior")])

        ratios.append(base / mine)
        print(f"{str(target):<14} {base:>10.4f} ({base_n:2d} fits, {base_m:2d} modes)"
              f" {mine:>10.4f} ({mine_n:2d} fits, {mine_m:2d} modes)"
              f" {base / mine:>6.2f}x")

    print(f"\nA single well-chosen guess is never worse, wins on "
          f"{sum(r > 1.1 for r in ratios)} of {len(ratios)} rows here (up to "
          f"{max(ratios):.1f}x),\nand costs less than half the fits. It is not "
          f"magic: on the rows where it ties,\nthe circles had already found "
          f"the same optimum.")

    # When you do NOT know the orientation, sweep it. This is the family the
    # default dictionary has no answer for -- circles carry no orientation at
    # all -- and it is opt-in because on most problems it does not pay.
    x, mass, rho, mu0, truth, sigma = build_row(TARGETS[0])
    z = rng.normal(size=(PROBES, x.shape[1]))
    y = z @ truth
    oriented = lgpsf.oriented_ladder(x, mu0, num_rungs=3,
                                     angles_degrees=[0.0, 45.0, 90.0, 135.0],
                                     aspect=4.0)
    error, count, modes = fit(x, mass, rho, mu0, truth, z, y, 0, oriented)
    print(f"\noriented_ladder: {len(oriented)} guesses over 4 angles x 3 scales"
          f" -> {error:.4f} ({count} fits, {modes} modes)")
    print("Use it when the shape is anisotropic and you cannot say which way it "
          "points.\nThe cost is one fit per entry, so keep the grid coarse.")


if __name__ == "__main__":
    main()
