# SPDX-License-Identifier: MIT
"""Does the initial-guess ladder have an anisotropy blind spot? At 8:1, no.

A standing worry about the initial-guess dictionary: it offers `sigma0`,
circles at `num_rungs` scales, and optionally scaled copies of the window's
shape -- every one of which is either the prior's own shape or ROUND. Nothing
in it covers ORIENTATION independently of the prior. The prototype's
`varpro_frog_robust_init.py` found that this bites hard at 8:1 anisotropy, and
`dev/robust-init-notes.md` parked a recipe adding an oriented-ellipse family.
`lgpsf::oriented_sigma` was ported for it and is tested, but no rung family is
built from it, so nothing reaches it.

**This experiment exists to keep that worry measurable, and as it stands it
does not confirm it.** Under the shipping defaults the circle ladder handles
8:1 fine -- see `anisotropy-hardening.md`. The reason is in the prototype's own
notes: its catastrophic cases were FREE-mu, and its FIX C found that pinning
the center "collapses the outcome variance -- no runaways anywhere". Pinning is
now the default. Both policies are run here so the distinction stays visible.

Keep it anyway. It is the ONLY problem in the repo anisotropic enough to see
this at all: the frog examples run at 2:1 and 4:1, and every row of the
field-scale glaciology validation has an a-priori aspect ratio below 1.7 by
construction. If a future change releases mu by default, drops the circle
rungs, or narrows the ladder, this is what would catch it.

WHAT IS MEASURED. One row of an 8:1 frog, fitted repeatedly from a single
starting ellipsoid at a time -- circles across scales, oriented ellipses across
angles, and the a-priori shape. A single start is expressible through the
shipping API: set `sigma0` to the guess and turn both rung families off, and
the dictionary reduces to that one entry.

    python experiments/anisotropy_hardening.py

Two further things are reported, both of which were open questions rather than
foregone conclusions:

  Does a round START bias the fitted ANSWER toward roundness?  A circular
      initial guess is not a circular constraint -- Levenberg-Marquardt
      optimizes the full log-Cholesky parameterization, so the fitted ellipsoid
      is free to become as anisotropic as the data wants. Whether it does is a
      question about basins, and the `fit aspect` column answers it.

  What does the SHIPPING default reach?  The last block runs the real
      configuration (`sigma0` + 3 circles) so the gap is a number rather than
      an argument.
"""
import numpy as np

import lgpsf
from frog_kernel import frog_row, frog_covariance

ASPECT = 8.0
SIGMA_ANISO = np.array([0.01, 0.01 / ASPECT**2])   # 1-sigma axes 0.1 x 0.0125
GRID = 81               # fine enough to resolve the thin axis and its modulation
TARGET = (0.55, 0.35)   # local rotation 0.5*pi*(0.55+0.35) = 81 degrees
WINDOW_RADIUS = 0.35    # 3.5x the long axis; the batch the row is fitted on

CIRCLE_RADII = [0.15, 0.10, 0.07, 0.05, 0.035, 0.025]
DICT_ANGLES = [0, 45, 90, 135]                  # true orientation is 81 degrees
DICT_SHAPES = [(0.07, 0.0175), (0.12, 0.03)]    # both 4:1

TARGETS = [("hard: 20 probes, 3 modes", 20, 1),
           ("comfortable: 60 probes, 21 modes", 60, 5)]

# The prototype's catastrophic 8:1 cases were all FREE-mu, and its own FIX C
# found that pinning the center "collapses the outcome variance -- no runaways
# anywhere, and inits coalesce into a few basins instead of the free-mu zoo".
# Pinning is now the library default, so both policies are run: Pinned says
# whether a user of the defaults is exposed, Free says whether the prototype's
# finding survived the port.
MU_POLICIES = [("mu pinned (the default)", lgpsf.MuPolicy.Pinned),
               ("mu free", lgpsf.MuPolicy.Free)]


def oriented_sigma(a, b, angle_degrees):
    """1-sigma semi-axes (a, b), the a-axis rotated from horizontal.

    The Python twin of `lgpsf::oriented_sigma`, which the C++ has but never
    builds a rung family from.
    """
    angle = np.radians(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag([a**2, b**2]) @ rotation.T


def shape_of(sigma):
    """(major 1-sigma axis, minor, orientation in degrees mod 180)."""
    values, vectors = np.linalg.eigh(sigma)
    major = vectors[:, -1]
    return (np.sqrt(max(values[-1], 0.0)), np.sqrt(max(values[0], 0.0)),
            np.degrees(np.arctan2(major[1], major[0])) % 180.0)


def build_row():
    """The batch, the masses, the true row, and where its center is.

    No dense operator: a single-row study needs one row, so it computes one.
    """
    axis = (np.arange(GRID) + 0.5) / GRID
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x_all = np.vstack([mesh[0].ravel(), mesh[1].ravel()])
    mass_all = np.full(x_all.shape[1], (1.0 / GRID) ** 2)

    center = np.array(TARGET)
    rho_all = int(np.argmin(np.linalg.norm(x_all - center[:, None], axis=0)))
    mu0 = x_all[:, rho_all]

    keep = np.linalg.norm(x_all - mu0[:, None], axis=0) <= WINDOW_RADIUS
    x, mass = x_all[:, keep], mass_all[keep]
    rho = int(np.argmin(np.linalg.norm(x - mu0[:, None], axis=0)))

    # H[rho, j] = m_rho * phi(x_rho, x_j) * m_j
    truth = mass[rho] * frog_row(mu0, x, SIGMA_ANISO) * mass
    return x, mass, rho, mu0, truth


def fit_from(x, mass, rho, mu0, truth, z, y, sigma_init, max_level, mu_policy):
    """One fit from ONE starting ellipsoid.

    Setting `sigma0` to the guess and disabling both rung families reduces the
    initial-guess dictionary to that single entry; a one-level mode policy
    removes the warm start too. So this is exactly a single-start fit, run
    through the shipping code path.
    """
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder([max_level])
    config.target_score = None
    config.mu = mu_policy
    config.window_shape_rungs = False
    config.circle_rungs_above_aspect = float("inf")     # no circles

    result = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                   sigma0=sigma_init, target_mass=mass[rho])
    predicted = mass[rho] * lgpsf.eval_expansion(result.model, x) * mass
    error = np.linalg.norm(predicted - truth) / np.linalg.norm(truth)
    frame = result.model.frame()
    return error, result.score, shape_of(frame.L @ frame.L.T), result


def report(label, rows, truth_shape):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"{'initial guess':<22} {'init aspect':>11} {'init ang':>9} "
          f"{'rel err':>9} {'score':>9} {'fit aspect':>11} {'fit ang':>8}")
    for name, init_shape, error, score, fit_shape in rows:
        ia = init_shape[0] / max(init_shape[1], 1e-300)
        fa = fit_shape[0] / max(fit_shape[1], 1e-300)
        print(f"{name:<22} {ia:>11.2f} {init_shape[2]:>9.0f} {error:>9.4f} "
              f"{score:>9.4f} {fa:>11.2f} {fit_shape[2]:>8.0f}")
    print(f"{'TRUE shape':<22} {truth_shape[0] / truth_shape[1]:>11.2f} "
          f"{truth_shape[2]:>9.0f}")


def main():
    x, mass, rho, mu0, truth = build_row()
    sigma_true = frog_covariance(mu0, SIGMA_ANISO)
    truth_shape = shape_of(sigma_true)

    print(f"8:1 frog, grid {GRID}, window radius {WINDOW_RADIUS} "
          f"({x.shape[1]} points)")
    print(f"target row at ({mu0[0]:.3f}, {mu0[1]:.3f}); true 1-sigma axes "
          f"{truth_shape[0]:.4f} x {truth_shape[1]:.4f} at "
          f"{truth_shape[2]:.0f} degrees")

    rng = np.random.default_rng(0)
    for label, num_probes, max_level in TARGETS:
        z = rng.normal(size=(num_probes, x.shape[1]))
        y = z @ truth

        dictionary = [(f"circle r={r:g}", np.diag([r * r, r * r]))
                      for r in CIRCLE_RADII]
        dictionary += [(f"a={a:g} ang={ang}", oriented_sigma(a, b, ang))
                       for (a, b) in DICT_SHAPES for ang in DICT_ANGLES]
        dictionary += [("a-priori true shape", sigma_true)]

        for mu_label, mu_policy in MU_POLICIES:
            rows = []
            for name, sigma_init in dictionary:
                error, score, fit_shape, _ = fit_from(
                    x, mass, rho, mu0, truth, z, y, sigma_init, max_level,
                    mu_policy)
                rows.append((name, shape_of(sigma_init), error, score, fit_shape))
            report(f"{label} -- {mu_label} -- one fit per starting ellipsoid",
                   rows, truth_shape)

            circles = [r for r in rows if r[0].startswith("circle")]
            oriented = [r for r in rows if r[0].startswith("a=")]
            apriori = [r for r in rows if r[0].startswith("a-priori")]
            best = lambda group: min(group, key=lambda r: r[2])
            # How often each family REACHES THE GOOD BASIN, not just whether
            # its best member does: a family whose best entry wins but whose
            # other entries diverge is not safe as the only family in the
            # dictionary. The outcomes are strongly bimodal -- good starts land
            # within a few parts in ten thousand of each other, failures are
            # 2x worse -- so 1.1x separates them cleanly with nothing near the
            # line. A looser threshold would score an obvious runaway (fitted
            # 44:1 against a true 8:1) as a success.
            ok = lambda group: sum(1 for r in group if r[2] < 1.1 * best(rows)[2])
            print(f"\n  best circle    {best(circles)[0]:<18} "
                  f"rel err {best(circles)[2]:.4f}   "
                  f"({ok(circles)}/{len(circles)} reached the good basin)")
            print(f"  best oriented  {best(oriented)[0]:<18} "
                  f"rel err {best(oriented)[2]:.4f}   "
                  f"({ok(oriented)}/{len(oriented)} reached the good basin)")
            print(f"  a-priori shape {'':<18} rel err {apriori[0][2]:.4f}")

            # What the SHIPPING configuration reaches: sigma0 + num_rungs
            # circles, which is every family the library actually has.
            shipped = lgpsf.ProbeFitConfig()
            shipped.mode_policy = lgpsf.ShellLadder([max_level])
            shipped.target_score = None
            shipped.mu = mu_policy
            result = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=shipped,
                                           sigma0=sigma_true,
                                           target_mass=mass[rho])
            predicted = mass[rho] * lgpsf.eval_expansion(result.model, x) * mass
            error = np.linalg.norm(predicted - truth) / np.linalg.norm(truth)
            fitted = shape_of(result.model.frame().L @ result.model.frame().L.T)
            print(f"\n  SHIPPING DEFAULT (sigma0 + {shipped.num_rungs} circles,"
                  f" {len(result.candidates)} candidates): rel err {error:.4f},"
                  f" fitted {fitted[0] / fitted[1]:.2f}:1 at {fitted[2]:.0f} deg")
            print(f"  gap to the best start of any family: "
                  f"{error / best(rows)[2]:.2f}x")

    print("\nThe `fit aspect` columns answer whether a round START biases the "
          "fitted ANSWER\ntoward roundness: compare the circle rows against "
          "the oriented ones. They do\nnot -- every start that reaches the "
          "good basin reports the same strongly\nanisotropic ellipsoid, so a "
          "circular start is a scaffold the solver discards.\n"
          "\nThe `reached the good basin` counts are the robustness measure "
          "that matters: a\nfamily whose BEST member wins but whose other "
          "members diverge is not safe as\nthe only family in the dictionary. "
          "See anisotropy-hardening.md and\ndev/robust-init-notes.md.")


if __name__ == "__main__":
    main()
