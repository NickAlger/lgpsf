# What the operator-level QC measures, and why probe scaling matters

`docs/varpro-whitening-notes.tex` ("Operator-level quality control") derives
the natural error of a fitted operator: the relative Hilbert-Schmidt norm

    |M^-1/2 (H - B) M^-1/2|_F / |M^-1/2 H M^-1/2|_F,

estimated by an energy ratio over held-out probes drawn i.i.d. IN THE
WHITENED VARIABLES -- that is, `z = M^-1/2 * randn`, the FEM representation
of L^2-white noise. A coordinate-i.i.d. draw sits one factor of M away from
white noise. On a quasi-uniform mesh the difference hides inside constants
that cancel; this example builds a mesh where it CANNOT hide, and checks
everything against the dense truth.

The problem is the frog kernel on a polar mesh of a disk: cell-centered
polar quadrature weights `m = r * dr * dtheta` span a factor that GROWS
under refinement (23x to 67x here), so any misplaced power of M drifts as
the mesh refines instead of cancelling. That makes this both a lesson and
an integration test of the fit -> qcE pipeline: the estimator must track
the exact dual-norm error at every resolution, and the exact error of the
properly-probed fit must approach a mesh-independent plateau.

What the numbers below show, honestly. First, the estimator works: `qcE`
(a handful of held-out probes) tracks the exact dual error within its
sampling noise at every resolution, while the mass ratio triples. Second,
the exact dual error of the white-noise-probed fit is flat across
refinement -- the number MEANS the same thing on every mesh, which is the
entire point of the scaling. Third, a genuine trade-off, not a bug: the
white-noise-probed fit wins in the dual (function-space) metric, and the
coordinate-i.i.d.-probed fit wins in the raw Frobenius metric. Each probe
family importance-samples its own norm. Neither is "wrong" -- but only the
dual metric survives mesh refinement with its meaning intact, so it is the
one a mesh-independent quality target can be calibrated against.

## Output

```text
    n  m ratio | exact dual A   qcE A | exact dual B |   raw B   raw A
  432     23.0 |       0.4482  0.4204 |       0.4582 |  0.4046  0.4128
  867     33.0 |       0.4482  0.4646 |       0.4655 |  0.4218  0.4288
 1728     47.0 |       0.4544  0.4716 |       0.4693 |  0.4283  0.4329
 3468     67.0 |       0.4456  0.4059 |       0.4619 |  0.4179  0.4195

exact dual A is flat while the mass ratio triples: the scaled metric
is mesh-independent.  qcE tracks it with 5-probe sampling noise.
A wins the dual metric, B wins the raw metric: each probe family
importance-samples its own norm; only the dual one keeps its meaning
under refinement.
```

## Program

```python
import numpy as np

import lgpsf
from frog_kernel import frog_covariance, frog_row

CENTER = np.array([0.5, 0.5])
RADIUS = 0.42
K_FIT, K_QC = 25, 5


def disk_problem(nr, ntheta):
    """Cell-centered polar mesh of the disk: points (2, n), masses (n,).

    The masses are the polar quadrature weights r*dr*dtheta -- exact, smooth,
    and strongly nonuniform: max/min ~ 2*nr, growing under refinement.
    """
    dr, dth = RADIUS / nr, 2.0 * np.pi / ntheta
    r = (np.arange(nr) + 0.5) * dr
    th = (np.arange(ntheta) + 0.5) * dth
    R, T = np.meshgrid(r, th, indexing="ij")
    x = np.vstack([CENTER[0] + (R * np.cos(T)).ravel(),
                   CENTER[1] + (R * np.sin(T)).ravel()])
    return x, (R * dr * dth).ravel()


def dense_truth(x, mass):
    n = x.shape[1]
    H = np.empty((n, n))
    for i in range(n):
        H[i, :] = mass[i] * frog_row(x[:, i], x) * mass
    return H


def fit_from_probes(x, mass, sigma, Z, Y):
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False              # mesh-resolved kernel; no diagonal spike
    config.row.mode_policy = lgpsf.ShellLadder(list(range(9)))
    config.row.target_score = None
    fit = lgpsf.fit_operator(x, mass, mass, Z.T, Y.T, sigma, config=config)
    return np.asarray(lgpsf.assemble_sparse(fit.model, np.inf).todense())


def rel_dual(H, B, mass):
    """Exact relative Hilbert-Schmidt error in the natural norms."""
    w = 1.0 / np.sqrt(mass)
    return (np.linalg.norm((H - B) * np.outer(w, w))
            / np.linalg.norm(H * np.outer(w, w)))


def qc_energy(H, B, mass, Zq):
    """The implemented estimator: dual-norm energy ratio on held-out probes."""
    Rz, Yz = (B - H) @ Zq, H @ Zq
    inv_m = 1.0 / mass
    return np.sqrt(np.sum(Rz**2 * inv_m[:, None])
                   / np.sum(Yz**2 * inv_m[:, None]))


def main():
    rng = np.random.default_rng(20260813)
    print(f"{'n':>5} {'m ratio':>8} | {'exact dual A':>12} {'qcE A':>7} | "
          f"{'exact dual B':>12} | {'raw B':>7} {'raw A':>7}")
    for nr, ntheta in [(12, 36), (17, 51), (24, 72), (34, 102)]:
        x, mass = disk_problem(nr, ntheta)
        n = x.shape[1]
        sigma = [frog_covariance(x[:, i]) for i in range(n)]
        H = dense_truth(x, mass)

        omega = rng.standard_normal((n, K_FIT + K_QC))
        sm = 1.0 / np.sqrt(mass)

        # A: white-noise probes (z = M^-1/2 randn); B: coordinate-i.i.d.
        Za = omega[:, :K_FIT] * sm[:, None]
        Zb = omega[:, :K_FIT]
        Ba = fit_from_probes(x, mass, sigma, Za, H @ Za)
        Bb = fit_from_probes(x, mass, sigma, Zb, H @ Zb)
        Zq = omega[:, K_FIT:] * sm[:, None]   # held-out: the FIXED metric family

        raw = lambda B: np.linalg.norm(H - B) / np.linalg.norm(H)  # noqa: E731
        print(f"{n:>5} {mass.max()/mass.min():>8.1f} | "
              f"{rel_dual(H, Ba, mass):>12.4f} {qc_energy(H, Ba, mass, Zq):>7.4f} | "
              f"{rel_dual(H, Bb, mass):>12.4f} | "
              f"{raw(Bb):>7.4f} {raw(Ba):>7.4f}")
    print()
    print("exact dual A is flat while the mass ratio triples: the scaled metric")
    print("is mesh-independent.  qcE tracks it with 5-probe sampling noise.")
    print("A wins the dual metric, B wins the raw metric: each probe family")
    print("importance-samples its own norm; only the dual one keeps its meaning")
    print("under refinement.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/operator_qc_mesh_study.py`](../../examples/operator_qc_mesh_study.py); the output and figures above come from actually running it.*
