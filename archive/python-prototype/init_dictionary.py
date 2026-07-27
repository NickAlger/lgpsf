"""Initial-ellipsoid dictionary generation: the hypothesis side of the
probe-fit search, as a standalone library (also used directly by
research sweep scripts, which used to hand-build these).

Everything here is geometry + policy over the caller's batch -- no
probes, no fitting. Thetas are produced in the FIXED-mu (log-Cholesky
shape-only) encoding of ellipsoid_transform; callers release them via
release_mu when they want the free encoding. Evidence behind the
families: docs/robust-init-notes.md (the portfolio recipe, the 8:1
anisotropy stress study) and the PIG slice-37 refits.
"""
import math

import numpy as np
from scipy.spatial import cKDTree


def theta_from_L(L):
    """Pack a lower-triangular Cholesky factor into the fixed-mu
    log-Cholesky theta (N log-diagonals, then strictly-lower entries
    row-major -- unpack_theta's convention)."""
    N = L.shape[0]
    th = [np.log(L[i, i]) for i in range(N)]
    for i in range(1, N):
        for j in range(i):
            th.append(L[i, j])
    return np.array(th)


def window_shape(x, m2_diag):
    """The window's own ellipsoid shape: mass-weighted covariance of the
    batch geometry, normalized to largest eigenvalue 1, as a lower
    Cholesky factor. Lumped masses ~ cell areas, so the mass weighting
    makes this the uniform-measure covariance of the window REGION,
    independent of mesh grading (an unweighted point covariance would
    measure where the mesh is dense instead). Eigenvalues are floored at
    1e-4 of the largest, capping the shape's aspect against degenerate
    (near-collinear or boundary-clipped) windows."""
    w = m2_diag / m2_diag.sum()
    xbar = x @ w
    d = x - xbar[:, None]
    S = np.einsum("j,ij,kj->ik", w, d, d)
    evals, evecs = np.linalg.eigh(S)
    evals = np.maximum(evals, 1e-4 * evals[-1]) / evals[-1]
    return np.linalg.cholesky((evecs * evals) @ evecs.T)


def oriented_sigma(a, b, angle_deg):
    """2D SPD covariance with 1-sigma semi-axes (a, b), the a-axis
    rotated angle_deg from horizontal -- building block for
    orientation-covering dictionaries (the circle family's blind spot)."""
    ang = math.radians(angle_deg)
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang), math.cos(ang)]])
    return R @ np.diag([a ** 2, b ** 2]) @ R.T


def local_spacing(x, mu0):
    """Mesh spacing at mu0: the nearest-neighbor distance of the batch
    point closest to mu0 -- the ladder's bottom scale."""
    pts = np.asarray(x, dtype=float).T
    tree = cKDTree(pts)
    _, i_near = tree.query(np.asarray(mu0, dtype=float), k=1)
    return float(tree.query(pts[i_near], k=2)[0][1])


def window_radius(x, mu0):
    """Radius of the batch around mu0 -- the ladder's top scale, and the
    window-containment admissibility bound (the window is conservative,
    so the true kernel fits inside it by construction)."""
    return float(np.linalg.norm(np.asarray(x, dtype=float)
                                - np.asarray(mu0, dtype=float)[:, None],
                                axis=0).max())


def ladder_scales(h_local, r_max, n_rungs):
    """Log-spaced scales from the local mesh spacing to the window
    radius: too-small and too-large inits both fail (differently), and a
    log ladder brackets every observed case."""
    return np.geomspace(max(h_local, 1e-12 * max(r_max, 1.0)),
                        max(r_max, h_local), n_rungs)


def mid_out(n):
    """Index order for a length-n ladder: middle first, then outward --
    mid scales win most often, extremes are insurance. Ordering matters
    because the absolute early-exit certificate fires on the first
    good-enough candidate."""
    return sorted(range(n), key=lambda i: (abs(i - (n - 1) / 2), i))


def circle_rungs(radii, N):
    """Labelled fixed-mu thetas for isotropic inits at the given radii,
    in mid-out order."""
    return [(f"circle r={radii[i]:.3g}",
             np.concatenate([np.full(N, np.log(radii[i])),
                             np.zeros(N * (N - 1) // 2)]))
            for i in mid_out(len(radii))]


def window_rungs(radii, L_shape):
    """Labelled fixed-mu thetas for scaled copies of the window shape
    (major semi-axis = rung radius), in mid-out order."""
    return [(f"window r={radii[i]:.3g}", theta_from_L(radii[i] * L_shape))
            for i in mid_out(len(radii))]
