"""The ellipsoid pullback T(theta, x) = L(theta)^{-1} (x - mu(theta)), and its
forward/reverse-mode derivatives with respect to theta.

T maps a physical point x into the normalized coordinate u = T(theta, x) that
eval_lg_nd/grad_eval_lg_nd (lg_functions.py) expect: u = L^{-1}(x - mu),
Sigma = L L^T the node's covariance ellipsoid. This is built as the
composition of two stages (design discussion, 2026-07-23; see
docs/design-notes.md for the layout convention used here):

  Stage 2 (theta -> (mu, L)): unpack_theta / jvp_unpack_theta /
    vjp_unpack_theta. theta is the log-Cholesky encoding of L (N diagonal
    log-entries so L_ii > 0 by construction, N(N-1)/2 free off-diagonal
    entries) with mu either the leading N entries of theta (mu0=None, "fit
    mu") or a fixed constant not touched by theta at all (mu0=<array>,
    "fixed mu") -- the two cases the VarPro design calls for.
  Stage 1 (mu, L, x -> T(x)): pullback / pullback_jvp / pullback_vjp. The
    only piece that ever touches the actual ellipsoid geometry; independent
    of how theta encodes (mu, L).

eval_T / jvp_T / vjp_T compose the two stages via the chain rule (forward
mode: push a theta-tangent through stage 2, then stage 1; reverse mode: pull
an output-cotangent through stage 1, then stage 2), exactly mirroring how an
autodiff system composes primitive rules -- so swapping the theta encoding
(fit mu vs fixed mu, or a future variant) only ever means writing a new
small stage-2 function, never touching stage 1.

Vectorized over x (a batch of any number of points, of any broadcastable
shape); NOT vectorized over theta -- theta is always a single parameter
vector. x, u, w and other point-indexed quantities are represented as
length-N tuples of arrays (one array per coordinate, the same
structure-of-arrays convention as eval_lg_nd's *u; see
docs/design-notes.md), so T's output can be fed straight into
eval_lg_nd(p, ell, m, *T(...)).
"""
import numpy as np


# --------------------------------------------------------------------------
# Small linear-algebra building blocks: solve triangular systems, batched
# over a point-tuple right-hand side, reused by both pullback (solve L) and
# its VJP (solve L^T).
# --------------------------------------------------------------------------

def solve_L(L, rhs):
    """Solve L y = rhs for y (L lower-triangular), rhs a length-N tuple of
    arrays. Forward substitution; the Python loop is over N (small, fixed),
    every arithmetic op is vectorized over whatever shape each rhs[i] is."""
    N = len(rhs)
    y = [None] * N
    for i in range(N):
        s = rhs[i]
        for j in range(i):
            s = s - L[i, j] * y[j]
        y[i] = s / L[i, i]
    return tuple(y)


def solve_LT(L, rhs):
    """Solve L^T z = rhs for z (L lower-triangular, so L^T is
    upper-triangular), rhs a length-N tuple of arrays. Backward
    substitution."""
    N = len(rhs)
    z = [None] * N
    for i in reversed(range(N)):
        s = rhs[i]
        for j in range(i + 1, N):
            s = s - L[j, i] * z[j]
        z[i] = s / L[i, i]
    return tuple(z)


# --------------------------------------------------------------------------
# Stage 1: the pullback itself, T(mu, L, x) = L^{-1}(x - mu), and its
# derivatives with respect to (mu, L) at fixed x. Never touches theta.
# --------------------------------------------------------------------------

def pullback(mu, L, x):
    """u = L^{-1}(x - mu). x: length-N tuple of arrays (broadcastable
    together); returns a length-N tuple of arrays, same convention."""
    N = len(x)
    v = tuple(np.asarray(x[i], dtype=float) - mu[i] for i in range(N))
    return solve_L(L, v)


def pullback_jvp(mu, L, dmu, dL, x):
    """Directional derivative of pullback(mu, L, x) w.r.t. (mu, L) in
    direction (dmu, dL), at fixed x. dmu: shape (N,); dL: shape (N, N)
    (only the lower-triangular part matters; stage 2 never populates the
    rest, see jvp_unpack_theta).

    du = -L^{-1} (dL @ u + dmu), u = pullback(mu, L, x): differentiate
    u = L^{-1} v (v = x - mu) via the standard matrix-inverse identity
    d(L^{-1}) = -L^{-1} dL L^{-1} and dv = -dmu, then substitute
    L^{-1} v = u.
    """
    N = len(x)
    u = pullback(mu, L, x)
    rhs = tuple(dmu[i] + sum(dL[i, j] * u[j] for j in range(N)) for i in range(N))
    y = solve_L(L, rhs)
    return tuple(-yi for yi in y)


def pullback_vjp(mu, L, x, w):
    """Reverse-mode: given a cotangent w (length-N tuple of arrays, same
    shape as x -- one cotangent per point), return (w_mu, w_L), the
    cotangents on (mu, L) such that for any (dmu, dL),
    sum_points w . pullback_jvp(mu, L, dmu, dL, x)
      == sum_points (w_mu . dmu + sum_ij w_L[i][j] * dL[i,j])
    (this identity is the adjoint-consistency test in
    test_ellipsoid_transform.py).

    w_mu: length-N tuple of arrays (batched over points, like w itself).
    w_L: N x N (nested list) of arrays, w_L[i][j] batched like w.

    z = L^{-T} w, w_mu = -z, w_L = -outer(z, u); derived from the adjoint
    of pullback_jvp's linear map (dmu, dL) -> du.
    """
    N = len(x)
    u = pullback(mu, L, x)
    z = solve_LT(L, w)
    w_mu = tuple(-zi for zi in z)
    w_L = [[-z[i] * u[j] for j in range(N)] for i in range(N)]
    return w_mu, w_L


# --------------------------------------------------------------------------
# Stage 2: theta -> (mu, L), a log-Cholesky encoding, with mu either free
# (leading N entries of theta) or fixed at a given constant mu0.
# --------------------------------------------------------------------------

def theta_size(N, mu0=None):
    """Number of free parameters in theta: N (mu, if free) + N (log-diagonal
    of L) + N(N-1)/2 (strictly-lower entries of L)."""
    n_mu = N if mu0 is None else 0
    return n_mu + N + N * (N - 1) // 2


def unpack_theta(theta, N, mu0=None):
    """theta -> (mu, L). mu0=None: mu is theta[:N] ("fit mu"). mu0=<array>:
    mu is fixed at mu0, unrelated to theta ("fixed mu"). Either way, the
    next N entries are L's log-diagonal, then N(N-1)/2 entries are L's
    strictly-lower-triangular part (row-major: (1,0), (2,0), (2,1), ...)."""
    theta = np.asarray(theta, dtype=float)
    idx = 0
    if mu0 is None:
        mu = theta[:N]
        idx = N
    else:
        mu = np.asarray(mu0, dtype=float)
    diag_log = theta[idx:idx + N]
    idx += N
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = np.exp(diag_log[i])
    for i in range(1, N):
        for j in range(i):
            L[i, j] = theta[idx]
            idx += 1
    return mu, L


def jvp_unpack_theta(theta, dtheta, N, mu0=None):
    """Directional derivative of unpack_theta w.r.t. theta in direction
    dtheta. Returns (dmu, dL), same shapes/convention as pullback_jvp
    expects. d(L_ii)/d(diag_log_i) = L_ii (chain rule through exp), the
    off-diagonal entries are theta components directly so their tangent is
    just the corresponding dtheta entry."""
    _, L = unpack_theta(theta, N, mu0)
    dtheta = np.asarray(dtheta, dtype=float)
    idx = 0
    if mu0 is None:
        dmu = dtheta[:N]
        idx = N
    else:
        dmu = np.zeros(N)
    ddiag_log = dtheta[idx:idx + N]
    idx += N
    dL = np.zeros((N, N))
    for i in range(N):
        dL[i, i] = L[i, i] * ddiag_log[i]
    for i in range(1, N):
        for j in range(i):
            dL[i, j] = dtheta[idx]
            idx += 1
    return dmu, dL


def vjp_unpack_theta(theta, N, w_mu, w_L, mu0=None):
    """Reverse-mode: given (w_mu, w_L) (the cotangents on (mu, L), batched
    over points as in pullback_vjp's output), return the batched covector
    on theta -- a length-theta_size(N, mu0) tuple of arrays. w_mu is simply
    dropped when mu0 is not None (mu isn't a function of theta, so it has
    no theta component to accumulate into)."""
    _, L = unpack_theta(theta, N, mu0)
    P = theta_size(N, mu0)
    result = [None] * P
    idx = 0
    if mu0 is None:
        for i in range(N):
            result[idx] = w_mu[i]
            idx += 1
    for i in range(N):
        result[idx] = w_L[i][i] * L[i, i]
        idx += 1
    for i in range(1, N):
        for j in range(i):
            result[idx] = w_L[i][j]
            idx += 1
    return tuple(result)


# --------------------------------------------------------------------------
# Composition: T(theta, x) and its theta-derivatives, chaining stage 2 then
# stage 1 (forward) or stage 1 then stage 2 (reverse).
# --------------------------------------------------------------------------

def eval_T(theta, N, x, mu0=None):
    """u = T(theta, x), the ellipsoid pullback. x: length-N tuple of
    arrays; returns a length-N tuple of arrays (feed straight into
    eval_lg_nd(p, ell, m, *eval_T(...)))."""
    mu, L = unpack_theta(theta, N, mu0)
    return pullback(mu, L, x)


def jvp_T(theta, dtheta, N, x, mu0=None):
    """Directional derivative of T(theta, x) w.r.t. theta in direction
    dtheta, at fixed x. Returns a length-N tuple of arrays, same
    convention as eval_T."""
    mu, L = unpack_theta(theta, N, mu0)
    dmu, dL = jvp_unpack_theta(theta, dtheta, N, mu0)
    return pullback_jvp(mu, L, dmu, dL, x)


def vjp_T(theta, N, x, w, mu0=None):
    """Reverse-mode: given a cotangent w (length-N tuple of arrays, one
    per point, same shape as x), return <w, dT/dtheta> as a length-P tuple
    of arrays (batched over points, P = theta_size(N, mu0)) -- deliberately
    per-point rather than summed, so the caller can sum (for a gradient of
    a scalar objective) or keep it as the rows of a batched Jacobian."""
    mu, L = unpack_theta(theta, N, mu0)
    w_mu, w_L = pullback_vjp(mu, L, x, w)
    return vjp_unpack_theta(theta, N, w_mu, w_L, mu0=mu0)


# --------------------------------------------------------------------------
# Testing utility: the full (batch_shape, N, P) Jacobian tensor, built two
# independent ways (forward and reverse sweeps). See
# test_ellipsoid_transform.py for the cross-checks this enables.
# --------------------------------------------------------------------------

def jacobian_tensor_forward(theta, N, x, mu0=None):
    """dT_i/dtheta_p at every point, via P calls to jvp_T with standard
    basis directions in theta-space. Shape (*batch_shape, N, P)."""
    P = theta_size(N, mu0)
    cols = []
    for p in range(P):
        dtheta = np.zeros(P)
        dtheta[p] = 1.0
        du = jvp_T(theta, dtheta, N, x, mu0=mu0)
        cols.append(np.stack(du, axis=-1))  # (*batch_shape, N)
    return np.stack(cols, axis=-1)  # (*batch_shape, N, P)


def jacobian_tensor_reverse(theta, N, x, mu0=None):
    """dT_i/dtheta_p at every point, via N calls to vjp_T with standard
    basis cotangents in output space. Shape (*batch_shape, N, P)."""
    bshape = np.broadcast(*[np.asarray(xi, dtype=float) for xi in x]).shape
    rows = []
    for i in range(N):
        w = tuple(
            np.full(bshape, 1.0 if k == i else 0.0) for k in range(N)
        )
        dtheta_batched = vjp_T(theta, N, x, w, mu0=mu0)
        rows.append(np.stack(dtheta_batched, axis=-1))  # (*batch_shape, P)
    return np.stack(rows, axis=-2)  # (*batch_shape, N, P)
