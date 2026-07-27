"""The ellipsoid pullback T(theta, x) = L(theta)^{-1} (x - mu(theta)), and its
forward/reverse-mode derivatives with respect to theta.

T maps a physical point x into the normalized coordinate u = T(theta, x) that
eval_lg_nd/grad_eval_lg_nd (lg_functions.py) expect: u = L^{-1}(x - mu),
Sigma = L L^T the node's covariance ellipsoid. This is built as the
composition of two stages (design discussion, 2026-07-23; see
docs/design-notes.md for the array layout convention used here):

  Stage 2 (theta -> (mu, L)): unpack_theta / jvp_unpack_theta /
    vjp_unpack_theta. theta is the log-Cholesky encoding of L (N diagonal
    log-entries so L_ii > 0 by construction, N(N-1)/2 free off-diagonal
    entries) with mu either the leading N entries of theta (mu0=None, "fit
    mu") or a fixed constant not touched by theta at all (mu0=<array>,
    "fixed mu") -- the two cases the VarPro design calls for.
  Stage 1 (mu, L, x -> T(x)): pullback / pullback_jvp / pullback_vjp. The
    only piece that ever touches the actual ellipsoid geometry; independent
    of how theta encodes (mu, L). Solves against L via its explicit inverse
    (formed once, since theta -- and hence L -- is never batched) rather
    than a triangular substitution loop: cheap (N <= 4) and turns every
    "solve" into a single batched matrix contraction.

eval_T / jvp_T / vjp_T compose the two stages via the chain rule (forward
mode: push a theta-tangent through stage 2, then stage 1; reverse mode: pull
an output-cotangent through stage 1, then stage 2), exactly mirroring how an
autodiff system composes primitive rules -- so swapping the theta encoding
(fit mu vs fixed mu, or a future variant) only ever means writing a new
small stage-2 function, never touching stage 1.

Vectorized over x (a batch of any number of points, of any broadcastable
shape); NOT vectorized over theta -- theta is always a single parameter
vector. Point-indexed quantities (x, u, w, ...) are arrays of shape
(N, *batch_shape) -- non-batch axes first, the point-batch axes last,
matching the numpy-natural layout convention in docs/design-notes.md -- so
T's output can be fed straight into eval_lg_nd(p, ell, m, T(...)). Loops
over N (spatial dimension) or P (theta's parameter count) stay as plain
Python loops -- both are small and fixed, and the governing rule (agreed
2026-07-23) is: vectorize over the point batch always, loops over every
other axis are fine, often preferable for memory.
"""
import numpy as np


def _apply_matrix(A, v):
    """A @ v, batched over v's trailing axes. A: (N, N); v: (N, *batch_shape);
    returns (N, *batch_shape)."""
    return np.einsum('ij,j...->i...', A, v)


def _broadcast_param(vec, ndim_extra):
    """Reshape a (N,) parameter vector so it broadcasts against a
    (N, *batch_shape) array with len(batch_shape) == ndim_extra."""
    return vec.reshape(vec.shape[0], *([1] * ndim_extra))


# --------------------------------------------------------------------------
# Stage 1: the pullback itself, T(mu, L, x) = L^{-1}(x - mu), and its
# derivatives with respect to (mu, L) at fixed x. Never touches theta.
# --------------------------------------------------------------------------

def pullback(mu, L, x):
    """u = L^{-1}(x - mu). x: array of shape (N, *batch_shape); returns an
    array of the same shape."""
    x = np.asarray(x, dtype=float)
    v = x - _broadcast_param(mu, x.ndim - 1)
    return _apply_matrix(np.linalg.inv(L), v)


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
    u = pullback(mu, L, x)
    rhs = _apply_matrix(dL, u) + _broadcast_param(dmu, u.ndim - 1)
    return -_apply_matrix(np.linalg.inv(L), rhs)


def pullback_vjp(mu, L, x, w):
    """Reverse-mode: given a cotangent w (array of shape (N, *batch_shape),
    same shape as x -- one cotangent per point), return (w_mu, w_L), the
    cotangents on (mu, L) such that for any (dmu, dL),
    sum_points w . pullback_jvp(mu, L, dmu, dL, x)
      == sum_points (w_mu . dmu + sum_ij w_L[i,j] * dL[i,j])
    (this identity is the adjoint-consistency test in
    test_ellipsoid_transform.py).

    w_mu: shape (N, *batch_shape), like w. w_L: shape (N, N, *batch_shape).

    z = L^{-T} w, w_mu = -z, w_L = -outer(z, u); derived from the adjoint
    of pullback_jvp's linear map (dmu, dL) -> du.
    """
    w = np.asarray(w, dtype=float)
    u = pullback(mu, L, x)
    z = _apply_matrix(np.linalg.inv(L).T, w)
    w_mu = -z
    w_L = -np.einsum('i...,j...->ij...', z, u)
    return w_mu, w_L


# --------------------------------------------------------------------------
# Stage 2: theta -> (mu, L), a log-Cholesky encoding, with mu either free
# (leading N entries of theta) or fixed at a given constant mu0. theta is
# never batched, so these loops over N/P are outside the batch-vectorization
# rule entirely and stay as plain Python loops.
# --------------------------------------------------------------------------

def theta_size(N, mu0=None):
    """Number of free parameters in theta: N (mu, if free) + N (log-diagonal
    of L) + N(N-1)/2 (strictly-lower entries of L)."""
    n_mu = N if mu0 is None else 0
    return n_mu + N + N * (N - 1) // 2


def release_mu(theta_fixed, mu0):
    """Convert a fixed-mu theta to the free-mu encoding, seeding mu at
    mu0 -- e.g. to warm-start a free-mu fit from a fixed-mu stage's
    result. Inverse of freeze_mu. This is a plain concatenation because
    the free-mu encoding is literally [mu, <the fixed-mu encoding>] (see
    unpack_theta); the equivalence eval_T(release_mu(th, mu0), N, x) ==
    eval_T(th, N, x, mu0=mu0) is pinned in test_ellipsoid_transform.py."""
    return np.concatenate([np.asarray(mu0, dtype=float),
                           np.asarray(theta_fixed, dtype=float)])


def freeze_mu(theta_free, N):
    """Split a free-mu theta into (theta_fixed, mu0) -- e.g. to continue
    a free-mu result with mu pinned at its fitted value. Inverse of
    release_mu."""
    theta_free = np.asarray(theta_free, dtype=float)
    return theta_free[N:].copy(), theta_free[:N].copy()


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
    on theta -- an array of shape (theta_size(N, mu0), *batch_shape). w_mu
    is simply dropped when mu0 is not None (mu isn't a function of theta,
    so it has no theta component to accumulate into)."""
    _, L = unpack_theta(theta, N, mu0)
    P = theta_size(N, mu0)
    result = [None] * P
    idx = 0
    if mu0 is None:
        for i in range(N):
            result[idx] = w_mu[i]
            idx += 1
    for i in range(N):
        result[idx] = w_L[i, i] * L[i, i]
        idx += 1
    for i in range(1, N):
        for j in range(i):
            result[idx] = w_L[i, j]
            idx += 1
    return np.stack(result, axis=0)


# --------------------------------------------------------------------------
# Composition: T(theta, x) and its theta-derivatives, chaining stage 2 then
# stage 1 (forward) or stage 1 then stage 2 (reverse).
# --------------------------------------------------------------------------

def eval_T(theta, N, x, mu0=None):
    """u = T(theta, x), the ellipsoid pullback. x: array of shape
    (N, *batch_shape); returns an array of the same shape (feed straight
    into eval_lg_nd(p, ell, m, eval_T(...)))."""
    mu, L = unpack_theta(theta, N, mu0)
    return pullback(mu, L, x)


def jvp_T(theta, dtheta, N, x, mu0=None):
    """Directional derivative of T(theta, x) w.r.t. theta in direction
    dtheta, at fixed x. Returns an array of shape (N, *batch_shape), same
    convention as eval_T."""
    mu, L = unpack_theta(theta, N, mu0)
    dmu, dL = jvp_unpack_theta(theta, dtheta, N, mu0)
    return pullback_jvp(mu, L, dmu, dL, x)


def vjp_T(theta, N, x, w, mu0=None):
    """Reverse-mode: given a cotangent w (array of shape (N, *batch_shape),
    one per point, same shape as x), return <w, dT/dtheta> as an array of
    shape (P, *batch_shape) (P = theta_size(N, mu0)) -- deliberately
    per-point rather than summed, so the caller can sum (for a gradient of
    a scalar objective) or keep it as the rows of a batched Jacobian."""
    mu, L = unpack_theta(theta, N, mu0)
    w_mu, w_L = pullback_vjp(mu, L, x, w)
    return vjp_unpack_theta(theta, N, w_mu, w_L, mu0=mu0)


# --------------------------------------------------------------------------
# Testing utility: the full (N, P, *batch_shape) Jacobian tensor, built two
# independent ways (forward and reverse sweeps). See
# test_ellipsoid_transform.py for the cross-checks this enables.
# --------------------------------------------------------------------------

def jacobian_tensor_forward(theta, N, x, mu0=None):
    """dT_i/dtheta_p at every point, via P calls to jvp_T with standard
    basis directions in theta-space. Shape (N, P, *batch_shape)."""
    P = theta_size(N, mu0)
    cols = []
    for p in range(P):
        dtheta = np.zeros(P)
        dtheta[p] = 1.0
        cols.append(jvp_T(theta, dtheta, N, x, mu0=mu0))  # (N, *batch_shape)
    return np.stack(cols, axis=1)  # (N, P, *batch_shape)


def jacobian_tensor_reverse(theta, N, x, mu0=None):
    """dT_i/dtheta_p at every point, via N calls to vjp_T with standard
    basis cotangents in output space. Shape (N, P, *batch_shape)."""
    x = np.asarray(x, dtype=float)
    bshape = x.shape[1:]
    rows = []
    for i in range(N):
        w = np.zeros((N,) + bshape)
        w[i] = 1.0
        rows.append(vjp_T(theta, N, x, w, mu0=mu0))  # (P, *batch_shape)
    return np.stack(rows, axis=0)  # (N, P, *batch_shape)
