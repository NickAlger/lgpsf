"""The theta-dependent smooth feature phi_i(x; theta) = psi_i(T(theta, x)),
and its forward/reverse-mode derivatives w.r.t. theta -- purely a
composition of lg_functions.py (the LG modes and their spatial gradient)
with ellipsoid_transform.py (the pullback T and its theta-derivatives) via
the chain rule. No new math: this module owns none of it, it just wires the
two together for a list of modes at once (so the pullback -- the same for
every mode at a given theta -- is computed once, not once per mode).

Deliberately mass-free: knows nothing about M1/M2. See
docs/varpro-whitening-notes.tex for how this layer composes with the
whitening interface (whitening.py) that sits on top of it.

Vectorized over the point batch only (x: array of shape (N, *batch_shape));
loops over the mode list and over N are plain Python loops, per the
2026-07-23 governing principle (docs/design-notes.md).
"""
import numpy as np

from lg_functions import eval_lg_nd, grad_eval_lg_nd
from ellipsoid_transform import eval_T, jvp_T, vjp_T


def eval_feature(theta, N, x, modes, mu0=None):
    """phi_i(x; theta) = psi_i(T(theta, x)) for each (p, ell, m) in modes.
    Returns an array of shape (len(modes), *batch_shape)."""
    u = eval_T(theta, N, x, mu0=mu0)
    return np.stack([eval_lg_nd(p, ell, m, u) for (p, ell, m) in modes], axis=0)


def jvp_feature(theta, dtheta, N, x, modes, mu0=None):
    """Directional derivative of eval_feature w.r.t. theta in direction
    dtheta, at fixed x. Chain rule:
        d/dtheta psi_i(T(theta,x)) = grad_u(psi_i)(u) . jvp_T(theta,dtheta,x).
    Returns an array of shape (len(modes), *batch_shape)."""
    u = eval_T(theta, N, x, mu0=mu0)
    du = jvp_T(theta, dtheta, N, x, mu0=mu0)  # (N, *batch_shape)
    out = []
    for (p, ell, m) in modes:
        grad = grad_eval_lg_nd(p, ell, m, u)  # (N, *batch_shape)
        out.append(np.sum(grad * du, axis=0))  # (*batch_shape,)
    return np.stack(out, axis=0)


def vjp_feature(theta, N, x, w, modes, mu0=None):
    """Reverse-mode: given a cotangent w (shape (len(modes), *batch_shape),
    matching eval_feature's output), return <w, d(eval_feature)/dtheta> as
    an array of shape (P, *batch_shape). Combine every mode's spatial
    gradient weighted by its own cotangent entry into a single
    u-space cotangent, then push that through vjp_T -- the reverse-mode
    counterpart of jvp_feature's per-mode chain rule."""
    u = eval_T(theta, N, x, mu0=mu0)
    combined = np.zeros_like(u)  # (N, *batch_shape)
    for k, (p, ell, m) in enumerate(modes):
        grad = grad_eval_lg_nd(p, ell, m, u)  # (N, *batch_shape)
        combined = combined + w[k] * grad
    return vjp_T(theta, N, x, combined, mu0=mu0)
