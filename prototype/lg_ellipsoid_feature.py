"""The theta-dependent smooth feature phi_i(x; theta) = psi_i(T(theta, x)),
and its forward/reverse-mode derivatives w.r.t. theta -- purely a
composition of lg_functions.py (the LG modes and their spatial gradient)
with ellipsoid_transform.py (the pullback T and its theta-derivatives) via
the chain rule. No new math: this module owns none of it, it just wires the
two together for a list of modes at once.

Everything shared across the mode list is computed once: the pullback
here (the same for every mode at a given theta), and inside
lg_functions.eval_lg_basis / grad_lg_basis the Gaussian, r^2, each
harmonic polynomial (shared across radial orders p) and each Laguerre
recurrence (shared across angular indices m). Those functions return
per-mode results; the contraction against du / dU / the cotangent stays
here, where it belongs -- the LG layer knows nothing about theta.

Deliberately mass-free: knows nothing about M1/M2. See
docs/varpro-whitening-notes.tex for how this layer composes with the
whitening interface (whitening.py) that sits on top of it.

Vectorized over the point batch only (x: array of shape (N, *batch_shape));
loops over the mode list and over N are plain Python loops, per the
2026-07-23 governing principle (docs/design-notes.md).
"""
import numpy as np

from lg_functions import eval_lg_basis, grad_lg_basis
from ellipsoid_transform import eval_T, jvp_T, vjp_T, jacobian_tensor_forward


def eval_feature(theta, N, x, modes, mu0=None):
    """phi_i(x; theta) = psi_i(T(theta, x)) for each (p, ell, m) in modes.
    Returns an array of shape (len(modes), *batch_shape)."""
    u = eval_T(theta, N, x, mu0=mu0)
    return eval_lg_basis(modes, u)


def jvp_feature(theta, dtheta, N, x, modes, mu0=None):
    """Directional derivative of eval_feature w.r.t. theta in direction
    dtheta, at fixed x. Chain rule:
        d/dtheta psi_i(T(theta,x)) = grad_u(psi_i)(u) . jvp_T(theta,dtheta,x).
    Returns an array of shape (len(modes), *batch_shape)."""
    u = eval_T(theta, N, x, mu0=mu0)
    du = jvp_T(theta, dtheta, N, x, mu0=mu0)  # (N, *batch_shape)
    grad = grad_lg_basis(modes, u)             # (len(modes), N, *batch_shape)
    out = []
    for i in range(len(modes)):
        out.append(np.sum(grad[i] * du, axis=0))  # (*batch_shape,)
    return np.stack(out, axis=0)


def jac_feature(theta, N, x, modes, mu0=None):
    """Full theta-Jacobian of eval_feature at every point: shape
    (len(modes), P, *batch_shape), entry [i, q] = d(phi_i)/d(theta_q).

    This is the interface the exact (Golub-Pereyra) VarPro Jacobian
    variant consumes -- its second term has no reverse-mode collapse, so
    it needs the full uncontracted tensor. (The default Kaufman variant
    goes through vjp_feature instead; see docs/design-notes.md.)

    The Levenberg-Marquardt Jacobian needs all P coordinate directions at
    every trial theta, and the spatial LG gradients don't depend on the
    theta direction -- so they're computed once, in a single
    grad_lg_basis call, rather than once per direction as calling
    jvp_feature in a loop over P would do. What remains per-direction is
    cheap: dT/dtheta_q (P small jvp_T calls inside
    jacobian_tensor_forward) and the final grad . dT/dtheta_q contraction.
    """
    u = eval_T(theta, N, x, mu0=mu0)
    dU = jacobian_tensor_forward(theta, N, x, mu0=mu0)  # (N, P, *batch_shape)
    P = dU.shape[1]
    grad = grad_lg_basis(modes, u)  # (len(modes), N, *batch_shape)
    out = []
    for i in range(len(modes)):
        rows = [np.sum(grad[i] * dU[:, q], axis=0) for q in range(P)]
        out.append(np.stack(rows, axis=0))  # (P, *batch_shape)
    return np.stack(out, axis=0)


def vjp_feature(theta, N, x, w, modes, mu0=None):
    """Reverse-mode: given a cotangent w (shape (len(modes), *batch_shape),
    matching eval_feature's output), return <w, d(eval_feature)/dtheta> as
    an array of shape (P, *batch_shape). Combine every mode's spatial
    gradient weighted by its own cotangent entry into a single
    u-space cotangent, then push that through vjp_T -- the reverse-mode
    counterpart of jvp_feature's per-mode chain rule."""
    u = eval_T(theta, N, x, mu0=mu0)
    grad = grad_lg_basis(modes, u)  # (len(modes), N, *batch_shape)
    combined = np.zeros_like(u)  # (N, *batch_shape)
    for k in range(len(modes)):
        combined = combined + w[k] * grad[k]
    return vjp_T(theta, N, x, combined, mu0=mu0)
