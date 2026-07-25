"""The noise-whitening interface layer: the only place M1 (row mass) and M2
(column mass) appear anywhere in this codebase. See
docs/varpro-whitening-notes.tex for the full derivation.

(Vocabulary note: "row"/"row_mass" reflect the operator-row application
the derivation was written for -- fitting one row of H = M1 Phi M2. The
math only requires a target dof with a mass; higher layers (probe_fit)
use the general "target function known through probe inner products"
framing and route their masses through here unchanged.)

Everything below is per-row: row_mass is a single scalar (M1)_rho,rho for
whichever row is currently being fit; m2_diag is the array of column masses
(M2)_jj for the row's local neighbor/support batch, same shape as the batch
(matches x's trailing axes).

    z_hat = whiten_probes(z, m2_diag)                     = M2^(1/2) z
    y_hat = whiten_data(y, row_mass)                       = y / sqrt(row_mass)
    E_hat = whiten_extra(E, row_mass, m2_diag)             = sqrt(row_mass) M2^(-1/2) E
    phi_hat = whitened_eval_feature(..., row_mass, m2_diag) = sqrt(row_mass) M2^(1/2) phi

so that y_hat = sum_i c_i (phi_hat_i . z_hat) + sum_d s_d (E_hat_d . z_hat)
is an ordinary (mass-free) linear model, and orthogonalizing phi_hat against
E_hat with the plain Euclidean inner product is exactly correct.

The whitened basis and its JVP/VJP wrap lg_ellipsoid_feature.py (itself
mass-free) with sqrt(row_mass) * M2^(+-1/2), a fixed, theta-independent,
symmetric operator -- so it composes with the existing JVP/VJP chain with
no new derivative math: JVP gets the operator applied to the raw JVP; VJP
gets the operator applied to the incoming cotangent first (symmetric, so
it's the same operator on both sides), before the existing VJP chain runs.
"""
import numpy as np

from lg_ellipsoid_feature import eval_feature, jvp_feature, vjp_feature, jac_feature


def whiten_probes(z, m2_diag):
    """z_hat = M2^(1/2) z. z, m2_diag: same shape (the row's batch)."""
    return np.sqrt(m2_diag) * z


def whiten_data(y, row_mass):
    """y_hat = y / sqrt(row_mass). y: any shape; row_mass: scalar."""
    return y / np.sqrt(row_mass)


def whiten_extra(E, row_mass, m2_diag):
    """E_hat = sqrt(row_mass) * M2^(-1/2) E. E: shape (num_extra, *batch_shape);
    m2_diag: shape (*batch_shape,); row_mass: scalar."""
    return np.sqrt(row_mass) * E / np.sqrt(m2_diag)


def whitened_eval_feature(theta, N, x, row_mass, m2_diag, modes, mu0=None):
    """phi_hat_i(theta) = sqrt(row_mass) * M2^(1/2) * phi_i(x; theta).
    Returns an array of shape (len(modes), *batch_shape)."""
    raw = eval_feature(theta, N, x, modes, mu0=mu0)
    return np.sqrt(row_mass) * np.sqrt(m2_diag) * raw


def whitened_jvp_feature(theta, dtheta, N, x, row_mass, m2_diag, modes, mu0=None):
    """Directional derivative of whitened_eval_feature w.r.t. theta in
    direction dtheta, at fixed x. The whitening operator is theta-independent,
    so this is just the operator applied to jvp_feature's raw output."""
    raw = jvp_feature(theta, dtheta, N, x, modes, mu0=mu0)
    return np.sqrt(row_mass) * np.sqrt(m2_diag) * raw


def whitened_jac_feature(theta, N, x, row_mass, m2_diag, modes, mu0=None):
    """Full theta-Jacobian of whitened_eval_feature: shape
    (len(modes), P, *batch_shape), entry [i, q] = d(phi_hat_i)/d(theta_q).
    This is the derivative interface the VarPro fitting core consumes (see
    jac_feature for why the jac, not the jvp, is the interface). The
    whitening operator is theta-independent, so it's just the operator
    applied to jac_feature's raw output -- the m2_diag scaling broadcasts
    over the trailing batch axes, indifferent to the extra P axis."""
    raw = jac_feature(theta, N, x, modes, mu0=mu0)
    return np.sqrt(row_mass) * np.sqrt(m2_diag) * raw


def whitened_basis(N, x, row_mass, m2_diag, modes, mu0=None):
    """Convenience closure builder: the whitened basis callables for one
    target, in the exact shapes the fitting layers consume --
    (eval(theta) -> (n_modes, K), vjp(theta, w_hat) -> (P, K),
    jac(theta) -> (n_modes, P, K)). Kills the per-caller
    functools.partial boilerplate."""
    common = dict(N=N, x=x, row_mass=row_mass, m2_diag=m2_diag,
                  modes=modes, mu0=mu0)

    def b_eval(theta):
        return whitened_eval_feature(theta, **common)

    def b_vjp(theta, w_hat):
        return whitened_vjp_feature(theta, w_hat=w_hat, **common)

    def b_jac(theta):
        return whitened_jac_feature(theta, **common)

    return b_eval, b_vjp, b_jac


def whitened_vjp_feature(theta, N, x, row_mass, m2_diag, w_hat, modes, mu0=None):
    """Reverse-mode: given a cotangent w_hat matching
    whitened_eval_feature's output (shape (len(modes), *batch_shape)),
    return <w_hat, d(whitened_eval_feature)/dtheta>, shape (P, *batch_shape).
    The whitening operator is symmetric, so apply it to the cotangent first,
    then reuse the existing (raw) VJP chain unchanged."""
    w = np.sqrt(row_mass) * np.sqrt(m2_diag) * w_hat
    return vjp_feature(theta, N, x, w, modes, mu0=mu0)
