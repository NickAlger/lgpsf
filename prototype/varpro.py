"""API sketch (not yet implemented) for the generic, mass-free VarPro
fitting core: fit the ellipsoid parameters theta of one mesh row by
Levenberg-Marquardt, eliminating the linear (smooth + extra) coefficients
at every trial theta via the whitened inner least-squares solve.

This module knows nothing about LG modes, ellipsoids, or mass matrices --
every problem-specific piece comes in as an array (already whitened, see
docs/varpro-whitening-notes.tex) or as a callable (the whitened smooth
basis and its derivatives, e.g. functools.partial over whitening.py's
functions, closed over this row's own N, x, row_mass, m2_diag, modes,
mu0). That is the actual point of the layering from last session: this
file is the piece that stays identical across rows, across different
choices of extra basis, and eventually across the C++ port.

Deliberately just the input/output contract for now -- fit_varpro raises
NotImplementedError. Not implementing the Levenberg-Marquardt loop or the
inner whitened least-squares solve tonight, per request; this is for
reviewing the shape of the interface first.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


@dataclass
class VarProOptions:
    """Tuning knobs for the outer Levenberg-Marquardt loop (passed through
    to scipy.optimize.least_squares, method="lm") and the inner whitened
    linear solve. Defaults are placeholders, not yet chosen carefully."""

    ridge: float = 1e-8
    """Ridge regularization added to the inner (whitened, already
    orthogonalized against e_hat) least-squares normal equations, after
    per-column equilibration -- guards against a smooth feature that's
    nearly unresolved on this row's mesh (lg-split-method-notes.tex)."""

    x_scale: Union[str, np.ndarray] = "jac"
    """Per-theta-component trust-region scaling for LM. "jac" (scipy's
    automatic column-norm scaling from the analytic Jacobian) is a
    reasonable default now that the Jacobian is analytic rather than
    finite-differenced; pass an array to override if some theta
    components are known to need different step scales (e.g. log-diagonal
    vs. off-diagonal Cholesky entries)."""

    max_nfev: int = 50
    """Cap on residual (theta trial) evaluations."""

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    """Convergence tolerances, scipy.optimize.least_squares convention:
    ftol on relative cost change, xtol on relative step size, gtol on
    gradient orthogonality to the residual."""


@dataclass
class VarProResult:
    """Output of a single row's VarPro fit."""

    theta: np.ndarray
    """(P,) fitted ellipsoid parameters."""

    c: np.ndarray
    """(n_modes,) fitted smooth (LG) basis coefficients at theta."""

    s: np.ndarray
    """(num_extra,) fitted extra-basis coefficients at theta -- e.g. the
    spike weight, if the caller's extra basis includes the diagonal
    one-hot."""

    residual: np.ndarray
    """(k,) whitened residual at the optimum, y_hat - model(theta, c, s)."""

    cost: float
    """0.5 * ||residual||^2 (scipy's convention)."""

    success: bool
    n_iterations: int
    n_function_evals: int
    message: str
    """Diagnostics from the underlying optimizer call."""

    jacobian: Optional[np.ndarray] = None
    """(k, P) reduced-residual Jacobian at the optimum, if requested --
    for future diagnostics (e.g. a cheap well-determinedness check), not
    needed for the fit itself."""


def fit_varpro(
    z_hat: np.ndarray,
    y_hat: np.ndarray,
    e_hat: np.ndarray,
    basis_eval: Callable[[np.ndarray], np.ndarray],
    basis_jac: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray,
    options: Optional[VarProOptions] = None,
) -> VarProResult:
    """Fit theta (and the linear coefficients c, s) for one mesh row.

    All array inputs are already whitened (docs/varpro-whitening-notes.tex,
    whitening.py); this function is mass-free and knows nothing about
    M1/M2, LG modes, or the ellipsoid parameterization -- everything
    problem-specific is either an array here or baked into the closures.

    Parameters
    ----------
    z_hat : (k, K) array
        Whitened random probe vectors, restricted to this row's
        neighbor/support batch. k = number of probes, K = batch size.
    y_hat : (k,) array
        This row's whitened response to each probe.
    e_hat : (num_extra, K) array
        Whitened extra (theta-independent) basis functions. num_extra may
        be 0 (an empty (0, K) array) if there is no extra basis at all.
    basis_eval : theta (P,) -> (n_modes, K)
        Whitened smooth-basis values at theta, e.g.
        functools.partial(whitened_eval_feature, N=N, x=x,
        row_mass=row_mass, m2_diag=m2_diag, modes=modes, mu0=mu0).
    basis_jac : theta (P,) -> (n_modes, P, K)
        Full theta-Jacobian of basis_eval, all P coordinate directions at
        once, e.g. functools.partial(whitened_jac_feature, ...). The
        Levenberg-Marquardt Jacobian needs every direction at each theta
        anyway, and only the feature layer can share the direction-
        independent work (each mode's spatial LG gradient) across them --
        so the interface asks for the jac, even though the jvp remains
        the primitive it's built from and tested against (decision
        2026-07-24, docs/design-notes.md). Reverse mode (the old
        basis_vjp) is not needed here at all: with the Jacobian explicit,
        the cost gradient is J^T r, a matvec on an already-built matrix.
        The vjp survives in the lower layers as a verification
        instrument, and the fitting-core tests use it directly from
        whitening.py for adjoint-consistency checks of the Jacobian
        machinery -- it just doesn't pass through this signature.
    theta_init : (P,) array
        Starting guess for the outer Levenberg-Marquardt iteration (e.g.
        from a smoothed-beta heuristic ellipsoid) -- domain-specific, so
        computing it is the caller's job, not this function's.
    options : VarProOptions, optional

    Returns
    -------
    VarProResult

    Deliberately out of scope here (a caller's decision, not VarPro's):
    whether to accept this fit over some baseline (e.g. a linear,
    fixed-Sigma fit) -- that needs held-out data this function never
    sees; and post-hoc bound checking (e.g. "reject if a fitted axis
    ended up 3x the initial guess") -- a diagnostic on the returned
    theta, not a constraint enforced during optimization (scipy's
    method="lm" doesn't support bounds; the research prototype only ever
    checked this after the fact too).
    """
    raise NotImplementedError(
        "API sketch only -- the Levenberg-Marquardt loop and the inner "
        "whitened least-squares solve (orthogonalize basis_eval's output "
        "against e_hat, solve for c and s, form the reduced residual) "
        "are not implemented yet."
    )
