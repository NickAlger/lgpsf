"""The generic, mass-free VarPro fitting core: fit the ellipsoid
parameters theta of one mesh row by Levenberg-Marquardt, eliminating the
linear (smooth + extra) coefficients at every trial theta via the
whitened inner least-squares solve.

This module knows nothing about LG modes, ellipsoids, or mass matrices --
every problem-specific piece comes in as an array (already whitened, see
docs/varpro-whitening-notes.tex) or as a callable (the whitened smooth
basis and its Jacobian, e.g. functools.partial over whitening.py's
functions, closed over this row's own N, x, row_mass, m2_diag, modes,
mu0). That is the actual point of the layering: this file is the piece
that stays identical across rows, across different choices of extra
basis, and eventually across the C++ port.

The mathematical shape (in the whitened variables): stack the k probes
into Z_hat, so the smooth basis at a trial theta gives a design matrix
A(theta) = Z_hat @ basis_eval(theta).T (k x n_modes), the extra basis a
constant block B = Z_hat @ e_hat.T (k x num_extra), and the row fit is

    min over theta, c, s of  || y_hat - A(theta) c - B s ||^2 .

VarPro = three ideas, each implemented as its own visible piece here:
 1. Projection: at fixed theta the optimal (c, s) is a linear solve, and
    the leftover ("reduced") residual is y_hat minus its projection onto
    the columns of [A(theta), B] -- so the outer optimizer only ever
    searches over theta (_inner_solve).
 2. The constant block B is projected out once, up front, not once per
    trial theta (Frisch-Waugh-Lovell: residualize y_hat and every A(theta)
    against an orthonormal basis of range(B); fit the smooth block alone;
    recover s by one small back-solve at the end). _orthonormal_range +
    _project_out.
 3. The reduced residual's Jacobian comes from differentiating the
    projector (Golub-Pereyra formula, with Kaufman's residual-order
    simplification as the cheap default).

Implementation status: the inner linear-algebra layer and the reduced
problem (residual + both Jacobian variants, _ReducedProblem) are
implemented and tested; fit_varpro itself (the outer Levenberg-Marquardt
wiring, the s back-solve, and result packaging) still raises
NotImplementedError.
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

    jacobian: str = "kaufman"
    """Which reduced-residual Jacobian the outer loop gets: "kaufman"
    (default) or "golub-pereyra". See _ReducedProblem.jacobian for the
    formulas and the exact relationship (both give the exact gradient;
    Kaufman's Gauss-Newton model is the Schur complement of the
    full-space GN Hessian, Golub-Pereyra's adds a PSD O(||r||^2) term
    and is the exact Jacobian of the reduced residual)."""

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


# --------------------------------------------------------------------------
# Inner linear-algebra layer: the "projection" in variable projection.
# Plain Euclidean everywhere -- whitening (whitening.py) already made that
# exactly correct, so nothing here knows about masses or geometry.
# --------------------------------------------------------------------------

def _project_out(Q, V):
    """V minus its projection onto the columns of Q: (I - Q Q^T) V.
    Q: (k, m) with orthonormal columns (m may be 0, in which case this is
    the identity); V: (k,) or (k, anything). Used three ways: residualizing
    y_hat and each trial A(theta) against the extra block's range, and (in
    the Jacobian) projecting onto range(A_tilde)'s orthogonal complement."""
    return V - Q @ (Q.T @ V)


def _orthonormal_range(B):
    """Orthonormal basis for range(B), as a (k, rank) matrix. Via SVD with
    a rank cutoff rather than plain QR, so exactly-dependent columns (e.g.
    a caller passing the same extra function twice) collapse instead of
    contaminating the basis; ports to Eigen's BDCSVD directly. B with zero
    columns (no extra basis at all) gives a (k, 0) result, and everything
    downstream treats projecting out nothing as the identity."""
    B = np.asarray(B, dtype=float)
    if B.shape[1] == 0:
        return np.zeros((B.shape[0], 0))
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    if s[0] == 0.0:
        return U[:, :0]
    tol = max(B.shape) * np.finfo(float).eps * s[0]
    return U[:, s > tol]


@dataclass
class _InnerSolve:
    """Everything the current trial theta's linear solve produced -- the
    coefficients and residual the outer loop needs, plus the factorization
    pieces the Jacobian reuses (so nothing is factored twice per theta)."""

    c: np.ndarray
    """(n_modes,) smooth coefficients: the (ridge-regularized) least-squares
    minimizer of ||y_tilde - A_tilde c||."""

    residual: np.ndarray
    """(k,) y_tilde - A_tilde c. With ridge=0 this is exactly the projection
    of y_tilde onto range(A_tilde)'s orthogonal complement -- the reduced
    residual r(theta) that VarPro's outer loop minimizes."""

    U: np.ndarray
    """(k, rank) orthonormal basis of range(A_tilde) (numerical rank, from
    the SVD's singular-value cutoff). The Jacobian's projector: both the
    Kaufman term (project dA c onto range's complement) and the second
    Golub-Pereyra term (via the pseudoinverse below) are built from this."""

    sigma: np.ndarray
    """(rank,) retained singular values of the *equilibrated* matrix."""

    Vt: np.ndarray
    """(rank, n_modes) retained right singular vectors (equilibrated)."""

    col_scale: np.ndarray
    """(n_modes,) column norms divided out before the SVD. Together with
    (U, sigma, Vt) this gives the pseudoinverse action the second
    Golub-Pereyra Jacobian term needs, with no second factorization:
    A_tilde = (U diag(sigma) Vt) diag(col_scale), so
    pinv(A_tilde)^T w = U diag(1/sigma) Vt (w / col_scale)."""


def _inner_solve(A_tilde, y_tilde, ridge):
    """The inner least-squares solve at one trial theta -- the "projection"
    in variable projection. A_tilde: (k, n_modes), already residualized
    against the extra block; y_tilde: (k,), likewise.

    One SVD powers everything: the (ridge-regularized) coefficient solve,
    the orthonormal range basis the Jacobian's projector needs, and rank
    detection for smooth features the probes can't distinguish. Columns are
    equilibrated to unit norm first, so `ridge` acts on singular values of
    a matrix whose scale is known (sigma ~ O(1)) rather than depending on
    the caller's units, and so the rank cutoff is meaningful.

    Note the deliberate asymmetry: `ridge` regularizes the *coefficients*
    (c stays bounded when a feature is nearly unresolved), but the residual
    returned is the plain model misfit y_tilde - A_tilde c at those
    coefficients, and the range basis U is cut off by numerical rank alone.
    With ridge > 0 the residual therefore differs from the exact projection
    by O(ridge) -- negligible at the default 1e-8, and the step-2 Jacobian
    tests pin down exactness against finite differences at ridge=0."""
    A_tilde = np.asarray(A_tilde, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)
    col_scale = np.linalg.norm(A_tilde, axis=0)
    col_scale = np.where(col_scale == 0.0, 1.0, col_scale)
    A_s = A_tilde / col_scale

    U, sigma, Vt = np.linalg.svd(A_s, full_matrices=False)
    sig_max = sigma[0] if sigma.size else 0.0
    tol = max(A_s.shape) * np.finfo(float).eps * sig_max
    keep = sigma > tol
    U, sigma, Vt = U[:, keep], sigma[keep], Vt[keep]

    c_scaled = Vt.T @ ((sigma / (sigma**2 + ridge)) * (U.T @ y_tilde))
    c = c_scaled / col_scale
    residual = y_tilde - A_tilde @ c
    return _InnerSolve(c=c, residual=residual, U=U, sigma=sigma, Vt=Vt,
                       col_scale=col_scale)


class _ReducedProblem:
    """The reduced (theta-only) nonlinear least-squares problem VarPro
    hands the outer optimizer: r(theta) = y_tilde - A_tilde(theta) c*(theta)
    with c* from _inner_solve, plus r's Jacobian by differentiating the
    projector.

    Holds everything that is fixed across trial thetas -- the probes, the
    extra block B = Z_hat E_hat^T and its orthonormal range Q_B, the
    residualized data y_tilde -- plus a one-entry cache of the latest
    theta's inner solve: optimizers call residual(theta) and
    jacobian(theta) separately but back to back at the same theta, and
    both need the same factorization.

    Jacobian formulas (A~ = A_tilde, P_perp = I - U U^T the projector onto
    range(A~)'s orthogonal complement, r the reduced residual, dA~_q the
    q-th theta-derivative of A~ -- which is just P_B_perp dA_q, since the
    extra block's projector doesn't move with theta):

        Golub-Pereyra (exact):  dr_q = -P_perp (dA~_q c) - pinv(A~)^T dA~_q^T r
        Kaufman:                dr_q = -P_perp (dA~_q c)

    How they relate (each verified numerically in test_varpro.py):
      - The dropped term is O(||r||): the variants coincide exactly at a
        zero-residual fit.
      - The dropped term's columns lie in range(A~), the kept term's in
        range(A~)^perp -- so the term is orthogonal to r, and both
        variants produce the SAME exact gradient J^T r. Kaufman changes
        the quadratic model only, never the descent direction's
        first-order information.
      - Consequently their Gauss-Newton models differ by the PSD matrix
        J2^T J2 = O(||r||^2): Kaufman's J^T J is exactly the Schur
        complement of the full-space Gauss-Newton Hessian (linearize,
        then eliminate the linear variables), Golub-Pereyra's is the
        Gauss-Newton Hessian of the reduced problem (eliminate, then
        linearize). The operations don't commute, by exactly this term.
    """

    def __init__(self, z_hat, y_hat, e_hat, basis_eval, basis_jac, ridge):
        self.z_hat = np.asarray(z_hat, dtype=float)
        self.y_hat = np.asarray(y_hat, dtype=float)
        e_hat = np.asarray(e_hat, dtype=float)
        self.B = self.z_hat @ e_hat.T                  # (k, num_extra)
        self.Q_B = _orthonormal_range(self.B)
        self.y_tilde = _project_out(self.Q_B, self.y_hat)
        self.basis_eval = basis_eval
        self.basis_jac = basis_jac
        self.ridge = ridge
        self._cached_theta = None
        self._cached_solve = None

    def design_matrix(self, theta):
        """A(theta) = Z_hat basis_eval(theta)^T, shape (k, n_modes) --
        before residualization (the back-solve for s needs the raw A)."""
        return self.z_hat @ self.basis_eval(theta).T

    def _solve_at(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self._cached_theta is None or not np.array_equal(theta, self._cached_theta):
            A_tilde = _project_out(self.Q_B, self.design_matrix(theta))
            self._cached_solve = _inner_solve(A_tilde, self.y_tilde, self.ridge)
            self._cached_theta = theta.copy()
        return self._cached_solve

    def residual(self, theta):
        """r(theta), shape (k,). What the outer loop minimizes half the
        squared norm of."""
        return self._solve_at(theta).residual

    def jacobian(self, theta, variant="kaufman"):
        """dr/dtheta, shape (k, P), by the formula selected in `variant`
        (see class docstring). Built column by column -- the loop over P is
        small and fixed, per the batch-only vectorization principle."""
        sol = self._solve_at(theta)
        dPhi = self.basis_jac(theta)                   # (n_modes, P, K)
        P = dPhi.shape[1]
        cols = []
        for q in range(P):
            dA_tilde = _project_out(self.Q_B, self.z_hat @ dPhi[:, q].T)
            col = -_project_out(sol.U, dA_tilde @ sol.c)
            if variant == "golub-pereyra":
                # -pinv(A~)^T dA~_q^T r, from the stored SVD pieces:
                # pinv(A~)^T w = U diag(1/sigma) Vt (w / col_scale)
                w = dA_tilde.T @ sol.residual          # (n_modes,)
                col = col - sol.U @ ((sol.Vt @ (w / sol.col_scale)) / sol.sigma)
            elif variant != "kaufman":
                raise ValueError(f"unknown jacobian variant: {variant!r}")
            cols.append(col)
        return np.stack(cols, axis=1)


def fit_varpro(
    z_hat: np.ndarray,
    y_hat: np.ndarray,
    basis_eval: Callable[[np.ndarray], np.ndarray],
    basis_jac: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray,
    e_hat: Optional[np.ndarray] = None,
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
    e_hat : (num_extra, K) array, optional
        Whitened extra (theta-independent) basis functions, e.g. the
        diagonal spike. None (the default) means no extra basis at all --
        equivalent to passing an empty (0, K) array; the returned s is
        then empty.
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
        "The inner linear-algebra layer and the reduced problem "
        "(_ReducedProblem: residual + both Jacobian variants) are "
        "implemented and tested, but the outer Levenberg-Marquardt "
        "wiring, the s back-solve, and result packaging are not."
    )
