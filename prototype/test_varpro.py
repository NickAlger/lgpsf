"""Tests for varpro.py's fitting machinery below the outer loop.

Step 1 -- the inner linear-algebra layer, pure linear algebra on random
matrices, each structured path checked against a brute-force reference
(numpy lstsq on the full/joint/augmented systems):

  - _inner_solve at ridge=0 vs plain lstsq (coefficients, residual,
    residual-orthogonal-to-range, range basis).
  - _inner_solve at ridge>0 vs the augmented least-squares system
    [A_s; sqrt(ridge) I] -- an independent formula for the same ridge
    solution.
  - Rank deficiency: SVD cutoff finds the right rank and still produces
    the exact least-squares residual.
  - The Frisch-Waugh-Lovell path (orthonormalize the extra block, project
    it out of everything, fit the smooth block alone, back-solve for s)
    vs the joint fit over [A, B] -- coefficient-by-coefficient and
    residual-by-residual. This is the identity that lets fit_varpro handle
    the extra basis entirely in preprocessing.

Step 2 -- the reduced problem (_ReducedProblem), exercised through the
real whitened LG feature closures (functools.partial over whitening.py),
i.e. exactly the callables fit_varpro will receive:

  - The Golub-Pereyra Jacobian vs central finite differences of the
    reduced residual -- the strict test that the projector-differentiation
    formula, including the theta-dependence of the inner solve's c*, is
    exact (at ridge=0, where the residual is exactly the projection).
  - The structural identities relating Kaufman to Golub-Pereyra: same
    gradient J^T r, Kaufman's columns in range(A~)^perp, the dropped term's
    columns in range(A~), and exact coincidence at a zero-residual fit.
  - Adjoint consistency of the gradient: J^T r (via the built Jacobian)
    vs an independent reverse-mode contraction through
    whitened_vjp_feature -- the house check that has caught real
    sign/transpose bugs at every other layer of this project.
  - The one-sweep reverse-mode Kaufman build (one batched vjp with
    cotangent w_hat[i,j] = c_i) vs an inline forward-mode reference built
    column by column from the full basis_jac tensor, to machine
    precision; plus the basis_jac-is-optional contract (default variant
    needs none; golub-pereyra without one is a clear error).
  - The one-entry cache: residual(theta) then jacobian(theta) at the same
    theta triggers exactly one basis evaluation / inner solve.

Step 3 -- fit_varpro end to end:

  - Exact recovery: y_hat built from the model at a known
    (theta*, c*, s*), fit started from a perturbed theta -- recovers all
    three to tight tolerance with near-zero cost, for both Jacobian
    variants, and with and without an extra basis.
  - Result packaging self-consistency on a noisy (no-exact-fit) problem:
    the returned residual/cost recompute from (theta, c, s); the
    back-solved s leaves the residual orthogonal to the extra block; the
    final cost does not exceed the starting cost.
  - Input validation: golub-pereyra without basis_jac, and fewer probes
    than theta parameters, fail fast with clear errors.

Run directly (`python test_varpro.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import theta_size
from harmonic_polynomials import num_harmonics
from whitening import whiten_extra, whitened_basis
from varpro import (
    VarProOptions,
    _ReducedProblem,
    _inner_solve,
    _orthonormal_range,
    _project_out,
    fit_varpro,
)


def test_project_out():
    rng = np.random.default_rng(0)
    k = 12
    Q = _orthonormal_range(rng.standard_normal((k, 3)))
    V = rng.standard_normal((k, 5))

    W = _project_out(Q, V)
    assert np.max(np.abs(Q.T @ W)) < 1e-12          # lands in range(Q)^perp
    assert np.allclose(_project_out(Q, W), W)        # idempotent

    v = rng.standard_normal(k)                       # plain vectors too
    assert np.max(np.abs(Q.T @ _project_out(Q, v))) < 1e-12

    Q_empty = np.zeros((k, 0))                       # no extra basis at all
    assert np.allclose(_project_out(Q_empty, V), V)


def test_orthonormal_range():
    rng = np.random.default_rng(1)
    k = 10
    B = rng.standard_normal((k, 3))

    Q = _orthonormal_range(B)
    assert Q.shape == (k, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.allclose(Q @ (Q.T @ B), B)             # range preserved

    # exactly-dependent columns collapse to the true rank
    B_deficient = np.column_stack([B[:, 0], 2.0 * B[:, 0], B[:, 1]])
    Q2 = _orthonormal_range(B_deficient)
    assert Q2.shape == (k, 2)
    assert np.allclose(Q2 @ (Q2.T @ B_deficient), B_deficient)

    assert _orthonormal_range(np.zeros((k, 0))).shape == (k, 0)


def test_inner_solve_matches_lstsq():
    rng = np.random.default_rng(2)
    k, n = 30, 8
    # uneven column scales, to make sure equilibration is exercised (and
    # undone) rather than a no-op
    A = rng.standard_normal((k, n)) * rng.uniform(0.05, 20.0, size=n)
    y = rng.standard_normal(k)

    sol = _inner_solve(A, y, ridge=0.0)
    c_ref = np.linalg.lstsq(A, y, rcond=None)[0]

    assert np.allclose(sol.c, c_ref, rtol=1e-9, atol=1e-12)
    assert np.allclose(sol.residual, y - A @ c_ref, rtol=1e-9, atol=1e-12)
    # the defining property of the projection: residual orthogonal to
    # every column of A
    assert np.max(np.abs(A.T @ sol.residual)) < 1e-9
    # range basis: orthonormal, full rank here, spans range(A)
    assert sol.U.shape == (k, n)
    assert np.allclose(sol.U.T @ sol.U, np.eye(n), atol=1e-12)
    assert np.allclose(sol.U @ (sol.U.T @ A), A)


def test_inner_solve_ridge_matches_augmented_system():
    """Independent formula for the same object: the ridge solution on the
    equilibrated matrix solves the augmented least-squares problem
    [A_s; sqrt(ridge) I] c_s ~ [y; 0]."""
    rng = np.random.default_rng(3)
    k, n = 25, 6
    A = rng.standard_normal((k, n)) * rng.uniform(0.05, 20.0, size=n)
    y = rng.standard_normal(k)
    ridge = 1e-3  # big enough that forgetting it would fail loudly

    sol = _inner_solve(A, y, ridge=ridge)

    col_scale = np.linalg.norm(A, axis=0)
    A_aug = np.vstack([A / col_scale, np.sqrt(ridge) * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    c_ref = np.linalg.lstsq(A_aug, y_aug, rcond=None)[0] / col_scale

    assert np.allclose(sol.c, c_ref, rtol=1e-9, atol=1e-12)
    # residual is the plain model misfit at the ridge coefficients
    assert np.allclose(sol.residual, y - A @ sol.c, rtol=1e-9, atol=1e-12)


def test_inner_solve_rank_deficient():
    rng = np.random.default_rng(4)
    k, n = 20, 5
    A = rng.standard_normal((k, n))
    A[:, 3] = 2.0 * A[:, 1]  # exact rank deficiency
    y = rng.standard_normal(k)

    sol = _inner_solve(A, y, ridge=0.0)
    assert sol.U.shape[1] == n - 1  # numerical rank found

    # the residual is still the exact least-squares residual: projection
    # doesn't care which of the dependent columns "gets" the coefficient
    c_ref = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(sol.residual, y - A @ c_ref, rtol=1e-9, atol=1e-12)
    assert np.max(np.abs(A.T @ sol.residual)) < 1e-9


def test_fwl_residualize_then_fit_matches_joint_fit():
    """The Frisch-Waugh-Lovell identity, which is what lets fit_varpro
    handle the theta-independent extra block entirely in preprocessing:
    project range(B) out of y and A once, fit the smooth block alone,
    back-solve for s -- and get exactly the joint minimizer over [A, B],
    same c, same s, same residual."""
    rng = np.random.default_rng(5)
    k, n = 40, 7
    for num_extra in [0, 1, 2]:
        A = rng.standard_normal((k, n))
        B = rng.standard_normal((k, num_extra))
        y = rng.standard_normal(k)

        # brute-force joint fit
        coeff = np.linalg.lstsq(np.column_stack([A, B]), y, rcond=None)[0]
        c_joint, s_joint = coeff[:n], coeff[n:]
        r_joint = y - A @ c_joint - B @ s_joint

        # FWL path, as fit_varpro will do it
        Q_B = _orthonormal_range(B)
        y_tilde = _project_out(Q_B, y)
        A_tilde = _project_out(Q_B, A)
        sol = _inner_solve(A_tilde, y_tilde, ridge=0.0)
        s = np.linalg.lstsq(B, y - A @ sol.c, rcond=None)[0]

        assert np.allclose(sol.c, c_joint, rtol=1e-8, atol=1e-10), num_extra
        assert np.allclose(s, s_joint, rtol=1e-8, atol=1e-10), num_extra
        assert np.allclose(sol.residual, r_joint, rtol=1e-8, atol=1e-10), num_extra


# --------------------------------------------------------------------------
# Step 2: the reduced problem, through the real whitened LG feature closures.
# --------------------------------------------------------------------------

def _random_theta(N, mu0, rng):
    P = theta_size(N, mu0)
    theta = rng.uniform(-0.3, 0.3, size=P)
    idx = N if mu0 is None else 0
    theta[idx:idx + N] = rng.uniform(-0.3, 0.3, size=N)
    return theta


def _some_modes(N, max_ell=2, max_p=1):
    modes = []
    for ell in range(max_ell + 1):
        for m in range(num_harmonics(N, ell)):
            for p in range(max_p + 1):
                modes.append((p, ell, m))
    return modes


def _make_problem(rng, N=2, mu0=None, num_extra=1, k=24, K=16):
    """One synthetic row, wired exactly the way fit_varpro will wire it:
    whitened probes/extra basis as arrays, the whitened smooth basis and
    its Jacobian as partial-applied closures over the row's own data."""
    modes = _some_modes(N)
    theta = _random_theta(N, mu0, rng)
    row_mass = rng.uniform(0.5, 2.0)
    x = rng.uniform(-1.5, 1.5, size=(N, K))
    m2_diag = rng.uniform(0.5, 2.0, size=K)
    z_hat = rng.standard_normal((k, K))

    E = np.zeros((num_extra, K))
    for d in range(num_extra):
        E[d, d] = 1.0  # one-hot spikes at the first num_extra batch points
    e_hat = whiten_extra(E, row_mass, m2_diag)

    basis = whitened_basis(N, x, row_mass, m2_diag, modes, mu0=mu0)

    return dict(
        theta=theta,
        P=theta_size(N, mu0),
        z_hat=z_hat,
        e_hat=e_hat,
        basis=basis,
        k=k,
    )


class _ValuesAndVjpOnly:
    """A basis evaluation exposing only values() and vjp() -- the minimal
    contract the default (Kaufman) path is allowed to need."""

    def __init__(self, at):
        self._at = at

    def values(self):
        return self._at.values()

    def vjp(self, w_hat):
        return self._at.vjp(w_hat)


def _without_jac(basis):
    return lambda theta: _ValuesAndVjpOnly(basis(theta))


def test_reduced_jacobian_golub_pereyra_matches_fd():
    """The strict exactness test: the Golub-Pereyra formula must reproduce
    central finite differences of the reduced residual -- which sees the
    full theta-dependence, c*(theta) and projector included. Random y_hat,
    so the residual is large and the second (residual-proportional) term
    genuinely participates: Kaufman alone could not pass this."""
    rng = np.random.default_rng(6)
    fd_step = 1e-6
    for N, fit_mu in [(1, True), (2, True), (2, False)]:
        mu0 = None if fit_mu else rng.uniform(-1, 1, size=N)
        prob = _make_problem(rng, N=N, mu0=mu0)
        y_hat = rng.standard_normal(prob["k"])
        rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                             prob["basis"], ridge=0.0)
        theta = prob["theta"]

        J = rp.jacobian(theta, variant="golub-pereyra")
        scale = max(1.0, np.max(np.abs(J)))
        for q in range(prob["P"]):
            theta_p, theta_m = theta.copy(), theta.copy()
            theta_p[q] += fd_step
            theta_m[q] -= fd_step
            fd = (rp.residual(theta_p) - rp.residual(theta_m)) / (2 * fd_step)
            err = np.max(np.abs(J[:, q] - fd)) / scale
            assert err < 1e-5, f"N={N} mu0={mu0} q={q}: rel err={err:.3e}"


def test_kaufman_vs_golub_pereyra_structure():
    """The structural identities from the Hessian discussion, numerically:
    Kaufman's columns live in range(A~)^perp, the dropped term's columns in
    range(A~) (so the two ranges are orthogonal), and -- the consequence
    that matters for optimization -- both variants give the SAME gradient
    J^T r. Kaufman never changes first-order information."""
    rng = np.random.default_rng(7)
    prob = _make_problem(rng, N=2)
    y_hat = rng.standard_normal(prob["k"])  # large residual, so the term matters
    rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                         prob["basis"], ridge=0.0)
    theta = prob["theta"]

    J_K = rp.jacobian(theta, variant="kaufman")
    J_GP = rp.jacobian(theta, variant="golub-pereyra")
    r = rp.residual(theta)
    U = rp._solve_at(theta).U

    diff = J_GP - J_K
    assert np.max(np.abs(diff)) > 1e-3            # the term is genuinely nonzero here
    assert np.max(np.abs(U.T @ J_K)) < 1e-10      # Kaufman: in range(A~)^perp
    assert np.max(np.abs(_project_out(U, diff))) < 1e-10  # dropped term: in range(A~)
    assert np.allclose(J_K.T @ r, J_GP.T @ r, rtol=1e-9, atol=1e-11)  # same gradient


def test_variants_coincide_at_zero_residual():
    """Build y_hat exactly from the model at theta (a perfect fit), so the
    reduced residual vanishes there -- and with it the entire difference
    between the two Jacobian variants."""
    rng = np.random.default_rng(8)
    prob = _make_problem(rng, N=2)
    theta = prob["theta"]
    A0 = prob["z_hat"] @ prob["basis"](theta).values().T
    B = prob["z_hat"] @ prob["e_hat"].T
    c_true = rng.uniform(-1, 1, size=A0.shape[1])
    s_true = rng.uniform(-1, 1, size=B.shape[1])
    y_hat = A0 @ c_true + B @ s_true

    rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                         prob["basis"], ridge=0.0)
    r = rp.residual(theta)
    assert np.max(np.abs(r)) < 1e-9 * max(1.0, np.max(np.abs(y_hat)))

    J_K = rp.jacobian(theta, variant="kaufman")
    J_GP = rp.jacobian(theta, variant="golub-pereyra")
    assert np.max(np.abs(J_K - J_GP)) < 1e-8 * max(1.0, np.max(np.abs(J_K)))


def test_reduced_gradient_adjoint_consistency():
    """The house adjoint check, at the reduced-problem level. Forward mode:
    g = J^T r with the built Jacobian (either variant -- they agree). Reverse
    mode, independently: g_q = -(dA_q c)^T r contracts to a single
    whitened_vjp_feature call with cotangent w[i,j] = c_i (Z_hat^T r)_j,
    summed over the batch. Two entirely different code paths through the
    derivative machinery must meet at the same P numbers."""
    rng = np.random.default_rng(9)
    for fit_mu in [True, False]:
        mu0 = None if fit_mu else rng.uniform(-1, 1, size=2)
        prob = _make_problem(rng, N=2, mu0=mu0)
        y_hat = rng.standard_normal(prob["k"])
        rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                             prob["basis"], ridge=0.0)
        theta = prob["theta"]

        r = rp.residual(theta)
        g_fwd_K = rp.jacobian(theta, variant="kaufman").T @ r
        g_fwd_GP = rp.jacobian(theta, variant="golub-pereyra").T @ r

        c = rp._solve_at(theta).c
        w_feat = c[:, None] * (prob["z_hat"].T @ r)[None, :]  # (n_modes, K)
        g_rev = -np.sum(prob["basis"](theta).vjp(w_feat), axis=1)

        assert np.allclose(g_fwd_K, g_rev, rtol=1e-8, atol=1e-10), mu0
        assert np.allclose(g_fwd_GP, g_rev, rtol=1e-8, atol=1e-10), mu0


def test_kaufman_reverse_mode_matches_jac_built():
    """The one-sweep reverse-mode identity behind the default Jacobian:
    G = basis_vjp with cotangent w_hat[i,j] = c_i, then
    J_K = -P_perp(P_B_perp(Z_hat G^T)), must equal the forward path
    -P_perp(dA~_q c) built column by column from the full basis_jac
    tensor -- reimplemented inline here as an independent reference, so
    the production code path and the reference share nothing but the
    feature layer underneath. Also: the default variant must work with no
    basis_jac at all, and asking for golub-pereyra without one must be a
    clear error, not a crash."""
    rng = np.random.default_rng(11)
    for fit_mu in [True, False]:
        mu0 = None if fit_mu else rng.uniform(-1, 1, size=2)
        prob = _make_problem(rng, N=2, mu0=mu0)
        y_hat = rng.standard_normal(prob["k"])
        # a basis WITHOUT jac(): the default path must not need it
        rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                             _without_jac(prob["basis"]), ridge=0.0)
        theta = prob["theta"]

        J_K = rp.jacobian(theta)  # default variant, reverse-mode build

        sol = rp._solve_at(theta)
        dPhi = prob["basis"](theta).jac()  # (n_modes, P, K)
        cols = []
        for q in range(prob["P"]):
            dA_tilde = _project_out(rp.Q_B, prob["z_hat"] @ dPhi[:, q].T)
            cols.append(-_project_out(sol.U, dA_tilde @ sol.c))
        J_ref = np.stack(cols, axis=1)

        scale = max(1.0, np.max(np.abs(J_ref)))
        assert np.max(np.abs(J_K - J_ref)) / scale < 1e-12, mu0

        try:
            rp.jacobian(theta, variant="golub-pereyra")
            assert False, "expected ValueError for golub-pereyra without jac()"
        except ValueError:
            pass


def test_inner_solve_cached_between_residual_and_jacobian():
    """residual(theta) then jacobian(theta) at the same theta must reuse
    one basis evaluation / inner solve; a new theta must trigger a fresh
    one. (This is the contract that makes handing residual and jacobian to
    scipy as separate callables not cost double.)"""
    rng = np.random.default_rng(10)
    prob = _make_problem(rng, N=2)
    y_hat = rng.standard_normal(prob["k"])
    n_evals = {"count": 0}

    def counting_basis(theta):
        n_evals["count"] += 1
        return prob["basis"](theta)

    rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                         counting_basis, ridge=0.0)
    theta = prob["theta"]

    rp.residual(theta)
    rp.jacobian(theta)
    assert n_evals["count"] == 1

    theta2 = theta.copy()
    theta2[0] += 0.01
    rp.residual(theta2)
    assert n_evals["count"] == 2


def test_fit_varpro_recovers_synthetic_row():
    """The end-to-end check: build y_hat exactly from the row model at a
    known (theta*, c*, s*), start LM from a perturbed theta, and recover
    everything. Runs with both Jacobian variants (the Kaufman/GP
    distinction vanishes as the residual -> 0, so both must converge to
    the same answer) and without any extra basis at all (e_hat=None)."""
    rng = np.random.default_rng(12)
    for variant in ["kaufman", "golub-pereyra"]:
        for use_extra in [True, False]:
            prob = _make_problem(rng, N=2, num_extra=1)
            theta_true = prob["theta"]
            A0 = prob["z_hat"] @ prob["basis"](theta_true).values().T
            c_true = rng.uniform(-1, 1, size=A0.shape[1])
            y_hat = A0 @ c_true
            e_hat = None
            s_true = np.zeros(0)
            if use_extra:
                e_hat = prob["e_hat"]
                B = prob["z_hat"] @ e_hat.T
                s_true = rng.uniform(-1, 1, size=B.shape[1])
                y_hat = y_hat + B @ s_true

            theta_init = theta_true + rng.uniform(-0.05, 0.05, size=prob["P"])
            result = fit_varpro(
                prob["z_hat"], y_hat,
                prob["basis"],
                theta_init,
                e_hat=e_hat,
                options=VarProOptions(ridge=0.0, jacobian=variant),
            )

            label = f"variant={variant} extra={use_extra}"
            assert result.success, f"{label}: {result.message}"
            assert np.max(np.abs(result.theta - theta_true)) < 1e-6, label
            assert np.max(np.abs(result.c - c_true)) < 1e-6, label
            assert result.s.shape == s_true.shape, label
            assert np.max(np.abs(result.s - s_true), initial=0.0) < 1e-6, label
            assert result.cost < 1e-14, f"{label}: cost={result.cost:.3e}"


def test_fit_varpro_result_self_consistency():
    """On a noisy problem with no exact fit, the packaging invariants must
    hold regardless of where LM stopped: the returned residual and cost
    recompute exactly from (theta, c, s); the back-solved s makes the
    residual orthogonal to the extra block (its normal equations); and the
    final cost does not exceed the starting cost (LM is monotone)."""
    rng = np.random.default_rng(13)
    prob = _make_problem(rng, N=2, num_extra=2)
    y_hat = rng.standard_normal(prob["k"])
    theta_init = prob["theta"]

    result = fit_varpro(
        prob["z_hat"], y_hat,
        prob["basis"],
        theta_init,
        e_hat=prob["e_hat"],
    )  # default options: kaufman, ridge=1e-8

    A = prob["z_hat"] @ prob["basis"](result.theta).values().T
    B = prob["z_hat"] @ prob["e_hat"].T
    r_recomputed = y_hat - A @ result.c - B @ result.s
    assert np.allclose(r_recomputed, result.residual, rtol=1e-9, atol=1e-11)
    assert abs(result.cost - 0.5 * (r_recomputed @ r_recomputed)) < 1e-11
    assert np.max(np.abs(B.T @ result.residual)) < 1e-8

    rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                         prob["basis"], ridge=1e-8)
    cost_init = 0.5 * float(np.sum(rp.residual(theta_init) ** 2))
    assert result.cost <= cost_init + 1e-12


def test_fit_varpro_callback_traces_iterates():
    """The callback traces the outer iteration path: first recorded theta
    is theta_init, last is exactly result.theta (scipy's final Jacobian
    evaluation at the solution), the recorded costs are non-increasing
    (rejected trial steps never evaluate the Jacobian, so they never fire
    the callback), and the last recorded cost matches result.cost."""
    rng = np.random.default_rng(15)
    prob = _make_problem(rng, N=2)
    theta_true = prob["theta"]
    A0 = prob["z_hat"] @ prob["basis"](theta_true).values().T
    c_true = rng.uniform(-1, 1, size=A0.shape[1])
    y_hat = A0 @ c_true
    theta_init = theta_true + rng.uniform(-0.05, 0.05, size=prob["P"])

    history = []
    result = fit_varpro(
        prob["z_hat"], y_hat,
        prob["basis"],
        theta_init,
        options=VarProOptions(ridge=0.0),
        callback=lambda th, c, r: history.append((th, c, 0.5 * float(r @ r))),
    )

    assert result.success
    assert len(history) >= 2
    assert np.array_equal(history[0][0], theta_init)
    assert np.array_equal(history[-1][0], result.theta)
    costs = [h[2] for h in history]
    assert all(costs[i + 1] <= costs[i] + 1e-14 for i in range(len(costs) - 1))
    assert abs(costs[-1] - result.cost) < 1e-12
    # shapes of the snapshots
    assert all(h[0].shape == (prob["P"],) for h in history)
    assert all(h[1].shape == (A0.shape[1],) for h in history)


def test_reduced_problem_survives_overflow_theta():
    """A trial theta with tiny log-Cholesky diagonals makes the pullback
    huge and the LG evaluation overflow to nan. The reduced problem must
    score such a point as 'model contributes nothing' (residual = y_tilde,
    the worst finite cost) rather than crash in the SVD -- this is how the
    outer optimizer gets to reject a wild trial step and back off."""
    rng = np.random.default_rng(16)
    prob = _make_problem(rng, N=2)
    y_hat = rng.standard_normal(prob["k"])
    rp = _ReducedProblem(prob["z_hat"], y_hat, prob["e_hat"],
                         prob["basis"], ridge=0.0)

    theta_bad = prob["theta"].copy()
    theta_bad[2] = theta_bad[3] = -400.0  # log-diagonals: L ~ e^-400
    with np.errstate(over="ignore", invalid="ignore"):
        r_bad = rp.residual(theta_bad)
    assert np.all(np.isfinite(r_bad))
    assert np.array_equal(r_bad, rp.y_tilde)

    # ...and the cache moves on: a sane theta afterwards solves normally
    r_ok = rp.residual(prob["theta"])
    assert np.all(np.isfinite(r_ok))
    assert np.linalg.norm(r_ok) < np.linalg.norm(r_bad)


def test_fit_varpro_input_validation():
    rng = np.random.default_rng(14)
    prob = _make_problem(rng, N=2)
    y_hat = rng.standard_normal(prob["k"])

    # golub-pereyra on a basis without jac() must fail fast, before optimizing
    try:
        fit_varpro(prob["z_hat"], y_hat, _without_jac(prob["basis"]),
                   prob["theta"], e_hat=prob["e_hat"],
                   options=VarProOptions(jacobian="golub-pereyra"))
        assert False, "expected ValueError"
    except ValueError:
        pass

    # fewer probes than theta parameters: method='lm' can't run
    few = 2
    try:
        fit_varpro(prob["z_hat"][:few], y_hat[:few], prob["basis"],
                   prob["theta"])
        assert False, "expected ValueError"
    except ValueError:
        pass

    # a theta_init that overflows the basis fails fast with a clear error
    theta_bad = prob["theta"].copy()
    theta_bad[2] = theta_bad[3] = -400.0
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            fit_varpro(prob["z_hat"], y_hat, prob["basis"], theta_bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_project_out()
    test_orthonormal_range()
    test_inner_solve_matches_lstsq()
    test_inner_solve_ridge_matches_augmented_system()
    test_inner_solve_rank_deficient()
    test_fwl_residualize_then_fit_matches_joint_fit()
    test_reduced_jacobian_golub_pereyra_matches_fd()
    test_kaufman_vs_golub_pereyra_structure()
    test_variants_coincide_at_zero_residual()
    test_reduced_gradient_adjoint_consistency()
    test_kaufman_reverse_mode_matches_jac_built()
    test_inner_solve_cached_between_residual_and_jacobian()
    test_fit_varpro_recovers_synthetic_row()
    test_fit_varpro_result_self_consistency()
    test_fit_varpro_callback_traces_iterates()
    test_reduced_problem_survives_overflow_theta()
    test_fit_varpro_input_validation()
    print("all varpro checks passed")
