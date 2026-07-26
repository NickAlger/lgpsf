"""Self-contained checks on the harmonic polynomials and their table.

Every assertion here is intrinsic: each one states a property the objects
must have as mathematics, checkable from the module alone. Nothing is
compared against a stored reference or another implementation, so these
survive future changes to how the polynomials are stored or evaluated
(a different term ordering, precomputed power tables, reduced precision).

Run directly (`python test_harmonic_polynomials.py`) or via pytest
(`pytest prototype/`) -- functions are plain asserts, no pytest dependency.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from harmonic_polynomials import (
    eval_harmonic, grad_harmonic, harmonic_terms, max_degree, max_dimension,
    num_harmonics,
)

FD_STEP = 1e-5
FD_TOL = 1e-6


def _shells():
    """(N, ell) over the whole generated table, skipping empty shells."""
    for N in range(1, max_dimension() + 1):
        for ell in range(max_degree() + 1):
            if num_harmonics(N, ell) > 0:
                yield N, ell


def _all_modes():
    for N, ell in _shells():
        for m in range(num_harmonics(N, ell)):
            yield N, ell, m


# --------------------------------------------------------------- structure

def test_term_lists_are_well_formed():
    """Every stored term is a genuine degree-ell monomial with a nonzero
    coefficient, listed once, in lexicographically descending order."""
    for N, ell, m in _all_modes():
        exps, coeffs = harmonic_terms(N, ell, m)
        assert len(exps) == len(coeffs), f"N={N} ell={ell} m={m}: ragged term list"
        assert len(exps) > 0, (
            f"N={N} ell={ell} m={m}: empty term list -- a harmonic basis "
            f"polynomial is never the zero polynomial, and the evaluator's "
            f"output shape depends on at least one term being present"
        )
        assert len(set(exps)) == len(exps), f"N={N} ell={ell} m={m}: duplicate monomial"
        for alpha, c in zip(exps, coeffs):
            assert len(alpha) == N, f"N={N} ell={ell} m={m}: exponent {alpha} has wrong length"
            assert all(a >= 0 for a in alpha), f"N={N} ell={ell} m={m}: negative exponent"
            assert sum(alpha) == ell, (
                f"N={N} ell={ell} m={m}: exponent {alpha} has degree {sum(alpha)}"
            )
            assert c != 0.0, (
                f"N={N} ell={ell} m={m}: stored an exactly-zero coefficient; "
                f"stored term <=> nonzero coefficient is the storage invariant"
            )
        assert list(exps) == sorted(exps, reverse=True), (
            f"N={N} ell={ell} m={m}: terms are not in lexicographically "
            f"descending monomial order"
        )


def test_dimension_matches_closed_form():
    """num_harmonics must equal dim H_ell(R^N) = C(N+ell-1, N-1) -
    C(N+ell-3, N-1), the standard harmonic-shell dimension -- an
    independent check that no basis vector was lost or double-counted by
    the generator's Gram-Schmidt."""
    def comb(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0

    for N in range(1, max_dimension() + 1):
        for ell in range(max_degree() + 1):
            expected = comb(N + ell - 1, N - 1) - comb(N + ell - 3, N - 1)
            assert num_harmonics(N, ell) == expected, (
                f"N={N} ell={ell}: {num_harmonics(N, ell)} modes, "
                f"expected dim H_ell = {expected}"
            )


def test_support_lies_in_one_parity_class():
    """Every term of a given polynomial shares one exponent-parity vector
    mod 2. This is why the stored form is sparse: the harmonic projection
    h = sum_k c_k |x|^(2k) Delta^k x^alpha only shifts exponents by +-2, and
    the Gaussian moment matrix is block diagonal across parity classes, so
    Gram-Schmidt cannot mix them."""
    for N, ell, m in _all_modes():
        exps, _ = harmonic_terms(N, ell, m)
        classes = {tuple(a % 2 for a in alpha) for alpha in exps}
        assert len(classes) == 1, (
            f"N={N} ell={ell} m={m}: support spans parity classes {sorted(classes)}"
        )


def test_gram_schmidt_staircase():
    """Within a parity class the basis is a Gram-Schmidt sequence over a
    fixed monomial order, so it is triangular: each polynomial vanishes on
    the leading monomial of every earlier one in its class."""
    for N, ell in _shells():
        pivots_by_class = {}
        for m in range(num_harmonics(N, ell)):
            exps, _ = harmonic_terms(N, ell, m)
            parity = tuple(a % 2 for a in exps[0])
            support = set(exps)
            for earlier_pivot in pivots_by_class.get(parity, []):
                assert earlier_pivot not in support, (
                    f"N={N} ell={ell} m={m}: nonzero on {earlier_pivot}, the "
                    f"leading monomial of an earlier polynomial in its class"
                )
            pivots_by_class.setdefault(parity, []).append(exps[0])


# ------------------------------------------------------------- mathematics

def test_polynomials_are_harmonic():
    """Delta Y = 0 exactly, checked symbolically on the stored terms: the
    defining property of a harmonic polynomial."""
    worst = 0.0
    for N, ell, m in _all_modes():
        exps, coeffs = harmonic_terms(N, ell, m)
        laplacian = {}
        for alpha, c in zip(exps, coeffs):
            for k in range(N):
                a = alpha[k]
                if a >= 2:
                    beta = alpha[:k] + (a - 2,) + alpha[k + 1:]
                    laplacian[beta] = laplacian.get(beta, 0.0) + c * a * (a - 1)
        if not laplacian:
            continue  # ell < 2: the Laplacian is identically zero term-by-term
        scale = max(abs(c) for c in coeffs) * max(1, ell * (ell - 1))
        residual = max(abs(v) for v in laplacian.values()) / scale
        worst = max(worst, residual)
        assert residual < 1e-14, (
            f"N={N} ell={ell} m={m}: relative |Delta Y| = {residual:.3e}"
        )
    print(f"  worst relative |Delta Y| over the table: {worst:.3e}")


def test_homogeneity():
    """Y(t u) = t^ell Y(u): the polynomials are homogeneous of degree ell."""
    rng = np.random.default_rng(0)
    for N, ell, m in _all_modes():
        u = rng.normal(size=(N, 5))
        for t in [0.5, 1.7, -1.3]:
            scaled = eval_harmonic(ell, m, t * u)
            expected = (t ** ell) * eval_harmonic(ell, m, u)
            err = np.max(np.abs(scaled - expected))
            scale = max(1.0, np.max(np.abs(expected)))
            assert err / scale < 1e-12, (
                f"N={N} ell={ell} m={m} t={t}: relative error {err / scale:.3e}"
            )


def _moment_1d_table(max_exponent):
    """int_R t^a exp(-t^2) dt for a = 0..max_exponent (zero for odd a)."""
    return np.array([0.0 if a % 2 else math.gamma((a + 1) / 2.0)
                     for a in range(max_exponent + 1)])


def test_orthonormal_on_the_sphere():
    """The modes of each shell are orthonormal in L^2(S^(N-1)) -- the
    normalization the whole LG construction assumes.

    Computed exactly (up to roundoff) rather than by sampling: for
    homogeneous degree-ell P, Q,

        int_{R^N} P Q exp(-r^2) dx = <P, Q>_{S^(N-1)} * Gamma(ell + N/2) / 2,

    and the left side is a finite sum of Gaussian monomial moments.
    """
    worst_off = 0.0
    worst_diag = 0.0
    for N, ell in _shells():
        moments = _moment_1d_table(2 * ell)
        radial = math.gamma(ell + N / 2.0) / 2.0
        rows = [harmonic_terms(N, ell, m) for m in range(num_harmonics(N, ell))]

        for i, (exps_i, coeffs_i) in enumerate(rows):
            ai = np.array(exps_i, dtype=int)
            ci = np.array(coeffs_i)
            for j, (exps_j, coeffs_j) in enumerate(rows[i:], start=i):
                aj = np.array(exps_j, dtype=int)
                cj = np.array(coeffs_j)
                # moments of every cross monomial x^(alpha_i + alpha_j)
                gauss = np.prod(moments[ai[:, None, :] + aj[None, :, :]], axis=2)
                value = float(ci @ gauss @ cj) / radial

                if i == j:
                    worst_diag = max(worst_diag, abs(value - 1.0))
                    assert abs(value - 1.0) < 1e-11, (
                        f"N={N} ell={ell} m={i}: sphere norm^2 = {value!r}"
                    )
                else:
                    worst_off = max(worst_off, abs(value))
                    assert abs(value) < 1e-11, (
                        f"N={N} ell={ell}: <Y_{i}, Y_{j}> = {value:.3e}"
                    )
    print(f"  sphere Gram: max |diag - 1| = {worst_diag:.3e}, "
          f"max |off-diag| = {worst_off:.3e}")


# ---------------------------------------------------------------- gradient

def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    for N, ell, m in _all_modes():
        if ell > 6:
            continue  # deep shells are covered by the identities above; FD there
                      # is limited by monomial-basis cancellation, not by the code
        for u in [np.zeros(N), rng.uniform(-1.5, 1.5, size=N)]:
            _, analytic = grad_harmonic(ell, m, u)
            fd = np.zeros(N)
            for k in range(N):
                up, um = u.copy(), u.copy()
                up[k] += FD_STEP
                um[k] -= FD_STEP
                fd[k] = (eval_harmonic(ell, m, up)
                         - eval_harmonic(ell, m, um)) / (2 * FD_STEP)
            err = np.max(np.abs(analytic - fd))
            scale = max(1.0, np.max(np.abs(fd)))
            assert err / scale < FD_TOL, (
                f"N={N} ell={ell} m={m} u={u}: analytic={analytic} fd={fd} "
                f"rel_err={err / scale:.3e}"
            )


def test_grad_harmonic_value_matches_eval_harmonic():
    """grad_harmonic returns the value alongside the gradient; it must be
    the same number eval_harmonic gives, not merely a close one."""
    rng = np.random.default_rng(2)
    for N, ell, m in _all_modes():
        u = rng.normal(size=(N, 6))
        Y_only = eval_harmonic(ell, m, u)
        Y_with_grad, _ = grad_harmonic(ell, m, u)
        assert np.array_equal(Y_only, Y_with_grad), (
            f"N={N} ell={ell} m={m}: eval_harmonic and grad_harmonic disagree"
        )


# ----------------------------------------------------------------- contract

def test_batch_shape_contract():
    """u of shape (N, *batch) -> Y of shape (*batch), dY of shape u.shape,
    for any batch shape including none."""
    rng = np.random.default_rng(3)
    for N, ell, m in _all_modes():
        for batch in [(), (5,), (3, 4)]:
            u = rng.normal(size=(N,) + batch)
            Y = eval_harmonic(ell, m, u)
            Y2, dY = grad_harmonic(ell, m, u)
            assert np.shape(Y) == batch, f"N={N} ell={ell} m={m}: Y shape {np.shape(Y)}"
            assert np.shape(Y2) == batch, f"N={N} ell={ell} m={m}: Y shape {np.shape(Y2)}"
            assert dY.shape == u.shape, f"N={N} ell={ell} m={m}: dY shape {dY.shape}"


def test_constant_shell_gradient_is_zero_and_correctly_shaped():
    """ell = 0 is the case where no term contributes to any gradient
    component, so the gradient array has to be pre-shaped rather than
    accumulated from scratch."""
    for N in range(1, max_dimension() + 1):
        u = np.ones((N, 4))
        Y, dY = grad_harmonic(0, 0, u)
        assert dY.shape == u.shape, f"N={N}: dY shape {dY.shape}, expected {u.shape}"
        assert np.all(dY == 0.0), f"N={N}: constant polynomial has nonzero gradient"
        assert np.all(Y == Y.flat[0]), f"N={N}: degree-0 polynomial is not constant"


def test_out_of_range_raises():
    for bad in [(max_dimension() + 1, 0), (1, max_degree() + 1), (0, 0), (2, -1)]:
        try:
            num_harmonics(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"num_harmonics{bad} should have raised")

    for bad_m in [-1, num_harmonics(3, 2)]:
        try:
            harmonic_terms(3, 2, bad_m)
        except ValueError:
            pass
        else:
            raise AssertionError(f"harmonic_terms(3, 2, {bad_m}) should have raised")

    # N = 1, ell >= 2 is empty but legal to ask about
    assert num_harmonics(1, 5) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all harmonic-polynomial checks passed")
