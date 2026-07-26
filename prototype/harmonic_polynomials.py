"""Real harmonic polynomials Y_{ell,m} on R^N: evaluation and gradient.

Y_{ell,m} is a homogeneous polynomial of degree ell in N variables with
Delta Y = 0. For each (N, ell) the modes m = 0 .. num_harmonics(N, ell) - 1
form an orthonormal basis of that space under the surface measure on
S^(N-1); restricted to the sphere they are the N-dimensional
generalization of the spherical harmonics (for N = 2 they are
r^ell {cos, sin}(ell*theta), the angular part of the classical
Laguerre-Gaussian modes -- see lg_functions.eval_lg).

The coefficients come from lg_harmonics_table.py, generated offline in
exact rational arithmetic (generate_lg_harmonics_table.py).

**This module is the only place the table's storage format appears.**
Everything else -- lg_functions.py, mode_policy.py, the tests, the
examples -- goes through the four functions below, the same confinement
discipline whitening.py applies to the mass matrices. Changing how
harmonic polynomials are represented or evaluated (a different term
ordering, precomputed power tables, a Horner scheme, a reduced-precision
path) is then a change to this file alone, testable in isolation against
test_harmonic_polynomials.py's intrinsic invariants.

The stored form is a sparse term list per polynomial: only the monomials
with a nonzero coefficient, since 91.5% of the dense coefficients are
exactly zero for structural reasons (exponent parity classes and the
Gram-Schmidt staircase -- see the generator's build_shell docstring and
docs/design-notes.md). Every polynomial has at least one term; the
generator raises if that is ever violated, and
test_harmonic_polynomials.py pins it.

Point batches follow the repo-wide convention (docs/design-notes.md):
u has shape (N, *batch_shape), non-batch axes first, and N is read off
u.shape[0]. Loops over the term list and over the spatial dimension are
plain Python loops on purpose -- only the point batch is vectorized.
"""
import numpy as np

from lg_harmonics_table import TABLE

_MAX_DIMENSION = max(N for (N, _) in TABLE)
_MAX_DEGREE = max(ell for (_, ell) in TABLE)


def max_dimension():
    """Largest spatial dimension N the generated table covers."""
    return _MAX_DIMENSION


def max_degree():
    """Largest harmonic degree ell the generated table covers."""
    return _MAX_DEGREE


def num_harmonics(N, ell):
    """Number of degree-ell harmonic modes in N dimensions, i.e. the
    number of valid m values -- dim H_ell(R^N).

    Zero only for N == 1, ell >= 2: there is no harmonic function of one
    variable of degree >= 2 (S^0 is two points, so ell in {0, 1} is all
    there is). Raises ValueError if (N, ell) lies outside the generated
    table, which is a caller error rather than an empty answer.
    """
    return len(_shell(N, ell)[1])


def harmonic_terms(N, ell, m):
    """The polynomial Y_{ell,m} on R^N as its list of nonzero terms:
    (exponents, coefficients), two equal-length lists holding the
    degree-ell exponent tuples and their coefficients.

    Terms absent from the list have an exactly zero coefficient (see the
    module docstring). Exponent tuples appear in lexicographically
    descending order, and the list is never empty.
    """
    exponent_rows, coeff_rows = _shell(N, ell)
    if not 0 <= m < len(coeff_rows):
        raise ValueError(
            f"N={N} ell={ell} has {len(coeff_rows)} harmonic mode(s); "
            f"m={m} is out of range"
        )
    return exponent_rows[m], coeff_rows[m]


def eval_harmonic(ell, m, u):
    """Y_{ell,m}(u) at points u of shape (N, *batch_shape), N read off
    u.shape[0]. Returns an array of shape (*batch_shape)."""
    u = np.asarray(u, dtype=float)
    exponents, coeffs = harmonic_terms(u.shape[0], ell, m)

    Y = 0.0
    for mono, coeff in zip(exponents, coeffs):
        term = coeff
        for k, a in enumerate(mono):
            term = term * u[k]**a
        Y = Y + term
    return Y


def grad_harmonic(ell, m, u):
    """Y_{ell,m}(u) together with its spatial gradient, at points u of
    shape (N, *batch_shape). Returns (Y, dY) with Y of shape
    (*batch_shape) and dY of shape (N, *batch_shape).

    Both come from one pass over the term list: the value and the
    gradient share the monomial walk, and every caller that wants the
    gradient (the LG product rule) needs the value too.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[0]
    exponents, coeffs = harmonic_terms(N, ell, m)

    Y = 0.0
    dY = np.zeros(u.shape)  # (N, *batch_shape); pre-shaped so ell == 0, whose
                            # terms contribute to no gradient component, still
                            # returns a correctly-shaped array of zeros.
    for mono, coeff in zip(exponents, coeffs):
        term = coeff
        for k, a in enumerate(mono):
            term = term * u[k]**a
        Y = Y + term

        for k in range(N):
            a_k = mono[k]
            if a_k == 0:
                continue
            dterm = coeff * a_k
            for j in range(N):
                power = mono[j] - 1 if j == k else mono[j]
                if power != 0:
                    dterm = dterm * u[j]**power
            dY[k] = dY[k] + dterm
    return Y, dY


def eval_harmonic_basis(shells, u):
    """Y_{ell,m}(u) for a LIST of (ell, m) at once, shape
    (len(shells), *batch_shape). Order matches `shells`.

    The batched form exists because every polynomial in the list is built
    from the same per-dimension monomial powers u[k]**a, which the
    one-at-a-time form recomputes for each. They are computed once here
    and shared. Values are bit-identical to calling eval_harmonic in a
    loop: the same powers multiplied in the same order, with the a == 0
    factors (which are exactly 1.0) skipped rather than materialized.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[0]
    terms = [harmonic_terms(N, ell, m) for (ell, m) in shells]
    powers = _power_tables(u, max((ell for (ell, _) in shells), default=0))

    Y = np.zeros((len(shells),) + u.shape[1:])
    for i, (exponents, coeffs) in enumerate(terms):
        for alpha, coeff in zip(exponents, coeffs):
            term = coeff
            for k, a in enumerate(alpha):
                if a:
                    term = term * powers[k][a]
            Y[i] = Y[i] + term
    return Y


def grad_harmonic_basis(shells, u):
    """eval_harmonic_basis together with the gradients: (Y, dY) with Y of
    shape (len(shells), *batch_shape) and dY of shape
    (len(shells), N, *batch_shape).

    Per-mode gradients, deliberately NOT contracted against anything --
    the three consumers in lg_ellipsoid_feature.py contract them
    differently (against a cotangent, against du, against the theta
    Jacobian), and the exact Golub-Pereyra VarPro variant needs the
    uncontracted tensor.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[0]
    terms = [harmonic_terms(N, ell, m) for (ell, m) in shells]
    powers = _power_tables(u, max((ell for (ell, _) in shells), default=0))

    Y = np.zeros((len(shells),) + u.shape[1:])
    dY = np.zeros((len(shells),) + u.shape)
    for i, (exponents, coeffs) in enumerate(terms):
        for alpha, coeff in zip(exponents, coeffs):
            term = coeff
            for k, a in enumerate(alpha):
                if a:
                    term = term * powers[k][a]
            Y[i] = Y[i] + term

            for k in range(N):
                a_k = alpha[k]
                if a_k == 0:
                    continue
                dterm = coeff * a_k
                for j in range(N):
                    power = alpha[j] - 1 if j == k else alpha[j]
                    if power != 0:
                        dterm = dterm * powers[j][power]
                dY[i, k] = dY[i, k] + dterm
    return Y, dY


def _power_tables(u, max_exponent):
    """powers[k][a] = u[k]**a for a = 1..max_exponent, per dimension k.

    Index 0 is a placeholder: an a == 0 factor is exactly 1.0, so callers
    skip it instead of multiplying by an array of ones (which the
    one-at-a-time evaluator does, and which profiling showed costs a few
    percent of a whole row fit in numpy alone).

    Uses `**` rather than repeated multiplication so the results are
    bit-identical to the one-at-a-time path. The C++ port should use
    repeated multiplication instead -- faster and no less accurate for
    small integer exponents -- which is why cross-language tests are
    tolerance-based.
    """
    powers = []
    for k in range(u.shape[0]):
        column = [None, u[k]]
        for a in range(2, max_exponent + 1):
            column.append(u[k]**a)
        powers.append(column)
    return powers


def _shell(N, ell):
    """(exponent_rows, coeff_rows) for one (N, ell) shell, or ValueError
    if the generated table does not cover it."""
    try:
        return TABLE[(N, ell)]
    except KeyError:
        raise ValueError(
            f"harmonic table covers N in 1..{_MAX_DIMENSION} and "
            f"ell in 0..{_MAX_DEGREE}; (N={N}, ell={ell}) is outside it "
            f"(extend generate_lg_harmonics_table.py's DIMENSIONS/MAX_ELL "
            f"and regenerate)"
        ) from None


