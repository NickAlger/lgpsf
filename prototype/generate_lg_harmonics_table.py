"""Generate the harmonic-polynomial coefficient tables for the N-dimensional
Laguerre-Gaussian modes (see lg_functions.py for the 2D reference).

Eigenfunctions of -Delta + |x|^2 in R^N separate as

    psi_{p,ell,m}(x) = C_{p,ell} * Y_{ell,m}(u) * L_p^(alpha)(r^2) * exp(-r^2/2),
    alpha = ell + N/2 - 1,

with Y_{ell,m} an orthonormal (surface-measure) basis of the degree-ell
harmonic homogeneous polynomials in N variables (for N=2 these are exactly
r^ell * {cos, sin}(ell*theta), the angular part already in eval_lg). This
script builds that basis for ell = 0..MAX_ELL and N in DIMENSIONS and writes
it out as a literal Python table (lg_harmonics_table.py): computed once,
offline, in exact rational arithmetic, so the runtime evaluation code never
needs a linear-algebra dependency.

Construction, per (N, ell):
  1. Harmonic-project every monomial of degree ell via the recursion
     h = sum_k c_k |x|^(2k) Delta^k(x^alpha),
     c_0 = 1, c_{j+1} = -c_j / (2(j+1)(N + 2(ell-j-2))).
     This spans H_ell but is redundant (dim H_ell < monomial count once
     ell >= 2). For N=1, ell >= 2 every projection is identically zero --
     there is no degree->=2 harmonic function of one variable (S^0 is two
     points, so ell in {0,1} is all there is; this matches the classical
     even/odd Hermite split, see the N-D LG design discussion).
  2. Orthogonalize the redundant spanning set by exact-rational Gram-Schmidt
     (skipping exactly-dependent vectors -- an unambiguous exact-zero test,
     not a numerical tolerance) against the Gaussian-moment inner product,
     stripped of its common sqrt(pi)^N factor: every nonzero Gaussian moment
     of a monomial carries exactly one factor of sqrt(pi) per dimension, so
     that factor is pulled out and applied once at the end rather than
     carried through the orthogonalization.
  3. Rescale from "unit full-space Gaussian norm" (what the rational
     Gram-Schmidt above naturally produces) to "unit surface-measure norm on
     S^(N-1)" -- the convention eval_lg's combining constant C_{p,ell}
     assumes. The two differ by a basis-vector-independent radial factor;
     see build_shell.

Moment matrices of this kind are classically ill-conditioned (like
Hilbert/Hankel matrices), which is exactly why exact rational arithmetic is
used throughout rather than floating point -- correctness does not depend on
conditioning here, only performance does, and it was checked empirically
(the worst case in this table, N=4 ell=10 with 286 spanning monomials,
Gram-Schmidt in under 15s; see dev notes).
"""
import math
from fractions import Fraction as Fr

DIMENSIONS = [1, 2, 3, 4]
MAX_ELL = 10


def multi_indices(N, ell):
    """All length-N tuples of nonnegative integers summing to ell, in a
    fixed deterministic order."""
    if N == 1:
        return [(ell,)]
    return [(a1,) + rest
            for a1 in range(ell, -1, -1)
            for rest in multi_indices(N - 1, ell - a1)]


def _laplacian(poly, N):
    result = {}
    for alpha, coeff in poly.items():
        for i in range(N):
            a_i = alpha[i]
            if a_i >= 2:
                beta = alpha[:i] + (a_i - 2,) + alpha[i + 1:]
                result[beta] = result.get(beta, 0) + coeff * a_i * (a_i - 1)
    return result


def _times_r2(poly, N):
    result = {}
    for alpha, coeff in poly.items():
        for i in range(N):
            beta = alpha[:i] + (alpha[i] + 2,) + alpha[i + 1:]
            result[beta] = result.get(beta, 0) + coeff
    return result


def harmonic_project(alpha, N, ell):
    """Harmonic projection of the monomial x^alpha (degree ell), as a dict
    {multi_index: Fraction} over degree-ell monomials."""
    deltas = [{tuple(alpha): Fr(1)}]
    for _ in range(ell // 2):
        deltas.append(_laplacian(deltas[-1], N))
    c = [Fr(1)]
    for j in range(ell // 2):
        c.append(Fr(-c[j], 2 * (j + 1) * (N + 2 * (ell - j - 2))))
    h = {}
    for k, dk in enumerate(deltas):
        term = dk
        for _ in range(k):
            term = _times_r2(term, N)
        for beta, coeff in term.items():
            h[beta] = h.get(beta, 0) + c[k] * coeff
    return h


def _moment_1d(a):
    """Rational factor c such that int_R t^a exp(-t^2) dt = c * sqrt(pi)
    (zero for odd a)."""
    if a % 2 == 1:
        return Fr(0)
    m = a // 2
    return Fr(math.factorial(2 * m), 4 ** m * math.factorial(m))


def build_shell(N, ell):
    """Orthonormal (surface-measure) basis of the degree-ell harmonic
    homogeneous polynomials in N variables.

    Returns (monomials, coeff_rows): monomials is the length-D list of
    degree-ell multi-indices (D = C(N+ell-1, ell)); coeff_rows is a list of
    length-D float lists, one per basis polynomial, its coefficients over
    that monomial basis. Empty (monomials still length D, coeff_rows = [])
    when dim H_ell = 0 (only N=1, ell >= 2).
    """
    monos = multi_indices(N, ell)
    D = len(monos)
    index_of = {m: i for i, m in enumerate(monos)}

    vectors = []
    for alpha in monos:
        h = harmonic_project(alpha, N, ell)
        v = [Fr(0)] * D
        for beta, coeff in h.items():
            v[index_of[beta]] = coeff
        vectors.append(v)

    moment_cache = {}

    def moment(a):
        if a not in moment_cache:
            moment_cache[a] = _moment_1d(a)
        return moment_cache[a]

    M = [[Fr(0)] * D for _ in range(D)]
    for i in range(D):
        for j in range(i, D):
            prod = Fr(1)
            for d in range(N):
                fac = moment(monos[i][d] + monos[j][d])
                if fac == 0:
                    prod = Fr(0)
                    break
                prod *= fac
            M[i][j] = M[j][i] = prod

    def matvec(v):
        return [sum(M[i][j] * v[j] for j in range(D) if v[j] != 0) for i in range(D)]

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v) if a != 0 and b != 0)

    accepted = []  # (vector, M @ vector, self M-inner-product)
    for v in vectors:
        w = list(v)
        for uk, Muk, ukk in accepted:
            coeff = dot(w, Muk)
            if coeff != 0:
                factor = coeff / ukk
                w = [wi - factor * uki for wi, uki in zip(w, uk)]
        Mw = matvec(w)
        wMw = dot(w, Mw)
        if wMw != 0:
            accepted.append((w, Mw, wMw))

    alpha_gamma = ell + N / 2.0
    radial0 = 0.5 * math.gamma(alpha_gamma)  # int_0^inf r^(2 ell+N-1) exp(-r^2) dr
    coeff_rows = []
    for w, _Mw, wMw in accepted:
        scale = math.sqrt(radial0 / (math.pi ** (N / 2.0) * float(wMw)))
        coeff_rows.append([float(wi) * scale for wi in w])

    return monos, coeff_rows


def build_table():
    table = {}
    for N in DIMENSIONS:
        for ell in range(MAX_ELL + 1):
            table[(N, ell)] = build_shell(N, ell)
    return table


def write_table_module(table, path):
    with open(path, "w") as f:
        f.write(
            '"""Generated by generate_lg_harmonics_table.py -- do not hand-edit.\n\n'
            "TABLE[(N, ell)] = (monomials, coeff_rows): monomials is the list of\n"
            "degree-ell exponent tuples (length N) spanning the standard monomial\n"
            "basis; coeff_rows is a list of orthonormal (surface-measure) harmonic\n"
            "polynomials, each a list of coefficients over that monomial basis,\n"
            "in the same order. len(coeff_rows) may be less than len(monomials)\n"
            '(and 0 for N=1, ell >= 2 -- see the generator module docstring).\n"""\n\n'
            "TABLE = {\n"
        )
        for (N, ell), (monos, coeff_rows) in table.items():
            f.write(f"    ({N}, {ell}): (\n")
            f.write(f"        {monos!r},\n")
            f.write("        [\n")
            for row in coeff_rows:
                f.write(f"            {row!r},\n")
            f.write("        ],\n")
            f.write("    ),\n")
        f.write("}\n")


if __name__ == "__main__":
    import os
    import time

    t0 = time.time()
    table = build_table()
    out_path = os.path.join(os.path.dirname(__file__), "lg_harmonics_table.py")
    write_table_module(table, out_path)
    print(f"wrote {out_path} in {time.time() - t0:.1f}s")
