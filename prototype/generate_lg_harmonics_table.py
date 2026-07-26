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

    Returns (exponent_rows, coeff_rows): one entry per basis polynomial,
    each holding only that polynomial's NONZERO terms -- exponent_rows[m]
    is its list of degree-ell multi-indices (in the canonical
    multi_indices order) and coeff_rows[m] the matching float
    coefficients. Both are empty lists when dim H_ell = 0 (only N=1,
    ell >= 2).

    The sparsity is exact and structural, not a numerical threshold: the
    Gram-Schmidt below runs in exact rational arithmetic, so a dropped
    term is an exact Fraction(0) (the drop test is `!= 0` on a Fraction,
    never a tolerance on a float). Two mechanisms produce it. First,
    harmonic_project builds h = sum_k c_k |x|^(2k) Delta^k x^alpha, and
    both Delta and multiplication by |x|^2 shift one exponent by +-2, so
    every term of h shares alpha's exponent parity vector mod 2; the
    Gaussian moment matrix is block diagonal across those parity classes,
    so Gram-Schmidt never mixes them and each basis polynomial stays
    inside one class. Second, within a class the accepted vectors are a
    Gram-Schmidt basis over a fixed monomial order, giving the usual QR
    staircase (each vector vanishes on the pivot monomial of every
    earlier one). Measured over the whole generated table: 91.5% of the
    dense coefficients are exactly zero, and the smallest surviving
    |coefficient| is 0.225 -- there is no continuum of small values near
    the cut, because there is no cut. See docs/design-notes.md.
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
    exponent_rows = []
    coeff_rows = []
    for w, _Mw, wMw in accepted:
        scale = math.sqrt(radial0 / (math.pi ** (N / 2.0) * float(wMw)))
        exps = [monos[i] for i, wi in enumerate(w) if wi != 0]
        vals = [float(wi) * scale for wi in w if wi != 0]
        if not exps:
            raise AssertionError(
                f"N={N} ell={ell}: accepted basis vector has no nonzero terms"
            )
        if any(v == 0.0 for v in vals):
            # An exact-rational nonzero that underflowed to 0.0 would break the
            # "stored term <=> nonzero coefficient" invariant the evaluator and
            # its tests rely on. Cannot happen at this table's magnitudes
            # (smallest |coefficient| is 0.225); checked rather than assumed.
            raise AssertionError(
                f"N={N} ell={ell}: nonzero rational coefficient rounded to 0.0"
            )
        exponent_rows.append(exps)
        coeff_rows.append(vals)

    return exponent_rows, coeff_rows


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
            "TABLE[(N, ell)] = (exponent_rows, coeff_rows): one entry per\n"
            "orthonormal (surface-measure) degree-ell harmonic polynomial in N\n"
            "variables. exponent_rows[m] lists that polynomial's NONZERO terms as\n"
            "degree-ell exponent tuples (length N, canonical multi_indices order);\n"
            "coeff_rows[m] holds the matching coefficients, same length, same\n"
            "order. Both lists are empty when dim H_ell = 0 (only N=1, ell >= 2).\n\n"
            "Terms absent from a row have an EXACTLY zero coefficient -- the\n"
            "generator drops them on an exact-rational `!= 0` test, never a\n"
            "numerical tolerance. See the generator's build_shell docstring for the\n"
            "two structural mechanisms (exponent parity classes, Gram-Schmidt\n"
            "staircase) that make 91.5% of the dense coefficients vanish.\n\n"
            "Read this module through prototype/harmonic_polynomials.py, which is\n"
            'its only intended consumer.\n"""\n\n'
            "TABLE = {\n"
        )
        for (N, ell), (exponent_rows, coeff_rows) in table.items():
            f.write(f"    ({N}, {ell}): (\n")
            f.write("        [\n")
            for exps in exponent_rows:
                f.write(f"            {exps!r},\n")
            f.write("        ],\n")
            f.write("        [\n")
            for row in coeff_rows:
                f.write(f"            {row!r},\n")
            f.write("        ],\n")
            f.write("    ),\n")
        f.write("}\n")


def _wrap(values, per_line, indent="    "):
    """Emit a comma-separated literal list, per_line entries per line."""
    lines = []
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        lines.append(indent + ", ".join(chunk) + ",")
    return "\n".join(lines) if lines else indent + "0"


def write_table_header(table, path):
    """Emit the same exact-rational table as a C++ header.

    Layout (see docs/cpp-port-plan.md): NONZERO-MAJOR with two PARALLEL
    blobs -- a double per term and dim int8 exponents per term -- walked
    in lockstep, one linear pass per polynomial with no indirection.
    Deliberately not interleaved into a {double; int8[N];} struct, whose
    12-byte stride at N=4 would misalign every double after the first.

    Flat `inline constexpr` arrays plus offset tables rather than nested
    aggregate initializers: one large initializer of scalars is what g++
    compiles cheaply, which matters under this project's compile-memory
    rules.

    Shells are indexed s = (N - 1) * (MAX_ELL + 1) + ell, so lookup is
    arithmetic, not a map. Rows (basis polynomials) are numbered globally
    in shell order.

    Coefficients are written with %.17g, which round-trips IEEE double
    exactly -- the emitted header holds the identical values to
    lg_harmonics_table.py, and both are produced in the same run from the
    same exact-rational computation so they cannot drift.
    """
    shell_row_begin = [0]
    row_term_begin = [0]
    row_exp_begin = [0]
    coefficients = []
    exponents = []

    for N in DIMENSIONS:
        for ell in range(MAX_ELL + 1):
            exponent_rows, coeff_rows = table[(N, ell)]
            for exps, vals in zip(exponent_rows, coeff_rows):
                for alpha, coeff in zip(exps, vals):
                    coefficients.append("%.17g" % coeff)
                    exponents.extend(str(a) for a in alpha)
                row_term_begin.append(len(coefficients))
                row_exp_begin.append(len(exponents))
            shell_row_begin.append(len(row_term_begin) - 1)

    n_shells = len(DIMENSIONS) * (MAX_ELL + 1)
    n_rows = len(row_term_begin) - 1
    assert len(shell_row_begin) == n_shells + 1

    with open(path, "w") as f:
        f.write(f"""#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief GENERATED harmonic-polynomial coefficient table -- do not hand-edit.
///
/// Produced by `python prototype/generate_lg_harmonics_table.py`, which writes
/// this header and the Python table in the same run from the same
/// exact-rational computation (monomial harmonic projection + Gram-Schmidt
/// against Gaussian moments), so the two cannot drift.
///
/// Each degree-ell harmonic polynomial in N variables is stored as its NONZERO
/// terms only: 91.5% of the dense coefficients are exactly zero for structural
/// reasons (exponent parity classes and the Gram-Schmidt staircase), and the
/// generator drops them on an exact-rational `!= 0` test, never a numerical
/// tolerance. See docs/design-notes.md.
///
/// Two parallel blobs, walked in lockstep: one double per term, and `dim`
/// int8 exponents per term. Shell s = (N - 1) * (MAX_DEGREE + 1) + ell; rows
/// (basis polynomials) are numbered globally in shell order.
///
/// Read this through lgpsf/harmonic_polynomials.hpp, its only intended
/// consumer.

namespace lgpsf {{
namespace detail {{

inline constexpr int kHarmonicMaxDimension = {max(DIMENSIONS)};
inline constexpr int kHarmonicMaxDegree    = {MAX_ELL};
inline constexpr int kHarmonicNumShells    = {n_shells};
inline constexpr int kHarmonicNumRows      = {n_rows};
inline constexpr int kHarmonicNumTerms     = {len(coefficients)};

/// First row index of each shell; shell s owns rows [begin[s], begin[s + 1]).
inline constexpr int kHarmonicShellRowBegin[kHarmonicNumShells + 1] = {{
{_wrap([str(v) for v in shell_row_begin], 12)}
}};

/// Term range of each row, into kHarmonicCoefficient.
inline constexpr int kHarmonicRowTermBegin[kHarmonicNumRows + 1] = {{
{_wrap([str(v) for v in row_term_begin], 12)}
}};

/// Exponent range of each row, into kHarmonicExponent (dim entries per term).
inline constexpr int kHarmonicRowExponentBegin[kHarmonicNumRows + 1] = {{
{_wrap([str(v) for v in row_exp_begin], 12)}
}};

/// Term exponents, dim consecutive entries per term.
inline constexpr signed char kHarmonicExponent[{len(exponents)}] = {{
{_wrap(exponents, 32)}
}};

/// Term coefficients, %.17g so they round-trip exactly.
inline constexpr double kHarmonicCoefficient[kHarmonicNumTerms] = {{
{_wrap(coefficients, 6)}
}};

}} // end namespace detail
}} // end namespace lgpsf
""")


if __name__ == "__main__":
    import os
    import time

    t0 = time.time()
    table = build_table()
    elapsed = time.time() - t0

    here = os.path.dirname(os.path.abspath(__file__))
    # Both outputs, one run, one exact-rational computation -- the Python
    # table and the C++ header cannot drift apart.
    py_path = os.path.join(here, "lg_harmonics_table.py")
    hpp_path = os.path.join(here, os.pardir, "include", "lgpsf", "detail",
                            "lg_harmonics_table.hpp")
    write_table_module(table, py_path)
    write_table_header(table, os.path.normpath(hpp_path))
    print(f"built the table in {elapsed:.1f}s; wrote")
    print(f"  {py_path}")
    print(f"  {os.path.normpath(hpp_path)}")
