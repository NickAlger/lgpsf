"""Real Laguerre-Gaussian (2D quantum-harmonic-oscillator) basis functions.

psi_{p,ell}(u) is the family used as the smooth part of the LG-PSF kernel
model: the polar-coordinates eigenfunctions of the 2D harmonic oscillator,

    psi_{p,ell}(u) ~ r^|ell| L_p^|ell|(r^2) exp(-r^2/2) * {cos, sin}(ell*theta),

indexed by radial order p >= 0 and angular order ell (any integer). The
sign of ell selects the real branch: ell > 0 -> cos(ell*theta), ell < 0 ->
sin(|ell|*theta), ell == 0 -> the single angle-independent radial mode. The
oscillator level is 2*p + abs(ell).

Modes are normalized to be orthonormal in L^2(R^2, du) (plain Lebesgue
measure -- the Gaussian envelope is part of the function itself, as for
Hermite functions, not a separate weight). This is a different
normalization from the node-local N_0 = 1/(2 pi sqrt(det Sigma)) mass-unit
scaling used when these modes are pulled back onto a physical ellipsoid as
kernel features; that scaling belongs to the feature-construction step, not
here.
"""
import math

import numpy as np

from harmonic_polynomials import (
    eval_harmonic, grad_harmonic, max_degree, num_harmonics,
)


def genlaguerre(p, alpha, x):
    """Generalized Laguerre polynomial L_p^alpha(x), via the standard
    three-term recurrence

        L_0^alpha = 1,  L_1^alpha = 1 + alpha - x,
        (k+1) L_{k+1}^alpha = (2k+1+alpha-x) L_k^alpha - (k+alpha) L_{k-1}^alpha.

    No special-function library calls, so this ports directly to C++: for
    each evaluation point, run the same scalar recurrence in a loop over k.
    """
    x = np.asarray(x, dtype=float)
    if p == 0:
        return np.ones_like(x)
    L_prev = np.ones_like(x)       # L_0^alpha
    L_curr = 1.0 + alpha - x       # L_1^alpha
    for k in range(1, p):
        L_next = ((2 * k + 1 + alpha - x) * L_curr - (k + alpha) * L_prev) / (k + 1)
        L_prev, L_curr = L_curr, L_next
    return L_curr


def factorial_ratio(p, a):
    """p! / (p+a)!, for nonnegative integers p, a.

    Computed as the exact product of reciprocals 1/(p+1) * ... * 1/(p+a)
    rather than via a gamma function: every partial product stays in
    [0, 1], so this is stable at the small p, a used here, and it is the
    natural form to port to C++ (an integer loop, no special functions).
    """
    ratio = 1.0
    for k in range(p + 1, p + a + 1):
        ratio /= k
    return ratio


def eval_lg(p, ell, u1, u2):
    """Evaluate the real Laguerre-Gaussian mode (p, ell) at points u = (u1, u2).

    p : int >= 0, radial order.
    ell : int, angular order; sign selects the cos/sin branch (see module
        docstring); ell == 0 gives the single angle-independent mode.
    u1, u2 : array_like, broadcastable together, the evaluation points.

    Orthonormal in L^2(R^2): the integral of psi_{p,ell} * psi_{p',ell'} du
    is 1 if (p, ell) == (p', ell') and 0 otherwise.
    """
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    r2 = u1 * u1 + u2 * u2
    a = abs(ell)

    norm = np.sqrt(factorial_ratio(p, a) / np.pi)
    if a > 0:
        norm *= np.sqrt(2.0)

    radial = r2 ** (a / 2.0) * genlaguerre(p, a, r2) * np.exp(-0.5 * r2)

    if ell == 0:
        angular = 1.0
    else:
        theta = np.arctan2(u2, u1)
        angular = np.cos(a * theta) if ell > 0 else np.sin(a * theta)

    return norm * radial * angular


def lg_norm(p, ell, N):
    """The combining constant C_{p,ell,N} that makes psi_{p,ell,m}
    orthonormal in L^2(R^N), given a surface-orthonormal Y_{ell,m}:

        C = sqrt(2 * p! / Gamma(p + alpha + 1)),   alpha = ell + N/2 - 1.

    alpha is an integer or a half-integer, so the gamma is elementary; C++
    gets it from std::tgamma.
    """
    alpha = ell + N / 2.0 - 1.0
    return math.sqrt(2.0 * math.factorial(p) / math.gamma(p + alpha + 1.0))


def eval_lg_nd(p, ell, m, u):
    """Evaluate the real Laguerre-Gaussian mode (p, ell, m) in N dimensions
    at points u, an array of shape (N, *batch_shape) -- N is read off
    u.shape[0], batch_shape can be anything (a flat list of points, a grid,
    even scalar). Returns an array of shape (*batch_shape).

    The mode is the product of the three separable factors

        psi = C_{p,ell,N} * Y_{ell,m}(u) * L_p^alpha(r^2) * exp(-r^2/2),

    with alpha = ell + N/2 - 1: the harmonic polynomial (angular part,
    harmonic_polynomials.py), the generalized-Laguerre radial profile, and
    the normalization constant.

    ell >= 0 is the harmonic-polynomial (angular) degree; m indexes the
    orthonormal basis of that degree (there is no canonical cos/sin-style
    labeling once N > 2, m is just an index -- for N == 2 it reproduces
    eval_lg's ell > 0/ell < 0 branches up to which index is which). N == 1
    has no modes for ell >= 2.

    Orthonormal in L^2(R^N), same convention as eval_lg.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[0]
    alpha = ell + N / 2.0 - 1.0
    r2 = np.sum(u * u, axis=0)
    Y = eval_harmonic(ell, m, u)
    radial = genlaguerre(p, alpha, r2) * np.exp(-0.5 * r2)
    return lg_norm(p, ell, N) * Y * radial


def grad_eval_lg_nd(p, ell, m, u):
    """Spatial gradient (w.r.t. u) of eval_lg_nd(p, ell, m, u).

    u: array of shape (N, *batch_shape), same convention as eval_lg_nd.
    Returns an array of shape (N, *batch_shape) (one gradient component per
    leading-axis entry, at every point). The product rule on
    psi = C * Y(u) * L(r^2) * exp(-r^2/2) gives

        grad psi = C exp(-r^2/2) [ L grad_Y - u Y (L - 2 L') ],

    using d/dt L_p^alpha(t) = -L_{p-1}^(alpha+1)(t) (the classical Laguerre
    derivative identity, checked symbolically against the recurrence before
    use), so the radial derivative is genlaguerre itself at (p-1, alpha+1)
    -- L_{-1} = 0 by convention, matching L_0 being constant. No new
    special-function code: grad_harmonic supplies Y and grad_Y from one
    pass, and both Laguerre values come from the same recurrence.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[0]
    alpha = ell + N / 2.0 - 1.0
    r2 = np.sum(u * u, axis=0)
    Y, dY = grad_harmonic(ell, m, u)
    L = genlaguerre(p, alpha, r2)
    dL_dt = -genlaguerre(p - 1, alpha + 1.0, r2) if p >= 1 else 0.0

    prefactor = lg_norm(p, ell, N) * np.exp(-0.5 * r2)
    return prefactor * (L * dY - u * Y * (L - 2.0 * dL_dt))


def modes_up_to_level(N, max_level, ell_max=None):
    """All (p, ell, m) with oscillator level 2p + ell <= max_level --
    complete shells in energy order, the single-knob mode family used
    throughout for nested mode ladders. ell_max caps the angular order
    (an ell-capped WEDGE: deep radial, shallow angular -- the PIG
    slice-38 finding is that PSF-like kernels want radial depth, and
    full shells waste budget on high-ell modes). None = no cap
    (complete shells).

    Raises ValueError if the requested angular orders exceed the
    generated harmonic table (N <= 4, ell <= 10). An earlier version
    stopped silently at the table's edge, which turned an unsatisfiable
    request into a quietly truncated mode set.
    """
    top = max_level if ell_max is None else min(max_level, ell_max)
    if top > max_degree():
        raise ValueError(
            f"modes_up_to_level(N={N}, max_level={max_level}, "
            f"ell_max={ell_max}) needs harmonics up to ell={top}, but the "
            f"generated table stops at ell={max_degree()}; lower max_level, "
            f"set ell_max, or extend the table"
        )
    modes = []
    for ell in range(top + 1):
        for m in range(num_harmonics(N, ell)):
            for p in range((max_level - ell) // 2 + 1):
                modes.append((p, ell, m))
    return modes
