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

from lg_harmonics_table import TABLE


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


def eval_lg_nd(p, ell, m, *u):
    """Evaluate the real Laguerre-Gaussian mode (p, ell, m) in N dimensions
    at points u = (u_1, ..., u_N) (broadcastable arrays; N = len(u)).

    ell >= 0 is the harmonic-polynomial (angular) degree; m indexes the
    d_N(ell)-dimensional orthonormal basis of that degree from
    lg_harmonics_table.TABLE (there is no canonical cos/sin-style labeling
    once N > 2, m is just an index -- for N == 2 it reproduces eval_lg's
    ell > 0/ell < 0 branches up to which index is which). N == 1 has no
    modes for ell >= 2 (see lg_harmonics_table's generator docstring).

    Orthonormal in L^2(R^N), same convention as eval_lg.
    """
    N = len(u)
    monos, rows = TABLE[(N, ell)]
    if m >= len(rows):
        raise ValueError(
            f"N={N} ell={ell} has {len(rows)} harmonic mode(s); m={m} is out of range"
        )
    row = rows[m]

    us = [np.asarray(ui, dtype=float) for ui in u]
    r2 = sum(ui * ui for ui in us)

    Y = 0.0
    for mono, coeff in zip(monos, row):
        term = coeff
        for ui, a in zip(us, mono):
            term = term * ui**a
        Y = Y + term

    alpha = ell + N / 2.0 - 1.0
    radial = genlaguerre(p, alpha, r2) * np.exp(-0.5 * r2)
    norm = math.sqrt(2.0 * math.factorial(p) / math.gamma(p + alpha + 1.0))

    return norm * Y * radial


def grad_eval_lg_nd(p, ell, m, *u):
    """Spatial gradient (w.r.t. u) of eval_lg_nd(p, ell, m, *u).

    Returns a length-N tuple of arrays (one per component of u, same shape
    as the broadcast inputs). Product rule on
    psi = C * Y(u) * L_p^alpha(r^2) * exp(-r^2/2), reusing exactly the
    pieces eval_lg_nd already has -- no new special-function code:

      - dY/du_k is elementary term-by-term monomial calculus over the same
        coefficient table;
      - d/dt L_p^alpha(t) = -L_{p-1}^(alpha+1)(t) (the classical Laguerre
        derivative identity, checked symbolically against the recurrence
        before use), so the radial derivative is genlaguerre itself at
        (p-1, alpha+1) -- L_{-1} = 0 by convention, matching L_0 being
        constant.
    """
    N = len(u)
    monos, rows = TABLE[(N, ell)]
    if m >= len(rows):
        raise ValueError(
            f"N={N} ell={ell} has {len(rows)} harmonic mode(s); m={m} is out of range"
        )
    row = rows[m]

    us = [np.asarray(ui, dtype=float) for ui in u]
    r2 = sum(ui * ui for ui in us)

    Y = 0.0
    dY = [0.0] * N
    for mono, coeff in zip(monos, row):
        term = coeff
        for ui, a in zip(us, mono):
            term = term * ui**a
        Y = Y + term

        for k in range(N):
            a_k = mono[k]
            if a_k == 0:
                continue
            dterm = coeff * a_k
            for j in range(N):
                power = mono[j] - 1 if j == k else mono[j]
                if power != 0:
                    dterm = dterm * us[j] ** power
            dY[k] = dY[k] + dterm

    alpha = ell + N / 2.0 - 1.0
    R = genlaguerre(p, alpha, r2)
    dR_dt = -genlaguerre(p - 1, alpha + 1.0, r2) if p >= 1 else 0.0
    gaussian = np.exp(-0.5 * r2)
    norm = math.sqrt(2.0 * math.factorial(p) / math.gamma(p + alpha + 1.0))

    prefactor = norm * gaussian
    return tuple(
        prefactor * (R * dY[k] - us[k] * Y * (R - 2.0 * dR_dt)) for k in range(N)
    )
