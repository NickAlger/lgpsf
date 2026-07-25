"""Estimators of a target function from its probe data alone -- no
fitting. The target is any function on the batch known only through
inner products with random probe fields (e.g. one row of an implicitly
available operator); the estimators here are the zero-extra-matvec
initial-guess / QC companions to probe_fit.

(The parked whole-mesh generalization -- conservative ellipsoid FIELDS
for every target at once via the squared-kernel moment identity -- lives
in docs/probe-moment-ellipsoids.md and would land in this module.)
"""
import numpy as np


def backproject(z, y):
    """Unbiased estimate of the target's raw coefficient vector
    restricted to the batch, from probe pairs: r_hat = (1/k) sum_l y_l
    z_l. Requires the probe entries to be iid standard normal in the raw
    dof basis (E[z z^T] = I), which is the convention of the intended
    pipeline. z: (k, K) raw probe fields on the batch; y: (k,) raw
    target responses. Noise per entry is O(||target|| / sqrt(k)) -- see
    raw_moments' thresholds."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return z.T @ y / z.shape[0]


def raw_moments(x, r_raw, spike_index=None, rel_threshold=0.05,
                noise_mad=0.0):
    """Weighted mean and covariance of the target kernel's magnitude,
    from RAW values (measured or backprojected). Returns
    (mu_hat, Sigma_hat).

    No mass vector: r_raw[j] = m_target * m_j * phi(x_j), so |r_raw|
    already carries the lumped-mass quadrature weight -- summing
    |r_raw|-weighted point statistics IS the mesh approximation of the
    continuum moments of |phi|. That is the whole mass subtlety,
    resolved by using raw values rather than kernel values.

    spike_index excludes the target's own point: its raw value contains
    the mesh-unresolvable spike, which belongs to the discrete
    correction, not to the smooth kernel whose moments we want -- left
    in, it drags mu_hat toward that point and shrinks Sigma_hat.

    Two thresholds zero out small weights (a backprojected target has a
    flat O(||target||/sqrt(k)) noise floor; on a support window that is
    mostly far field, those noise entries outnumber the signal and drag
    mu_hat toward the window centroid and inflate Sigma_hat):
      - rel_threshold: relative to max|r|;
      - noise_mad: in robust noise sigmas, sigma_noise estimated as
        1.4826 * median|r|. Recommended ~3-4 for backprojected targets
        on CONSERVATIVE support windows, where most entries are
        far-field noise and the median estimates it; on tight windows
        where most entries carry signal, the median overestimates the
        noise and rel_threshold alone is safer.
    Set both to 0.0 for exact (measured) targets.
    """
    x = np.asarray(x, dtype=float)
    w = np.abs(np.asarray(r_raw, dtype=float)).copy()
    if spike_index is not None:
        w[spike_index] = 0.0
    if rel_threshold > 0.0:
        w[w < rel_threshold * w.max()] = 0.0
    if noise_mad > 0.0:
        sigma_noise = 1.4826 * np.median(np.abs(np.asarray(r_raw)))
        w[w < noise_mad * sigma_noise] = 0.0
    w = w / w.sum()
    mu_hat = x @ w
    d = x - mu_hat[:, None]
    sigma_hat = np.einsum("j,ij,kj->ik", w, d, d)
    return mu_hat, sigma_hat
