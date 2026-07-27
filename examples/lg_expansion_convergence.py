"""Convergence of a Laguerre-Gaussian expansion for f(x) = gaussian(x) * smooth(x).

f is a Gaussian with a WIDTH DELIBERATELY MISMATCHED against the LG basis's
native width 1, times a smooth (but non-polynomial, non-Gaussian) tilt along
the first axis -- so the expansion never terminates exactly and the plot
shows genuine convergence, exercising both the radial ladder (width
mismatch) and the harmonic-polynomial/angular part (the asymmetric tilt).

Modes are orthonormal in L^2(R^N) (see eval_lg_nd), so the best truncated
approximation's coefficients are plain inner products c_i = integral(f *
psi_i), evaluated here by a straightforward grid quadrature (Riemann sum) --
the same reason N=4 is skipped: a dense grid becomes too expensive by 4D on
a laptop, so this script only covers N = 1, 2, 3, growing the mode set one
oscillator level at a time and plotting relative L2 error vs mode count.

Needs matplotlib, so run with the `tttt` conda env rather than `t3toolbox`:
    python examples/lg_expansion_convergence.py
"""
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from harmonic_polynomials import max_degree, num_harmonics
from lg_functions import eval_lg_nd

WIDTH = 1.3   # deliberately mismatched vs the LG basis's native width 1
AMP = 0.3
FREQ = 1.5

# (max_level, grid half-width, grid points per axis), tuned so the highest
# mode used still has grid self-norm accurate to ~1e-6 while staying under
# a few seconds per dimension.
SETTINGS = {
    1: dict(max_level=20, half_width=8.0, n_grid=3000),
    2: dict(max_level=10, half_width=6.0, n_grid=250),
    3: dict(max_level=6, half_width=5.0, n_grid=45),
}


def target_f(coords):
    """coords: array of shape (N, *batch_shape)."""
    r2 = np.sum(coords * coords, axis=0)
    gaussian = np.exp(-r2 / (2.0 * WIDTH**2))
    smooth = 1.0 + AMP * np.sin(FREQ * coords[0])
    return gaussian * smooth


def enumerate_modes_by_level(N, max_level):
    """List of levels, each a list of (p, ell, m) triples with 2p+ell equal
    to that level, in increasing level order -- the natural "add one more
    degree of freedom group" sequence for a convergence study."""
    levels = []
    for level in range(max_level + 1):
        modes_this_level = []
        for p in range(level // 2 + 1):
            ell = level - 2 * p
            if ell < 0:
                continue
            if N == 1 and ell >= 2:
                continue  # analytically zero (S^0 is two points), not a table limit
            if ell > max_degree():
                raise ValueError(
                    f"ell={ell} > generated table range ({max_degree()}); "
                    f"reduce max_level or extend the table"
                )
            for m in range(num_harmonics(N, ell)):
                modes_this_level.append((p, ell, m))
        if modes_this_level:
            levels.append(modes_this_level)
    return levels


def run_convergence(N, max_level, half_width, n_grid):
    axes = [np.linspace(-half_width, half_width, n_grid) for _ in range(N)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)  # (N, n_grid, ..., n_grid)
    cell_volume = (axes[0][1] - axes[0][0]) ** N

    f_vals = target_f(mesh)
    f_norm = math.sqrt(np.sum(f_vals**2) * cell_volume)

    levels = enumerate_modes_by_level(N, max_level)

    recon = np.zeros_like(f_vals)
    mode_counts = []
    errors = []
    total_modes = 0
    for level_modes in levels:
        for (p, ell, m) in level_modes:
            psi = eval_lg_nd(p, ell, m, mesh)
            c = np.sum(f_vals * psi) * cell_volume
            recon = recon + c * psi
            total_modes += 1
        err = math.sqrt(np.sum((f_vals - recon) ** 2) * cell_volume) / f_norm
        mode_counts.append(total_modes)
        errors.append(err)

    return mode_counts, errors


def main():
    fig, ax = plt.subplots(figsize=(6, 5))
    for N, cfg in SETTINGS.items():
        mode_counts, errors = run_convergence(N, **cfg)
        print(f"N={N}: {mode_counts[-1]} modes, final relative L2 error = {errors[-1]:.4e}")
        ax.semilogy(mode_counts, errors, marker="o", label=f"N={N}")

    ax.set_xlabel("number of LG modes")
    ax.set_ylabel("relative $L^2$ reconstruction error")
    ax.set_title(r"LG expansion convergence: $f(x) = \mathrm{gaussian}(x) \cdot \mathrm{smooth}(x)$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "lg_expansion_convergence.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
