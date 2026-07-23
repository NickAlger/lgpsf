"""Grid of plots of the real Laguerre-Gaussian modes psi_{p,ell}.

Rows are radial order p, columns are angular order ell (negative = sin
branch, positive = cos branch, 0 = the angle-independent mode), matching
the (p, ell) table in lg-split-method-notes.tex. Saves the figure next to
this script; does not call plt.show() so it runs headless.

Needs matplotlib, so run with the `tttt` conda env rather than `t3toolbox`:
    /home/nick/miniconda3/envs/tttt/bin/python examples/plot_lg_modes.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from lg_functions import eval_lg

P_MAX = 3
ELL_MAX = 3
EXTENT = 5.0
GRID_N = 201


def main():
    u = np.linspace(-EXTENT, EXTENT, GRID_N)
    u1, u2 = np.meshgrid(u, u)

    ps = list(range(P_MAX + 1))
    ells = list(range(-ELL_MAX, ELL_MAX + 1))

    fig, axes = plt.subplots(
        len(ps), len(ells),
        figsize=(2.0 * len(ells), 2.0 * len(ps)),
        squeeze=False,
    )

    for i, p in enumerate(ps):
        for j, ell in enumerate(ells):
            ax = axes[i, j]
            psi = eval_lg(p, ell, u1, u2)
            vmax = np.max(np.abs(psi))
            ax.imshow(
                psi, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin="lower",
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"$\\ell={ell}$", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$p={p}$", fontsize=10)

    fig.suptitle(r"Real Laguerre-Gaussian modes $\psi_{p,\ell}$", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(os.path.dirname(__file__), "lg_modes_grid.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
