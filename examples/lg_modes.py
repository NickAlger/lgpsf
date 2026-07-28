# SPDX-License-Identifier: MIT
"""What a Laguerre-Gaussian mode looks like.

Every fit lgpsf produces is a weighted sum of these. Before meeting them as
config vocabulary -- `ShellLadder`, `max_level`, `Mode(p, ell, m)` -- it helps
to see what they are.

A mode is indexed by `(p, ell, m)`: `ell` is the ANGULAR order (how many lobes
go around), `p` the RADIAL order (how many rings go out), and `m` selects among
the `ell`-th harmonics -- in 2-D there are two for each `ell > 0`, the cos and
sin branches. All of them are modulated by the same Gaussian, and they are
orthonormal on R^N.

The energy of a mode is its OSCILLATOR LEVEL, `2p + ell`. That single number
orders the whole basis and is what "level" means everywhere else in the
library: `modes_up_to_level(dim, L)` returns every mode with `2p + ell <= L`.
Cheap, smooth modes first.

    python examples/lg_modes.py

Writes lg_modes.png if matplotlib is available; prints the mode table either
way.
"""
import numpy as np

import lgpsf

P_MAX, ELL_MAX = 3, 3
HALF_WIDTH, RESOLUTION = 4.0, 161


def mode_grid(p, ell, half_width=HALF_WIDTH, resolution=RESOLUTION):
    """One mode sampled on a square, as a (resolution, resolution) image.

    `eval_lg` is the 2-D reference form and takes the two coordinates
    separately, as flat arrays; the sign of `ell` picks the branch (positive =
    cos, negative = sin). For real work use `eval_lg_basis`, which evaluates a
    whole mode SET at once and shares the work between modes.
    """
    axis = np.linspace(-half_width, half_width, resolution)
    u1, u2 = np.meshgrid(axis, axis, indexing="ij")
    return lgpsf.eval_lg(p, ell, u1.ravel(), u2.ravel()).reshape(u1.shape)


def main():
    print(f"{'mode':>16}  {'level 2p+ell':>12}  {'max |psi|':>10}")
    for p in range(P_MAX + 1):
        for ell in range(ELL_MAX + 1):
            field = mode_grid(p, ell)
            print(f"  (p={p}, ell={ell})  {2 * p + ell:>12}  "
                  f"{np.abs(field).max():>10.4f}")

    # The library's own enumeration, in the order a ShellLadder walks it.
    for level in range(3):
        modes = lgpsf.modes_up_to_level(2, level)
        print(f"modes_up_to_level(2, {level}) -> {len(modes)} modes")

    try:
        import matplotlib
    except ImportError:
        print("\n(matplotlib not installed -- skipping the figure)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(range(-ELL_MAX, ELL_MAX + 1))
    fig, axes = plt.subplots(P_MAX + 1, len(columns),
                             figsize=(1.5 * len(columns), 1.5 * (P_MAX + 1)))
    for row, p in enumerate(range(P_MAX + 1)):
        for col, ell in enumerate(columns):
            field = mode_grid(p, ell)
            limit = np.abs(field).max()
            ax = axes[row, col]
            ax.imshow(field.T, origin="lower", cmap="RdBu_r",
                      vmin=-limit, vmax=limit)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$\\ell={ell}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"$p={p}$", fontsize=9)
    fig.suptitle("Laguerre-Gaussian modes: angular order across, "
                 "radial order down\n(negative $\\ell$ is the sin branch)")
    fig.tight_layout()
    fig.savefig("examples/lg_modes.png", dpi=130)
    print("\nwrote examples/lg_modes.png")


if __name__ == "__main__":
    main()
