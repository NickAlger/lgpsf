# SPDX-License-Identifier: MIT
"""The single driver behind the "LGPSF Heat Inverse Problem Tutorial".

Every figure and every number in `heat-pipeline.tex` is produced here, and
every code snippet the paper shows is EXTRACTED from this file (between
`# snippet:name` / `# end snippet` markers) rather than transcribed -- the
committed PDF is built from the exact code that made its figures.

The problem is `examples/heat_inversion.py` at grid = 32 (n = 1024).
Stages map to the paper's items and run independently:

    python make_figures.py --items problem,psfs     # slice 1: items 1-4
    python make_figures.py --items fit              # slice 2: item 5 + QC

Environment: numpy + scipy + matplotlib for the early items (the `tttt`
env on the maintainer's machine); the fitting items additionally need the
lgpsf bindings on PYTHONPATH (rebuild `build-py39` at library tip for the
matplotlib-bearing environment). Fits are cached under `cache/` (not
committed); delete it to regenerate everything from scratch.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
from heat_inversion import (build_problem, conductivity, probes,  # noqa: E402
                            regions)

FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "cache")
GRID = 32
NOISE = 0.015          # relative noise on the observed final-time field
SEED = 20260803
KS = (10, 20, 50, 150)  # the probe-budget ladder (items 5 and 8)
QC_SEED = 101           # held-out probes: never the fitter's seed


def problem_instance():
    """The problem plus the paper's synthetic truth and data."""
    problem = build_problem(grid=GRID)
    x, g = problem["x"], GRID

    # snippet:true_ic
    # The true initial condition carries features at TWO scales: one broad
    # blob spanning the conductive slab's boundary, and four sharp spots --
    # in the pocket, the background, and the slab. The scale separation is
    # what the final figure of the paper is built to expose.
    def gaussian(center, width, amplitude=1.0):
        d2 = ((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
        return amplitude * np.exp(-0.5 * d2 / width**2)

    u0 = (gaussian((0.42, 0.62), 0.16)
          + 0.9 * gaussian((0.75, 0.30), 0.02)     # in the insulating pocket
          + 0.8 * gaussian((0.70, 0.75), 0.02)     # in the background
          - 0.7 * gaussian((0.22, 0.35), 0.02)     # in the slab, signed
          + 0.8 * gaussian((0.15, 0.80), 0.02))    # in the slab
    # end snippet

    # snippet:data
    # Observe the diffused field everywhere at the final time, with iid
    # noise at 1.5% of the signal's RMS -- the data a real inversion holds.
    rng = np.random.default_rng(SEED)
    observed = problem["propagate"](u0)                        # A u0
    noise = NOISE * np.sqrt(np.mean(observed**2))
    data = observed + noise * rng.standard_normal(observed.shape)
    # end snippet
    problem.update(u0=u0, data=data, noise_level=noise)
    return problem


def field(ax, values, title, cmap="viridis", log=False, vmax=None):
    img = values.reshape(GRID, GRID).T
    kw = dict(origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
    if log:
        from matplotlib.colors import LogNorm
        handle = ax.imshow(img, norm=LogNorm(), **kw)
    else:
        vmax = vmax if vmax is not None else np.abs(img).max()
        handle = ax.imshow(img, vmin=-vmax, vmax=vmax, **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(handle, ax=ax, fraction=0.046, pad=0.03)


def item_problem(problem):
    """Items 1-3: the truth, the conductivity, and the noisy data."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    field(axes[0], problem["u0"], "true initial condition $u_0$",
          cmap="RdBu_r")
    field(axes[1], problem["kappa"], r"conductivity $\kappa$ (log scale)",
          cmap="magma", log=True)
    field(axes[2], problem["data"],
          f"noisy data $d = A u_0 + \\eta$  ({100*NOISE:.1f}% noise)",
          cmap="RdBu_r")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "problem.png"), dpi=180)
    plt.close(fig)
    print("wrote figures/problem.png")


def item_psfs(problem):
    """Item 4: sampled impulse responses -- local, but not Gaussian."""
    # snippet:psf_rows
    # A row of the Hessian is the point-spread function of "diffuse for
    # time 2T", anchored at that row's grid point. Sampling one costs two
    # PDE solves -- H_d e_i by the matrix-free apply.
    def psf(index):
        impulse = np.zeros(problem["count"])
        impulse[index] = 1.0
        return problem["apply_Hd"](impulse)
    # end snippet

    x, kappa = problem["x"], problem["kappa"]

    def nearest(px, py):
        return int(np.argmin((x[0] - px) ** 2 + (x[1] - py) ** 2))

    spots = [  # (label, index, zoom half-width)
        ("slab interior", nearest(0.20, 0.55), 0.34),
        ("slab-side interface", nearest(0.39, 0.45), 0.20),
        ("background-side interface", nearest(0.42, 0.45), 0.20),
        ("background", nearest(0.52, 0.75), 0.17),
        ("pocket edge", nearest(0.60, 0.30), 0.14),
        ("pocket interior", nearest(0.75, 0.25), 0.14),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0))
    axes[1, 3].axis("off")
    ax = axes[0, 0]
    img = np.log10(kappa).reshape(GRID, GRID).T
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    for tag, (label, idx, _) in zip("abcdef", spots):
        ax.plot(x[0, idx], x[1, idx], "o", color="cyan", markersize=5)
        ax.annotate(tag, (x[0, idx], x[1, idx]), color="cyan",
                    textcoords="offset points", xytext=(5, 4), fontsize=11)
    ax.set_title(r"locations on $\log_{10}\kappa$", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    panels = [axes[0, 1], axes[0, 2], axes[0, 3],
              axes[1, 0], axes[1, 1], axes[1, 2]]
    for tag, (label, idx, pad), ax in zip("abcdef", spots, panels):
        row = psf(idx)
        img = row.reshape(GRID, GRID).T
        ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="inferno",
                  vmin=0.0, vmax=img.max())
        ax.plot(x[0, idx], x[1, idx], "+", color="cyan", markersize=8)
        # zoom to the PSF's own neighborhood so the shapes are legible
        ax.set_xlim(max(0, x[0, idx] - pad), min(1, x[0, idx] + pad))
        ax.set_ylim(max(0, x[1, idx] - pad), min(1, x[1, idx] + pad))
        ax.set_title(f"({tag}) {label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "psfs.png"), dpi=180)
    plt.close(fig)
    print("wrote figures/psfs.png")


def fitted_operator(problem, k):
    """The paper's fit at probe budget k, cached on disk together with its
    probe pairs (the corrections layer archives those later)."""
    path = os.path.join(CACHE, f"fit_k{k}.npz")
    if os.path.exists(path):
        blob = np.load(path)
        B = sp.csr_matrix((blob["data"], blob["indices"], blob["indptr"]),
                          shape=tuple(blob["shape"]))
        return B, blob["V"], blob["HV"]

    # snippet:fit
    # The fitter's entire budget: k random probes and their Hessian
    # applies. Each response costs one H_d application (two PDE solves);
    # nothing else about H_d is ever observed.
    import lgpsf
    V, HV = probes(problem, k)
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4])
    config.row.target_score = None
    fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"],
                             V, HV, problem["sigma"], config=config)
    B = lgpsf.assemble_sparse(fit.model, 6.0, lgpsf.Symmetrize.Weighted)
    # end snippet

    B = sp.csr_matrix(B)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, data=B.data, indices=B.indices,
                        indptr=B.indptr, shape=np.array(B.shape),
                        V=V, HV=HV)
    return B, V, HV


def item_fit(problem):
    """Item 5: one interface PSF against its fits as the budget grows,
    plus the held-out QC table by region."""
    # A slab-side interface row, chosen because its held-out scores track
    # the interface-region MEDIAN at every budget -- a representative hard
    # row, not a best case. (Individual rows are not guaranteed monotone
    # in k; the QC score is what flags the exceptions.)
    x = problem["x"]
    idx = int(np.argmin((x[0] - 0.39) ** 2 + (x[1] - 0.58) ** 2))
    truth_row = problem["H"][idx]

    # snippet:qc
    # Held-out quality control: score the assembled operator's rows
    # against probes the fitter never saw (the same referee the library
    # uses internally, applied to the deployed object).
    V_qc, HV_qc = probes(problem, 30, seed=QC_SEED)
    def qc_scores(B):
        residual = np.linalg.norm(B @ V_qc.T - HV_qc.T, axis=1)
        return residual / np.linalg.norm(HV_qc.T, axis=1)
    # end snippet

    fits, scores = {}, {}
    for k in KS:
        fits[k], _, _ = fitted_operator(problem, k)
        scores[k] = qc_scores(fits[k])

    # -- the figure: truth vs fits at the slab-side interface row ---------
    fig, axes = plt.subplots(1, 1 + len(KS), figsize=(14.5, 3.1),
                             constrained_layout=True)
    vmax = np.abs(truth_row).max()          # one scale, anchored to truth
    panels = [("sampled row (truth)", truth_row, None)]
    panels += [(f"fit, k = {k}", np.asarray(fits[k][idx].todense()).ravel(),
                scores[k][idx]) for k in KS]
    pad = 0.20
    for ax, (title, row, q) in zip(axes, panels):
        img = row.reshape(GRID, GRID).T
        handle = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot(x[0, idx], x[1, idx], "+", color="k", markersize=8)
        ax.set_xlim(x[0, idx] - pad, x[0, idx] + pad)
        ax.set_ylim(x[1, idx] - pad, x[1, idx] + pad)
        label = title if q is None else f"{title}  (row QC {q:.2f})"
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(handle, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(os.path.join(FIG, "psf_fit.png"), dpi=180)
    plt.close(fig)
    print("wrote figures/psf_fit.png")

    # -- the QC table: median held-out score per region, per budget -------
    masks = regions(problem)
    order = ["slab", "interface", "background", "pocket"]
    print(f"{'k':>5}  " + "".join(f"{name:>11}" for name in order)
          + f"{'overall':>11}")
    rows = []
    for k in KS:
        q = scores[k]
        cells = [np.median(q[masks[name]]) for name in order]
        cells.append(np.median(q))
        rows.append((k, cells))
        print(f"{k:>5}  " + "".join(f"{c:>11.3f}" for c in cells))

    out = os.path.join(HERE, "snippets")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "qc_table.tex"), "w") as handle:
        handle.write("\\begin{tabular}{r cccc c}\n\\hline\n")
        handle.write("$k$ & " + " & ".join(order)
                     + " & overall \\\\\n\\hline\n")
        for k, cells in rows:
            handle.write(f"{k} & "
                         + " & ".join(f"{c:.3f}" for c in cells)
                         + " \\\\\n")
        handle.write("\\hline\n\\end{tabular}\n")
    print("wrote snippets/qc_table.tex")


def extract_snippets():
    """Write snippets/<name>.tex from the markers in this file."""
    out = os.path.join(HERE, "snippets")
    os.makedirs(out, exist_ok=True)
    lines = open(__file__).read().splitlines()
    name, body = None, []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# snippet:"):
            name, body = stripped.split(":", 1)[1], []
        elif stripped == "# end snippet" and name:
            indent = min((len(b) - len(b.lstrip()) for b in body if b.strip()),
                         default=0)
            with open(os.path.join(out, name + ".tex"), "w") as handle:
                handle.write("\\begin{verbatim}\n")
                handle.write("\n".join(b[indent:] for b in body) + "\n")
                handle.write("\\end{verbatim}\n")
            print(f"wrote snippets/{name}.tex")
            name = None
        elif name is not None:
            body.append(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="problem,psfs,fit")
    args = parser.parse_args()
    os.makedirs(FIG, exist_ok=True)

    problem = problem_instance()
    stages = {"problem": item_problem, "psfs": item_psfs, "fit": item_fit}
    for item in args.items.split(","):
        stages[item](problem)
    extract_snippets()


if __name__ == "__main__":
    main()
