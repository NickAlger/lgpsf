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
    python make_figures.py --items lcurve,recon     # slice 3: items 6-7

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
from heat_inversion import (build_problem, conductivity, pcg,  # noqa: E402
                            probes, regions)

FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "cache")
GRID = 32
NOISE = 0.015          # relative noise on the observed final-time field
SEED = 20260803
KS = (10, 20, 50, 150)  # the probe-budget ladder (items 5 and 8)
QC_SEED = 101           # held-out probes: never the fitter's seed

FIT_K = 50              # the paper's canonical probe budget
VP_APPLIES = 30         # value-pass budget on top of the fit
A0 = 1e-5               # build shift: the smallest a we ANTICIPATE needing
SWEEP = np.logspace(-2, -7, 11)   # top-down; the tail below A0 is diagnostic


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
    # The smooth LG expansion is the fit. The additive spike term is a
    # trick for undermeshed regimes; this problem's narrowest PSFs are
    # still about a cell wide, so it stays off.
    config.spike = False
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


def corrected_operator(problem):
    """The paper's canonical build on the k = FIT_K fit: certify positive
    definiteness at A0, then spend VP_APPLIES true Hessian applies pricing
    the fit's residual error. Returns the struct and the module handle."""
    import lgpsf
    corr = lgpsf.corrections
    B, V, HV = fitted_operator(problem, FIT_K)
    B = sp.csc_matrix(B)

    # snippet:build
    # One certified operator struct from the fit and its probe archive.
    # make_pd flips the few pencil modes below the build threshold (an
    # Hr-solve currency -- no PDE solves) and certifies a floor: the
    # shifts a at which B + E + a Hr is provably positive definite. The
    # value pass then spends a small budget of TRUE Hessian applies
    # pricing the error the fit left behind.
    archive = corr.ProbeArchive()
    archive.Z, archive.Y = V.T.copy(), HV.T.copy()
    A = corr.make_shifted_operator(corr.sparse_op(B), archive,
                                   corr.sparse_hr_oracle(problem["Hr"]), A0)
    report = corr.make_pd(A, max_iters=3000)
    assert report.certified
    Hd_op = corr.SymmetricOp(problem["count"], problem["apply_Hd"])
    corr.value_pass(A, Hd_op, VP_APPLIES, corr.ValuePassMode.V2)
    # end snippet

    print(f"build at a0 = {A0:.0e}: {report.flipped} flips, certified "
          f"floor {-A.lambda_floor:.2e}, block rank {A.block.rank}")
    return A, corr


def regularization_sweep(problem):
    """Items 6-7 compute: the a-sweep on ONE build, cached to disk."""
    path = os.path.join(CACHE, "sweep.npz")
    if os.path.exists(path):
        return dict(np.load(path))

    A, corr = corrected_operator(problem)
    data, Hr = problem["data"], problem["Hr"]
    rhs = problem["spacing"] ** 2 * problem["propagate"](data)
    target = problem["noise_level"] * np.sqrt(problem["count"])
    floor_before = -A.lambda_floor

    rows, solutions = [], []

    def solve_at(a):
        u, history = pcg(lambda v: problem["apply_Hd"](v) + a * (Hr @ v),
                         lambda r: corr.solve(A, r[:, None], a,
                                              mode=corr.SolveMode.TwoLevel,
                                              rtol=1e-10).X.ravel(),
                         rhs, rtol=1e-10, return_solution=True)
        return u, len(history) - 1

    def record(a, u, iterations):
        mis = np.linalg.norm(problem["propagate"](u) - data)
        semi = np.sqrt(u @ (Hr @ u))
        rows.append((a, iterations, mis, semi))
        solutions.append(u)
        print(f"  a {a:.2e}  its {iterations:3d}  misfit {mis:.4e}  "
              f"seminorm {semi:.4f}")

    # snippet:sweep
    # The whole sweep runs on the one build: changing the shift is scalar
    # arithmetic, zero refactorization. Each solve is outer PCG on the
    # TRUE operator (one PDE-based Hessian apply per iteration) with the
    # corrected struct as the preconditioner. When the diagnostic tail
    # crosses below the certified floor the solve is REFUSED, and
    # rebuild_at re-anchors every contract from the archive: new flips
    # cost Hr-solves only, and the value-pass pairs refold in exactly,
    # with zero new PDE solves.
    for a in SWEEP:
        if corr.classify_shift(A, a).zone == corr.Zone.Refused:
            report = corr.rebuild_at(A, SWEEP.min(), max_iters=3000)
        u, iterations = solve_at(a)
        record(a, u, iterations)
    # end snippet

    rebuild_flips = report.flip.flipped if "report" in locals() else 0
    print(f"rebuild_at({SWEEP.min():.0e}): {rebuild_flips} new flips, "
          f"floor {floor_before:.2e} -> {-A.lambda_floor:.2e}, value pairs "
          "refolded with 0 new PDE solves")

    a_grid, its, mis, semi = (np.array([r[c] for r in rows])
                              for c in range(4))
    # the discrepancy-principle corner: interpolate the crossing, solve it
    j = int(np.argmax(mis[::-1] >= target))          # sweep is descending
    hi, lo = len(mis) - j - 1, len(mis) - j
    frac = (np.log(target / mis[lo])
            / np.log(mis[hi] / mis[lo]))
    a_star = float(np.exp(np.log(a_grid[lo])
                          + frac * np.log(a_grid[hi] / a_grid[lo])))
    u_star, its_star = solve_at(a_star)

    rebuilt_from = (int(np.argmax(a_grid <= floor_before))
                    if (a_grid <= floor_before).any() else len(a_grid))
    os.makedirs(CACHE, exist_ok=True)
    blob = dict(a=a_grid, its=its, misfit=mis, seminorm=semi,
                target=target, a_star=a_star, u_star=u_star,
                its_star=its_star,
                mis_star=np.linalg.norm(problem["propagate"](u_star) - data),
                semi_star=np.sqrt(u_star @ (Hr @ u_star)),
                rebuilt_from=rebuilt_from, rebuild_flips=rebuild_flips,
                floor_before=floor_before, floor_after=-A.lambda_floor,
                u_small=solutions[-1], u_large=solutions[1])
    np.savez_compressed(path, **blob)
    return blob


def item_lcurve(problem):
    """Item 6: the L-curve from the one-build sweep, corner marked."""
    S = regularization_sweep(problem)
    a, mis, semi = S["a"], S["misfit"], S["seminorm"]
    split = int(S["rebuilt_from"])

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.loglog(mis, semi, "-", color="0.8", zorder=1)
    ax.loglog(mis[:split], semi[:split], "o", color="C0", zorder=2,
              label="solves on the corner-anchored build")
    ax.loglog(mis[split:], semi[split:], "s", color="C1", zorder=2,
              label=f"after rebuild_at({SWEEP.min():.0e})")
    ax.loglog(S["mis_star"], S["semi_star"], "k*", markersize=15, zorder=3,
              label=f"discrepancy corner, a* = {S['a_star']:.1e}")
    ax.axvline(S["target"], color="0.5", linestyle="--", linewidth=1)
    for j in (0, len(a) - 1):
        ax.annotate(f"a = {a[j]:.0e}", (mis[j], semi[j]), fontsize=8,
                    textcoords="offset points", xytext=(6, -4))
    ax.set_xlabel(r"data misfit  $\|Au_0 - d\|$")
    ax.set_ylabel(r"regularity  $|u_0|_{H_r}$")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lcurve.png"), dpi=180)
    plt.close(fig)
    print("wrote figures/lcurve.png")


def item_recon(problem):
    """Item 7: reconstructions at too-small / corner / too-large shifts."""
    S = regularization_sweep(problem)
    panels = [
        ("true $u_0$", problem["u0"]),
        (f"$a = ${SWEEP.min():.0e}  (too small)", S["u_small"]),
        (f"$a^* = ${S['a_star']:.1e}  (discrepancy)", S["u_star"]),
        (f"$a = ${SWEEP[1]:.0e}  (too large)", S["u_large"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.2))
    for ax, (title, u) in zip(axes, panels):
        field(ax, u, title, cmap="RdBu_r",
              vmax=np.abs(problem["u0"]).max())
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "recon.png"), dpi=180)
    plt.close(fig)
    print("wrote figures/recon.png")


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
    parser.add_argument("--items", default="problem,psfs,fit,lcurve,recon")
    args = parser.parse_args()
    os.makedirs(FIG, exist_ok=True)

    problem = problem_instance()
    stages = {"problem": item_problem, "psfs": item_psfs, "fit": item_fit,
              "lcurve": item_lcurve, "recon": item_recon}
    for item in args.items.split(","):
        stages[item](problem)
    extract_snippets()


if __name__ == "__main__":
    main()
