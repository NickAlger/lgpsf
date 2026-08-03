# SPDX-License-Identifier: MIT
"""Heat-equation initial-condition inversion: the shared problem for the
CORRECTIONS examples.

Not a lesson in itself -- the target the corrections examples operate on,
the way `frog_kernel.py` is the target for the fitting examples. The frog is
deliberately NON-symmetric (that is its lesson); the corrections layer is
about symmetric data-misfit Hessians, so it gets its own problem, and one
worth having: a small but genuine PDE-constrained inverse problem, end to
end.

**The problem.** Recover the initial condition of the heat equation from a
noisy observation of the solution everywhere at a short final time. The
thermal conductivity `kappa(x)` is piecewise constant with SHARP JUMPS --
three orders of magnitude between the conductive slab and the insulating
pocket. The data-misfit Hessian is

    H_d = h^2 A^T A,     A = (I + dt L_kappa)^{-steps}   (implicit Euler),

with `L_kappa = -div(kappa grad .)` (5-point stencil, harmonic face
averages, zero Dirichlet). A row of `H_d` is the point-spread function of
"diffuse for time 2T": a bump whose WIDTH goes like sqrt(T kappa) and whose
MAGNITUDE varies over orders of magnitude with kappa -- so the row scales
the weighted symmetrization exists for arise from the physics, not from a
contrivance. The jumps are what keeps the fit honest: away from them every
PSF is a clean anisotropy-free Gaussian the level-0 LG mode nails; a row
NEAR an interface has a two-sided, skewed PSF (heat floods into the
conductive side and stalls at the insulating one) that the higher modes
have to earn, and the residual error those rows leave behind is what the
deflation examples price.

The regularization operator `Hr` is the (kappa-free) Laplacian on the same
grid -- the H^1 seminorm a real inversion would use, and the right-hand
side of every pencil in the corrections layer. With `Hr = I` instead, the
pencil is just the Euclidean spectrum and the layer's central claim (a
mesh-independent threshold for the flip) has nothing to bite on.

As in `frog_kernel.py`, the dense `H` is built ONLY so the examples can
report errors against ground truth; `apply_Hd` is the object a real
application would have (each application = 2*steps sparse solves), and
`V`/`HV` from `probes` is all the fitter ever sees.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

KAPPA_BACKGROUND = 0.1
KAPPA_SLAB = 1.0        # conductive slab: PSFs broad and shallow
KAPPA_POCKET = 0.005    # insulating pocket: PSFs narrow and tall
TIME = 0.01             # diffusion time per propagator application
STEPS = 8               # implicit-Euler steps per application


def conductivity(x):
    """Piecewise-constant kappa at points x of shape (2, K): a conductive
    slab on the left, an insulating pocket lower-right, background between.
    The jumps are sharp on purpose (see the module docstring)."""
    kappa = np.full(x.shape[1], KAPPA_BACKGROUND)
    kappa[x[0] < 0.42] = KAPPA_SLAB
    kappa[(x[0] > 0.58) & (x[0] < 0.92) & (x[1] > 0.08) & (x[1] < 0.5)] = \
        KAPPA_POCKET
    return kappa


def diffusion_operator(grid, kappa_cells):
    """-div(kappa grad .) on the cell-centered grid, zero Dirichlet outside.

    Face conductivities are HARMONIC means of the neighboring cells -- the
    correct average across a jump (the arithmetic mean lets heat tunnel
    through an insulating cell).
    """
    h = 1.0 / grid
    kap = kappa_cells.reshape(grid, grid)
    rows, cols, vals = [], [], []
    index = np.arange(grid * grid).reshape(grid, grid)

    def face(ka, kb):
        return 2.0 * ka * kb / (ka + kb)

    diag = np.zeros((grid, grid))
    for axis in (0, 1):
        left = kap
        right = np.roll(kap, -1, axis=axis)
        conductance = face(left, right) / h**2
        for i in range(grid):
            for j in range(grid):
                if (axis == 0 and i == grid - 1) or (axis == 1 and j == grid - 1):
                    continue
                a = index[i, j]
                b = index[i + 1, j] if axis == 0 else index[i, j + 1]
                c = conductance[i, j]
                rows += [a, b]
                cols += [b, a]
                vals += [-c, -c]
                diag[i, j] += c
                diag.flat[b] += c
    # Dirichlet: boundary cells also couple to the zero exterior
    for i in range(grid):
        for j in range(grid):
            k = kap[i, j] / h**2
            edges = (i == 0) + (i == grid - 1) + (j == 0) + (j == grid - 1)
            diag[i, j] += edges * k
    rows += list(index.ravel())
    cols += list(index.ravel())
    vals += list(diag.ravel())
    return sp.csc_matrix((vals, (rows, cols)),
                         shape=(grid * grid, grid * grid))


def build_problem(grid=24, seed=0):
    """The grid, kappa, the dense truth, the matrix-free apply, and Hr."""
    axis = (np.arange(grid) + 0.5) / grid
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])       # (2, K)
    count = x.shape[1]
    spacing = 1.0 / grid
    mass = np.full(count, spacing**2)
    kappa = conductivity(x)

    L = diffusion_operator(grid, kappa)
    stepper = spla.splu(sp.eye(count, format="csc") + (TIME / STEPS) * L)

    def propagate(U):
        """A U for a block of columns: `steps` implicit-Euler solves each."""
        U = np.asarray(U, dtype=float)
        squeeze = U.ndim == 1
        V = U.reshape(count, -1)
        for _ in range(STEPS):
            V = stepper.solve(V)
        return V[:, 0] if squeeze else V

    def apply_Hd(U):
        """H_d U = h^2 A(A U): the matrix-free Hessian apply (A = A^T)."""
        return spacing**2 * propagate(propagate(U))

    A_dense = propagate(np.eye(count))
    H = spacing**2 * (A_dense.T @ A_dense)
    H = 0.5 * (H + H.T)                        # exactly symmetric

    # the a-priori ellipsoid field a physicist would supply: isotropic, width
    # from the LOCAL kappa -- honest away from the jumps, wrong next to them,
    # which is exactly the situation the circle rungs exist for
    sig2 = 2.0 * (2.0 * TIME) * kappa + (0.75 * spacing) ** 2
    sigma = np.stack([s * np.eye(2) for s in sig2])

    Hr = diffusion_operator(grid, np.ones(count)) * spacing**2

    return dict(x=x, mass=mass, H=H, sigma=sigma, count=count,
                spacing=spacing, seed=seed, kappa=kappa, Hr=Hr,
                apply_Hd=apply_Hd, propagate=propagate, grid=grid)


def probes(problem, num_probes, seed=None):
    """`num_probes` random probes and their responses -- all the fitter sees."""
    rng = np.random.default_rng(problem["seed"] if seed is None else seed)
    V = rng.normal(size=(num_probes, problem["count"]))       # (num_probes, K)
    return V, V @ problem["H"].T                              # (num_probes, R)


def lean_fit(problem, num_probes=30):
    """The shared starting point of the corrections examples: a deliberately
    LEAN fit (probes for ~5% of the rows -- the realistic budget regime),
    assembled with the weighted symmetrization. Returns the sparse matrix
    and the probe pairs, which the corrections layer archives."""
    import lgpsf
    V, HV = probes(problem, num_probes)
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4])
    config.row.target_score = None
    fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"],
                             V, HV, problem["sigma"], config=config)
    B_sparse = lgpsf.assemble_sparse(fit.model, 6.0,
                                     lgpsf.Symmetrize.Weighted)
    return B_sparse, V.T.copy(), HV.T.copy()   # pairs in COLUMNS convention


def pcg(apply_A, solve_P, b, maxit=300, rtol=1e-12, return_solution=False):
    """Textbook preconditioned CG, recording the relative residual history --
    the examples' referee for "how good is this preconditioner". With
    `return_solution` it also returns the iterate (the tutorial's consumer)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = solve_P(r)
    p_dir = z.copy()
    rz = r @ z
    history = [1.0]
    for _ in range(maxit):
        Ap = apply_A(p_dir)
        alpha = rz / (p_dir @ Ap)
        x += alpha * p_dir
        r -= alpha * Ap
        history.append(np.linalg.norm(r) / np.linalg.norm(b))
        if history[-1] < rtol:
            break
        z = solve_P(r)
        rz_next = r @ z
        p_dir = z + (rz_next / rz) * p_dir
        rz = rz_next
    if return_solution:
        return x, np.array(history)
    return np.array(history)


def iterations_to(history, tolerances):
    """First iteration at which the residual drops below each tolerance."""
    return [int(np.argmax(history < t)) if (history < t).any() else -1
            for t in tolerances]


def regions(problem):
    """Masks for the narrative tables: slab, pocket, background, and the
    one-cell-wide band on each side of a jump (where the PSFs are skewed)."""
    x, kappa, h = problem["x"], problem["kappa"], problem["spacing"]
    near = np.zeros(problem["count"], dtype=bool)
    for shift in ([h, 0], [-h, 0], [0, h], [0, -h]):
        shifted = x + np.asarray(shift)[:, None]
        inside = ((shifted >= 0) & (shifted <= 1)).all(axis=0)
        moved = conductivity(np.clip(shifted, 1e-9, 1 - 1e-9)) != kappa
        near |= inside & moved
    return dict(slab=(kappa == KAPPA_SLAB) & ~near,
                pocket=(kappa == KAPPA_POCKET) & ~near,
                background=(kappa == KAPPA_BACKGROUND) & ~near,
                interface=near)
