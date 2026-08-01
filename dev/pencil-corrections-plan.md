# Pencil corrections layer — implementation plan

Plan for a post-fit corrections layer: the machinery that turns a fitted
sparse operator into a deployable SPD preconditioner / proposal covariance
for systems $H_d + a H_r$, with $H_r$ a consumer-supplied SPD
regularization operator. Distilled from the maintainer's deflation research program on
the Pine Island Glacier (PIG) ice-sheet Hessian. The design was worked
out in a joint session 2026-07-31; this document is self-contained —
every observation that motivates a design choice is quoted inline, so a
fresh session can implement from it without access to anything private.
Mathematical companion:
[`pencil-corrections-notes.pdf`](pencil-corrections-notes.pdf) — the
operations, ingredients, and formulas from the basics, including the
two-coefficient mode block and the contradictions that forced it.

**Scope decision.** Three things move into the library: (1) weighted
symmetrization, into the existing assembly path; (2) PSD-ification
(eigenvalue flip / relu) of the fit; (3) two deflation constructions —
free-residual Rayleigh–Ritz and the two-step value pass. Everything else
from the research program stays research (§10). The corrections layer is
on-theme for lgpsf because its key input — the fit's deployment residuals
— is a byproduct that exists only where the fit lives; a downstream
consumer cannot reconstruct it without re-plumbing lgpsf internals.

---

## 1. Design principles

**P1 — every spectral operation is a pencil operation.** The typical
target $H_d$ is an integral-kernel operator with numerical rank $r$ fixed
by the physics, discretized at dimension $N$ with $N \to \infty$ under
mesh refinement (PIG coarse: $r \sim 10^3$–$2\cdot10^3$, $N \sim 6.5
\cdot 10^3$; Antarctica: $r \sim 10^4$, $N \sim 10^6$). Euclidean
spectral surgery on the fit $\tilde B$ is ill-posed in this limit: the
Euclidean spectrum accumulates at 0 from both sides, with $N - r$
jitter-split noise eigenvalues and no principled threshold. The
generalized pencil $(\tilde B, H_r)$ is well-posed: its eigenvalues are
dimensionless data-to-prior ratios, mesh-independent, accumulating at 0
with only the physics-determined informed modes away from it. The
deflation pencil $(H_d + aH_r,\ \tilde B + aH_r)$ likewise accumulates at
1. **Rule: never enumerate modes near an accumulation point.** Every
operation below touches only a physics-bounded, mesh-independent set.

**P2 — no $N$-sized factorization in the required path.** Sparse Cholesky
of $\tilde B + aH_r$ is only scalable in 2-D, and the fit's interesting
regime — informative smooth directions, hence large ellipsoids, hence
wide stencils — is sparse factorization's worst case. The foundation is
four primitives: apply $\tilde B$ (a caller-supplied symmetric operator —
P4, §7), apply $H_r$ (cheap), **solve $H_r$ to a tolerance** (the
consumer's oracle — in practice Krylov preconditioned by multigrid), and
rank-sized dense algebra. $\rho \times \rho$ dense Cholesky of
capacitance matrices is fine; $N$-sized sparse Cholesky is an optional
backend (§8).

**P3 — one durable struct, functions over it.** The durable data is: a
reference to the sparse fit, the probe archive, ONE $H_r$-orthonormal
low-rank mode block with provenance tags, and a few scalars. Within the
block, distinguish *cache* (modes recomputable from $\tilde B$ + oracle:
the fit's own pencil modes, flip modes) from *information* (deflation /
value-pass modes, whose only ground truth is the probe archive).
Correctness never depends on cache freshness; the block is extendable and
rebuildable on demand.

**P4 — the layer is operator-blind.** Decided 2026-07-31, after auditing
every operation below: nothing in the required path reads a matrix entry.
Pencil Lanczos, the flip, both deflations, the zone checks, and both
solve modes need only the four primitives of P2, so the layer takes
$\tilde B$ as an opaque *symmetric operator* — dimension plus block
matvec — and never sees the sparse format. Four reasons: the audit says
nothing is lost; the same layer then serves other approximation formats
(a block-low-rank format, a dense test operator, a shell around someone
else's code); an entry-free layer ports to distributed memory by swapping
vector types, because its $N$-sized objects are tall-skinny blocks with
rank-sized Gram reductions — exactly the shapes that distribute; and the
restriction enforces operator-level thinking, with no entry tricks that
cannot scale. Two boundary consequences. First, **symmetrization stays
format-side**: the weighted formula of §2 is per-entry and cannot be
composed from matvecs, so each format owns producing its symmetric
operator (lgpsf's is `assemble_sparse` with `Symmetrize::Weighted`).
Second, **no transpose matvec** crosses the boundary: everything here is
symmetric, and the layer *verifies* that with a seeded stochastic check
($x^T(By)$ vs $y^T(Bx)$, matvecs only) at construction — catching an
as-fitted operator handed in by mistake, which the assembly layer itself
cannot do. A cost-model corollary for the docs: GLR-mode deployment (§6)
performs zero $\tilde B$ applies, so an expensive matvec only pays during
builds and in two-level mode.

---

## 2. Weighted symmetrization (core change, independent of the rest)

The raw assembled fit $A$ is nonsymmetric (each row is fitted
independently). Plain $(A + A^T)/2$ lets a strong row's noise overwrite a
weak row's signal where their stencils overlap. The scale-aware weighted
symmetrization is a per-entry convex average with inverse-row-energy
weights:

$$
\tilde B_{ij} \;=\; \frac{w_i^2 A_{ij} + w_j^2 A_{ji}}{w_i^2 + w_j^2},
\qquad
w_i^2 \;=\; \frac{1}{\lVert A_{i,:}\rVert_2^2 +
  \bigl(10^{-2}\,\mathrm{median}_i \lVert A_{i,:}\rVert_2\bigr)^2},
$$

i.e. the weak row owns the entries a strong row only grazes. Exactly
symmetric by construction; observed insensitive to the $10^{-2}$ floor
over two decades on the PIG Hessian, where plain averaging let strong
rows' tails overwrite weak rows' signal. The scipy formula the method was
validated with is pinned in-repo by the binding test.

**Change:** add `Symmetrize::Weighted` to the existing enum consumed by
`assemble_sparse`, implemented sparsely in C++, and make it the
documented recommendation for SPD downstream use. Small, self-contained,
first implementation slice.

---

## 3. The durable struct

`ShiftedOperator` (renamed from the earlier working name `ShiftedFit` —
it no longer holds a fit). Fields:

| field | contents | durability |
|---|---|---|
| `op` | owned type-erased handle to the symmetric operator $\tilde B$ (§7) | never serialized — the caller re-supplies it on load; for lgpsf, reassembly from the saved fit is bit-identical |
| `archive` | probe pairs $(Z, Y)$, $Y = H_d Z$; QC pairs; any value-pass pairs $(Q, H_d Q)$ | irreplaceable — only trace of $H_d$ |
| `block` | $H_r$-orthonormal vectors $V$ ($V^T H_r V = I$), symmetric coefficient matrix $C$, per-mode provenance tag (`pencil-cache` / `flip` / `deflation` / `value-pass`) | rebuildable cache + derived information |
| scalars | $a_0$ (build shift), $\gamma$ (flip safety factor), $\lambda_{\mathrm{floor}}$ (leftmost surviving pencil eigenvalue), clamp floor | contract parameters |

The operator the struct represents is
$\;\tilde B + H_r V C V^T H_r + a H_r\;$ for a *caller-supplied* $a$
(§7). The block is one object: flip modes, inner-preconditioner pencil
modes, and deflation modes are all the same kind — $H_r$-orthonormal
vectors with coefficients — merged by $H_r$-Gram re-orthonormalization
(rank-sized Gram, $H_r$ applies only; coefficients transform by
congruence).

**Why $H_r$-orthonormal:** with $W = H_r V$ and $M_k(a) = aH_r + WCW^T$,
the Woodbury inverse closes over the oracle with *diagonal* $a$-dependence.
Eigendecompose $C = U \Theta U^T$ once (rank-sized); then

$$
M_k(a)^{-1} x \;=\; \tfrac1a H_r^{-1} x
\;-\; V U\, \operatorname{diag}\!\Bigl(\tfrac{\theta_i}{a(a+\theta_i)}\Bigr)\,
U^T V^T x ,
$$

one oracle solve plus $O(N\rho)$ per application, and changing $a$ costs
scalar arithmetic — no refactorization anywhere in an L-curve sweep.
$M_k(a) \succ 0 \iff a > -\min_i \theta_i$: an analytic PD certificate.

---

## 4. Spectral operations

### 4.1 `extend_modes` — generalized Lanczos on $(\tilde B, H_r)$

Both ends of the pencil spectrum come from one machine: Lanczos (or
LOBPCG) on $H_r^{-1} \tilde B$ using oracle solves, with $H_r$-inner
products, full reorthogonalization against the existing block, and a
solve-tolerance argument passed through to the oracle. Rightmost modes
feed the GLR preconditioner (§6); leftmost modes feed the flip (§4.2).
Cost is $O(\#\text{modes})$ oracle solves + sparse applies — no $H_d$
access, mesh-independent counts (P1). Modes land in the block tagged
`pencil-cache`.

### 4.2 `make_pd` — flip / relu at an $a_0$-tied threshold

Let $(\lambda_i, v_i)$ be pencil eigenpairs, $\tilde B v_i = \lambda_i
H_r v_i$. With $w_i = H_r v_i$, correct exactly the modes below the
threshold $-\gamma a_0$, $\gamma \in (0,1)$ (default $\tfrac12$):

$$
\tilde B_{\mathrm{pd}} \;=\; \tilde B \;-\; c \sum_{\lambda_i < -\gamma a_0}
\lambda_i\, w_i w_i^T ,
\qquad c = 2 \ (\text{flip}) \ \text{or}\ c = 1 \ (\text{relu}),
$$

implemented as block entries (tag `flip`) — the sparse matrix is never
modified. The slightly-negative noise tail is deliberately left in place:
the deployment shift absorbs it. Record $\lambda_{\mathrm{floor}} \in
[-\gamma a_0, 0]$, the leftmost surviving pencil eigenvalue (a byproduct
of the same Lanczos run). The contract is then **exact and
data-dependent**:

$$
\tilde B_{\mathrm{pd}} + a H_r \;\succ\; 0
\quad\Longleftrightarrow\quad a > -\lambda_{\mathrm{floor}},
\qquad\text{and}\qquad
\tilde B_{\mathrm{pd}} + a_0 H_r \;\succeq\; (1-\gamma)\,a_0 H_r .
$$

Exact PSD of $\tilde B$ alone ($a_0 = 0$) is documented as out of scope:
in the $N \to \infty$ limit it requires touching unboundedly many noise
modes. The contract is always relative to $(H_r, \gamma a_0)$.

*Observed (PIG, dense comparator):* flip vs relu differed by ≤ 1 PCG
iteration at every depth measured. Default: flip; relu as an option.

### 4.3 `deflate_free` — Rayleigh–Ritz on the fit's own residuals

Zero extra operator access. The deployment residuals $R = Y - \tilde B Z$
are exact samples of the error action. In the metric $M_0 =
\tilde B_{\mathrm{pd}} + a_0 H_r$ (whitened error $D = M_0^{-1/2} (H_d -
\tilde B_{\mathrm{pd}}) M_0^{-1/2}$), the square-root-free form: compute
$M_0^{-1} R$ by the solve path of §6, $M_0$-orthonormalize it to
$\tilde Q$ (Cholesky of the Gram matrix $R^T M_0^{-1} R$), then

$$
T \;=\; \operatorname{sym}\!\bigl((\tilde Q^T R)\,
(\tilde Q^T M_0 Z)^{+}_{\mathrm{rcond}}\bigr),
$$

eigendecompose $T$, keep the top-$r$ modes by $|\theta|$, clamp
eigenvalues below at the clamp floor (default $-0.95$), and store the
correction $\widehat E = (M_0 \tilde Q)\, W \Theta W^T (M_0 \tilde Q)^T$
in the block (tag `deflation`, merged by $H_r$-Gram; $k$ inner solves
total). The truncated pseudoinverse is essential: with the raw
pseudoinverse the estimated values reached $27.7$ / $-10.3$ against a
certified extreme of $3.10$ and the corrected operator lost definiteness.

*Observed (PIG, $a_0 = 10^{-4}$):* the residual subspace is accurate
(principal cosines vs certified directions: maxima $0.989$–$0.994$); the
*values* are the limiting factor. Iteration counts (to $10^{-6}$, vs
undeflated 34): best $k{=}200$ setting ($\mathrm{rcond} = 3\cdot10^{-2}$,
$r \ge 35$) gave 22; U-shaped in rcond with the optimum drifting toward
stronger truncation as $k$ shrinks; **no setting helped at $k \le 20$**.
Ship with these caveats in the docs verbatim.

### 4.4 `value_pass` — spend $m$ true applies on the values

The consumer supplies `apply_Hd`; the basis is free. Default construction
(V1 of the research program): take the top-$r$ left singular basis of the
whitened residuals (Gram-based, no square roots), un-whiten to probe
directions, spend $m = r$ applies $H_d q_i$, form the small Rayleigh
matrix $T = Q^T D Q$ exactly on that basis, eigendecompose, clamp, store
(tag `value-pass`). The new pairs $(Q, H_d Q)$ enter the archive — they
are secant information, reusable by any later rebuild. A power-iteration
variant (V2: re-orthonormalize $D \cdot$ basis, $m = 2m_1$) is worth
implementing behind an option; a fresh Gaussian sketch (V3) was measured
and is not worth shipping.

*Observed (PIG):* at $k{=}100$, $m{=}50$: offline study predicted 6 / 18
(to $10^{-2}/10^{-6}$); the end-to-end rerun with true operator applies
measured **5 / 12 / 18 / 24** (to $10^{-2}/10^{-4}/10^{-6}/10^{-8}$) vs
undeflated 12 / 26 / 36 / 45 — surrogate-to-truth transfer within one
iteration, zero clamp events. At $k{=}20$, V1 saturates when $m$ reaches
the free-basis dimension; V2 keeps improving. Equal-budget allocation
observed: $k{=}50$ fit + 50 value applies beat a $k{=}100$ fit alone
(30 vs 39 at $10^{-6}$). Guidance for docs (not code): fit to
$k \sim 50$–$100$, then spend on the value pass.

---

## 5. Varying $a$ — semantics

The struct carries $a_0$; every operation on the struct takes $a$ as an
argument. Three zones, checked in this order:

| zone | status | mechanism |
|---|---|---|
| $a \ge a_0$ | guaranteed | build contract (§4.2 + clamp in $M_0$) |
| $-\lambda_{\mathrm{floor}} < a < a_0$ | allowed with warning | flip still PD-certified analytically; the *clamp* guarantee (set in the $a_0$ metric) no longer applies — certify at runtime (below) |
| $a \le -\lambda_{\mathrm{floor}}$ | refused | sparse part indefinite; offer `rebuild_at` |

Runtime certification in the warning zone, cheapest first: (i) analytic
sufficient condition — since $\tilde B_{\mathrm{pd}} \succeq
\lambda_{\mathrm{floor}} H_r$ and the block's pencil eigenvalues against
$H_r$ are $\operatorname{eig}(C)$ exactly ($V$ is $H_r$-orthonormal), the
full operator is PD if $\min_i \theta_i(C_{\mathrm{defl}}) > -(a +
\lambda_{\mathrm{floor}})$; (ii) if that fails, the exact check is a
rank-sized projected eigenproblem after $\rho$ oracle solves, and CG on
an indefinite operator diagnoses itself in any case.

**`rebuild_at(a_1)`, $a_1 < a_0$** — restores full guarantees with *no
new $H_d$ access*: (i) incremental leftmost Lanczos flips only the new
modes in $(-\gamma a_0, -\gamma a_1)$, warm-started by deflating the
already-found ones; (ii) `deflate_free` re-runs in the new metric from
the archived residuals; (iii) archived value-pass pairs fold in as secant
data. Cost: oracle solves and rank-sized algebra only. Usage guidance:
build at the smallest $a$ you anticipate — the guarantee covers the
entire upward sweep for free — and `rebuild_at` covers the unanticipated
case.

---

## 6. The solve path

Two deployment modes, both via the block; no $N$-sized factorization.

**GLR mode (default).** The preconditioner applied to the consumer's
Krylov solve of $(H_d + aH_r)x = b$ is $M_k(a)^{-1}$ from §3 directly:
one oracle solve + $O(N\rho)$ per application, trivially re-shiftable in
$a$. This is the architecture validated end-to-end in the ymir
integration at PIG scale.

**Two-level mode (accuracy upgrade, measure before defaulting).** Apply
$P(a)^{-1} = (\tilde B_{\mathrm{pd}} + \widehat E + aH_r)^{-1}$ by an
inner Krylov iteration on $P(a)$ itself (sparse + oracle-cheap applies),
preconditioned by $M_k(a)^{-1}$. Inner conditioning is governed by $1 +
\lambda_{k+1}/a$ with $\lambda_{k+1}$ the first pencil eigenvalue *not*
in the block — pin it by extending modes (§4.1), which costs no $H_d$
access. The outer solver must then be flexible (FCG/FGMRES); document
this contract prominently. This mode uses the sparse apply's
beyond-rank-$k$ accuracy, which GLR mode discards.

The division of labor worth stating in the docs: the sparse fit is the
*information-compression* device ($k \sim 100$ probe applies stand in for
$r \sim 10^3$ spectral applies), and its pencil decomposition —
obtainable from sparse applies and oracle solves alone — is the
*deployment* device.

---

## 7. The operator boundary and API (C++17, header-only)

The layer lives in its own subdirectory and namespace, a **sibling** of
the fit stack rather than a downstream consumer of it:

```
include/lgpsf/corrections/
  symmetric_op.hpp      SymmetricOp + adapters + symmetry check   (bottom, peer)
  hr_oracle.hpp         HrOracle + sparse reference adapter       (bottom, peer)
  mode_block.hpp        needs hr_oracle
  pencil_lanczos.hpp    needs both
  shifted_operator.hpp  the struct of §3 + GLR-mode apply/solve
  deflation.hpp         deflate_free + value_pass
  cholesky_backend.hpp  optional; the only file allowed to see entries
```

C++ namespace `lgpsf::corrections`; Python `lgpsf.corrections`
submodule. `corrections/*` includes no fit header and no fit header
includes `corrections/*` — enforced mechanically in
`tools/check_dependencies.py`. lgpsf's native path into the layer is one
adapter call: `assemble_sparse(fit, tau, Symmetrize::Weighted)` wrapped
by `sparse_op`.

The two boundary types are small **type-erased classes**, not duck-typed
templates: dispatch cost is nil at block-matvec granularity, the struct
needs an owned storable handle, the layer compiles once rather than per
operator type, and the Python adapter becomes trivial. Blocks are
**columns**, $(N, m)$ — the corrections layer speaks linear algebra,
unlike the fit layer's probes-as-rows convention; the adapters are where
the transposition happens, documented there.

```cpp
class SymmetricOp {          // type-erased: dimension + block matvec
  Eigen::Index dim() const;
  Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& X) const;
};
SymmetricOp sparse_op(Eigen::SparseMatrix<double> A);   // owns its copy
SymmetricOp dense_op(Eigen::MatrixXd A);                // testing
double symmetry_defect(const SymmetricOp& B, int pairs, unsigned seed);

class HrOracle {             // type-erased: apply + tolerance solve
  Eigen::Index dim() const;
  Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& X) const;
  Eigen::MatrixXd solve(const Eigen::Ref<const Eigen::MatrixXd>& B,
                        double tol) const;
};
HrOracle sparse_hr_oracle(Eigen::SparseMatrix<double> Hr);
// SimplicialLLT inside: the reference/small-N adapter. The production
// oracle is the consumer's multigrid-preconditioned Krylov solver,
// wrapped from C++ or Python.

ShiftedOperator make_shifted_operator(SymmetricOp B, ProbeArchive arch,
                                      HrOracle hr, double a0, Options opt);
// verifies symmetry of B at construction (seeded stochastic check)
void extend_modes(ShiftedOperator& A, int n_right, double lambda_min_target);
FlipReport make_pd(ShiftedOperator& A, double gamma /*=0.5*/, FlipMode mode);
DeflateReport deflate_free(ShiftedOperator& A, double rcond, int rank,
                           double clamp /*=-0.95*/);
DeflateReport value_pass(ShiftedOperator& A, const SymmetricOp& Hd, int m,
                         double clamp, ValuePassMode mode /*=V1*/);
void solve(const ShiftedOperator& A, const VectorXd& b, VectorXd& x,
           double a, SolveOpts opt);   // zones of §5; GLR or two-level
void apply(const ShiftedOperator& A, const VectorXd& x, VectorXd& y, double a);
void rebuild_at(ShiftedOperator& A, double a1);
// serialization: save/load of {archive, block, scalars}; the operator is
// never serialized -- the caller re-supplies it on load.
```

Reports carry what the research program taught us to watch: modes touched, clamp
events, $\lambda_{\mathrm{floor}}$, estimated remaining-error window.
All spectral internals honor the oracle tolerance; defaults chosen so
that PIG-scale behavior reproduces (§9).

---

## 8. Cholesky backend (optional, capability-gated)

The one entry-level component, and it activates only when the caller
*additionally* supplies sparse forms of $\tilde B$ and $H_r$ — the core
boundary of §7 never mentions entries. For small-$N$ / 2-D problems where
sparse factorization of $\tilde B + aH_r$ is viable: factor per $a$ with
a keyed cache (the block correction wraps the factorization by Woodbury),
use it in place of the two-level inner solve, and attempt-Cholesky
becomes an alternative exact PD certificate. This is
also the *validation* backend — the maintainer's offline PIG validation
is exactly this path, so backend-vs-oracle agreement is a built-in
correctness test, not just a convenience.

---

## 9. Validation plan

In-repo (runs in CI, no external data): the frog-kernel synthetic
operator is the testbed, as elsewhere in the repo. Unit properties:
wsym exactness/symmetry and weight behavior; **adapter equivalence** (the
whole layer run on the same matrix wrapped by `dense_op` and `sparse_op`
gives identical results — a mechanical format-independence check);
`symmetry_defect` flags an as-fitted operator and passes a
weighted-symmetrized one; $H_r$-orthonormality of the
block under merges; Woodbury identity $M_k(a) M_k(a)^{-1} = I$ to solver
tolerance across a sweep of $a$ with zero refactorization; flip contract
($\lambda_{\mathrm{floor}}$ certificate honored at the PD boundary);
zone semantics incl. warning + refusal; `rebuild_at` equivalence to a
fresh build at $a_1$; GLR vs two-level vs Cholesky backend agreement.

Maintainer-side (private PIG data; acceptance numbers recorded here so
the implementing session knows the targets):

1. **Flip-count study** — the motivating claim of P1. Dense generalized
   eigendecomposition of (weighted-sym fit, $H_r$): count Euclidean
   flip's touched modes vs pencil modes below $-\gamma a_0$
   ($\gamma = \tfrac12$, $a_0 = 10^{-4}$). Expect orders of magnitude
   apart; expect pencil-threshold flip within ~1 PCG iteration of the
   dense Euclidean flip (flip-vs-relu was already ≤ 1).
2. **Free deflation reproduction** — $k{=}200$,
   $\mathrm{rcond}=3\cdot10^{-2}$: 34 → 22 at $10^{-6}$; no gain at
   $k \le 20$.
3. **Value-pass reproduction** — $k{=}100$, $m{=}50$: 5/12/18/24 against
   the true operator (vs undeflated 12/26/36/45); zero clamps.
4. **Exact-value ceiling context** — certified-correction truncations
   gave 4/9/13 at $r{=}25$ and 3/5/7 at $r{=}50$ (to
   $10^{-2}/10^{-4}/10^{-6}$); useful for judging how much of the gap a
   given build closes.

## 10. Not included, and why

Measured and set aside (the numbers quoted are from the PIG study;
full equations and observations are in the maintainer's private
records): importance-ordered multisecant / SR-1
(interpolatory values; remaining-error $\lambda_{\min}$ reached $-222$,
counts worse than no deflation nearly everywhere); Riemannian fixed-rank
least squares (data residual falls monotonically while the fitted
eigenvalue range blows up to $[-19.7, 13.2]$ vs certified $[-0.92,
3.10]$; counts worse at every rank); damped Nyström (helped, superseded
by the value pass at equal budget); holdout-value estimation
(underdetermined/noisy at practical holdout sizes). Research
methodology that stays research: the certified deflated spectral
surrogate ($H_{\mathrm{surr}}$) used as referee, and the allocation-curve
studies — the latter inform the §4.4 guidance but ship as documentation,
not code.

## 11. Implementation slices (ordered, reviewable)

1. `Symmetrize::Weighted` in assembly (§2) + tests + docs. Independent.
   **Landed 2026-07-31** (see `dev/design-notes.md` for the decisions).
2. `corrections/symmetric_op.hpp` + `corrections/hr_oracle.hpp`: the two
   type-erased boundary classes, `sparse_op` / `dense_op` /
   `sparse_hr_oracle` adapters, `symmetry_defect`; the
   `lgpsf.corrections` binding submodule; the include-direction rule in
   `tools/check_dependencies.py`. **Landed 2026-08-01.**
3. `corrections/mode_block.hpp`: $H_r$-orthonormal block, Gram merge,
   congruence, provenance, serialization. **Landed 2026-08-01** (plain
   public arrays, numpy persistence per library convention — no bespoke
   format; both $V$ and $H_r V$ stored).
4. `corrections/shifted_operator.hpp`: struct + GLR-mode apply/solve +
   analytic PD certificates + per-$a$ scalar arithmetic (frog-kernel
   tests). **Landed 2026-08-01** (`eig(C)` recomputed per solve call —
   microseconds against an oracle solve, and it cannot go stale against
   the caller-visible block; `BuildOptions` symmetry gate at
   construction).
5. `corrections/pencil_lanczos.hpp`: generalized Lanczos both ends;
   `extend_modes`; `make_pd` with $\gamma$, $\lambda_{\mathrm{floor}}$,
   `FlipReport`. **Landed 2026-08-01** (restart-in-complement on
   breakdown; salted internal seed; `make_pd` resumable, certifies only
   on a CONVERGED leftmost survivor; leftmost discovery belongs to
   `make_pd` alone — run it before caching leftmost modes).
6. Zone semantics + warnings + `solve(…, a)` dispatch (§5).
   **Landed 2026-08-01** (`corrections/solve.hpp`; the warning-zone
   certificate bounds post-certification corrections exactly via a
   `C_corr` snapshot taken at certification).
7. Two-level solve mode (§6) with the flexible-outer contract documented.
   **Landed 2026-08-01** (inner PCG in `corrections/solve.hpp`;
   nonpositive curvature throws rather than converging wrong).
8. `deflate_free` (§4.3) in `corrections/deflation.hpp`.
   **Landed 2026-08-01** (residuals taken against the CURRENT corrected
   operator; requires a certified struct; exact-recovery test on a
   planted low-rank error).
9. `value_pass` (§4.4) + archive growth. **Landed 2026-08-01** (V1 and
   V2; a test pins V2 reaching past the residual span where V1
   saturates).
10. `rebuild_at` (§5) + persistence round-trip. **Landed 2026-08-01**
    (`corrections/rebuild.hpp` + `fold_value_pairs`; persistence is
    numpy arrays end to end, no bespoke format).
11. `corrections/cholesky_backend.hpp` (§8, capability-gated) + three-way
    backend agreement test. **Landed 2026-08-01** (construction verifies
    the supplied matrices against the operator handles; the per-shift
    cache follows the block's rank).
12. Maintainer-side validation runs (§9) and recording of results here.

Each slice lands with tests and doc updates; bindings track the C++ per
repo convention. Dependency direction (`tools/check_dependencies.py`)
keeps the corrections headers a **sibling** of the fit stack:
`corrections/*` includes no fit header and no fit header includes
`corrections/*`.
