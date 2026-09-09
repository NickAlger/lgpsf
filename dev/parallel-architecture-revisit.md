# Revisit: the parallel architecture of the fit

Parked 2026-09-08, at the maintainer's request, while implementing the
fitting-only row redistribution (`row-balance-plan.md`). Nothing here is a
decision. It records why that work felt more complicated than it should, so a
later session can judge whether the complication is essential or an artefact
of the architecture it is being retrofitted onto.

## The observation

Slice 3 of the balance plan needs: a two-callback delegate inside
`fit_operator` with a specific call ordering; a `RowFitProblem` that is
non-owning on the fused path and owning on the delegated one; an assignment
computed at one particular point in the loop (after the window pre-pass,
before phase A) because that is the only place where a good weight is free and
the package does not yet exist; and a second window gather for every delegated
row. That is a lot of machinery, and it buys a bounded win: the resident share
of 0.084 caps the fit's improvement near fivefold and a full-Hessian rebuild
near 2.5x.

None of the individual pieces is wrong. The question is whether they are
forced by the problem or by the shape of what they are being added to.

## The root, stated plainly

The distributed fit's unit of ownership is a RANK, and a rank owns rows and
columns together. `dist_fit` takes a rank's columns and a rank's rows, builds
a halo sized from those rows' windows, and fits them. Every improvement we
have wanted since then fights that arrangement:

- **Load balancing** wants to move a row's work away from its columns. Hence
  D1/D2, the delegate, the package, the return exchange.
- **Splitting one row's search across ranks** (`probe_fit.hpp`'s candidates in
  a level are independent and the truncation is re-appliable, so it is
  bit-identical) would remove the whole-row granularity floor that `T >=
  max_i w_i` imposes and that already binds at 192 ranks. It has no natural
  home in a rank-owns-everything design.
- **Distributing the full-window re-score** -- the thing that actually caps
  the payoff, since the score is a mass-weighted quadrature and is therefore
  reducible -- is a cross-rank reduction over one row's window, which is
  exactly what the current halo makes awkward.

All three are natural if the unit of work is a self-contained ROW TASK with an
explicit input (centre, prior, quadrature, responses) and an explicit output
(candidate parameters), placed by a scheduler, with the points a task needs
pulled rather than pre-planned per rank. The balance plan is, in effect, that
design smuggled in through a callback.

## What a revisit would have to weigh

- The halo plan is genuinely good at what it does: one allgather of box cuts,
  neighbourhood point exchange, exact membership resolved by a dual-tree
  descent, and it is already parameterized by an arbitrary window list rather
  than by ownership (`halo_exchange.hpp`, and `lgh_fit_smooth_field` uses it
  with an unrelated ellipsoid set). A row-task design would still want most of
  it, as a pull.
- Bit-identity is the property everything rests on and the reason the retrofit
  is safe. Any rewrite has to keep it, and the existing gate (serial reference
  on rank 0, dense row images compared bitwise at n = 1/2/4) is what makes
  that checkable. Do not start without extending that gate first.
- The consumer's three `(row - first_row)` index expressions and its CSR
  assembly assume a rank holds exactly its own rows. The balance plan is
  careful to keep that true. A row-task design would have to keep it true too,
  or move the assembly.
- Cost. `dist_fit`, `halo_exchange` and the consumer's `fit_once` are the
  files at stake, and the ice-sheet application is mid-campaign against them.

## When to look at this

Not while the campaign is running. The natural moments are: after the balance
work lands and has field numbers, so the complexity has a measured payoff to
be judged against; or when the granularity floor starts binding at larger
allocations, since that is the first thing the retrofit cannot fix.
