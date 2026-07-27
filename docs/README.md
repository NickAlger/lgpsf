# Documentation

Everything here is for **using** lgpsf. Notes for people changing it live in
[`../dev/`](../dev/).

| | |
|---|---|
| [validation.md](validation.md) | What the library was tested against at field scale, and which defaults came out of it. Read this to make the "PIG" references in the headers resolve, and to judge whether the method suits your problem. |
| [reproducibility.md](reproducibility.md) | What is bit-exact and what is not — identical across threads, runs, callers and the two inner-solve paths; **not** across builds with different compiler flags, where ~0.08% of rows can land on a different local minimum. Includes a table for reading a failed comparison. |
| [varpro-whitening-notes.pdf](varpro-whitening-notes.pdf) ([.tex](varpro-whitening-notes.tex)) | The mathematics: how the smooth and spike bases combine without the fitting code ever touching a mass matrix. Needed to follow `whitening.hpp`; not needed to call the library. |

Running example:
[`../examples/operator_fit_frog.py`](../examples/operator_fit_frog.py) fits a
whole operator from random matvecs and plots the error against the probe
budget. It is self-contained and doubles as the integration test.

> **Being written.** Installation, a quickstart, an API tour of the three
> layers, and a page collecting the defaults with the evidence behind them.
> Until those land, [`../CLAUDE.md`](../CLAUDE.md) is the fullest description
> of the architecture and the headers are the API reference.
