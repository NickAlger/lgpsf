# `dev/` — developer notes

Working notes for people changing lgpsf: current state, decisions and their
rationale, and the record of threads that have closed. None of it is needed to
*use* the library — that is [`docs/`](../docs/).

| | |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Where things stand, what is in flight, what is owed, what is parked. Start here. |
| [`archive/`](archive/) | Closed threads: completed plans, session records, superseded decisions. Kept for provenance; not current. |

Two neighbours are easy to confuse with this directory:

- [`../experiments/`](../experiments/) — measurements *about* the library
  (benchmarks and their write-ups), rather than notes about developing it.
- [`../archive/python-prototype/`](../archive/python-prototype/) — the frozen
  Python implementation the method was built in. History, not a reference.

`dev/` is tracked, so what lands here is public. Raw data dumps, compiled
harnesses and profiler output are gitignored and should stay that way — they
regenerate, and one of them was 100 MB. Anything naming a private problem
should name it in prose, never as a path; `tools/check_dependencies.py`
enforces the distinction.
