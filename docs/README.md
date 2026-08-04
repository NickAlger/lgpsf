# Documentation

For **using** lgpsf. Notes for people changing it are in [`../dev/`](../dev/).

## Start here

| | |
|---|---|
| [installation.md](installation.md) | pip, CMake, and the two build warnings worth heeding. |
| [quickstart.md](quickstart.md) | What you must supply, the fit, and how to check it worked. |
| [heat-pipeline/heat-pipeline.pdf](heat-pipeline/heat-pipeline.pdf) | **The tutorial**: the complete pipeline — probe, fit, certify, deflate, deploy — on one heat-equation inverse problem, with every figure and snippet generated from [heat-pipeline/make_figures.py](heat-pipeline/make_figures.py). |

## Reference

| | |
|---|---|
| [api-guide.md](api-guide.md) | The three layers, the array conventions that trip people up, and the two encodings of `theta`. |
| [defaults.md](defaults.md) | Every knob's default and the measurement behind it — including why the mode policy has no default, and why dead rows should not be gated. |

## Judging it

| | |
|---|---|
| [validation.md](validation.md) | What the library was tested against at field scale, and how it compares to the alternative. Read this to make the "PIG" references in the headers resolve. |
| [reproducibility.md](reproducibility.md) | What is bit-exact and what is not — identical across threads, runs, callers and the two inner-solve paths; **not** across builds with different compiler flags. Includes a table for reading a failed comparison. |

## Going deeper

| | |
|---|---|
| [varpro-whitening-notes.pdf](varpro-whitening-notes.pdf) ([.tex](varpro-whitening-notes.tex)) | The mathematics: how the smooth and spike bases combine without the fitting code ever touching a mass matrix. |
| [examples/](examples/) | **Every example as a page**: the program, its real output, and the figures it draws. Generated from the code, so it cannot drift. |
| [`../examples/`](../examples/) | The example sources themselves, with a guide to what each teaches and the order to read them in. |
| [`../experiments/`](../experiments/) | Measurements about the library — including whether refining the mesh costs more probes (it does not). |
