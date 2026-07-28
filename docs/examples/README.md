# Examples

Each page is a complete program, its real output, and the figures it draws — regenerated from the code by `docs/generate_examples.py`.

See [`../../examples/README.md`](../../examples/README.md) for what each one teaches and the order to read them in.

## Python

- [The counting rule: why a cost near zero is not good news](counting_rule.md)
- [Using a fitted operator: the four ways, and what truncation does](deploying_a_fit.md)
- [What `theta` is, and the two encodings that catch people out](ellipsoid_theta.md)
- [Fit ONE point-spread function from random probes](fit_one_psf.md)
- [Why a SHORT Laguerre-Gaussian expansion is worth fitting](lg_expansion_convergence.md)
- [What a Laguerre-Gaussian mode looks like](lg_modes.md)
- [Choosing which modes to add, and in what order](mode_policies.md)
- [Reading a fit: which rows worked, which did not, and how you would know](operator_diagnostics.md)
- [Fit a whole operator from random matvecs, and watch the error fall with k](operator_fit_frog.md)
- [An LGOperator you build yourself, with no fitting at all](operator_without_fitting.md)
- [What a fit is FOR: preconditioning a regularized inverse problem](preconditioner.md)
- [What the fitter tried, what it kept, and why it stopped](reading_a_row_fit.md)
- [When rows and columns live on different meshes](rectangular_operator.md)
- [The fitting core on its own: VarPro, whitening, and the two Jacobians](varpro_custom_basis.md)

## C++

- [Fit ONE point-spread function from random probes -- the row layer in C++](fit_one_psf-cpp.md)
- [Minimal smoke test for the VS Code build/debug setup: links against lgpsf](hello_world-cpp.md)
- [Fit a whole operator from random matvecs -- the complete pipeline in C++](operator_fit_frog-cpp.md)
- [What a fit is FOR: preconditioning a regularized inverse problem](preconditioner-cpp.md)
