# docs

- [design-notes.md](design-notes.md) -- running log of design decisions relevant to the C++ port.
- [varpro-whitening-notes.pdf](varpro-whitening-notes.pdf) ([.tex](varpro-whitening-notes.tex)) -- noise-whitening derivation: combining theta-dependent smooth and theta-independent extra basis functions without mass matrices in the VarPro fit.
- [robust-init-notes.md](robust-init-notes.md) -- PARKED design note: the portfolio/init-dictionary recipe for robust per-row fitting, the anisotropy-stress evidence behind it, and the enrichment-saddle fact.
- [probe-moment-ellipsoids.md](probe-moment-ellipsoids.md) -- PARKED idea sketch: conservative per-row ellipsoid fields from probe data alone (squared-kernel moment identity, iterated window shrinkage, cross-row smoothing), with the feasibility analysis and the free decisive experiment.
- [operator-api-plan.md](operator-api-plan.md) -- design record for the whole-operator fitting API, implemented as `prototype/operator_fit.py` (H~ = M1 Phi~ M2 + M1 S two-component type structure, spike convention, inputs incl. the sigma/tau_window resolution, OperatorFit padded arrays, helper table, baseline-always-on guard, symmetry-as-assembly-policy).
