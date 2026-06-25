# Configs

Experiment JSON files select the flow case, objective, optimizer, mesh settings, and Fluent session strategy.

Final FFS production configs live in `ffs_final/`. There is one config for
each Reynolds number (2000, 3000, 4000, and 6000). They optimize `geko_csep`
and `geko_cnw` using 48 Gaussian-process Bayesian evaluations: 16 Sobol
initial points followed by 32 model-guided proposals.

The final objective compares `Ux` and `total-turbulent-kinetic-energy` on a
uniform common grid with the FFS solid step removed. It uses the GEDCP
coefficient preference with `lambda_preference = 0.5`.

Run from the repository root, for example:

```powershell
geko-opt run configs/ffs_final/ffs_re6000_csep_cnw.json
```

Historical FFS configs are retained under `ffs_retired/` for provenance only.
