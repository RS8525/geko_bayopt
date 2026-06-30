# Configs

Experiment JSON files select the flow case, objective, optimizer, mesh settings, and Fluent session strategy.

Final FFS production configs live in `ffs_final/`. The `2_param/` configs
optimize `geko_csep` and `geko_cnw` with 48 Gaussian-process Bayesian
evaluations: 16 Sobol initial points followed by 32 model-guided proposals.
The `all_param/` configs optimize `geko_csep`, `geko_cnw`, `geko_cjet`,
`geko_cturb`, and `geko_cmix` with 100 evaluations: 32 Sobol initial points
followed by 68 model-guided proposals.

The final objective compares `Ux` and `total-turbulent-kinetic-energy` on a
uniform common grid with the FFS solid step removed. It uses the GEDCP
coefficient preference with `lambda_preference = 0.5`.

Completed final FFS result bundles are consolidated under
`results/ffs_final_runs/all_param_runs/` and
`results/ffs_final_runs/two-param-runs/`, one directory per `experiment_id`.

Final plotting configs live in `../scripts/ffs/plots/final/`. Each completed
run has a default-GEKO and optimized-GEKO config. Those configs write plots
back into the corresponding result bundle and include normalized-error plots
for both L1 and L2 field norms with fixed per-field color scales.
Metadata convergence plotting configs live in
`../scripts/ffs/plots/final_metadata/`. They compare each completed
all-parameter run against the matching two-parameter run rescored with the
all-parameter GEDCP preference convention, and also include standalone
two-parameter convergence plots with the default-GEKO row as the baseline.

Run from the repository root, for example:

```powershell
geko-opt run configs/ffs_final/all_param/ffs_re6000_all_param.json
```

Historical FFS configs are retained under `ffs_retired/` for provenance only.
