# Scripts

This folder holds ad hoc analysis and helper scripts outside the main package.

FFS-specific diagnostics live under `scripts/ffs/`.

`ffs/diagnose_ffs_objective.py` compares the FFS DNS CSV with an existing Fluent ASCII export to inspect coordinate ranges, field ranges, interpolation coverage, and per-field objective behavior.

`ffs/decompose_ffs_scores.py` recomputes per-field contributions from saved FFS metadata and ASCII files.

`ffs/localize_ffs_error.py` bins the FFS `Ux` error spatially to show where a score contribution comes from.

`ffs/common_grid_ffs_error.py` evaluates both DNS and RANS fields on the same physical grid, which is the preferred diagnostic for LES/RANS mesh comparisons.

`ffs/plot_ffs_fields.py` creates config-driven FFS field and comparison plots.
Configs live under `scripts/ffs/plots/`; only `ffs_default.json` is tracked.
Simulation and DNS columns should be selected by their header strings. Numeric
simulation column indices remain supported for old local configs.
