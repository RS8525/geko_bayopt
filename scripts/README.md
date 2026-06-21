# Scripts

This folder holds ad hoc analysis and helper scripts outside the main package.

`diagnose_ffs_objective.py` compares the FFS DNS CSV with an existing Fluent ASCII export to inspect coordinate ranges, field ranges, interpolation coverage, and per-field objective behavior.

`decompose_ffs_scores.py` recomputes per-field contributions from saved FFS metadata and ASCII files.

`localize_ffs_error.py` bins the FFS `Ux` error spatially to show where a score contribution comes from.

`common_grid_ffs_error.py` evaluates both DNS and RANS fields on the same physical grid, which is the preferred diagnostic for LES/RANS mesh comparisons.
