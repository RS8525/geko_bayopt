# Objective

This folder contains loss functions and error calculators used by the Bayesian optimization loop.

For FFS `ffs_csep_v3`, `__init__.py::objective_geko` composes per-field errors from `field_error.py`, optional integral errors, and the GEKO default-coefficient preference from `GEDCP.py`.

`field_error.py` supports `area_weight_mode`: use `structured` for tensor-product DNS grids, `density` for unstructured LES/RANS point-cloud comparison, `uniform` for node-weighted diagnostics, or `auto` to choose from DNS geometry.

For fundamentally different LES/RANS meshes, prefer `evaluation_mode = "common_grid"` so both datasets are interpolated onto the same physical grid before error computation. FFS configs should also set `common_grid_floor = "ffs_step"` to exclude the solid step region.
