# Objective

This folder contains loss functions and error calculators used by the Bayesian optimization loop.

For final FFS runs, `__init__.py::objective_geko` composes the normalized
`Ux` and total-TKE field errors from `field_error.py`. Optional integral and
GEKO default-coefficient preference terms remain available. The final
two-parameter configurations enable the preference term with
`lambda_preference = 0.5`.

On the common grid, each field contribution is its RMSE divided by the DNS
standard deviation for that field. GEDCP sums those normalized field
contributions and then applies the multiplicative coefficient-preference
factor.

`field_error.py` supports `area_weight_mode`: use `structured` for tensor-product DNS grids, `density` for unstructured LES/RANS point-cloud comparison, `uniform` for node-weighted diagnostics, or `auto` to choose from DNS geometry.

For fundamentally different LES/RANS meshes, prefer `evaluation_mode = "common_grid"` so both datasets are interpolated onto the same physical grid before error computation. FFS configs should also set `common_grid_floor = "ffs_step"` to exclude the solid step region.
