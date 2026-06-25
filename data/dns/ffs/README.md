# FFS DNS Data

This folder stores forward-facing-step LES/DNS reference data.

Raw `FFS_Reh*_SBES_Node` files are 3D exports. Convert them to the
spanwise-averaged `*_2D.csv` references with:

```bash
python data/dns/ffs/average_z_dns_ffs.py
```

The active Re=6000 reference is
`FFS_Reh6000_SBES_Node_NEW_2D.csv`, generated from the updated source export
that includes `mean-tke_tot-dataset`.

The converter identifies the 20 primary spanwise planes and maps them to the
canonical first-plane mesh before applying trapezoidal averaging. Do not use
exact `(x, y)` grouping: the exports contain coordinate noise and sparse
intermediate z-levels.

Before spanwise averaging, it creates `total-turbulent-kinetic-energy` on
every raw 3D row. It uses `mean-tke_tot-dataset` directly where available.
For legacy exports lacking that source column, the converter reconstructs it as:

```text
k_total = k_modeled + 0.5 * (u_rms^2 + v_rms^2 + w_rms^2)
```

For the reconstructed case, calculate the squared velocity fluctuations
before averaging over z.

Use `--overwrite` to regenerate an existing output. `--z-planes` and
`--plane-tolerance-fraction` are available if a future export uses a different
spanwise mesh.

The generated references are compared against Fluent RANS exports on a
different mesh, so diagnostics should verify coordinate extents, field ranges,
units, and interpolation coverage before interpreting objective scores.
