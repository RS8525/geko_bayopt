# FFS DNS Data

This folder stores forward-facing-step LES/DNS reference data.

Raw `FFS_Reh*_SBES_Node` files are 3D exports. Convert them to the
spanwise-averaged `*_2D.csv` references with:

```bash
python data/dns/ffs/average_z_dns_ffs.py
```

The converter identifies the 20 primary spanwise planes and maps them to the
canonical first-plane mesh before applying trapezoidal averaging. Do not use
exact `(x, y)` grouping: the exports contain coordinate noise and sparse
intermediate z-levels.

Use `--overwrite` to regenerate an existing output. `--z-planes` and
`--plane-tolerance-fraction` are available if a future export uses a different
spanwise mesh.

The generated references are compared against Fluent RANS exports on a
different mesh, so diagnostics should verify coordinate extents, field ranges,
units, and interpolation coverage before interpreting objective scores.
