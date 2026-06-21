# FFS DNS Data

This folder stores forward-facing-step LES/DNS reference data.

`FFS_Reh6000_SBES_Node_2D.csv` is the reference file used by `configs/ffs_csep_v3.json`. It is compared against Fluent RANS exports on a different mesh, so diagnostics should verify coordinate extents, field ranges, and interpolation coverage before interpreting objective scores.
