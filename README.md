# geko_bayesopt

Bayesian optimization of ANSYS Fluent GEKO turbulence model coefficients,
calibrated against DNS reference data (Xiao et al., 2020 — parameterized periodic hills).

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
geko-opt run configs/periodic_hills_csep.json
```

## FFS Final Runs

Final forward-facing-step production configs live under `configs/ffs_final/`.

## DNS Data

Download from: https://github.com/xiaoh/para-database-for-PIML
Place the coarse-mesh OpenFOAM case for periodic hills (Re=5600, α=1.0) at:
`data/dns/periodic_hills/`

## Project Structure

See `AGENTS.md` for full architecture documentation, data contracts, and agent instructions.
