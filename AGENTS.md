# AGENTS.md — geko_bayesopt Project Context

> This file is the authoritative context document for AI agents (Copilot, Claude, Cursor, etc.)
> working in this repository. Read it fully before making any changes.

---

## Project Purpose

Bayesian optimization of ANSYS Fluent GEKO turbulence model coefficients, calibrated
against DNS reference data from the parameterized periodic hills dataset (Xiao et al., 2020).
The long-term goal is to extend this into data-driven turbulence modeling (field inversion,
ML correction models), so the architecture must stay modular and extension-friendly.

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| CFD solver | ANSYS Fluent (local license) | Automated via `ansys-fluent-core` (pyfluent) |
| CFD input | `.msh.h5` or `.cas.h5` | Fluent native mesh/case format |
| DNS data | OpenFOAM field format (coarse mesh) | From Xiao et al. para-database-for-PIML |
| DNS parsing | `fluidfoam` | Reads OpenFOAM field files into numpy arrays |
| Interpolation | `scipy.interpolate.LinearNDInterpolator` | DNS → RANS grid, 2D unstructured |
| Optimization | `scikit-optimize` (`gp_minimize`) | Will upgrade to BoTorch for >3 parameters |
| Config | JSON + `pydantic` v2 | Validated at load time, one file per experiment |
| Storage | `pandas` + `h5py` | Run metadata (CSV-like) + field snapshots (HDF5) |
| Python | ≥ 3.10 | Uses dataclasses, match statements |
| Build | `hatchling` | `pyproject.toml`, editable install via `pip install -e .` |

---

## Repository Layout

```
geko_bayesopt/
│
├── configs/                        # One JSON file per experiment configuration
│   └── periodic_hills_csep.json
│
├── cases/                          # Per-flow-case directories (mesh + base Fluent case)
│   └── periodic_hills/
│       ├── mesh/                   # .msh.h5 file lives here
│       ├── base_case/              # .cas.h5 base case (no GEKO params patched yet)
│       └── case_config.json        # Case-level metadata (Re, geometry alpha, etc.)
│
├── data/
│   ├── dns/
│   │   └── periodic_hills/         # Raw OpenFOAM-format DNS case (Xiao et al.)
│   │       ├── 0/                  # Field files: U, p, R (Reynolds stress), epsilon
│   │       └── constant/           # mesh (polyMesh)
│   └── processed/                  # Cached: interpolated DNS arrays saved as .h5
│
├── results/
│   └── experiments/                # One subdirectory per experiment run
│       └── <experiment_id>/
│           ├── runs/               # Per-iteration Fluent run directories
│           ├── metadata.csv        # Parameters + scores for every iteration
│           └── optimizer.pkl       # Serialized skopt optimizer (for resume)
│
├── src/
│   └── geko_bayesopt/
│       ├── __init__.py
│       ├── cli.py                  # Entry point: `geko-opt run configs/foo.json`
│       ├── config.py               # Pydantic models for experiment JSON schema
│       │
│       ├── workflow/
│       │   ├── __init__.py
│       │   ├── experiment.py       # Main optimization loop (ask → run → tell → store)
│       │   ├── store.py            # Checkpoint/resume: write after every iteration
│       │   └── types.py            # RunResult dataclass — the central data contract
│       │
│       ├── cases/
│       │   ├── __init__.py
│       │   └── periodic_hills.py   # Concrete FlowCase: prepares Fluent run dir,
│       │                           # extracts RunResult from completed run
│       │
│       ├── dns/
│       │   ├── __init__.py
│       │   └── periodic_hills_loader.py  # Loads coarse-mesh OpenFOAM DNS via fluidfoam
│       │                                 # Returns: coords (x,y), U, p, R, k arrays
│       │
│       ├── ansys/
│       │   ├── __init__.py
│       │   ├── journal_writer.py   # Writes .jou files that patch GEKO params
│       │   ├── fluent_runner.py    # Launches Fluent, monitors convergence, returns exit
│       │   └── result_extractor.py # Reads Fluent output → RunResult
│       │
│       ├── objective/
│       │   ├── __init__.py
│       │   └── field_error.py      # Interpolates DNS → RANS grid, computes weighted
│       │                           # L2 error over selected fields and x/H stations
│       │
│       ├── optimization/
│       │   ├── __init__.py
│       │   └── bayesopt_adapter.py # Thin wrapper: skopt ask/tell, bounds from config
│       │
│       └── utils/
│           ├── __init__.py
│           ├── paths.py            # Resolves all runtime paths from config + experiment_id
│           └── logging.py          # Structured logging setup (timestamps, run_id prefix)
│
├── tests/
│   ├── conftest.py
│   ├── test_dns_loader.py
│   ├── test_field_error.py
│   └── test_bayesopt_adapter.py    # Uses mock Fluent (analytic test function)
│
├── pyproject.toml
├── AGENTS.md                       # ← this file
└── README.md
```

---

## Central Data Contract: `RunResult`

Every component speaks `RunResult`. Do not pass raw dicts between modules.

```python
# src/geko_bayesopt/workflow/types.py
@dataclass
class RunResult:
    run_id: str                          # e.g. "exp_001_iter_007"
    parameters: dict[str, float]         # e.g. {"C_sep": 1.23}
    fields: dict[str, np.ndarray]        # e.g. {"Ux": arr, "k": arr, "uv": arr}
    grid_coords: np.ndarray              # shape (N_cells, 2) — x, y cell centers
    converged: bool                      # False if Fluent hit iteration limit
    cost_seconds: float                  # wall-clock time for this run
```

---

## DNS Data Notes

**Source:** https://github.com/xiaoh/para-database-for-PIML
**Format:** OpenFOAM field files (coarse mesh, RANS-comparable resolution)
**Fields available:** `U` (mean velocity), `p` (mean pressure), `R` (Reynolds stress tensor,
6 components), `epsilon` (dissipation, 2021 uploads only)
**Derived:** `k = 0.5 * (R_uu + R_vv + R_ww)` — compute in loader, do not store separately

**Reading pattern (fluidfoam):**
```python
coords = fluidfoam.readmesh(dns_path)          # (3, N) → take [:2] for 2D
U      = fluidfoam.readfield(dns_path, "0", "U")   # (3, N)
R      = fluidfoam.readfield(dns_path, "0", "R")   # (6, N) — xx,yy,zz,xy,xz,yz
```

**Critical:** DNS coordinates are non-dimensionalized by hill height H. Verify that
Fluent RANS output uses the same reference length before computing any error metric.
This is the first thing to check/plot — do it before any optimizer code runs.

---

## Optimization Strategy

**Current phase:** 1D sweep on `C_sep` ∈ [0.5, 2.0], 8 initial random points + 32 BO iterations.

**Objective:** Weighted L2 error of `Ux` profiles at x/H = {0.5, 2.0, 4.0, 6.0, 8.0}
stations. Weight `k` error at 0.3 when TKE data is validated.

**Upgrade path:**
- Add `C_nw` as second parameter → keep skopt
- Add `C_mix`, `C_sep2` → switch to BoTorch (optional `[ml]` dependency group)
- Data-driven extension → field inversion using DNS `R` tensor as target

**Checkpoint/resume:** The optimizer is serialized with `pickle` after every `tell()`.
On restart, `store.resume_optimizer()` reloads it and returns the iteration count.
Never re-run iterations — load from `metadata.csv` and re-tell.

---

## GEKO Model Parameters (ANSYS Fluent)

GEKO (Generalized k-ω) has 6 free coefficients. Tuning order:

| Parameter | Default | Physical role | Tune order |
|---|---|---|---|
| `C_sep` | 1.75 | Separation sensitivity | **1st** |
| `C_nw` | 0.5 | Near-wall behavior | 2nd |
| `C_mix` | 0.0 | Mixing / free shear | 3rd |
| `C_sep2` | — | Secondary separation | 4th |
| `C_jet` | — | Jet flows | later |
| `C_corner` | — | Corner flows | later |

Parameters are patched via Fluent journal files (`.jou`), not via TUI interactively.

---

## Key Constraints for Agents

1. **Do not modify `workflow/types.py` `RunResult` fields** without updating all consumers
   (`result_extractor.py`, `field_error.py`, `store.py`). This is the load-bearing interface.

2. **Do not add OpenFOAM as a runtime dependency.** DNS data is read with `fluidfoam` only.
   We do not run OpenFOAM simulations.

3. **Fluent automation uses `ansys-fluent-core`** (the `pyfluent` library). Do not use
   subprocess + journal file execution as the primary path — use the Python API. Journal
   files are used only for GEKO parameter patching where the API has no direct method.

4. **All paths must go through `utils/paths.py`.** No hardcoded paths anywhere else.
   `PathResolver` takes the config and experiment_id and returns all runtime paths.

5. **Write after every iteration.** `store.save_run()` is called in the experiment loop
   immediately after the objective score is computed — before `optimizer.tell()`.
   A crash after `save_run()` but before `tell()` is recoverable. The reverse is not.

6. **Interpolation lives in `objective/field_error.py`**, not in the DNS loader.
   The loader returns raw DNS arrays + coordinates. Interpolation happens at evaluation
   time when both the DNS grid and the RANS grid are available simultaneously.

7. **The `processed/` directory is a cache.** If a `.h5` file for a DNS case already
   exists there, the loader reads from cache. If not, it reads OpenFOAM files and writes
   the cache. Never commit processed files to git.

---

## Not Yet Implemented (Planned)

- `cases/forward_facing_step.py` — second flow case
- `optimization/parameter_space.py` — generalized bounds/scaling for multi-param
- `objective/weighted_objective.py` — configurable field/station weights
- `[ml]` dependency group: `torch`, `botorch`, `gpytorch`
- Data-driven correction: Reynolds stress discrepancy modeling (field inversion approach)

Do not implement these until the periodic hills BayesOpt loop is validated end-to-end.
