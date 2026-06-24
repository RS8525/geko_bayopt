# geko_bayopt — Architecture Notes

Living document. ALWAYS Update when structure changes. Code is the source of truth; this file explains the *why*.

---

## Project goal

Bayesian optimization of the GEKO turbulence-model coefficients in ANSYS Fluent, scored against DNS reference data. Single-machine, single-user, thesis-scale. Two flow cases implemented today (periodic hill, forward-facing step); more can be added easily.

---

## Three swappable axes

The code is organized around three independent variation points, each selectable via JSON config:

| Axis | What changes | Where it lives | Dispatcher |
|------|--------------|----------------|------------|
| **Flow case** | Boundary conditions, DNS format, geometry | `cases/<name>/` | `cases/__init__.py::build_flow_case` |
| **Loss function** | How sim is scored vs DNS | `objective/<name>.py` | `objective/__init__.py::build_loss_fn` |
| **Optimizer** | How next params are proposed | `optimizer.py` | `optimizer.py::build_optimizer` |

Adding any of the above is: write the new module, register it in the corresponding dispatcher, extend the `Literal[...]` in `config.py`. Zero changes elsewhere.

---

## Package layout

```
src/geko_bayesopt/
├── __init__.py
├── cli.py                  CLI entry (geko-opt run config.json)
├── config.py               Pydantic schemas for experiment JSON
├── types.py                RunResult dataclass (shared data contract)
├── experiment.py           BO loop (composition only, no logic)
├── store.py                metadata.csv + optimizer.pkl persistence
├── optimizer.py            Optimizer dispatcher (skopt today, BoTorch later)
│
├── fluent/                 Generic Fluent automation (case-agnostic)
│   ├── case_config.py      CaseConfig dataclass
│   ├── mesh_config.py      MeshConfig dataclass
│   ├── mesh_generator.py   2D Meshing workflow
│   ├── solver.py           PeriodicHillSolver (start/run_trial/close)
│   ├── runner.py           Helpers: run_case, open_session, run_geko_trial
│   └── extract.py          ASCII -> RunResult parser
│
├── cases/                  Per-flow-case knowledge
│   ├── base.py             FlowCase abstract base class
│   ├── periodic_hills/
│   │   └── case.py         PeriodicHillsCase (BCs + Laizet DNS loader)
│   └── ffs/
│       └── case.py         ForwardFacingStepCase (Velocity inlet + Pressure outlet)
│
└── objective/              Loss functions
    ├── field_error.py      FieldErrorCalculator (core MSE math)
    ├── types.py            LossFn type alias
    ├── mse.py              mse_field, mse_cp factories
    └── weighted.py         weighted_multi_field factory
```

---

## Data contract

Every component that produces or consumes simulation output speaks `RunResult` (defined in `types.py`):

```python
@dataclass
class RunResult:
    run_id: str
    parameters: dict[str, float]      # {"geko_csep": 1.85, ...}
    grid_coords: np.ndarray           # (N, 2), non-dim x/H, y/H
    fields: dict[str, np.ndarray]     # {"Ux": ..., "Uy": ..., "p": ..., "cp": ...}
    converged: bool
    cost_seconds: float
    ascii_path: Path | None           # debug only, not load-bearing
```

Producers: `fluent/extract.py::build_run_result`.
Consumers: loss functions in `objective/`.

Do not invent ad-hoc dict shapes for passing simulation results between modules. If you need a new field, add it to `RunResult.fields` with a documented name.

---

## Coordinates and units

`RunResult.grid_coords` is **non-dimensional** (x/H, y/H). `RunResult.fields["Ux"]`, `"Uy"` are in units of `U_bulk`. Pressure in `rho * U_bulk^2`. This is done by `fluent/extract.py` at the moment of extraction, so DNS comparison is unit-free.

The flow case is responsible for declaring `hill_height` and the case's derived `u_bulk` (computed from Re_h). If you change reference conventions, do it there — never in the loss function.

For unstructured FFS LES/RANS comparisons, prefer `objective.options.evaluation_mode = "common_grid"` with `common_grid_floor = "ffs_step"` so DNS and simulation are evaluated on the same physical grid. If evaluating directly on DNS points, `area_weight_mode` should be `"density"` rather than the periodic-hill structured-grid formula.

---

## File layout convention

The repository separates *inputs* (things you provide) from *outputs* (things the code produces):

```
<repo>/
├── data/                        ← inputs you provide
│   ├── geometry/                  CAD files (.dsco)
│   └── dns/                       reference DNS data
└── results/                     ← outputs the code writes
    ├── fluent/<experiment_id>/  mesh, .cas, .dat, .ascii files
    └── experiments/<experiment_id>/  metadata.csv, optimizer.pkl
```

Both `geometry_path` and `dns_path` in the experiment JSON are resolved relative to the repo root when they're not absolute. The repo root is auto-detected as the grandparent of the config file (`<root>/configs/<name>.json` → `<root>`).

---

## Save-before-tell ordering

The experiment loop runs in this order per trial:

1. `optimizer.ask()` → parameter vector
2. `solver.run_trial(params)` → ASCII + Fluent state
3. `extract.build_run_result(...)` → RunResult
4. `loss_fn(run_result)` → score
5. `store.save_trial(result, score)` ← ON DISK before next step
6. `optimizer.tell(x, score)` ← optimizer learns
7. `store.save_optimizer(optimizer)` ← checkpoint

If a crash happens between (5) and (6), the next run replays completed trials from `metadata.csv` via `tell()` and continues. If we reversed (5) and (6), a crash there would leave the optimizer ahead of the durable record, which is unrecoverable.

---

## Session strategies

The JSON config's `session_strategy` field selects between:

- **`"live"`** — one Fluent process, reused across all trials. Saves ~30s of launch overhead per trial. Recommended unless your environment is flaky.
- **`"per_trial"`** — launch + exit Fluent per trial. Slower but resilient to license-server hiccups (Student edition only allows one session at a time).

Both go through the same `PeriodicHillSolver.run_trial()` API — the difference is whether the solver is constructed once or per trial.

---

## Adding things

### A new loss function

1. Create `objective/<name>.py` with a factory:
   ```python
   def my_loss(dns_coords, dns_fields, *, ...kwargs from JSON) -> LossFn:
       def loss(run: RunResult) -> float: ...
       return loss
   ```
2. Register in `objective/__init__.py::_REGISTRY`.
3. Add the string to `Literal[...]` in `config.py::ObjectiveSection`.

### A new optimizer

1. Add a `_make_<name>(parameters, ...)` builder in `optimizer.py`.
2. Add a branch in `build_optimizer`.
3. Extend `Literal[...]` in `config.py::OptimizerSection`.
4. The object must expose `ask()` and `tell(x, y)`. skopt and BoTorch already do.

### A new flow case

1. Create `cases/<name>/case.py` with a `FlowCase` subclass.
2. Implement `build_case_config`, `apply_boundary_conditions`, `load_dns`.
3. Register in `cases/__init__.py::_REGISTRY`.
4. Extend `Literal[...]` in `config.py::CaseSection`.

The solver (`fluent/solver.py`) is case-agnostic; it calls `flow_case.apply_boundary_conditions(...)` at the right moment in `start()`. For periodic hill this creates the periodic interface and mass-flow forcing; for forward-facing step it would set velocity-inlet and pressure-outlet.

---

## Fluent automation guarantees (lessons learned)

These are baked into `fluent/`:

- `cad_route="DSCO"` + `cad_route="disco"` for `.dsco` files. The `"Workbench"` + `"pmdb"` route is used for `.pmdb` files.
- `two_dim.write_2d_mesh` (workflow task), NOT `tui.file.write_mesh` (the latter produces files the solver rejects as "surface mesh").
- Mesh write is sanity-checked for file size > 0.5 MB to catch silent failures.
- `version="2d"` must NOT be passed to `launch_fluent` with `mode="meshing"` — that combination silently switches to the solver and breaks the workflow.
- GEKO coefficients live at `solver.settings.setup.models.viscous.geko.<coef>.value` on Fluent 2026 R1.
- Periodic forcing uses `/define/periodic-conditions/massflow-rate-specification?` with raw TUI (the structured paths drift between Fluent versions).
- Student license allows only one Fluent session at a time. If you see "Connection refused" on launch, check Task Manager for stray `fluent.exe` processes.
- For the Forward-Facing Step case `"mask_hill": false` is required to avoid a Fluent error about "overlapping periodic interfaces". This is a quirk of the FFS geometry, which has a small ledge at the inlet that collides with the periodic interface. The hill mask (which zeros out the loss contribution from the hill region) is not needed for FFS since the ledge doesn't affect the loss.
- For the Forward-Facing Step case, the `ceiling` zone must be converted to a `symmetry` boundary after mesh load and before inlet/outlet setup. Named selections can arrive from meshing as wall-type zones.
- For the Forward-Facing Step case, the outlet is a `pressure-outlet` with static gauge pressure from `case.options.outlet_static_pressure` (default `0.0` Pa). Configure this with PyFluent's structured pressure-outlet settings, not the TUI prompt stream.

---

## Out of scope (not implemented)

- Parallel trials (sequential only)
- BoTorch backend (skopt only; add via `optimizer.py::build_optimizer` when needed)
- Mesh sensitivity studies (mesh is fixed at experiment start)
- Anything ML beyond skopt (no PyTorch, no JAX)

---

## FFS plotting helper

- `scripts/ffs/plot_ffs_fields.py` is a standalone plotting helper for FFS DNS/simulation data.
- Configuration lives in `scripts/ffs/plots/*.json`, and figures are written to `scripts/ffs/plots/<config-name>/`.
- Simulation and DNS columns are selected by exported header strings; legacy numeric simulation indices are still accepted.
- It does not participate in the main `src/geko_bayesopt` config flow.
- Only `scripts/ffs/plots/ffs_default.json` is tracked; local working configs and generated figures stay ignored.
- Keep each JSON config in sync with the specific DNS and simulation exports you want to inspect.

## FFS DNS conversion helper

- `data/dns/ffs/average_z_dns_ffs.py` converts all raw `FFS_Reh*_SBES_Node` exports to `*_2D.csv`.
- It maps the 20 primary spanwise planes to the canonical first-plane mesh and applies trapezoidal averaging.
- Exact `(x, y)` grouping is invalid for these exports because coordinate noise fragments equivalent mesh points.
