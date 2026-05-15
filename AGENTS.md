# geko_bayopt — Architecture Notes

Living document. Update when structure changes. Code is the source of truth; this file explains the *why*.

---

## Project goal

Bayesian optimization of the GEKO turbulence-model coefficients in ANSYS Fluent, scored against DNS reference data. Single-machine, single-user, thesis-scale. One flow case today (periodic hill); more later (forward-facing step, etc.).

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
│   └── periodic_hills/
│       └── case.py         PeriodicHillsCase (BCs + Laizet DNS loader)
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

- `cad_route="DSCO"` for `.dsco` files. The `"Workbench"` route silently produces unmeshable shells with zero cells.
- `two_dim.write_2d_mesh` (workflow task), NOT `tui.file.write_mesh` (the latter produces files the solver rejects as "surface mesh").
- Mesh write is sanity-checked for file size > 0.5 MB to catch silent failures.
- `version="2d"` must NOT be passed to `launch_fluent` with `mode="meshing"` — that combination silently switches to the solver and breaks the workflow.
- GEKO coefficients live at `solver.settings.setup.models.viscous.geko.<coef>.value` on Fluent 2026 R1.
- Periodic forcing uses `/define/periodic-conditions/massflow-rate-specification?` with raw TUI (the structured paths drift between Fluent versions).
- Student license allows only one Fluent session at a time. If you see "Connection refused" on launch, check Task Manager for stray `fluent.exe` processes.

---

## Out of scope (not implemented)

- Parallel trials (sequential only)
- BoTorch backend (skopt only; add via `optimizer.py::build_optimizer` when needed)
- Forward-facing step or other cases (architectural slot exists; no code yet)
- Mesh sensitivity studies (mesh is fixed at experiment start)
- Anything ML beyond skopt (no PyTorch, no JAX)
