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
├── geko_defaults.py        Canonical Fluent GEKO baseline coefficients
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

Field-error objectives accept `objective.options.field_error_norm`. The default
`"l2"` is the historical area-weighted RMSE divided by DNS weighted standard
deviation. `"l1"` uses the area-weighted mean absolute residual divided by the
same DNS weighted standard deviation.

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

FFS production configs live below `configs/ffs_final/`, split into
`2_param/` and `all_param/`. Because the runner still uses the config
grandparent as the inferred repo root, nested configs must point back to the
real input tree with `../../data/...` paths. Run these configs from the
repository root so explicit result paths resolve under `<repo>/results/`.
Completed final FFS result bundles are consolidated under
`results/ffs_final_runs/`, split into `all_param_runs/<experiment_id>/` and
`two-param-runs/<experiment_id>/`. In those bundles, `fluent_work_dir` and
`results_dir` intentionally point to the same folder so the retained final
ASCII, default ASCII, solved `.cas.h5`/`.dat.h5`, `metadata.csv`, and
`optimizer.pkl` stay together. The unfinished `ffs_re6000_all_param_mae_final`
run still uses the historical split between `results/fluent/` and
`results/experiments/` until it completes.

---

## Save-before-tell ordering

When `evaluate_default_first` is enabled (the default), a fresh experiment
first runs Fluent with no GEKO coefficient overrides. This baseline is scored
and saved with `trial_role="baseline"`, but it is never passed to
`optimizer.tell()` and does not consume the optimizer's `n_calls` budget.

The optimizer loop then runs in this order per trial:

1. `optimizer.ask()` → parameter vector
2. `solver.run_trial(params)` → ASCII + Fluent state
3. `extract.build_run_result(...)` → RunResult
4. `loss_fn(run_result)` → score
5. `store.save_trial(result, score)` ← ON DISK before next step
6. `optimizer.tell(x, score)` ← optimizer learns
7. `store.save_optimizer(optimizer)` ← checkpoint

If a crash happens between (5) and (6), the next run replays completed trials from `metadata.csv` via `tell()` and continues. If we reversed (5) and (6), a crash there would leave the optimizer ahead of the durable record, which is unrecoverable.

`metadata.csv` records `trial_role` as either `baseline` or `optimizer`.
Resume replay and completed-call counting use only optimizer rows. Legacy
metadata without this column is treated as optimizer-only data. If an existing
experiment has optimizer trials but no baseline, rerunning it backfills only
the independent baseline without changing optimizer history or budget.

---

## Output retention

- `keep_only_best_case_files=true` keeps only the current-best solved
  `.cas.h5`/`.dat.h5` pair and removes initialization cases.
- `keep_all_ascii_files` defaults to the negation of
  `keep_only_best_case_files`.
- When `keep_all_ascii_files=false`, only the current-best optimizer ASCII and
  every baseline ASCII are retained.
- Baseline ASCII is always protected, independent of both retention settings.

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
- Do not include `x-coordinate` or `y-coordinate` in `PeriodicHillSolver.EXPORT_VARIABLES`. Fluent's ASCII exporter writes the coordinates automatically; requesting them explicitly duplicates both columns in the output.
- Student license allows only one Fluent session at a time. If you see "Connection refused" on launch, check Task Manager for stray `fluent.exe` processes.
- For the Forward-Facing Step case `"mask_hill": false` is required to avoid a Fluent error about "overlapping periodic interfaces". This is a quirk of the FFS geometry, which has a small ledge at the inlet that collides with the periodic interface. The hill mask (which zeros out the loss contribution from the hill region) is not needed for FFS since the ledge doesn't affect the loss.
- For the Forward-Facing Step case, the `ceiling` zone must be converted to a `symmetry` boundary after mesh load and before inlet/outlet setup. Named selections can arrive from meshing as wall-type zones.
- For the Forward-Facing Step case, the outlet is a `pressure-outlet` with static gauge pressure from `case.options.outlet_static_pressure` (default `0.0` Pa). Configure this with PyFluent's structured pressure-outlet settings, not the TUI prompt stream.
- Configure the FFS velocity inlet and pressure outlet entirely through structured PyFluent settings. All momentum and turbulence inputs must be value-based; empty profile references cause persistent interpolation warnings and a false continuity-residual plateau. Keep outlet target mass flow disabled.

---

## Out of scope (not implemented)

- Parallel trials (sequential only)
- BoTorch backend (skopt only; add via `optimizer.py::build_optimizer` when needed)
- Mesh sensitivity studies (mesh is fixed at experiment start)
- Anything ML beyond skopt (no PyTorch, no JAX)

---

## FFS plotting helper

- `scripts/ffs/plot_ffs_fields.py` is a standalone plotting helper for FFS DNS/simulation data.
- Configuration lives in `scripts/ffs/plots/*.json`; final-run reusable configs live in `scripts/ffs/plots/final/*.json`. Figures are written to `scripts/ffs/plots/<config-name>/`.
- Plot configs may set a repo-root-relative `output_dir`; final-run configs write plots into each run bundle below `results/ffs_final_runs/.../plots/`.
- Simulation and DNS columns are selected by exported header strings; legacy numeric simulation indices are still accepted.
- Normalized-error plots use the common-grid objective path and can report both L1 and L2 field contributions via `plots.normalized_error.norms`.
- Fixed normalized-error color scales are configured with `plots.normalized_error.error_limits`, either globally or per field/pair, so default-GEKO and optimized-GEKO plots can be compared directly.
- `scripts/ffs/plot_metadata_convergence.py` plots optimizer metadata convergence. Final metadata configs live in `scripts/ffs/plots/final_metadata/`.
- Metadata convergence configs can use `score_as` to benchmark 2-parameter runs under the matching all-parameter GEDCP preference convention. This relies on final FFS objectives having `lambda_integral = 0`, recovering field error from the stored score, then reapplying the target preference term with missing GEKO coefficients filled by defaults.
- Final metadata configs also include standalone 2-parameter convergence plots where the default-GEKO metadata row is drawn as the baseline.
- It does not participate in the main `src/geko_bayesopt` config flow.
- Tracked reusable configs cover the generic example plus default-GEKO and optimized-GEKO plots for every completed final FFS run. Other local working configs and generated figures stay ignored unless they are intentionally published.
- Keep each JSON config in sync with the specific DNS and simulation exports you want to inspect.

## FFS DNS conversion helper

- `data/dns/ffs/average_z_dns_ffs.py` converts all raw `FFS_Reh*_SBES_Node` exports to `*_2D.csv`.
- It maps the 20 primary spanwise planes to the canonical first-plane mesh and applies trapezoidal averaging.
- Exact `(x, y)` grouping is invalid for these exports because coordinate noise fragments equivalent mesh points.

## FFS total turbulent kinetic energy

- The canonical comparison key is `"total-turbulent-kinetic-energy"` in `m^2/s^2`.
- The active Re=6000 DNS/SBES reference is `FFS_Reh6000_SBES_Node_NEW_2D.csv`.
- FFS DNS preprocessing uses `mean-tke_tot-dataset` where available, including the active Re=6000 reference. For legacy exports without that column, it derives total TKE on each raw 3D row as modeled `k` plus `0.5 * (u_rms^2 + v_rms^2 + w_rms^2)`, then performs z averaging.
- For FFS RANS, Fluent's `turb-kinetic-energy` already represents total TKE and is exposed under the canonical key without changing the existing raw field.
- Periodic-hills extraction retains its existing non-dimensional `turb-kinetic-energy` behavior.

## Turbulent-viscosity findings

- Fluent's `viscosity-turb` is dynamic turbulent viscosity in `kg/(m s)`.
- Do not use `source-turbulent-viscosity`: its unit is `kg/(m s^2)`, so it is a turbulent-viscosity source rate rather than turbulent viscosity.
- A direct RANS-to-SBES turbulent-viscosity comparison is not physically equivalent. In RANS, eddy viscosity represents the effect of the fully modeled turbulence. In SBES, the exported turbulent viscosity represents only the modeled/subgrid contribution; resolved turbulent transport is not included in that field.
- Spanwise averaging can smooth localized turbulent-viscosity peaks, but it does not resolve the modeled-versus-resolved mismatch. Low SBES turbulent-viscosity values are therefore not, by themselves, evidence of an averaging error.
- The final FFS optimization does not export, load, or score turbulent viscosity. It uses total TKE instead: RANS `turb-kinetic-energy` versus SBES resolved plus modeled TKE.
- If turbulent viscosity is reintroduced later, define whether the intended comparison is modeled contribution only and establish a consistent dimensional or non-dimensional scaling before adding it to `RunResult.fields`.

## Final FFS optimization configs

- `configs/ffs_final/2_param/` contains one C_SEP/C_NW Bayesian optimization config for each of Re=2000, 3000, 4000, and 6000.
- `configs/ffs_final/all_param/` contains matching all-parameter configs named `ffs_re*_all_param*.json`; they optimize `geko_csep`, `geko_cnw`, `geko_cjet`, `geko_cturb`, and `geko_cmix`.
- Completed final FFS run artifacts live in `results/ffs_final_runs/all_param_runs/` and `results/ffs_final_runs/two-param-runs/`, one directory per `experiment_id`. `ffs_re6000_all_param_mae_final` is excluded from that publication bundle until it completes.
- Each config hardcodes a unique `base_case_name`, DNS reference, fluid viscosity, inlet turbulence intensity, and inlet viscosity ratio.
- The 2-parameter runs perform 48 evaluations: 16 Sobol initial points and 32 GP-guided proposals. The all-parameter runs perform 100 evaluations: 32 Sobol initial points and 68 GP-guided proposals. Epsilon early stopping is disabled so every case receives the full budget.
- All final FFS runs require continuity, velocity, `k`, and `omega` residuals
  to reach `1e-6`.
- The objective uses `Ux` and `total-turbulent-kinetic-energy` on a 360x120 common grid with the FFS step masked out.
- The final GEDCP configurations use `lambda_preference = 0.5`. Each field contribution defaults to common-grid RMSE divided by the common-grid DNS standard deviation; `field_error_norm` can switch this to L1 field error. The field contributions are summed and then multiplied by the GEKO default-coefficient preference factor.
- Historical FFS configs live under `configs/ffs_retired/` and are not production inputs.
