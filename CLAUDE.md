# CLAUDE.md — Implementation knowledge for this repository

This file records non-obvious constraints, workarounds, and design decisions that
are not visible from the code alone.  Read this before touching the optimizer or
visualization code.

---

## Project structure

```
src/geko_bayesopt/       Core optimizer library (ask/tell interface)
optimizer_visualization/ Benchmark plotting scripts
configs/optimizer_comparison_configs/  Real-run JSON configs (one per optimizer)
results/experiments/     Real CFD run outputs (metadata.csv per experiment)
```

The optimizer library is imported via `sys.path` injection in `_benchmark_core.py`
(adds `<project_root>/src/` at import time).  Run all scripts from the project root.

---

## Optimizer interface

All optimizers expose three methods:

```python
x: list[float]  = opt.ask()          # suggest next parameter vector
                  opt.tell(x, y)     # report objective value
done: bool      = opt.should_stop()  # convergence / budget check
```

`build_optimizer(section, parameters)` constructs the correct class from an
`OptimizerSection` config object.

---

## Finite Differences — known difficulties

### The optimizer does not distinguish probe evals from gradient-step evals

`FiniteDifferenceOptimizer` interleaves two types of evaluations in a fixed cycle:
- **D probe evals** — perturb each dimension by `step_size × range` to estimate ∂f/∂xᵢ
- **1 gradient step eval** — move `base − lr × grad`, clipped to bounds

The optimizer's `ask()`/`tell()` interface returns probe points and step points
identically.  For plotting, probe points are noise (they exist only to estimate the
gradient) and should be hidden.

**Workaround:** `_fd_show_mask(section, n, d)` in `_benchmark_core.py` analytically
reconstructs which indices are steps vs probes from the known cycle structure:

| Optimizer kind       | Pattern (D=1, 1D)                  | Pattern (D=2, 2D)               |
|----------------------|------------------------------------|---------------------------------|
| `finite_differences` | 0=base, 1=probe, 2=step, 3=probe… | 0=base, 1,2=probes, 3=step, …  |
| `hybrid_fd_bayes`    | FD phase as above; BO phase all show | same                          |
| `hybrid_bayes_fd`    | BO phase all show; FD starts with probe (base injected, not re-eval'd) | same |

`run_1d` and `run_2d` return a 3-tuple `(xs, ys, step_mask)`.  Pass `step_mask`
to `plot_1d_ax` / `plot_2d_ax` to filter automatically.

**Ideal fix (future):** `FiniteDifferenceOptimizer` should expose a method or
property (e.g. `pending_op: str`) so callers can query whether the last `ask()`
was a probe or a gradient step without relying on cycle arithmetic.

### Learning rate must match the function's Lipschitz constant

The correct step size is `lr = 1 / L` where L is the spectral norm of the Hessian
over the domain (Lipschitz constant of the gradient).

| Context              | L estimate | lr used   |
|----------------------|-----------|-----------|
| Benchmark 1D         | ≈ 67       | 0.015     |
| Benchmark 2D         | ≈ 67 (x₁) | 0.015     |
| Real CFD (standalone FD) | unknown | 0.075 (chosen empirically) |
| Real CFD (BO→FD exploitation) | unknown | 0.045 (60% of standalone) |

The x₂ direction of the 2D composite function is piecewise-linear (Hessian = 0
away from the kink at x₂ = 0.5).  Current benchmark lr = 0.015 is calibrated
to the x₁ direction and works well in practice.

### Boundary clipping — current workaround, not ideal

`_clip(x, params)` in `_benchmark_core.py` pins any out-of-bounds suggestion to
the nearest boundary before evaluation and `tell()`.  This handles two cases:
- **NM reflections** near the lower bound can produce x < low.
- **FD gradient steps** with large lr overshoot bounds.

Clipping is physically hard to justify: the optimizer proposes one point but we
evaluate (and report) a different one.  Alternatives to investigate:
- **Reflection:** mirror the overshoot back into the domain.
- **Backtracking line search:** halve `lr` until the step stays in bounds.
- **Rejection / re-ask:** discard the point and call `ask()` again.

This is tracked as an open task.

### FD on the real GEKO objective

`configs/optimizer_comparison_configs/03_fd_1d_ph2800.json` uses `learning_rate=0.075`
and `configs/optimizer_comparison_configs/08_bo_fd_1d_ph2800.json` uses `learning_rate=0.045`
(60% exploitation tightening).  Both use `step_size=0.05`.

---

## Nelder–Mead — known difficulties

NM can propose simplex vertices outside the parameter bounds (reflections near the
lower bound produce x < low).  The same `_clip()` workaround as FD is applied.
See the boundary clipping section above for open alternatives.

---

## Particle Swarm Optimization — equal budget, particle-coloured plots

PSO uses `n_particles × (1 + max_iter)` evaluations total.  Budget equals all other
optimizers in both dimensions:

| Dimension | n_particles | max_iter | Total evals |
|-----------|------------|---------|------------|
| 1D        | 4          | 4       | 20         |
| 2D        | 4          | 8       | 36         |

**Plotting:** PSO dots are coloured by particle index (not evaluation order) so each
particle's trajectory is visually traceable across swarm iterations.  Each particle
keeps the same colour across all iterations.  Consequently the sequential
evaluation-order colorbar is omitted from PSO figures — `_PARTICLE_COLORS` in
`_benchmark_core.py` defines the four colours (tab10 orange/green/purple/brown).

Pass `n_particles` to `plot_1d_ax` / `plot_2d_ax` (and to `make_individual_figure` in
`benchmark_individual.py`) to activate this mode.  The "N = function evals" count is
embedded in the top-right annotation instead of a separate text box.

---

## Hybrid optimizers — phase split alignment and exploitation

### Phase splits (benchmark)

For FD-based hybrids, the FD phase must span whole cycles
(1 base + D probes + 1 step per cycle, period = D+1).

| Optimizer       | 1D n_initial | 2D n_initial | Comment                                     |
|-----------------|-------------|-------------|---------------------------------------------|
| NM → BO         | 10          | 16          |                                             |
| FD → BO         | 9           | 16          | 9 = 1+4×2 (4 cycles, 1D); 16 = 1+5×3 (5 cycles, 2D) |
| BO → NM         | 10          | 20          | NM phase = 10 (1D) / 16 (2D) evals         |
| BO → FD         | 10          | 20          | FD starts with a probe (base injected, not re-eval'd) |

### Exploitation tightening in BO → NM and BO → FD

When BO locates a good region first, the subsequent NM or FD phase can exploit it
by starting with a tighter search radius.

**BO → NM:** `nm_options` accepts `simplex_scale` (float, default 1.0).  In the
benchmark and real configs, `simplex_scale=0.6` shrinks the NM startup simplex to
60% of its normal size (x₁ offsets ±0.15 instead of ±0.25; x₂ offset +0.06 instead
of +0.10).  Implemented in `HybridBayesNelderMeadOptimizer._build_nm_simplex`
(`src/geko_bayesopt/optimizer.py`).

**BO → FD:** Pass smaller `step_size` and `learning_rate` in `fd_options`:
- Benchmark: `step_size=0.03` (vs 0.05), `learning_rate=0.009` (vs 0.015)
- Real CFD:  `step_size=0.03`, `learning_rate=0.045` (vs 0.075)

The FD→BO and NM→BO hybrids use the same FD/NM parameters as the standalone
versions — the heuristic phase runs first with no prior knowledge of the optimum.

For `hybrid_bayes_fd`, the BO best is injected directly as the FD starting point
without re-evaluating it.  This means the first FD `ask()` returns a probe, not a
base.  `_fd_show_mask` accounts for this offset.

---

## 2-D test function

```
f(x₁, x₂) = f1d(x₁) + 3·|x₂ − 0.5|
```

Combines the 1-D objective along x₁ (same local/global minimum structure) with a
V-shaped linear penalty along x₂ centered at 0.5.

- Global minimum at (X1D_STAR, 0.5), f = Y1D_STAR.
- Local minimum at (≈0.81, 0.5).
- No clipping in contour plots — range is naturally bounded (≈ −8 to +6).

### Non-smoothness of the x₂ term

`|x₂ − 0.5|` has a kink (non-smooth point) at the exact ridge minimum x₂ = 0.5.
This is intentional: none of the optimizers use analytical gradients.  FD probes
straddling x₂ = 0.5 will estimate the gradient as ±3, which points toward the
minimum correctly but may cause oscillation if lr is too large.

---

## File roles

| File | Role | Notes |
|------|------|-------|
| `_benchmark_core.py` | Shared logic: test functions, optimizer configs, run helpers, plot helpers | Do not run directly |
| `benchmark_individual.py` | Saves one PNG per optimizer | Run from project root |
| `ot.py` | Plots real CFD results from `results/experiments/` | Requires actual run data; use `--fake` for synthetic curves |
| `src/geko_bayesopt/optimizer.py` | Optimizer implementations | Do not modify for benchmark purposes |
| `src/geko_bayesopt/config.py` | `OptimizerSection`, `ParameterSpec` dataclasses | |
| `src/geko_bayesopt/optimizers_readme.md` | Reference doc for all optimizer `kind` values and their options | |
| `configs/optimizer_comparison_configs/` | 8 JSON configs for real CFD runs (one per optimizer, 1D) | Numbered to match benchmark PNG slugs |

---

## Windows / encoding notes

- Shell: use PowerShell or Git Bash.  Path separator is `\` in PowerShell, `/` in Bash.
- Print statements must use ASCII-safe characters — the terminal uses cp1252 which
  does not support Unicode arrows (→) or dashes (–).  Use `->` and `-` in strings.
- Virtual environment: `.venv/Scripts/python.exe` (not `python` or `python3`).
