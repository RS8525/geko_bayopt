# optimizer_visualization

Plotting scripts for the GEKO Bayesian Optimisation project.
All scripts are run from the **project root** and write their output to `plots/`.

---

## Folder contents

| File | Purpose |
|---|---|
| `benchmark_combined.py` | One figure with all 8 optimizers arranged in rows (1-D left, 2-D right) |
| `benchmark_individual.py` | Same data, but saves a separate PNG per optimizer — useful for documentation slides |
| `ot.py` | Convergence comparison against **real CFD results** for the Periodic Hills Re=2800 case |
| `_benchmark_core.py` | Shared code imported by the two benchmark scripts — do not run directly |
| `plots/` | All output PNGs land here |

---

## How to run

All commands are executed from the **project root** using the project virtual environment.

### Synthetic benchmarks (no CFD required)

These scripts run the optimizers on two analytic test functions and are self-contained — no simulation data is needed.

```bash
# Combined figure  →  plots/benchmark_combined.png
.venv/Scripts/python.exe optimizer_visualization/benchmark_combined.py

# Individual figures  →  plots/01_bayesian_opt.png … plots/08_hybrid_bo_fd.png
.venv/Scripts/python.exe optimizer_visualization/benchmark_individual.py
```

Both scripts run all 8 optimizers sequentially and print progress to the terminal.
Runtime is roughly 30–60 seconds per script.

### Real-results comparison (requires CFD data)

```bash
# Produces three plots:  optimizer_comparison_linear.png
#                        optimizer_comparison_log.png
#                        optimizer_comparison_zoom.png
# (all saved to the project root)

.venv/Scripts/python.exe optimizer_visualization/ot.py

# Overlay synthetic convergence curves when no CFD data is available yet:
.venv/Scripts/python.exe optimizer_visualization/ot.py --fake
```

`ot.py` reads from `results/experiments/<experiment_id>/metadata.csv`.
Experiments that have not been run yet are silently skipped.

---

## Test functions used in the benchmarks

The two synthetic benchmark scripts use the same pair of functions for all optimizers:

**1-D** — evaluated on `geko_csep ∈ [0.5, 3.5]`, GEKO default 1.75:
```
f(x) = −2.5x · sin(2.5x)
```
Local minimum near x ≈ 0.81, global minimum near x ≈ 3.19.
The GEKO default (x = 1.75) lands near the local maximum between the two minima — a deliberate trap for gradient-based methods that start from the default.

**2-D** — evaluated on `geko_csep × geko_cnw ∈ [0.5, 3.5] × [0.1, 0.9]`, GEKO defaults (1.75, 0.5):
```
f(x₁, x₂) = (u² + v − 11)² + (u + v² − 7)²   [Himmelblau, scaled]
where  u = −5 + (x₁ − 0.5) · 10/3
       v = −5 + (x₂ − 0.1) · 12.5
```
This is the Himmelblau function mapped from its standard domain [−5, 5]² onto the GEKO parameter space.
It has **4 global minima at f = 0**, all within the parameter bounds.
The GEKO starting point (1.75, 0.5) sits at f ≈ 168, far from any minimum.

Both functions use GEKO parameter names (`geko_csep`, `geko_cnw`) because the Nelder-Mead and Finite-Difference optimizers look up their default starting points by parameter name.

---

## Output files

### `benchmark_individual.py`

| File | Optimizer |
|---|---|
| `01_bayesian_opt.png` | Bayesian Optimisation (GP + Sobol initialisation) |
| `02_nelder_mead.png` | Nelder–Mead simplex |
| `03_finite_differences.png` | Finite-difference gradient descent |
| `04_particle_swarm.png` | Particle Swarm Optimisation (PSO) |
| `05_hybrid_nm_bo.png` | Hybrid: Nelder–Mead warm-up → Bayesian Optimisation |
| `06_hybrid_fd_bo.png` | Hybrid: Finite Differences warm-up → Bayesian Optimisation |
| `07_hybrid_bo_nm.png` | Hybrid: Bayesian Optimisation → Nelder–Mead refinement |
| `08_hybrid_bo_fd.png` | Hybrid: Bayesian Optimisation → Finite Differences refinement |

### `benchmark_combined.py`

| File | Contents |
|---|---|
| `benchmark_combined.png` | All 8 optimizers in a single figure (8 rows × 2 columns) |

### `ot.py`

| File | Contents |
|---|---|
| `optimizer_comparison_linear.png` | Best-cost-so-far on a linear y-axis with a late-stage zoom inset |
| `optimizer_comparison_log.png` | Same on a log y-axis |
| `optimizer_comparison_zoom.png` | Late-stage view only, log y-axis |

---

## Reading the benchmark plots

- **Grey curve** — objective function shape (1-D plots only)
- **Red ★** — true optimum of the test function
- **Gold ★** — best point found by the optimizer
- **Dot colour** — evaluation order: dark purple = first evaluation, bright yellow = last
- **● circles** — Phase 1 evaluations (or all evaluations for single-phase optimizers)
- **▲ triangles** — Phase 2 evaluations (hybrid optimizers only, after the strategy switch)
- **White path line** — trajectory through parameter space (2-D plots)
