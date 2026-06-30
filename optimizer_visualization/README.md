# optimizer_visualization

Plotting scripts for the GEKO Bayesian Optimisation project.
All scripts are run from the **project root** and write their output to `optimizer_visualization/plots/`.

---

## Folder contents

| File | Purpose |
|---|---|
| `benchmark_individual.py` | Runs all 8 optimizers on the synthetic test functions and saves a separate PNG per optimizer |
| `optimizer_comparison.py` | Convergence comparison across all 8 optimizers using **real CFD results** (Periodic Hills Re=2800) |
| `ot.py` | Standalone convergence plotter for a single real CFD experiment — separate from the optimizer comparison |
| `_benchmark_core.py` | Shared code imported by the benchmark scripts — do not run directly |
| `documentation.md` | Presentation notes: what to communicate when showing the plots to executives |
| `plots/` | All output PNGs land here |

---

## How to run

All commands are executed from the **project root** using the project virtual environment.

### Synthetic benchmarks (no CFD required)

```bash
# Individual figures  ->  plots/01_bayesian_opt.png ... plots/08_hybrid_bo_fd.png
.venv/Scripts/python.exe optimizer_visualization/benchmark_individual.py
```

Runs all 8 optimizers sequentially and prints progress to the terminal.
Runtime is roughly 1-2 minutes.

### Real-results comparison (requires CFD data)

```bash
# 1-D results (default) -> plots/optimizer_comparison_1d_{linear,log,zoom}.png
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py

# 2-D results -> plots/optimizer_comparison_2d_{linear,log,zoom}.png
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py --dim 2d

# Overlay synthetic convergence curves when no CFD data is available yet:
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py --fake
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py --dim 2d --fake
```

`optimizer_comparison.py` reads `metadata.csv` from:
- 1-D: `results/experiments/optimizer_comparison/one-param-runs/<experiment_id>/metadata.csv`
- 2-D: `results/experiments/optimizer_comparison/two-param-runs/<experiment_id>/metadata.csv`

Experiments that have not been run yet are silently skipped.

---

## Test functions used in the benchmarks

**1-D** — evaluated on `geko_csep in [0.5, 3.5]`, GEKO default 1.75:
```
f(x) = -(2.5x) * sin(2.5x)
```
Local minimum near x = 0.81, global minimum near x = 3.19.
The GEKO default (x = 1.75) lands near the local maximum between the two minima — a deliberate trap for gradient-based methods that start from the default.

**2-D** — evaluated on `geko_csep x geko_cnw in [0.5, 3.5] x [0.1, 0.9]`:
```
f(x1, x2) = f1d(x1) + 3 * |x2 - 0.5|
```
Combines the 1-D objective along x₁ with a V-shaped linear penalty along x₂,
centred at x₂ = 0.5. Global minimum at (≈3.19, 0.5); local minimum at (≈0.81, 0.5).
The absolute-value term is non-smooth at x₂ = 0.5 (intentional — no optimizer
uses analytical gradients, so this tests robustness to non-smoothness).

Both functions use GEKO parameter names (`geko_csep`, `geko_cnw`) because the
Nelder-Mead and Finite-Difference optimizers look up their default starting points
by parameter name.

---

## Output files

| File | Optimizer |
|---|---|
| `01_bayesian_opt.png` | Bayesian Optimisation (GP + Sobol initialisation) |
| `02_nelder_mead.png` | Nelder-Mead simplex |
| `03_finite_differences.png` | Finite-difference gradient descent |
| `04_particle_swarm.png` | Particle Swarm Optimisation (PSO) |
| `05_hybrid_nm_bo.png` | Hybrid: Nelder-Mead warm-up -> Bayesian Optimisation |
| `06_hybrid_fd_bo.png` | Hybrid: Finite Differences warm-up -> Bayesian Optimisation |
| `07_hybrid_bo_nm.png` | Hybrid: Bayesian Optimisation -> Nelder-Mead refinement |
| `08_hybrid_bo_fd.png` | Hybrid: Bayesian Optimisation -> Finite Differences refinement |

---

## Reading the benchmark plots

- **Grey curve** — objective function shape (1-D plots only)
- **Red star** — true optimum of the test function
- **Gold star** — best point found by the optimizer
- **Dot colour** — evaluation order: dark purple = first, bright yellow = last (all optimizers except PSO)
- **Circle dots** — Phase 1 evaluations (or all evaluations for single-phase optimizers)
- **Triangle dots** — Phase 2 evaluations (hybrid optimizers only, after the strategy switch)
- **White path line** — trajectory through parameter space (2-D plots, non-PSO)

### PSO plots

PSO dots are coloured **by particle** (not by evaluation order).  Each of the 4
particles keeps its own colour across all swarm iterations so its trajectory is
visually traceable.  The sequential evaluation-order colorbar is therefore omitted
from PSO figures; a particle legend is shown instead.

### Note on Finite Differences plots

FD plots show **only the gradient-step evaluations**, not the finite-difference probe
evaluations used to estimate the gradient.  The number of visible dots is therefore
lower than the total evaluation count:
- 1-D: N = 20 function evals → 10 visible dots (10 gradient steps)
- 2-D: N = 36 function evals → 12 visible dots (12 gradient steps)

See `documentation.md` for a full explanation.

### Note on evaluation budgets

All optimizers use the same total budget: **20 evaluations in 1-D**, **36 in 2-D**.

PSO: 4 particles × (1 init + 4 swarm iters) = 20 evals (1-D); 4 × (1 + 8) = 36 (2-D).
