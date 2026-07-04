# optimizer_visualization

Plotting scripts for the GEKO Bayesian Optimisation project.
All scripts are run from the **project root** and write their output under `optimizer_visualization/plots/`:
`benchmark_individual.py` writes to `plots/individual/`, `optimizer_comparison.py` writes to `plots/comparison/`.

---

## Folder contents

| File | Purpose |
|---|---|
| `benchmark_individual.py` | Runs all 8 optimizers on the synthetic test functions and saves a separate PNG per optimizer |
| `optimizer_comparison.py` | Convergence comparison across BO/NM/FD/hybrids, plus a dedicated BO-vs-PSO comparison, using **real CFD results** (Periodic Hills Re=2800) |
| `ot.py` | Standalone convergence plotter for a single real CFD experiment — separate from the optimizer comparison |
| `_benchmark_core.py` | Shared code imported by the benchmark scripts — do not run directly |
| `documentation.md` | Presentation notes: what to communicate when showing the plots to executives |
| `plots/individual/` | PNGs from `benchmark_individual.py` |
| `plots/comparison/` | PNGs from `optimizer_comparison.py` |

---

## How to run

All commands are executed from the **project root** using the project virtual environment.

### Synthetic benchmarks (no CFD required)

```bash
# Individual figures  ->  plots/individual/01_bayesian_opt.png ... plots/individual/08_hybrid_bo_fd.png
.venv/Scripts/python.exe optimizer_visualization/benchmark_individual.py
```

Runs all 8 optimizers sequentially and prints progress to the terminal.
Runtime is roughly 1-2 minutes.

### Real-results comparison (requires CFD data)

```bash
# 1-D results (default) -> plots/comparison/1d/optimizer_comparison_1d_*.png
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py

# 2-D results -> plots/comparison/2d/optimizer_comparison_2d_*.png
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py --dim 2d
```

Each run produces two sets of plots, both driven entirely by named cutoff constants
at the top of `optimizer_comparison.py`:

**Main comparison** (`RUNS_1D_BO` / `RUNS_2D_BO` — BO, NM, FD, and all hybrids; PSO
excluded):
- **`_full`** — all iterations on a linear y-axis, y-range auto-fitted to the data.
- **`_after_iter<_CUT_ITER_1D/2D>`** — early iterations removed (iter > 7 for 1-D,
  iter > 13 for 2-D), i.e. shortly after the "BO sampling stops" marker.
- **`_after_iter<_CUT_ITER_1D/2D_STAGE2>`** — iterations removed up to the "BO swaps
  to NM/FD" marker (iter > 14 for 1-D, iter > 23 for 2-D).

**BO-vs-PSO comparison** (`RUNS_1D_BO_VS_PSO` / `RUNS_2D_BO_VS_PSO` — BO against the
PSO particle-count sweep: 3/5/7/9 particles in 1-D, 10/15/20 particles in 2-D),
filenames prefixed `_vs_pso_`:
- **`_full`** — all iterations on a linear y-axis, y-range auto-fitted to the data.
- **`_after_iter<sampling-stop>`** — cut at the same iteration where BO's own sampling
  phase ends (`_BO_SAMPLING_STOP`), i.e. iter > 7 for 1-D, iter > 13 for 2-D.
- **`_after_iter<_CUT_ITER_1D/2D_PSO>`** — cut further in, at iter > 28 (1-D) or
  iter > 51 (2-D), to compare late-stage convergence. In 1-D this cutoff falls past
  BO's own real-CFD budget (21 evals total), so BO has already finished and would
  only show a flat trailing line; that file is produced under a separate
  `_pso_only_` slug (`optimizer_comparison_1d_pso_only_after_iter28.png`) with BO
  dropped and a dedicated title, instead of as a `_vs_pso_` file. In 2-D, BO's
  larger budget survives the cut, so it remains a genuine `_vs_pso_` file.

The "BO swaps to NM/FD" marker is omitted from the BO-vs-PSO plots since none of
those runs ever swap to NM/FD.

Each legend entry is suffixed with the parameter value(s) at that run's best (argmin)
point, e.g. `Bayesian Opt (GP); Csep=0.886` in 1-D or
`Hybrid BO -> FD; Csep=0.888, Cnw=0.513` in 2-D. This is the global best found over
the whole run, so it's identical across the `_full` and `_after_iter*` variants of a
given comparison.

Raw per-iteration costs and prolongation dashes are not shown; each curve ends where its experiment ended.

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
