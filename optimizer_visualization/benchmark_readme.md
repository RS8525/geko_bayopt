# Optimizer Benchmark — Presentation Notes

This document records everything that must be communicated when presenting the
benchmark plots to an audience unfamiliar with the implementation details.

---

## Test functions

The benchmarks use two analytic functions chosen to be representative of the
challenges the optimizers face on the real GEKO objective.

**1-D** — `f(x) = −2.5x · sin(2.5x)` on `geko_csep ∈ [0.5, 3.5]`
- Has one **local minimum** near x ≈ 0.81 and a deeper **global minimum** near x ≈ 3.19.
- The GEKO default starting point (x = 1.75) sits near a local maximum between the two
  minima — a deliberate trap for gradient-based methods.
- Methods that follow gradients (Finite Differences, Nelder–Mead) will converge to
  the local minimum, not the global one, unless they explore broadly enough.

**2-D** — composite function on `geko_csep × geko_cnw ∈ [0.5, 3.5] × [0.1, 0.9]`:

```
f(x1, x2) = f1D(x1)  +  3 · |x2 − 0.5|
```

- The `x1` direction reuses the 1-D objective, preserving its **local minimum**
  (x1 ≈ 0.81) and **global minimum** (x1 ≈ 3.19) structure.
- The `x2` direction adds a symmetric, V-shaped cost for any deviation from x2 = 0.5.
  A deviation of 0.4 (the full half-range) adds 3 × 0.4 = 1.2 to the cost.
- Combined: **global minimum at (≈3.19, 0.5)**, local minimum at (≈0.81, 0.5).
- The GEKO starting point (1.75, 0.5) sits at f ≈ +4.1, near the local maximum of
  f1D — in the x2 valley but on the wrong side of the 1-D ridge.

**Non-smoothness note:** The `|x2 − 0.5|` term is non-smooth (has a kink) at the
exact minimum ridge x2 = 0.5.  This is intentional: none of the optimizers in this
benchmark use analytical gradients, so a non-smooth landscape is a legitimate test
case.  Note that the Finite Differences optimizer approximates gradients numerically —
probes straddling x2 = 0.5 will return a gradient signal of ±3, which points
correctly toward the minimum but may cause oscillation close to the ridge if the
learning rate is too large.

These functions were chosen to challenge different optimizer properties: the 1-D
direction tests local-minimum avoidance; the 2-D direction tests convergence toward
a linear valley with a non-smooth floor.

---

## Evaluation budgets

| Optimizer | 1-D evals | 2-D evals |
|-----------|----------|----------|
| Bayesian Opt. (GP) | 24 | 48 |
| Nelder–Mead | 24 | 48 |
| Finite Differences | 24 | 48 |
| Particle Swarm | 24 | 48 |
| Hybrid NM → BO | 24 (9 + 15) | 48 (15 + 33) |
| Hybrid FD → BO | 24 (9 + 15) | 48 (15 + 33) |
| Hybrid BO → NM | 24 (17 + 7) | 48 (32 + 16) |
| Hybrid BO → FD | 24 (17 + 7) | 48 (32 + 16) |

All optimizers use the same budget: 24 evaluations in 1-D, 48 in 2-D, with
`random_state = 42` throughout. Every optimizer with an initialization phase
uses `n_initial = 9` in 1-D and `n_initial = 15` in 2-D; the BO-first hybrids
allocate a longer BO phase (17 evals in 1-D, 32 in 2-D) so it covers both the
Sobol initialization and a model-guided stretch before the hand-off.
PSO is configured with 4 particles; the swarm-iteration counts are derived
from the budget: 5 iterations in 1-D (4 × 6 = 24) and 11 in 2-D (4 × 12 = 48),
which divides evenly so no partial swarm iterations occur.

---

## What the plot elements mean

| Element | Meaning |
|---------|---------|
| Grey curve (1-D only) | Shape of the objective function |
| Red ★ | True global minimum of the test function |
| Gold ★ | Best point found by the optimizer |
| Dot colour | Evaluation order: dark purple = first, bright yellow = last |
| ● circles | Phase 1 evaluations (or all evals for single-phase optimizers) |
| ▲ triangles | Phase 2 evaluations (hybrid optimizers only, after strategy switch) |
| White path line (2-D, non-PSO) | Trajectory through parameter space |
| Red ★ (2-D) | Global minimum of the 2-D test function |
| Particle colour (PSO only) | Each particle keeps one fixed colour across all swarm iterations so its path is traceable; the evaluation-order colorbar is omitted for PSO |

---

## Finite Differences — what the plots show and what they hide

The Finite Differences optimizer works in cycles:
1. Evaluate the function at the current base point.
2. Probe each parameter dimension individually (perturb by a small step) to
   estimate the gradient — these are **internal evaluations, not optimizer decisions**.
3. Take one gradient-descent step.
4. The stepped point becomes the new base and the cycle repeats.

**The "N = … function evals" label counts all function evaluations** including the
probe evaluations that are internal to the FD algorithm.  Only the base evaluation
and the gradient-step evaluations are shown as dots in the plot.  As a result, the
number of visible dots is smaller than the N shown: for example, N = 24 function
evaluations in 1-D produce 12 visible dots (1 base + 11 gradient steps), because
each gradient-descent step requires 1 additional probe evaluation to estimate the
derivative in that dimension.  In 2-D, N = 48 function evaluations produce 16 visible
dots (1 base + 15 gradient steps), because each step requires 2 probe evaluations
(one per parameter dimension).

This design is intentional and consistent: the N label reflects the true computational
cost, while the plot shows only the evaluations that represent actual optimizer decisions.

### Learning rate calibration

The gradient-descent step size (`learning_rate`) must be calibrated to the function's
steepness (Lipschitz constant of the gradient):

- **1-D:** learning rate = 0.015, chosen below 1/L ≈ 0.019 (Lipschitz constant
  of the gradient: L = max |f''| ≈ 52 on [0.5, 3.5]).
  FD converges visibly toward the local minimum at x ≈ 0.81.
- **2-D:** learning rate = 0.015 (same as 1-D).  The `x1` direction has the same
  Lipschitz constant L ≈ 52, so the rate is equally appropriate.  The `x2` direction
  is piecewise-linear (slope ±3, Hessian = 0 away from the ridge), so the gradient
  step in x₂ is bounded by lr × 3 = 0.045, well within the x₂ range of 0.8.

### Boundary handling

Neither Finite Differences nor Nelder-Mead clip proposals to the parameter bounds:
every suggested point (a gradient step, a probe, a simplex reflection/expansion/
contraction) is evaluated exactly where proposed, even when that lies outside the
nominal range. This is a deliberate decision, not an oversight. Clipping to the
nearest boundary would cause persistent re-evaluation of the same boundary point
whenever a step or reflection keeps landing out of bounds (a "boundary limit
cycle"), plus a mismatch between the point the optimizer *thinks* it evaluated
and the point actually scored. Comparisons are meant to expose this kind of
boundary behavior, not mask it: the walk may wander outside the bounds, the
objective returns a poor score there, and the optimizer retreats naturally. In
principle the walk can move far outside the bounds; on real CFD runs the only
hard limit is the point at which Fluent no longer accepts the coefficient
values (has not happened in practice yet). PSO is the only exception, and not
because of clipping: its absorption rule (particle pinned to the bound, that
velocity component zeroed) is part of the PSO algorithm itself.

---

## Nelder–Mead

A simplex method; does not use gradients.  Initialises from the GEKO defaults and
explores by reflecting, expanding, or contracting the simplex.  Well-suited to
moderate dimensions with smooth objectives but may stagnate in flat regions.

---

## Bayesian Optimisation

Uses a Gaussian Process surrogate to model the objective and Expected Improvement
to select the next evaluation point.  Starts with Sobol quasi-random sampling
(9 points in 1-D, 15 in 2-D) to build the initial surrogate, then switches to
model-guided search.
Naturally handles boundaries and multi-modal landscapes.

---

## Particle Swarm

A population-based method.  A swarm of particles explores the domain simultaneously,
guided by personal bests and the global best.

Configuration used:
- 1-D: 4 particles, 5 swarm iterations (24 total function evaluations).
- 2-D: 4 particles, 11 swarm iterations (48 total function evaluations).

---

## Hybrid optimizers

All four hybrids split the budget into two phases, shown as circles (Phase 1) and
triangles (Phase 2) in the plots.

| Optimizer | Phase 1 | Phase 2 | Rationale |
|-----------|---------|---------|-----------|
| NM → BO | Nelder–Mead warm-up | Bayesian Opt. | NM rapidly finds a good region; BO refines |
| FD → BO | Finite Differences warm-up | Bayesian Opt. | FD provides gradient-guided starting data for BO |
| BO → NM | Bayesian Opt. exploration | Nelder–Mead refinement | BO maps the landscape; NM converges from best found |
| BO → FD | Bayesian Opt. exploration | Finite Differences refinement | BO maps the landscape; FD follows gradient from best found |

In `BO → FD`, the FD phase starts from the BO best point without re-evaluating it,
so it immediately begins gradient probing rather than wasting an evaluation on the
already-known base.

### Exploitation tightening in BO → NM and BO → FD

When BO runs first, its best point is a well-informed starting location.  The
subsequent NM or FD phase can therefore afford to search a smaller neighbourhood —
prioritising exploitation over exploration.

**BO → NM:** The NM startup simplex is built around the BO best point with offsets
scaled to **60% of the default size** (offset ±0.06 instead of the uniform ±0.10
used in every dimension by the default startup simplex).  This tighter simplex makes NM refine locally rather than
re-exploring the full domain.  Controlled via the `simplex_scale` parameter in
`nm_options` (default 1.0; set to 0.6 here).

**BO → FD:** The FD `step_size` (finite-difference perturbation) and `learning_rate`
(gradient-descent step) are both reduced to **60% of the standalone FD values**:
`step_size = 0.03` (vs 0.05) and `learning_rate = 0.009` (vs 0.015).  The smaller
perturbation gives a more local gradient estimate; the smaller step avoids overshooting
near the minimum found by BO.

The NM→BO and FD→BO hybrids use the same FD/NM parameters as the standalone
versions, since in those cases the heuristic phase runs first with no prior knowledge
of the optimum location.

---

## Known limitations to flag

1. **FD convergence depends on learning rate calibration.**  The x₁ direction of the
   2-D test function has Lipschitz constant L ≈ 52; the x₂ direction is piecewise-
   linear (non-smooth at x₂ = 0.5).  The standalone FD learning rate 0.015 is
   calibrated to the x₁ direction; the BO→FD rate of 0.009 trades some speed for
   tighter local convergence.

2. **PSO uses the same budget as all other optimizers** (24 evals in 1-D, 48 in 2-D)
   with 4 particles in both cases.

3. **Local vs global minimum (1-D):** gradient-based methods (FD, NM, and their
   hybrids) are attracted to the local minimum at x ≈ 0.81.  Only methods that
   explore broadly (BO, PSO, or hybrids where BO runs first) reliably find the
   global minimum at x ≈ 3.19.

4. **No boundary clipping:** FD and NM evaluate proposals exactly where suggested,
   even outside the nominal bounds (see "Boundary handling" above). The gold star
   and best-value annotation reflect the actual point found, which may itself lie
   outside the nominal bounds if that is where the best score occurred.

---

## Real CFD comparison (`optimizer_comparison.py`)

The synthetic benchmark results above come from analytic test functions.  A
separate script, `optimizer_comparison.py`, plots the same convergence curves
using actual CFD runs of the Periodic Hills Re=2800 case.

### Where the data lives

All real-run results are stored under:

```
results/experiments/optimizer_comparison/
    one-param-runs/    <- 1-D cases (geko_csep only)
        BO_1D_ph2800/
        NM_1D_ph2800/
        FD_1D_ph2800/
        PSO_1D_ph2800_p3/   <- PSO particle-count sweep (3/5/7/9 particles)
        PSO_1D_ph2800_p5/
        PSO_1D_ph2800_p7/
        PSO_1D_ph2800_p9/
        NM_BO_1D_ph2800/
        FD_BO_1D_ph2800/
        BO_NM_1D_ph2800/
        BO_FD_1D_ph2800/
    two-param-runs/    <- 2-D cases (geko_csep + geko_cnw)
        BO_2D_ph2800/ ... (same naming pattern)
        PSO_2D_ph2800_p10/   <- PSO particle-count sweep (10/15/20 particles)
        PSO_2D_ph2800_p15/
        PSO_2D_ph2800_p20/
```

Each experiment folder contains `metadata.csv` (iteration history and scores),
`optimizer.pkl` (serialised optimizer state), and the Fluent case/data files for
the best result found.

### How to run

```bash
# 1-D comparison (default)
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py

# 2-D comparison
.venv/Scripts/python.exe optimizer_visualization/optimizer_comparison.py --dim 2d
```

Each run produces six output PNGs in `optimizer_visualization/plots/comparison/<dim>/`.
All cutoff iterations are named hyperparameters defined at the top of
`optimizer_comparison.py` (`_CUT_ITER_1D`, `_CUT_ITER_2D`, `_CUT_ITER_1D_STAGE2`,
`_CUT_ITER_2D_STAGE2`, `_CUT_ITER_1D_PSO`, `_CUT_ITER_2D_PSO`).

**Main comparison** — BO, NM, FD, and all hybrids (`RUNS_1D_BO` / `RUNS_2D_BO`; PSO
excluded, see below):

| File | Contents |
|------|----------|
| `optimizer_comparison_{1d,2d}_full.png` | All iterations, linear y-scale |
| `optimizer_comparison_1d_after_iter7.png` / `optimizer_comparison_2d_after_iter13.png` | Iterations after the early-phase cutoff (iter > 7 for 1-D, iter > 13 for 2-D) |
| `optimizer_comparison_1d_after_iter14.png` / `optimizer_comparison_2d_after_iter23.png` | Iterations after the "BO swaps to NM/FD" marker (iter > 14 for 1-D, iter > 23 for 2-D) |

**BO-vs-PSO comparison** — BO against the PSO particle-count sweep
(`RUNS_1D_BO_VS_PSO`: 3/5/7/9 particles; `RUNS_2D_BO_VS_PSO`: 10/15/20 particles):

| File | Contents |
|------|----------|
| `optimizer_comparison_{1d,2d}_vs_pso_full.png` | All iterations, linear y-scale |
| `optimizer_comparison_1d_vs_pso_after_iter7.png` / `optimizer_comparison_2d_vs_pso_after_iter13.png` | Iterations after BO's own sampling phase ends (`_BO_SAMPLING_STOP`); iter > 7 for 1-D, iter > 13 for 2-D |
| `optimizer_comparison_2d_vs_pso_after_iter51.png` | Deeper cutoff (`_CUT_ITER_2D_PSO`), iter > 51; BO's larger 2-D budget survives this cut, so BO stays in the plot |
| `optimizer_comparison_1d_pso_only_after_iter28.png` | Deeper cutoff (`_CUT_ITER_1D_PSO`), iter > 28; this lies past BO's real-CFD 1-D budget (21 evals), so BO has already ended and is dropped entirely — this file shows only the PSO particle-count runs, under a separate `pso_only` slug and title rather than `vs_pso` |

The "BO swaps to NM/FD" marker is omitted from the BO-vs-PSO plots since none of
those runs ever swap to NM/FD. Missing experiment folders are silently skipped so
partial result sets still produce a valid plot.

### Plot elements

Each curve is the **best cost found so far** (monotonically non-increasing).
Raw per-iteration costs are not shown.  Shorter runs are not prolonged — curves
end where the experiment ended.  The y-axis is automatically scaled to the range
of the plotted data (5 % padding on each side), so small differences between
converged curves are visible.

Each legend entry is suffixed with the argmin parameter value(s) for that run, e.g.
`Bayesian Opt (GP); Csep=0.886` (1-D) or `Hybrid BO -> FD; Csep=0.888, Cnw=0.513`
(2-D) — the `geko_csep`/`geko_cnw` values at the iteration with the lowest score.
This is the global best over the entire run (not just the visible window), so it
is the same across a comparison's `_full` and `_after_iter*` plots.
