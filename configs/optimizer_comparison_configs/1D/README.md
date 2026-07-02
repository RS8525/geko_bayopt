# 1D Optimizer Comparison — Design Rationale

These 8 configs run all optimizers on the single-parameter GEKO calibration problem
(`geko_csep`, Re=2800 periodic hills).  Each optimizer is a separate experiment so
results can be compared at equal cost.

---

## Parameter space

One free parameter: `geko_csep` in [0.7, 2.5].

---

## Evaluation budget

All configs use **`n_calls = 21`** as the hard evaluation limit, with `epsilon = null`
so no early stopping is applied.  The budget is chosen based on prior literature which
reports that roughly 20 Bayesian Optimisation iterations are sufficient to locate the
optimal GEKO parameters.  Pure BO (config 01) therefore serves as the global-search
benchmark: if any optimizer matches or beats its result within the same budget, that is
evidence the cheaper method is competitive.

---

## Optimizer-by-optimizer budget breakdown

### 01 — Bayesian Optimisation (BO)

`n_calls = 21`, `n_initial = 7` Sobol samples.

The first 7 evaluations probe the domain quasi-randomly (Sobol sequence) to build
an initial surrogate model.  The remaining 14 evaluations are GP-guided.  BO is the
global-search reference: it is known to work and has no exploitative bias toward any
starting region.

### 02 — Nelder–Mead (NM)

`n_calls = 21`.

In 1D, NM starts with 2 simplex vertices and refines through reflections, expansions,
and contractions.  No special allocation is needed; the budget applies uniformly.

### 03 — Finite Differences (FD)

`n_calls = 21`.

In 1D the FD cycle is:

```
eval 1 (odd) : base point   <- improvement
eval 2 (even): probe        <- gradient approximation only
eval 3 (odd) : gradient step <- improvement
eval 4 (even): probe
eval 5 (odd) : step
...
```

Odd-numbered evaluations (in 1-based counting) are the base point or gradient steps
that directly improve the solution; even-numbered evaluations are forward-difference
probes used only to estimate the gradient.  A budget of 21 yields 11 productive
evaluations and 10 probes.

### 04 — Particle Swarm Optimisation (PSO)

`n_calls = 42`, `n_particles = 6`, `max_iter = 6`.

PSO runs `n_particles × (1 + max_iter)` evaluations in total (6 × 7 = 42): each
particle occupies one initial position and then moves 6 times.  Compared with the
other optimizers, a budget of 21 gives only 4–5 swarm iterations, which is unlikely to
be competitive for a swarm-based method.  To give PSO a fair chance, 21 additional
evaluations are granted (42 total = 21 base + 21 extra).

**Parameter changes from initial run (4 particles, 8 iterations, 36 evals):**

The original run (Re=2800) showed a long plateau: `gbest` stayed at 2.4931 for
24 consecutive evaluations (iterations 1–6) before improving only marginally to
2.4824, while BO found the true optimum (2.4577 at Csep ≈ 0.886) in 23 evals.
Root-cause analysis identified four contributing factors, each addressed by one
parameter change:

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `n_particles` | 4 | 6 | 4 particles gave too little init diversity — the first random draw set `gbest` at Csep ≈ 0.96 (close to but not at the true minimum), and the social term locked the entire swarm onto that false attractor for 24 evals. 6 particles improve initial coverage of [0.7, 2.5] and reduce the probability of this happening. |
| `max_iter` | 8 | 6 | Adjusted to keep the total budget at 6 × 7 = 42 (roughly equal to the old 36 + margin for the larger swarm). |
| `w_end` | 0.4 | 0.2 | The final inertia of 0.4 left too much residual momentum in the later swarm iterations, causing particles to overshoot the narrow valley around Csep ≈ 0.886 (score range < 0.04 over the interval [0.75, 1.10]). Lowering to 0.2 makes the swarm brake more aggressively and settle finer in the exploitation phase. |
| `v_max_frac` | 0.2 | 0.1 | `v_max = 0.2 × 1.8 = 0.36` allowed jumps of ±20 % of the full range per step — too coarse to resolve the flat minimum. Halving to 0.1 (`v_max = 0.18`) reduces overshoot and gives finer resolution near the optimum without severely restricting exploration given the larger swarm. |

`w_start`, `c1`, `c2`, and `random_state` are unchanged.

### 05 — NM → BO (heuristic warmup, then BO)

`n_calls = 21`, `n_initial = 7` (NM phase).

NM runs for 7 evaluations to seed the BO surrogate with points that already cluster
near a promising region.  BO then uses the remaining 14 evaluations with a warm prior.

### 06 — FD → BO (gradient warmup, then BO)

`n_calls = 21`, `n_initial = 7` (FD phase).

FD runs a 7-evaluation warmup (1 base + 3 probe/step pairs), providing 3 gradient
steps that push toward a local minimum before BO takes over.  The 7-eval FD warmup
is efficient because it spans 3 complete FD cycles and delivers a directionally
informed starting region to BO.

### 07 — BO → NM (global search, then local refinement)

`n_calls = 21`, `n_initial = 14` (BO phase: 7 Sobol + 7 GP), NM phase = 7 evals.

BO first explores the domain globally for 14 evaluations (7 Sobol + 7 GP-guided).
NM then starts from the BO best point with a reduced simplex (`simplex_scale = 0.6`)
and exploits the located basin for the remaining 7 evaluations (~33% of total budget).

### 08 — BO → FD (global search, then gradient descent)

`n_calls = 21`, `n_initial = 14` (BO phase: 7 Sobol + 7 GP), FD phase = 7 evals.

Same split as BO→NM.  After BO locates the best region, the BO best is injected
directly as the FD base without re-evaluation, so the FD phase opens immediately with
a gradient probe.  The FD step size and learning rate are tightened relative to the
standalone FD config (`step_size = 0.03`, `lr = 0.045` vs. `0.05` / `0.075`) to
reflect exploitation of a known good neighbourhood rather than blind exploration.
7 FD evaluations yield 3 gradient steps from the BO best point.

---

## Summary table

| # | Optimizer | n_calls | Phase 1 | Phase 2 |
|---|-----------|---------|---------|---------|
| 01 | BO | 21 | 7 Sobol + 14 GP | — |
| 02 | NM | 21 | 21 NM | — |
| 03 | FD | 21 | 21 FD (11 steps, 10 probes) | — |
| 04 | PSO | 42 | 42 PSO (6 particles, 6 iterations) | — |
| 05 | NM → BO | 21 | 7 NM | 14 BO |
| 06 | FD → BO | 21 | 7 FD | 14 BO |
| 07 | BO → NM | 21 | 14 BO (7 Sobol + 7 GP) | 7 NM |
| 08 | BO → FD | 21 | 14 BO (7 Sobol + 7 GP) | 7 FD |
