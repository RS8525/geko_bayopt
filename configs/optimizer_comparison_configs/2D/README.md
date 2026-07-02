# 2D Optimizer Comparison — Design Rationale

These 8 configs run all optimizers on the two-parameter GEKO calibration problem
(`geko_csep` and `geko_cnw`, Re=2800 periodic hills).  Each optimizer is a separate
experiment so results can be compared at equal cost.

The benchmark runs (configs 01 and 02) were already executed with `n_calls=60` to
establish a reliable reference for what the global optimum looks like.  The
comparison configs below all use a constrained budget of approximately 35 evaluations
to test how well each optimizer performs under a realistic computation limit.

---

## Parameter space

Two free parameters:
- `geko_csep` in [0.7, 2.5]
- `geko_cnw` in [-2.0, 2.0]

---

## Evaluation budget

The target budget is **approximately 35 evaluations**, with `epsilon = null` so no
early stopping is applied.  Going from 1D to 2D the landscape is harder to explore,
so the budget is scaled up from 21 (1D) to ~35 to give all optimizers a reasonable
number of improvement steps.

---

## 2D-specific FD cycle structure

In 2D (D=2) the finite-difference cycle costs **3 evaluations per gradient step**:

```
eval 1       : base point
eval 2       : probe along x1  (gradient approximation)
eval 3       : probe along x2  (gradient approximation)
eval 4       : gradient step   <- improvement
eval 5       : probe along x1
eval 6       : probe along x2
eval 7       : gradient step   <- improvement
...
```

Only 1 in 3 evaluations after the base is an actual gradient step.  This is less
efficient than 1D FD (1 in 2) because the 2D gradient estimate requires two probes
instead of one.  Budget choices for FD phases are constrained to `1 + 3k` (whole
cycles from a base) or `3k` (whole cycles when BO injects the base, so no re-eval).

---

## Optimizer-by-optimizer budget breakdown

### 01 — Bayesian Optimisation (BO) — benchmark reference

`n_calls = 35`, `n_initial = 12` Sobol samples.

The first 12 evaluations probe the 2D domain quasi-randomly (Sobol sequence) to
build an initial surrogate model; the remaining 23 evaluations are GP-guided.
12 Sobol samples provide good 2D coverage (same count as in the 60-iteration
benchmark run).  BO is the global-search reference.

### 02 — Nelder–Mead (NM) — benchmark reference

`n_calls = 35`.

NM in 2D starts with 3 simplex vertices (n+1 = 3) and refines from there.  No
special allocation is needed.

### 03 — Finite Differences (FD)

`n_calls = 34`.

`34 = 1 + 11 × 3`: one initial base evaluation plus 11 complete probe-probe-step
cycles, yielding 11 gradient steps.  34 is used instead of 35 to avoid stopping
mid-cycle (35 would cut off after the first probe of cycle 12).

### 04 — Particle Swarm Optimisation (PSO)

`n_calls = 54`, `n_particles = 6`, `max_iter = 8`.

`54 = 6 × (1 + 8) = 6 × 9`: each of the 6 particles occupies one initial position
and then moves 8 times.  6 particles are used instead of 4 (as in 1D) to provide
better 2D coverage per swarm iteration.  As in 1D, PSO receives an extended budget
(54 vs ~35) because a swarm-based method needs enough iterations to converge; the
strict 35-eval cap would allow only ~4 swarm iterations, which is too few.

### 05 — NM → BO (heuristic warmup, then BO)

`n_calls = 35`, `n_initial = 10` (NM phase).

NM runs for 10 evaluations to explore the domain and seed the BO surrogate with
directionally informed points before GP-guided search takes over for the remaining
25 evaluations.

### 06 — FD → BO (gradient warmup, then BO)

`n_calls = 35`, `n_initial = 10` (FD phase).

`10 = 1 + 3 × 3`: one base evaluation plus 3 complete 2D FD cycles (3 gradient
steps).  This is the smallest clean multiple that completes whole cycles before BO
takes over for the remaining 25 evaluations.

### 07 — BO → NM (global search, then local refinement)

`n_calls = 35`, `n_initial = 23` (BO phase: 12 Sobol + 11 GP), NM phase = 12 evals.

BO explores the 2D domain globally for 23 evaluations (12 Sobol + 11 GP-guided).
NM then starts from the BO best with a reduced simplex (`simplex_scale = 0.6`) and
exploits the located basin for the remaining 12 evaluations (~34% of total budget).

### 08 — BO → FD (global search, then gradient descent)

`n_calls = 35`, `n_initial = 23` (BO phase: 12 Sobol + 11 GP), FD phase = 12 evals.

Same BO phase as BO→NM.  The BO best is injected directly as the FD base without
re-evaluation, so the 12 FD evaluations are pure gradient cycles:
`12 = 3 × 4` — 4 complete probe-probe-step cycles, yielding 4 gradient steps from
the BO best point.  FD step size and learning rate are tightened relative to the
standalone FD config (`step_size = 0.03`, `lr = 0.045` vs `0.05` / `0.075`) to
exploit the known good neighbourhood rather than explore blindly.

---

## Summary table

| # | Optimizer | n_calls | Phase 1 | Phase 2 |
|---|-----------|---------|---------|---------|
| 01 | BO | 35 | 12 Sobol + 23 GP | — |
| 02 | NM | 35 | 35 NM | — |
| 03 | FD | 34 | 34 FD (1 base + 11 cycles, 11 steps) | — |
| 04 | PSO | 54 | 54 PSO (6 particles, 8 iterations) | — |
| 05 | NM → BO | 35 | 10 NM | 25 BO |
| 06 | FD → BO | 35 | 10 FD (1 base + 3 cycles, 3 steps) | 25 BO |
| 07 | BO → NM | 35 | 23 BO (12 Sobol + 11 GP) | 12 NM |
| 08 | BO → FD | 35 | 23 BO (12 Sobol + 11 GP) | 12 FD (4 cycles, 4 steps) |
