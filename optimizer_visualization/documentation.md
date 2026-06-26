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
| Bayesian Opt. (GP) | 20 | 36 |
| Nelder–Mead | 20 | 36 |
| Finite Differences | 20 | 36 |
| Particle Swarm | 20 | 36 |
| Hybrid NM → BO | 20 (10 + 10) | 36 (16 + 20) |
| Hybrid FD → BO | 20 (9 + 11) | 36 (16 + 20) |
| Hybrid BO → NM | 20 (10 + 10) | 36 (20 + 16) |
| Hybrid BO → FD | 20 (10 + 10) | 36 (20 + 16) |

All optimizers use the same budget: 20 evaluations in 1-D, 36 in 2-D.
PSO is configured with 4 particles and 8 swarm iterations
(4 × 9 = 36), which divides evenly so no partial swarm iterations occur.

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
4. Set the best-seen point as the new base and repeat.

**The "N = … function evals" label counts all function evaluations** including the
probe evaluations that are internal to the FD algorithm.  Only the base evaluation
and the gradient-step evaluations are shown as dots in the plot.  As a result, the
number of visible dots is smaller than the N shown — for example, N = 20 function
evaluations in 1-D produce 10 visible dots (1 base + 9 gradient steps), because
each gradient-descent step requires 1 additional probe evaluation to estimate the
derivative in that dimension.  In 2-D, N = 36 function evaluations produce 12 visible
dots (1 base + 11 gradient steps), because each step requires 2 probe evaluations
(one per parameter dimension).

This design is intentional and consistent: the N label reflects the true computational
cost, while the plot shows only the evaluations that represent actual optimizer decisions.

### Learning rate calibration

The gradient-descent step size (`learning_rate`) must be calibrated to the function's
steepness (Lipschitz constant of the gradient):

- **1-D:** learning rate = 0.015, derived from the Lipschitz constant L ≈ 67.
  FD converges visibly toward the local minimum at x ≈ 0.81.
- **2-D:** learning rate = 0.015 (same as 1-D).  The `x1` direction has the same
  Lipschitz constant L ≈ 67, so the rate is equally appropriate.  The `x2` direction
  is piecewise-linear (slope ±3, Hessian = 0 away from the ridge), so the gradient
  step in x₂ is bounded by lr × 3 = 0.045, well within the x₂ range of 0.8.

### Boundary handling

When a gradient step or Nelder–Mead reflection proposes a parameter value outside
the valid bounds, the point is **clipped to the nearest boundary** before evaluation.
This is a practical workaround; physically, it means the optimizer effectively
evaluates the boundary point rather than the intended point.  Better alternatives
(reflection into the domain, backtracking line search) are under consideration.

---

## Nelder–Mead

A simplex method; does not use gradients.  Initialises from the GEKO defaults and
explores by reflecting, expanding, or contracting the simplex.  Well-suited to
moderate dimensions with smooth objectives but may stagnate in flat regions.

---

## Bayesian Optimisation

Uses a Gaussian Process surrogate to model the objective and Expected Improvement
to select the next evaluation point.  Starts with Sobol quasi-random sampling
(8 points) to build the initial surrogate, then switches to model-guided search.
Naturally handles boundaries and multi-modal landscapes.

---

## Particle Swarm

A population-based method.  A swarm of particles explores the domain simultaneously,
guided by personal bests and the global best.

Configuration used:
- 1-D: 4 particles, 4 swarm iterations (20 total function evaluations).
- 2-D: 4 particles, 8 swarm iterations (36 total function evaluations).

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
scaled to **60% of the default size** (x₁ offset ±0.15 instead of ±0.25; x₂ offset
+0.06 instead of +0.10).  This tighter simplex makes NM refine locally rather than
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
   2-D test function has Lipschitz constant L ≈ 67; the x₂ direction is piecewise-
   linear (non-smooth at x₂ = 0.5).  The standalone FD learning rate 0.015 is
   calibrated to the x₁ direction; the BO→FD rate of 0.009 trades some speed for
   tighter local convergence.

2. **PSO uses the same budget as all other optimizers** (20 evals in 1-D, 36 in 2-D)
   with 4 particles in both cases.

3. **Local vs global minimum (1-D):** gradient-based methods (FD, NM, and their
   hybrids) are attracted to the local minimum at x ≈ 0.81.  Only methods that
   explore broadly (BO, PSO, or hybrids where BO runs first) reliably find the
   global minimum at x ≈ 3.19.

4. **Boundary clipping:** out-of-bounds suggestions are silently pinned to the
   boundary.  The gold star and annotation reflect the best clipped value, not
   the intended step.
