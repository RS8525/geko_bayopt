# Optimizer Implementation Documentation

This document describes how each optimizer in `optimizer.py` works internally and,
more importantly, the non-obvious edge cases and design decisions baked into the
code. Read this together with `optimizers_readme.md` (the configuration reference)
before modifying any optimizer.

---

## Architecture shared by all optimizers

### The ask/tell contract

Every optimizer exposes:

```python
x: list[float]  = opt.ask()          # suggest the next parameter vector
                  opt.tell(x, y)     # report the objective value for x
done: bool      = opt.should_stop()  # epsilon convergence check
                  opt.test_config()  # raise ValueError on invalid options
```

`build_optimizer(section, parameters)` constructs the right class from the JSON
config and calls `test_config()` before returning, so invalid configurations fail
at build time, not mid-run.

The experiment loop strictly alternates `ask()` → evaluate → `tell()`. Several
internal mechanisms (hybrid phase routing, FD probe cycling) rely on this
alternation; none of the optimizers support batched or parallel asks.

### Resume by replay — the central invariant

A crashed or interrupted experiment is resumed by `_replay_into_optimizer` in
`experiment.py`: a **fresh** optimizer object receives every completed trial via
`tell(x, y)` — `ask()` is never called during replay. This leads to the central
design invariant:

> **Each optimizer's state machine must advance entirely inside `tell()`.**
> Anything that only happens inside `ask()` will not happen during a replay.

Status per optimizer (verified by test: a run resumed at any point continues
bit-for-bit identically to an uninterrupted run):

| Optimizer            | Resume behavior |
|----------------------|-----------------|
| `skopt_gp`           | Exact. skopt refits eagerly inside `tell()`. |
| `nelder_mead`        | Exact. Simplex is initialized and advanced inside `tell()`. |
| `finite_differences` | Exact. Probe/step cycle is driven by `tell()`. |
| `pso`                | Exact. Swarm updates fire inside `tell()`; the RNG stream is consumed at the same points as in the original run. |
| `hybrid_nm_bayes`    | Exact (both phases). |
| `hybrid_fd_bayes`    | Exact (both phases). |
| `hybrid_bayes_nm`    | Exact (both phases). |
| `hybrid_bayes_fd`    | Exact (both phases). |

There is deliberately no snapshot (`get_state`/`set_state`) API. It existed
once but was never called anywhere and the hybrid implementations were
incomplete, so it was removed — resume is always reconstruction-by-replay. If
snapshot checkpointing is ever needed (e.g. replay becomes too expensive),
implement it completely for every class, including the sub-optimizer state
inside the hybrids.

### Determinism and seeding

All optimizers are bitwise reproducible given the same config. Two details make
this work for the GP-based ones:

1. `_resolve_bo_base_estimator` builds the GP estimator **explicitly** instead of
   letting skopt cook it from the string `"GP"`. skopt's cooking draws from the
   Optimizer's shared RNG to seed the estimator, which would shift the Sobol
   initial-point sequence between configs with identical `random_state`.
2. The estimator itself is seeded with the config's `random_state`. With
   sklearn's default `random_state=None`, the L-BFGS restarts of every kernel
   hyperparameter fit draw from the **global numpy RNG**, which made every
   model-based proposal irreproducible across runs and resumes (this was a real
   bug: benchmark reruns produced different trajectories).

All five GP construction sites (standalone `skopt_gp` and the four hybrids) go
through `_resolve_bo_base_estimator`, so every optimizer's BO component uses the
identical estimator (Matern ν=2.5 kernel, `n_restarts_optimizer=10`).

### The epsilon stopping helper `_should_stop`

Compares `min(history[-window:])` against `min(history[:-window])` relative to
the prior best; requires at least `2 × window` entries. Two subtleties:

- The comparison uses `abs(recent_best − prior_best)`, so a *worsening* recent
  window (possible for FD, whose step series is not monotonic) produces a large
  difference and does **not** stop. Only stagnation stops.
- Each optimizer feeds a different "meaningful step" series into the check (NM
  iterations, FD gradient steps, PSO swarm iterations, raw evals for BO) — see
  `optimizers_readme.md` for the exact granularity.

---

## Nelder–Mead (`NelderMeadOptimizer`)

An explicit ask/tell state machine: startup phase emits the initial simplex
points one at a time; afterwards each ask/tell cycle performs one NM operation
(reflect / expand / contract / shrink step).

### Startup simplex

Built around the hard-coded GEKO defaults
`(csep=1.75, cnw=0.5, cmix=0.0, cwall=0.9)`, clipped into the bounds (the
search *starts* in the valid region; only later proposals may wander), with a
uniform offset of 0.1 in every dimension:

- 1D: `default ± 0.1` (2 points).
- ND: `default[0] ± 0.1` plus one point per remaining dimension at
  `default[d] + 0.1` (n_dim+1 points).

**Edge case:** parameters must use the canonical GEKO names — any other name
raises `KeyError` at construction (intentional; the defaults are the physical
starting point).

### Simplex initialization from history (best + D nearest)

When enough evaluations exist, `_init_simplex()` builds the active simplex from
the **evaluation history**: the best (lowest-y) point plus its `n_dim` nearest
neighbours by Euclidean distance in parameter space.

- In a fresh run the history contains exactly the n_dim+1 startup points at
  that moment, so all of them are selected — identical to the classic startup
  simplex.
- The best+nearest rule matters when the simplex must be rebuilt from a longer
  or mixed history (resume scenarios, hybrid handovers): it recovers a compact
  simplex around the best point seen so far.
- **Edge case:** duplicated points in the history (e.g. repeated boundary
  evaluations) can select coinciding vertices → degenerate simplex. Seed NM
  from distinct points.

Initialization happens inside `tell()`, which is what makes resume-by-replay
exact: replayed post-startup results drive `_process_result` through the same
deterministic operation sequence as the original run.

### Bounds are deliberately NOT enforced

NM's geometric steps (reflection/expansion/contraction/shrink) can propose
points outside the parameter bounds. Clipping them internally caused persistent
resampling: a clipped vertex kept regenerating the same out-of-bounds proposal,
re-running an identical expensive CFD case forever. Instead, out-of-bounds
proposals are evaluated as-is; the simulator returns a poor score and the
simplex retreats naturally. This is now the project-wide policy: **nothing
clips anywhere** — neither the optimizers (NM, FD) nor the benchmark harness.
Optimizer comparisons are meant to expose boundary behavior, not mask it.


### Other edge cases

- **1D degeneracy:** with only 2 vertices, "second worst" equals "best", so the
  accept-reflection branch (`f_best ≤ f_r < f_second_worst`) is unreachable;
  every reflection either triggers an expansion attempt or a contraction. This
  is inherent to NM in one dimension and harmless.
- **Non-standard defaults:** α=0.8 (textbook 1.0) and γ=1.5 (textbook 2.0) —
  deliberately less aggressive because the CFD cost surface near the optimum is
  approximately a smooth bowl and exploitation matters more than exploration.
- **Stopping granularity:** `_iter_best_y` records the best simplex value once
  per NM iteration (in `_prepare_reflect`); startup evals and the individual
  shrink evaluations do not advance the epsilon check.

---

## Finite Differences (`FiniteDifferenceOptimizer`)

Cycle structure (period D+1 after the initial base evaluation):
evaluate base → D forward-difference probes (one per dimension) → 1 gradient
descent step → the step point becomes the next base.

### Probes and steps look identical to callers

`ask()` returns probe points and step points indistinguishably. For plotting,
probes are noise; `_fd_show_mask` in `_benchmark_core.py` reconstructs the
probe/step pattern analytically from the cycle arithmetic. If you change the
cycle structure, update `_fd_show_mask` and its offset handling for
`hybrid_bayes_fd` (whose FD phase starts with a probe, not a base).

### Unconditional step acceptance (not elitist)

`_process_result` sets the base to whatever the gradient step evaluated to —
**not** the best point seen so far. This is intentional: probe and step points
are pure functions of the base, so a "keep best" rule would freeze the base
whenever a cycle failed to improve, and every later cycle would re-evaluate the
exact same points forever (a real historical bug). Consequences:

- `step_history_y` is not monotonic; never assume the last step improved.
- Near a smooth optimum the walk settles into a small limit cycle biased by
  roughly `delta/2` from the true optimum (forward-FD bias). Set `epsilon` on
  real CFD runs so the run terminates instead of oscillating through the budget.

### Boundary handling: bounds are NOT enforced

Like NM, FD does not enforce the parameter bounds: probes are always
`base + delta` (forward difference, `delta` clamped only in magnitude to
`[min_step, max_step]`), and the gradient step is `base − lr·grad` with no
clipping. The walk may leave the domain; the objective is evaluated there and
a poor score steers it back. The bounds are still used to *scale* the probe
perturbation (`delta = step_size × range`).

This replaced an earlier self-clipping scheme (probe direction flips near the
upper bound plus `np.clip` on the step), which produced **boundary limit
cycles**: a step pinned to the bound became the next base and every subsequent
cycle re-evaluated (nearly) the same boundary points, wasting the eval budget.
Letting FD wander makes bad boundary behavior *visible* in optimizer
comparisons instead of silently masking it.

### Learning rate

The correct step size is `lr ≈ 1/L` (L = Lipschitz constant of the gradient).
Benchmark: 0.015. Real CFD: 0.075 standalone, 0.045 in the BO→FD exploitation
phase. A too-large lr overshoots and, combined with unconditional acceptance,
produces a zig-zag walk.

---

## Particle Swarm (`ParticleSwarmOptimizer`)

Sequential ask/tell PSO: each swarm iteration is `n_particles` ask/tell cycles;
when the last result of an iteration arrives, `tell()` updates personal/global
bests and pre-computes all next positions.

- **Initial positions** come from a Sobol' sequence seeded exactly the way
  `skopt.Optimizer` seeds its own Sobol generator (`check_random_state` →
  single `randint` draw), so PSO and BO runs with the same `random_state` are
  comparable. Initial **velocities are zero**; the first move is driven by the
  cognitive/social terms.
- **Bounds:** absorption — an overshooting particle is placed on the bound and
  its velocity component zeroed. (This is part of the PSO algorithm itself,
  not a harness workaround, so it stays despite the no-clipping policy.)
- **Iteration count is derived, not configured:**
  `max_iter = n_calls // n_particles − 1` (init sweep + `max_iter` iterations
  exactly consume the `n_calls` budget). `test_config()` raises unless
  `n_calls` is divisible by `n_particles` and the budget fits at least the
  init sweep plus one swarm iteration (`n_calls ≥ 2 × n_particles`). A
  `max_iter` key in `kind_specific_options` is **ignored**.
- **Inertia:** decays linearly from `w_start` to `w_end`, exactly spanning the
  derived `max_iter`; `_current_w` clamps at `max_iter`, so any extra
  iterations run at `w_end`.
- **Stopping granularity:** one `gbest` entry per completed swarm iteration.
- **Resume:** exact. All swarm updates happen inside `tell()`, and the RNG is
  consumed only in `_compute_next_positions`, which replay triggers at the same
  points as the original run.

---

## Hybrid optimizers

All four hybrids run one sub-optimizer for the first `n_initial` evaluations and
another for the rest. Two shared implementation rules:

### Phase routing is by evaluation index, not by a phase flag

`tell()` decides which sub-optimizer receives a result by comparing
`len(history)` against `n_initial` — **not** by the `_phase` attribute (which is
only updated inside `ask()` and is stale during a resume replay, where `ask()`
is never called). Routing by index is what makes hybrid resume work; do not
"simplify" it back to `if self._phase == ...`.

### Warm-up hybrids (`hybrid_nm_bayes`, `hybrid_fd_bayes`)

During the heuristic phase every result is also told to the BO optimizer
(constructed with `n_initial_points=0`), which warm-starts the GP.

**Edge case — the out-of-bounds guard (both warm-up hybrids):** NM and FD can
propose points outside the bounds (see above), but skopt raises `ValueError`
when told a point outside its space. Out-of-bounds warm-up evaluations are
therefore silently excluded from the BO warm-start history. This is a
deliberate trade-off: the alternative (enlarging the BO space) was rejected.

**Edge case — starved BO warm-start (accepted, not handled):** because of the
guard above, the BO optimizer (built with `n_initial_points=0`) only receives
the *in-bounds* warm-up evaluations. If a pathological warm-up run spends most
of its budget outside the bounds, the GP enters its phase fitted on very few
points; in the extreme case of *zero* in-bounds warm-up points, the first
BO-phase `ask()` raises a `RuntimeError` from skopt (no data, no initial
points). This cannot happen under normal conditions — warm-ups start in-bounds
at the GEKO defaults — so it is deliberately left unhandled.

### BO-first hybrids (`hybrid_bayes_nm`, `hybrid_bayes_fd`)

The BO phase uses Sobol initial sampling (`n_initial_sobol`) followed by GP
proposals. At the transition the refinement optimizer is seeded from the BO
best point:

- **BO→NM:** NM's startup simplex is re-centred on the BO best
  (`_seed_nm_from_bo_best`), using the same uniform 0.1 offsets as the default
  startup simplex, scaled by `simplex_scale` (0.6 in the real configs for
  tighter exploitation). The seed's argmin is taken over the **first
  `n_initial` history entries only**, so a resume replay — where the history
  already contains NM-phase points — picks the same centre as the live run did.
  Seeding is idempotent and triggered from both `ask()` (live transition) and
  `tell()` (replay).
- **BO→FD:** the BO best is injected directly as the FD base
  (`_seed_fd_from_bo_best`) without re-evaluating it, so the FD phase starts
  with a probe rather than a base evaluation (`_fd_show_mask` accounts for this
  offset). The injected point is also appended to FD's own history — pure
  bookkeeping; FD's decision logic reads only `_base`/`_base_y` and
  `_step_history_y`. Like BO→NM, the seed argmin is restricted to the first
  `n_initial` history entries and seeding is triggered from both `ask()` and
  `tell()` (idempotent), which makes resume exact.

**Maintenance note:** both seeding helpers reach into their sub-optimizer's
private attributes (`_initial_points`; `_base`, `_base_y`, `_start_probing`).
Any refactor of NM/FD internals must update them. The planned fix is a public
seeding method on each sub-optimizer (e.g. `seed_initial_simplex(center, scale)`
and `seed_base(x, y)`).

### Stopping in hybrids

`should_stop()` also routes by evaluation index: the active phase's rule
applies (sliding window over all evals for BO, sub-optimizer granularity for
NM/FD). Note that the BO-phase check runs over the full mixed history, which is
fine because `_should_stop` compares minima.

---

## Known open issues

- **Seeding via private attributes:** `_seed_nm_from_bo_best` /
  `_seed_fd_from_bo_best` write sub-optimizer privates; the planned fix is a
  public seeding method on each sub-optimizer (e.g.
  `seed_initial_simplex(center, scale)` and `seed_base(x, y)`).
- **FD probe visibility:** `FiniteDifferenceOptimizer` should expose whether the
  pending `ask()` is a probe or a step (e.g. a public `pending_op`), so callers
  don't reconstruct the cycle arithmetic (`_fd_show_mask`).
