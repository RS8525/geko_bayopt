# Optimizer Configuration Reference

All optimizers are configured inside the `"optimizer"` block of a JSON config.

Every optimizer block has the same top-level shape:

```json
"optimizer": {
    "kind": "<optimizer_kind>",
    "stopping_criteria": {
        "n_calls": 32,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        ...
    }
}
```

`stopping_criteria` and `kind_specific_options` can be omitted entirely to use defaults.

### Stopping criteria

| Field | Default | Description |
|---|---|---|
| `n_calls` | `32` | Hard cap on total CFD evaluations. The experiment always stops here. |
| `epsilon` | `1e-4` | Relative-improvement threshold for early stopping (see below). Set to `null` to disable. |
| `window` | `3` | Number of "meaningful steps" to compare for the epsilon check (see below). |

**How epsilon stopping works**

After each trial, the optimizer checks whether improvement has stalled. The check compares the best value seen in the last `window` *meaningful steps* against the best value seen before those steps. If the relative improvement is below `epsilon`, the experiment stops early.

What counts as a *meaningful step* differs by optimizer type — raw function evaluations are not always meaningful, particularly for gradient and simplex methods:

- **BO (`skopt_gp`)**: every evaluation counts. The window slides over the last `window` objective values.
- **Nelder-Mead**: only one entry per NM iteration (i.e. per simplex operation that produces a new reflection candidate). The D+1 startup evaluations and individual probe calls within a shrink are not counted.
- **Finite differences**: only one entry per complete gradient cycle (the value of the accepted gradient step; acceptance is unconditional, so this series is **not** monotonic). The D probe evaluations used to estimate the gradient are invisible to the convergence check.
- **PSO**: only one entry per completed swarm iteration (the global-best value after all `n_particles` results in that iteration are processed). Individual particle evaluations are invisible to the convergence check.
- **Hybrid optimizers (warm-up → BO)**: during the warm-up phase (NM or FD) the sub-optimizer's logic above applies; during the BO phase all evaluations count.
- **Hybrid optimizers (BO → refinement)**: during the BO phase all evaluations count; during the refinement phase (NM or FD) the sub-optimizer's logic above applies.

With the default `window=3`, at least `2 × window = 6` meaningful steps must have occurred before the check can trigger, preventing spurious early stops at the very start.

### Configuration validation

`build_optimizer` validates every configuration at build time (each optimizer's `test_config()` is called before the optimizer is returned). Invalid options — e.g. a PSO `n_calls` that is not divisible by `n_particles`, a non-positive `learning_rate`, `epsilon <= 0`, or `window < 1` — raise a `ValueError` immediately instead of failing mid-run. The valid ranges are listed with each option below.

---

## Optimizer kinds

### `skopt_gp` — Bayesian Optimization with Gaussian Process surrogate

```json
"optimizer": {
    "kind": "skopt_gp",
    "stopping_criteria": {
        "n_calls": 20,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "n_initial": 8,
        "random_state": 42
    }
}
```

`kind_specific_options`:

| Option | Default | Description |
|---|---|---|
| `n_initial` | `8` | Number of initial Sobol samples before the GP surrogate takes over. Must be ≥ 1. Rule of thumb: 8 × D (dimension). |
| `random_state` | `42` | Seed for the Sobol sampler **and** for the GP hyperparameter-fit restarts, making runs fully reproducible. |

---

### `nelder_mead` — Nelder-Mead Simplex

Self-initializes a startup simplex (D+1 points) around the GEKO defaults `(csep=1.75, cnw=0.5, cmix=0.0, cwall=0.9)`. No `n_initial` needed.

Parameter bounds are **not enforced** on simplex operations: reflections/expansions may propose points outside the bounds, which are evaluated as-is (a poor score makes the simplex retreat naturally). Clipping proposals caused persistent re-evaluation of identical boundary points.

```json
"optimizer": {
    "kind": "nelder_mead",
    "stopping_criteria": {
        "n_calls": 20,
        "epsilon": 1e-5,
        "window": 3
    },
    "kind_specific_options": {
        "alpha": 0.8,
        "gamma": 1.5,
        "rho": 0.5,
        "sigma": 0.5
    }
}
```

`kind_specific_options`:

| Option | Default | Description |
|---|---|---|
| `alpha` | `0.8` | Reflection coefficient. Must be > 0. |
| `gamma` | `1.5` | Expansion coefficient. Must be > 1. |
| `rho` | `0.5` | Contraction coefficient. Must be in (0, 1). |
| `sigma` | `0.5` | Shrink coefficient. Must be in (0, 1). |

---

### `finite_differences` — Finite-Difference Gradient Descent

Starts from GEKO defaults, estimates the gradient by perturbing each dimension by `step_size`, then takes a descent step. No `n_initial` needed.

Parameter bounds are **not enforced** on probes or gradient steps (same policy as Nelder-Mead): the walk may leave the bounds and is steered back by poor scores. The bounds are only used to scale the probe perturbation (`delta = step_size × range`).

Each cycle costs D+1 evaluations: D probes to estimate the gradient, plus 1 gradient step. For a D-dimensional problem this means the epsilon convergence check advances by one step every D+1 evaluations.

```json
"optimizer": {
    "kind": "finite_differences",
    "stopping_criteria": {
        "n_calls": 20,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "step_size": 0.05,
        "learning_rate": 0.2,
        "min_step": 1e-5,
        "max_step": 0.25
    }
}
```

`kind_specific_options`:

| Option | Default | Description |
|---|---|---|
| `step_size` | `0.05` | FD perturbation as a fraction of each parameter's range. Must be > 0. |
| `learning_rate` | `0.2` | Gradient descent step size. Must be > 0. |
| `min_step` | `1e-5` | Minimum absolute perturbation. Must be > 0. |
| `max_step` | `0.25` | Maximum absolute perturbation. Must be ≥ `min_step`. |

---

### `hybrid_nm_bayes` — Nelder-Mead → Bayesian

Phase 1: first `n_initial` evaluations use Nelder-Mead (self-initialized from GEKO defaults).  
Phase 2: remaining `n_calls − n_initial` evaluations use a GP surrogate warm-started with the NM history.

The epsilon check uses NM simplex-iteration logic during Phase 1 and a sliding window over all evaluations during Phase 2.

```json
"optimizer": {
    "kind": "hybrid_nm_bayes",
    "stopping_criteria": {
        "n_calls": 22,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "n_initial": 10,
        "bo_options": {
            "bayesian_kind": "GP",
            "random_state": 42
        },
        "nm_options": {
            "alpha": 0.8,
            "gamma": 1.5,
            "rho": 0.5,
            "sigma": 0.5
        }
    }
}
```

Phase split example above: 10 NM evaluations, then 12 Bayesian evaluations (22 total).

`bo_options`:

| Option | Default | Description |
|---|---|---|
| `bayesian_kind` | `"GP"` | Phase-2 surrogate: `"GP"`, `"RF"`, `"ET"`, or `"GBRT"`. `"GP"` resolves to the same explicitly-built estimator used by `skopt_gp` (identical kernel/`n_restarts` settings). |
| `random_state` | `42` | Seed for the Bayesian phase, including the GP hyperparameter-fit restarts. |

`nm_options`: same options as the standalone `nelder_mead` kind.

---

### `hybrid_fd_bayes` — Finite-Difference → Bayesian

Same structure as `hybrid_nm_bayes` but uses finite-difference gradient descent in Phase 1.

The epsilon check uses FD gradient-step logic during Phase 1 and a sliding window over all evaluations during Phase 2.

```json
"optimizer": {
    "kind": "hybrid_fd_bayes",
    "stopping_criteria": {
        "n_calls": 60,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "n_initial": 20,
        "bo_options": {
            "bayesian_kind": "GP",
            "random_state": 42
        },
        "fd_options": {
            "step_size": 0.05,
            "learning_rate": 0.2,
            "min_step": 1e-5,
            "max_step": 0.25
        }
    }
}
```

`bo_options`: same as for `hybrid_nm_bayes`.  
`fd_options`: same options as the standalone `finite_differences` kind.

---

### `hybrid_bayes_nm` — Bayesian → Nelder-Mead

Phase 1: first `n_initial` evaluations use BO (Sobol sampling, then GP surrogate).  
Phase 2: remaining evaluations use Nelder-Mead, with its startup simplex re-centred on the best point found by BO (instead of the GEKO defaults).

The epsilon check uses a sliding window over all evaluations during Phase 1 and NM simplex-iteration logic during Phase 2.

```json
"optimizer": {
    "kind": "hybrid_bayes_nm",
    "stopping_criteria": {
        "n_calls": 30,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "n_initial": 10,
        "bo_options": {
            "n_initial_sobol": 5,
            "bayesian_kind": "GP",
            "random_state": 42
        },
        "nm_options": {
            "alpha": 0.8,
            "gamma": 1.5,
            "rho": 0.5,
            "sigma": 0.5,
            "simplex_scale": 0.6
        }
    }
}
```

Phase split example above: 10 BO evaluations (5 Sobol + 5 GP), then Nelder-Mead starting from the BO best.

`bo_options`:

| Option | Default | Description |
|---|---|---|
| `n_initial_sobol` | `min(5, n_initial)` | Number of Sobol samples before the GP surrogate takes over within the BO phase. |
| `bayesian_kind` | `"GP"` | Surrogate type: `"GP"`, `"RF"`, `"ET"`, or `"GBRT"`. `"GP"` resolves to the same explicitly-built estimator used by `skopt_gp`. |
| `random_state` | `42` | Seed for the Sobol sampler and the GP (including its hyperparameter-fit restarts). |

`nm_options`: same options as the standalone `nelder_mead` kind, plus:

| Option | Default | Description |
|---|---|---|
| `simplex_scale` | `1.0` | Scale factor applied to the NM startup simplex built around the BO best point. Values below 1 tighten local exploitation (e.g. `0.6` shrinks offsets to 60%). Has no effect on the standalone `nelder_mead` kind. |

---

### `hybrid_bayes_fd` — Bayesian → Finite Differences

Phase 1: first `n_initial` evaluations use BO (Sobol sampling, then GP surrogate).  
Phase 2: remaining evaluations use finite-difference gradient descent, starting from the best point found by BO (injected directly as the FD base — not re-evaluated; it is also recorded in the FD sub-optimizer's own history as bookkeeping, since FD's decision logic reads only its base and step history).

The epsilon check uses a sliding window over all evaluations during Phase 1 and FD gradient-step logic during Phase 2.

```json
"optimizer": {
    "kind": "hybrid_bayes_fd",
    "stopping_criteria": {
        "n_calls": 30,
        "epsilon": 1e-4,
        "window": 3
    },
    "kind_specific_options": {
        "n_initial": 10,
        "bo_options": {
            "n_initial_sobol": 5,
            "bayesian_kind": "GP",
            "random_state": 42
        },
        "fd_options": {
            "step_size": 0.05,
            "learning_rate": 0.2,
            "min_step": 1e-5,
            "max_step": 0.25
        }
    }
}
```

`bo_options`: same as for `hybrid_bayes_nm`.  
`fd_options`: same options as the standalone `finite_differences` kind.

**Exploitation tip:** Because BO has already found a good starting point, the FD phase
can use a smaller `step_size` and `learning_rate` to search more locally.  A ratio of
60% of the standalone values is a reasonable starting point (e.g. `step_size=0.03`,
`learning_rate` at 60% of the standalone value).  This tightens the gradient estimate
near the BO best and reduces the risk of overshooting the minimum.

---

### `pso` — Particle Swarm Optimization

A swarm of `n_particles` particles explores the parameter space simultaneously. Each particle tracks its own personal best and is attracted toward the global best. Particle evaluations are serialised (one `ask()`/`tell()` at a time); a full swarm iteration completes after `n_particles` evaluations.

Initial positions are drawn from a Sobol' sequence spanning the parameter bounds, using the same mechanism as the BO optimizers: `random_state` is passed through `sklearn.utils.check_random_state`, and a single `randint` draw from the resulting `RandomState` seeds `skopt.sampler.Sobol.generate` (mirroring `skopt.Optimizer`'s own internal Sobol initialisation exactly). This gives more even coverage of the space than uniform random draws, especially for small `n_particles`.

Initial velocities are zero. The first swarm move is driven entirely by the cognitive/social attraction terms once personal and global bests are known from the Sobol-sampled evaluations — there is no random "kick" at t=0.

If a particle overshoots a bound, it is placed on the boundary and its velocity in that dimension is zeroed (absorption). This boundary handling is part of the PSO algorithm itself (unlike NM/FD, which deliberately wander).

`v_max_frac` remains relevant despite the zero-velocity init: it is applied every iteration (not just at init) to clip the velocity computed from the inertia/cognitive/social terms in `_compute_next_positions`, preventing particles from picking up unbounded velocity as they accelerate toward pbest/gbest. Removing the random initial velocity draw does not remove this per-iteration clamp.

The number of swarm iterations is **derived from the evaluation budget** — there is no `max_iter` option. Internally `max_iter = n_calls // n_particles − 1` (the init sweep costs `n_particles` evaluations, each swarm iteration costs `n_particles` more), and the inertia weight decays linearly from `w_start` to `w_end` over exactly those iterations. Validation requires `n_calls` to be divisible by `n_particles` (so iterations use the full budget) and `n_calls ≥ 2 × n_particles` (init sweep plus at least one iteration). A `max_iter` key in `kind_specific_options` is ignored.

```json
"optimizer": {
    "kind": "pso",
    "stopping_criteria": {
        "n_calls": 200,
        "epsilon": 1e-4,
        "window": 5
    },
    "kind_specific_options": {
        "n_particles": 10,
        "w_start": 0.9,
        "w_end": 0.4,
        "c1": 1.5,
        "c2": 1.5,
        "v_max_frac": 0.2,
        "random_state": 42
    }
}
```

Config above: 10 particles, 200 evaluations → 1 init sweep + 19 swarm iterations. Inertia decays from 0.9 → 0.4 over those 19 iterations.

`kind_specific_options`:

| Option | Default | Description |
|---|---|---|
| `n_particles` | `10` | Number of particles in the swarm. Must be ≥ 2, and `n_calls` must be a multiple of it. |
| `w_start` | `0.9` | Initial inertia weight. Requires `0 < w_end ≤ w_start ≤ 1`. |
| `w_end` | `0.4` | Final inertia weight (reached at `max_iter`). |
| `c1` | `1.5` | Cognitive coefficient (pull toward personal best). Must be > 0. |
| `c2` | `1.5` | Social coefficient (pull toward global best). Must be > 0. |
| `v_max_frac` | `0.2` | Maximum velocity per dimension as a fraction of that dimension's range. Applied every iteration to clip velocity growth from the cognitive/social terms — still needed even though initial velocities are zero. |
| `random_state` | `42` | Seed used both for the Sobol' initial-position sequence (same scheme as the BO optimizers) and for the `r1`/`r2` random draws in the per-iteration velocity update. |

---
