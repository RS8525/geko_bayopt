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
- **Finite differences**: only one entry per complete gradient cycle (the best-so-far value after the gradient step is accepted). The D probe evaluations used to estimate the gradient are invisible to the convergence check.
- **Hybrid optimizers**: during the warm-up phase the sub-optimizer's check is used (NM or FD logic above); during the BO phase all evaluations count.

With the default `window=3`, at least `2 × window = 6` meaningful steps must have occurred before the check can trigger, preventing spurious early stops at the very start.

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
| `n_initial` | `8` | Number of initial Sobol samples before the GP surrogate takes over. Rule of thumb: 8 × D (dimension). |
| `random_state` | `42` | Seed for the Sobol sampler. |

---

### `nelder_mead` — Nelder-Mead Simplex

Self-initializes a startup simplex (D+1 points) around the GEKO defaults `(csep=1.75, cnw=0.5, cmix=0.0, cwall=0.9)`. No `n_initial` needed.

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
| `alpha` | `0.8` | Reflection coefficient. |
| `gamma` | `1.5` | Expansion coefficient. |
| `rho` | `0.5` | Contraction coefficient. |
| `sigma` | `0.5` | Shrink coefficient. |

---

### `finite_differences` — Finite-Difference Gradient Descent

Starts from GEKO defaults, estimates the gradient by perturbing each dimension by `step_size`, then takes a descent step. No `n_initial` needed.

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
| `step_size` | `0.05` | FD perturbation as a fraction of each parameter's range. |
| `learning_rate` | `0.2` | Gradient descent step size. |
| `min_step` | `1e-5` | Minimum absolute perturbation. |
| `max_step` | `0.25` | Maximum absolute perturbation. |

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
| `bayesian_kind` | `"GP"` | Phase-2 surrogate: `"GP"`, `"RF"`, `"ET"`, or `"GBRT"`. |
| `random_state` | `42` | Seed for the Bayesian phase. |

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
