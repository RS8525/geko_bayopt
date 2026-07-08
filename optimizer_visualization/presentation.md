# Optimizer Presentation Guide

This document is the presenter's guide for the optimizer part of the talk. It
contains two parts:

* **Part I: Particle Swarm Optimization (PSO)** covers the routine and every
  hyperparameter of [`PSO.py`](PSO.py) and its figure sequence in
  `plots/PSO/`.
* **Part II: Bayesian Optimization (BO)** specifies, in detail, what has to be
  included when explaining BO, mapped to the figure sequence produced by
  [`visualization_BayOpt.py`](visualization_BayOpt.py) in `plots/BayOpt/`.

---
---

# Part I: Particle Swarm Optimization

This part explains the Particle Swarm Optimizer implemented in
[`PSO.py`](PSO.py) and used to generate the figure sequence in `plots/PSO/`.
The script mirrors the update rule of the project optimizer
(`src/geko_bayesopt/optimizer.py`, `ParticleSwarmOptimizer`) on the 2-D
Goldstein-Price test function, so the same mechanics apply to the CFD objective.

The goal here is to explain *how the optimizer works*, not to squeeze out the
best possible objective value.

## 1. The idea in one paragraph

PSO optimizes a function by moving a population of candidate solutions
("particles") through the search space. Each particle remembers the best point
*it* has ever seen (its personal best), and the whole swarm shares the best
point *anyone* has ever seen (the global best). On every iteration each
particle is pulled a little toward its own personal best and a little toward
the global best, with some inertia carrying it in its current direction. Over
time the swarm collapses onto promising regions while still exploring early on.
No gradients are required, which is why PSO suits noisy, black-box objectives
such as a CFD simulation.

## 2. The optimization problem

The demo minimizes the **Goldstein-Price function**

```
f(x, y) = [1 + (x + y + 1)^2 (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
        * [30 + (2x - 3y)^2 (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]
```

on the domain `x in [-2, 2]`, `y in [-2, 2]`. Its global minimum is `f = 3` at
`(0, -1)`. It is a classic multimodal test: the surface is highly nonlinear with
several local minima and steep plateaus, and its values span many orders of
magnitude (from 3 up to roughly `10^6` across the domain), which is why the
figures use log-spaced contour levels.

In the real project the two coordinates are replaced by the GEKO tuning
parameters and `f` is replaced by the CFD error metric, but the algorithm is
identical.

The script can also visualize the swarm on two other classic 2-D benchmarks by
setting `FUNCTION` near the top of [`PSO.py`](PSO.py) to `"rosenbrock"` (minimum
0 at `(1, 1)`) or `"beale"` (minimum 0 at `(3, 0.5)`). All runs write to
`plots/PSO/`, so switching `FUNCTION` overwrites the previous run's frames.

## 3. The routine, step by step

### Step 0: Initialization

1. Draw `N_PARTICLES` low-discrepancy **Sobol** points in the unit square and
   scale them into the domain. Sobol sampling spreads the initial swarm more
   evenly than uniform random sampling, so the search starts with good
   coverage.
2. Set every particle's **velocity to zero**.
3. Evaluate `f` at each particle. Store each particle's position and value as
   its **personal best** (`pbest`).
4. The particle with the lowest value defines the initial **global best**
   (`gbest`).

### Step 1..N: Swarm iterations

Each iteration updates every particle in three moves.

**a) Velocity update**

```
v_i  <-  w * v_i
       +  c1 * r1 * (pbest_i - x_i)      # cognitive pull, toward own best
       +  c2 * r2 * (gbest   - x_i)      # social pull,   toward swarm best
```

* `w * v_i` is **inertia**: it keeps the particle moving in its current
  direction (momentum, promotes exploration).
* The **cognitive** term pulls the particle back toward the best spot it
  personally found.
* The **social** term pulls it toward the best spot the whole swarm found.
* `r1`, `r2` are fresh uniform random numbers in `[0, 1)`, drawn *per particle
  and per dimension*. They are what makes the swarm stochastic: two particles
  the same distance from `gbest` still take different steps.

**b) Velocity clamping**

Each velocity component is clipped to `[-v_max, v_max]`, where
`v_max = V_MAX_FRAC * (domain width)` per dimension. This prevents "velocity
explosion", where the momentum term makes particles overshoot the domain on
every step.

**c) Position update and boundary handling**

```
x_i  <-  x_i + v_i
```

If the new position leaves the domain, the particle is **absorbed**: the
offending coordinate is pinned to the boundary and that velocity component is
reset to zero, so the particle does not keep pushing into the wall.

**d) Bookkeeping**

After evaluating `f` at the new positions:

* If a particle improved on its own history, update its `pbest`.
* If any particle beat the swarm record, update `gbest`.

**e) Inertia decay**

The inertia weight is decreased linearly from `W_START` to `W_END` across the
run:

```
w(t) = W_START - (W_START - W_END) * t / N_ITERATIONS
```

Early iterations use high inertia (wide, exploratory moves); later iterations
use low inertia (short, exploitative moves that settle into the minimum). This
is the standard **exploration-to-exploitation** schedule.

### Termination

The demo simply runs a fixed number of iterations. In the project optimizer the
run also stops early once the best-so-far value stops improving by more than a
tolerance over a window of iterations.

## 4. What the figures show

Each frame in `plots/PSO/` is one snapshot of the routine above:

| Element | Meaning |
|---|---|
| Blue contour landscape | The cost function `f` (log-scaled levels). Brighter = lower cost, the bright basin holds the global minimum. |
| Black dots | The particles at their current positions. |
| Black arrows | Each particle's **next** step: the arrow length and direction equal the actual move it is about to make. |
| Orange star | The **current global best**: the best location found so far. It is a fixed point the swarm is attracted to, not a moving particle, so it is drawn on the top layer to stay visible. |
| Gold star | The true global optimum at `(0, -1)`, shown for reference. |

The opening frames build up the initial swarm one layer at a time:
`iteration_000.png` shows just the particles (and the true optimum);
`iteration_001.png` adds the current global best (orange star); and
`iteration_002.png` adds the first velocity arrows. `iteration_003.png` onward
are the subsequent swarm iterations. The frames carry no title.

## 5. Hyperparameters

All hyperparameters are exposed near the top of [`PSO.py`](PSO.py).

### Swarm and run size

| Name | Demo value | Meaning | Effect of increasing |
|---|---|---|---|
| `N_PARTICLES` | 6 | Number of particles in the swarm. | Better coverage and robustness, but each iteration costs `N_PARTICLES` function evaluations. |
| `N_ITERATIONS` | 20 | Number of swarm iterations after initialization. | More refinement and better convergence, at linear cost. Also sets the length of the inertia decay schedule. |
| `PLOT_EVERY` | 1 | Save a figure every `PLOT_EVERY`-th frame (the first and last are always saved). | Purely cosmetic; controls how many figures are written. |

The total number of objective evaluations is
`N_PARTICLES * (N_ITERATIONS + 1)` (one initialization sweep plus one sweep per
iteration).

### Movement coefficients

| Name | Demo value | Meaning | Effect |
|---|---|---|---|
| `W_START` | 0.7 | Inertia weight at the start. | Higher = more momentum and exploration early on. |
| `W_END` | 0.4 | Inertia weight at the end. | Lower = tighter, more exploitative convergence at the end. |
| `C1` | 1.5 | Cognitive coefficient (pull toward personal best). | Higher makes particles trust their own findings more, keeping the swarm diverse. |
| `C2` | 1.5 | Social coefficient (pull toward global best). | Higher makes particles rush to the swarm best faster, risking premature convergence. |
| `V_MAX_FRAC` | 0.08 | Velocity clamp as a fraction of each dimension's range. | Higher allows larger jumps; lower keeps steps short and stable. |

Rules of thumb: `W_START > W_END` gives the exploration-to-exploitation schedule;
`C1` and `C2` are typically of order 1-2; and a common balanced default is
`C1 = C2`, as used here. The demo keeps the balanced `C1 = C2 = 1.5` of the
project defaults but uses a softer inertia and tighter velocity clamp
(`W_START = 0.7` vs `0.9`, `V_MAX_FRAC = 0.08` vs `0.2`) so the on-screen arrows
stay short and readable.

### Reproducibility

| Name | Value | Meaning |
|---|---|---|
| `SEED` | 42 | Seeds both the Sobol initialization and the random-number stream for `r1`, `r2`. Fixing it makes every run identical. |

### Presentation only

`PARTICLE_COLOR`, `GBEST_COLOR`, `ARROW_COLOR`, `FIGSIZE` and the domain bounds
(`X_BOUNDS`, `Y_BOUNDS`) control the figures, not the optimization.

## 6. Running PSO in parallel

PSO is **embarrassingly parallel within each iteration**. The reason is the
data dependency structure:

* Computing the *next* position of every particle needs only the state from the
  *end of the previous iteration* (each particle's position, velocity, personal
  best, and the shared global best).
* Given those, the `N_PARTICLES` objective evaluations for one iteration are
  **completely independent of each other**.

This is exactly the expensive part for a CFD objective, where a single
evaluation of `f` is a full simulation that can take minutes or hours.

### Synchronous (bulk) parallel PSO

The simplest and most common scheme:

1. At the start of iteration `t`, compute all `N_PARTICLES` proposed positions
   (cheap, serial).
2. **Evaluate all `N_PARTICLES` objectives at once**, one worker per particle
   (a process pool, `joblib`, MPI ranks, or a cluster job array submitting one
   CFD case per particle).
3. Wait for all of them (a barrier), then update the personal bests and the
   global best.
4. Proceed to iteration `t + 1`.

With `W` workers the wall-clock time per iteration drops from
`N_PARTICLES * (cost of one evaluation)` to roughly
`ceil(N_PARTICLES / W) * (cost of one evaluation)`. Choosing `N_PARTICLES`
equal to a multiple of the number of available workers keeps them fully
utilized. Results are identical to the serial run as long as the per-particle
random draws are assigned deterministically (for example, one seeded sub-stream
per particle) rather than pulled from a single shared stream in completion
order.

### Asynchronous parallel PSO

If evaluation times vary a lot (some CFD cases converge faster than others), the
synchronous barrier wastes workers that finish early. Asynchronous PSO removes
the barrier: whenever a worker finishes, its particle's bests are updated
against the *current* global best and the particle is immediately relaunched.
This keeps every worker busy at the cost of a slightly "staler" global best and
a run that is no longer bit-for-bit reproducible.

### Practical mapping to this project

For the GEKO/CFD objective the natural setup is synchronous PSO with a job
queue: each iteration submits `N_PARTICLES` Fluent runs to the scheduler,
collects the error metrics when they return, updates `gbest`, and submits the
next batch. Because PSO never needs gradients and the batch is independent, it
parallelizes far more readily than inherently sequential optimizers.

---
---

# Part II: Bayesian Optimization

This part specifies **what has to be included** in the Bayesian Optimization
explanation of the presentation. It is written against the figure sequence
produced by [`visualization_BayOpt.py`](visualization_BayOpt.py), which lives in
`plots/BayOpt/`. Show the figures in numbered order and cover the concepts
below as you go.

## II.0 Learning goals: what the audience must take away

By the end of the BO section, a viewer should be able to state:

1. **When** to use BO: the objective is expensive to evaluate, is a black box
   (no formula, no gradients), and may be noisy. Each evaluation "costs" a full
   CFD run, so we want to find the minimum in as few evaluations as possible.
2. **What** the two moving parts are: a **surrogate model** that cheaply
   approximates the expensive function, and an **acquisition function** that
   decides where to sample next.
3. **How** the loop works: fit the surrogate to the data, optimize the
   acquisition function to pick the next point, evaluate the true function
   there, add the result, repeat.
4. **Why** it balances exploration and exploitation, and therefore tends to
   find the global minimum rather than getting stuck in a local one.

## II.1 Concepts that must be explained

These are the non-negotiable ideas. Every one of them should be spoken aloud at
least once, anchored to a figure.

### a) The setting

State explicitly that the true cost function is **unknown**: we can only probe
it point by point, and each probe is expensive. This is the whole motivation
for BO and separates it from classical optimization that assumes cheap
evaluations or gradients.

### b) The surrogate model (Gaussian Process)

* The surrogate is a **cheap statistical model** fit to the points observed so
  far. Here it is a **Gaussian Process (GP)**.
* A GP predicts, at every input, both a **mean** (its best guess of the
  function value, the dashed line) **and an uncertainty** (the shaded band).
* Emphasize the uncertainty: the band is **wide where we have no data** and
  **pinches to zero at every observed point**, because the model knows the true
  value there. This is the key property that drives the search.

### c) The acquisition function and exploration vs exploitation

* The acquisition function turns the surrogate (mean + uncertainty) into a
  single score that says "how promising is it to sample here next".
* This demo uses the **Upper Confidence Bound (UCB)** with parameter
  `kappa = 2.0`. In minimization terms the next point is where the **lower
  confidence bound** `mean - kappa * std` is smallest: places that are either
  predicted to be low (**exploitation**) or highly uncertain (**exploration**).
* `kappa` is the explicit exploration/exploitation knob: larger `kappa` weights
  uncertainty more (more exploration).
* On the slides this optimum is the **gold star** (minimum of the surrogate).

### d) The BO loop

Show it as a four-step cycle and make clear it repeats:

```
1. Fit the surrogate (GP) to all data collected so far.
2. Optimize the acquisition function -> next point to sample (gold star).
3. Evaluate the true, expensive function there (red star).
4. Add the new (x, f(x)) to the data set.  Go to 1.
```

### e) Why global

BO revisits its uncertainty at every step, so it keeps probing unexplored
regions instead of only walking downhill from one starting point. That is why
it escapes local minima and homes in on the **global** minimum, which is the
concluding message.

## II.2 Slide-by-slide script

Show the figures in this order. Each row lists the file, what is on screen, and
the point to make.

| # | Figure | On screen | What to say / concept |
|---|---|---|---|
| 1 | `01_objective.png` | The cost function alone. | "This is the function whose minimum we want to find. In reality we do **not** see this curve; it is unknown and expensive to evaluate." |
| 2 | `02_initial_sampling.png` | Cost function + a few red observations. | "We begin by probing the function at a handful of points (here chosen up front)." Introduce the idea that data comes point by point. |
| 3 | `03_observations_only.png` | Only the observations, no curve. | "This is all the algorithm actually knows: a few input/output pairs. Everything between them is unknown." |
| 3.1 | `04_surrogate.png` | Observations + surrogate mean (dashed) + uncertainty band. | Introduce the **surrogate model / GP**: cheap guess (mean) plus **uncertainty** (band wide far from data, zero at data). This is concept II.1(b). |
| 3.2 | `05_surrogate_with_min.png` | Same, plus a gold star at the surrogate minimum. | Introduce the **acquisition function**: the gold star marks where the surrogate says it is most promising to look next (low predicted value and/or high uncertainty). Mention `kappa` and exploration vs exploitation, concept II.1(c). |
| 4 | `06_first_evaluation.png` | Cost function again; previous points as diamonds, the new evaluation as a red star at the gold-star location. | "We spend one expensive evaluation exactly where the surrogate pointed. This is one turn of the BO loop." Note the red star sits on the true curve at the star's x. |
| 4.1 | `07_surrogate_with_min.png` | Surrogate refit on all points (the new one is now a diamond, the band pinched there) + a new gold star at the updated minimum. | "We refit the surrogate with the new data and it proposes a **new** place to look. The loop repeats." This closes concept II.1(d): the cycle is now explicit. Point out that the new star may jump to an uncertain region (exploration). |
| 5 | `08_many_observations.png` | Cost function with many observations. | "Skipping several iterations of that loop, the sampling concentrates around the promising region." Point out the cluster forming near the true minimum. |
| 5.1 | `09_interpolation_vs_truth.png` | True function (faded) + dense observations + surrogate mean. | "The surrogate now closely interpolates the true function, and the samples are densest exactly at the global minimum." Concept II.1(d) has paid off. |
| 6 | (reuse `09`) | Same figure, or a final emphasis. | **Conclude**: because BO always accounts for uncertainty, it explores globally and converges on the **global** minimum, using very few evaluations. Concept II.1(e). |

The standalone legend for all of these markers is `legend.png` (cost function,
surrogate model, 95% confidence band, observations, minimum of surrogate,
next evaluation). Put it on the first surrogate slide or keep it visible
throughout.

## II.3 Points to get right (and pitfalls)

* **Do not flip signs on stage.** Internally the code maximizes the negative
  cost and negates it back for the plots; present everything purely as
  minimization. The audience should never hear about the sign trick.
* **The band pinching at data points** is the single most important visual.
  Call it out explicitly; it explains both interpolation and where the model is
  still unsure.
* **Gold star = decision, red star = result.** The gold star (step 3.2) is
  where BO *decides* to look; the red star (step 4) is the *true value found*
  there. Keeping this distinction clear is what makes the loop click.
* **Few evaluations is the whole point.** Repeat that each red marker is one
  expensive CFD run, so the value of BO is reaching the minimum in as few
  markers as possible.
* **UCB is one choice among several.** It is worth one sentence that other
  acquisition functions exist (Expected Improvement, Probability of
  Improvement); UCB with `kappa` was chosen here because its
  exploration/exploitation trade-off is the easiest to explain.

## II.4 Reproducing the figures

Run in the `BayOpt_clean` environment:

```
conda activate BayOpt_clean
python optimizer_visualization/visualization_BayOpt.py
```

Key settings are near the top of [`visualization_BayOpt.py`](visualization_BayOpt.py):
`KAPPA` (UCB exploration weight), `RANDOM_STATE` (reproducibility), and
`N_EXTRA` (how many BO iterations to skip forward for the "many points" plot).
The figures share a common y-range and have their axes stripped, matching the
PSO figures.
