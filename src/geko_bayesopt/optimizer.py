"""
Optimizer dispatcher.

Each optimizer kind builds an object implementing the ``Optimizer`` protocol
below.  The raw ``skopt.Optimizer`` only provides ask/tell, so it is wrapped
(``SkoptGPOptimizer``) to add the rest.

Add a new optimizer by:
    1. Implementing a class with ``ask``, ``tell``, ``should_stop`` and
       ``test_config``.  ``build_optimizer`` calls ``test_config()``
       unconditionally, and the experiment loop polls ``should_stop()``.
       The state machine MUST advance inside ``tell()``: resumed runs are
       reconstructed by replaying every completed trial through ``tell()``
       without any ``ask()`` calls (``experiment._replay_into_optimizer``).
    2. Adding a builder branch in ``build_optimizer``.
    3. Extending ``Literal`` in ``config.OptimizerSection``.

There is deliberately no snapshot save/restore API: resume always
reconstructs the optimizer by replaying completed trials through ``tell()``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .config import OptimizerSection, ParameterSpec
from .geko_defaults import defaults_for_parameters


class Optimizer(Protocol):
    """Structural type for what the experiment loop and builder need.

    Note: the raw ``skopt.Optimizer`` does NOT satisfy this protocol (it has
    no ``should_stop``/``test_config``) — that is what the ``SkoptGPOptimizer``
    wrapper is for.  Custom optimizers (e.g. a future BoTorch wrapper) need
    all four methods.
    """

    def ask(self) -> list[float]: ...
    def tell(self, x: list[float], y: float) -> Any: ...
    def should_stop(self) -> bool: ...
    def test_config(self) -> None: ...


# --------------------------------------------------------------------- #
# Stopping Criteria Helper                                              #
# --------------------------------------------------------------------- #

def _should_stop(
    history_y: list[float],
    epsilon: float | None,
    *,
    window: int = 3,
) -> bool:
    """Return True when the best-so-far value has not improved by at least
    ``epsilon`` (relative) over the last ``window`` evaluations.

    Uniform across all optimizer kinds: ``history_y`` is the raw
    per-evaluation history, and the check compares the overall best against
    the best before the last ``window`` evaluations.  Requires more than
    ``window`` entries so a pre-window best exists; the check can therefore
    fire at the earliest after ``window + 1`` evaluations.
    """
    if epsilon is None or len(history_y) <= window:
        return False
    prior_best   = min(history_y[:-window])
    current_best = min(history_y)
    denom = max(abs(prior_best), 1e-10)
    return (prior_best - current_best) / denom < epsilon


def _resolve_bo_base_estimator(
    bayesian_kind: str, random_state: int | None = None
) -> Any:
    """Build the skopt base_estimator for a given ``bayesian_kind``.

    For "GP" this constructs the same GaussianProcessRegressor used by the
    standalone ``skopt_gp`` optimizer. Passing an already-built object (rather
    than the string "GP") into ``skopt.Optimizer`` is required for two reasons:
    skopt's own ``cook_estimator`` for "GP" uses different kernel/noise/
    n_restarts_optimizer defaults than ours, and — more subtly — it also draws
    from the Optimizer's shared RNG stream to seed the estimator, which shifts
    the subsequent Sobol initial-point sequence even when random_state is
    identical across configs. Other kinds pass through as strings for skopt's
    own cook_estimator to build (no equivalent mismatch to avoid there).

    ``random_state`` seeds the estimator itself (the L-BFGS restarts of the
    kernel hyperparameter fit).  It must be set explicitly: skopt only seeds
    estimators it cooks, and with sklearn's default ``random_state=None``
    every fit draws from the global numpy RNG, making each model-based
    proposal irreproducible across runs and resumes.  Seeding the estimator
    does not touch the skopt Optimizer's own RNG, so the Sobol initial-point
    sequence is unaffected.
    """
    if bayesian_kind == "GP":
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern
        return GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=random_state,
        )
    return bayesian_kind


# --------------------------------------------------------------------- #
# Nelder-Mead Optimizer                                                 #
# --------------------------------------------------------------------- #

class NelderMeadOptimizer:
    """Nelder-Mead optimizer implemented as an explicit ask/tell state machine.

    Phase 1: evaluates the n_dim+1 startup simplex points seeded around the
             GEKO defaults.
    Phase 2: runs one NM operation per ask/tell cycle. ask() returns the next
             candidate point; tell() receives its value and advances the simplex.
    """


    def __init__(
        self,
        parameters: list[ParameterSpec],
        epsilon: float | None = None,
        window: int = 3,
        options: dict[str, float] | None = None,
    ):
        self.parameters = parameters
        self.epsilon = epsilon
        self.window = window
        self.options = options or {}

        # Defaults are the canonical textbook coefficients; the comparison
        # study overrides alpha/gamma in its configs to weight exploitation.
        self._ALPHA = float(self.options.get("alpha", 1.0))
        self._GAMMA = float(self.options.get("gamma", 2.0))
        self._RHO   = float(self.options.get("rho",   0.5))
        self._SIGMA = float(self.options.get("sigma",  0.5))

        self.bounds = np.array([[p.low, p.high] for p in parameters])
        self.n_dim = len(parameters)

        # Bounds are NOT enforced on NM geometric steps (reflection, expansion,
        # contraction, shrink).  Clipping proposed points to the parameter
        # bounds caused persistent resampling: when a computed point lands
        # outside the bounds it is pinned to the boundary, and subsequent NM
        # steps may produce the same out-of-bounds value that clips to the same
        # boundary point — re-running an identical expensive CFD simulation for
        # no gain.  Without clipping, NM can briefly explore outside the nominal
        # bounds; the simulator returns a poor score there and NM retreats
        # naturally.
        #
        # The initial simplex is still built inside the bounds so the optimizer
        # starts in a valid region (see _build_initial_simplex).

        self._initial_points = self._build_initial_simplex()

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

        # Active simplex — None until the startup phase is complete.
        self._simplex_x: list[np.ndarray] | None = None
        self._simplex_y: list[float] | None = None

        # State machine bookkeeping.
        self._pending_x: np.ndarray | None = None   # point most recently proposed
        self._pending_op: str | None = None           # 'reflect'|'expand'|'contract'|'shrink'
        self._x0: np.ndarray | None = None            # centroid of best n vertices
        self._x_r: np.ndarray | None = None           # reflection point
        self._f_r: float | None = None                # f(x_r)
        self._x_e: np.ndarray | None = None           # expansion point
        self._x_c: np.ndarray | None = None           # contraction point
        self._contract_type: str | None = None         # 'outside' | 'inside'
        self._shrink_idx: int = 0                      # which shrink vertex is pending

    def ask(self) -> list[float]:
        # Phase 1: emit startup simplex points one by one.
        if len(self._history_x) < len(self._initial_points):
            return self._initial_points[len(self._history_x)]

        # Phase 2: tell() initializes the simplex when startup completes;
        # this fallback is defensive and unreachable in normal operation.
        if self._simplex_x is None:
            self._init_simplex()
            self._prepare_reflect()

        return list(self._pending_x)

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        if self._simplex_x is None:
            # Initialize the simplex as soon as enough evaluations exist.
            # This happens in tell() (not ask()) so that a resumed run, which
            # replays its completed trials through tell() without ever calling
            # ask(), reconstructs the simplex and then advances the state
            # machine through the remaining replayed results — instead of
            # silently dropping them and restarting from the initial points.
            if len(self._history_x) >= len(self._initial_points):
                self._init_simplex()
                self._prepare_reflect()
            return

        self._process_result(np.array(x, dtype=float), float(y))

    # ------------------------------------------------------------------ #
    # Simplex initialization                                              #
    # ------------------------------------------------------------------ #

    def _init_simplex(self) -> None:
        """Build the active simplex from the evaluation history.

        Selects the best (lowest-y) point in the history plus its n_dim
        nearest neighbours in parameter space (Euclidean distance).  In a
        fresh run the history contains exactly the n_dim+1 startup
        evaluations when this is called, so all of them are selected and the
        result is the classic startup simplex.  The best+nearest rule matters
        when the history is longer than the startup set (e.g. a simplex
        rebuilt from a history that also contains non-startup evaluations):
        it then recovers a compact simplex around the best point seen so far.

        Caveat: if the history contains duplicated points (e.g. repeated
        boundary evaluations), the selected vertices can coincide and the
        simplex degenerates.  Callers seeding NM from arbitrary histories
        should ensure the points are distinct.
        """
        xs = np.array(self._history_x, dtype=float)
        ys = np.array(self._history_y, dtype=float)
        best = int(np.argmin(ys))
        dist = np.linalg.norm(xs - xs[best], axis=1)
        dist[best] = -1.0   # guarantee the best point itself is selected
        idx = np.argsort(dist, kind="stable")[: self.n_dim + 1]
        self._simplex_x = [xs[i].copy() for i in idx]
        self._simplex_y = [float(ys[i]) for i in idx]

    # ------------------------------------------------------------------ #
    # NM state machine                                                    #
    # ------------------------------------------------------------------ #

    def _sort_simplex(self) -> None:
        order = np.argsort(self._simplex_y)
        self._simplex_x = [self._simplex_x[i] for i in order]
        self._simplex_y = [self._simplex_y[i] for i in order]

    def _centroid(self) -> np.ndarray:
        """Centroid of all vertices except the worst."""
        return np.mean(self._simplex_x[:-1], axis=0)

    def _prepare_reflect(self) -> None:
        """Start a new NM iteration: sort simplex and propose the reflection point."""
        self._sort_simplex()
        self._x0 = self._centroid()
        self._x_r = self._x0 + self._ALPHA * (self._x0 - self._simplex_x[-1])
        self._pending_op = 'reflect'
        self._pending_x = self._x_r

    def _process_result(self, x: np.ndarray, y: float) -> None:
        op = self._pending_op
        if op == 'reflect':
            self._handle_reflect(y)
        elif op == 'expand':
            self._handle_expand(y)
        elif op == 'contract':
            self._handle_contract(y)
        elif op == 'shrink':
            self._handle_shrink(x, y)

    def _handle_reflect(self, f_r: float) -> None:
        self._f_r = f_r
        f_best         = self._simplex_y[0]
        f_second_worst = self._simplex_y[-2]
        f_worst        = self._simplex_y[-1]

        if f_r < f_best:
            # Reflection improved on the best — try expansion.
            self._x_e = self._x0 + self._GAMMA * (self._x_r - self._x0)
            self._pending_op = 'expand'
            self._pending_x  = self._x_e

        elif f_r < f_second_worst:
            # Accept reflection.
            self._simplex_x[-1] = self._x_r
            self._simplex_y[-1] = f_r
            self._prepare_reflect()

        else:
            # Contraction.
            if f_r < f_worst:
                # Outside contraction (reflection was at least better than worst).
                self._x_c = self._x0 + self._RHO * (self._x_r - self._x0)
                self._contract_type = 'outside'
            else:
                # Inside contraction (reflection was worse than worst).
                self._x_c = self._x0 + self._RHO * (self._simplex_x[-1] - self._x0)
                self._contract_type = 'inside'
            self._pending_op = 'contract'
            self._pending_x  = self._x_c

    def _handle_expand(self, f_e: float) -> None:
        if f_e < self._f_r:
            self._simplex_x[-1] = self._x_e
            self._simplex_y[-1] = f_e
        else:
            self._simplex_x[-1] = self._x_r
            self._simplex_y[-1] = self._f_r
        self._prepare_reflect()

    def _handle_contract(self, f_c: float) -> None:
        accept = (
            (self._contract_type == 'outside' and f_c <= self._f_r) or
            (self._contract_type == 'inside'  and f_c <  self._simplex_y[-1])
        )
        if accept:
            self._simplex_x[-1] = self._x_c
            self._simplex_y[-1] = f_c
            self._prepare_reflect()
        else:
            # Shrink all vertices toward the best.
            self._shrink_idx = 1
            self._pending_op = 'shrink'
            self._pending_x  = self._next_shrink_point()

    def _next_shrink_point(self) -> np.ndarray:
        return self._simplex_x[0] + self._SIGMA * (self._simplex_x[self._shrink_idx] - self._simplex_x[0])

    def _handle_shrink(self, x: np.ndarray, y: float) -> None:
        self._simplex_x[self._shrink_idx] = x
        self._simplex_y[self._shrink_idx] = y
        self._shrink_idx += 1

        if self._shrink_idx <= self.n_dim:
            self._pending_x = self._next_shrink_point()
        else:
            self._prepare_reflect()

    # ------------------------------------------------------------------ #
    # Startup simplex construction                                        #
    # ------------------------------------------------------------------ #

    def _build_initial_simplex(self) -> list[list[float]]:
        """Build the Nelder-Mead startup simplex around GEKO defaults."""

        defaults_map = defaults_for_parameters(self.parameters)
        defaults = [defaults_map[parameter.name] for parameter in self.parameters]

        # Uniform startup offset of 0.1 in every dimension.
        offset = 0.1

        points: list[list[float]] = []
        if self.n_dim == 1:
            points = [
                [self._clip_value(defaults[0] - offset, 0)],
                [self._clip_value(defaults[0] + offset, 0)],
            ]
        else:
            lower = defaults.copy()
            upper = defaults.copy()
            lower[0] = self._clip_value(defaults[0] - offset, 0)
            upper[0] = self._clip_value(defaults[0] + offset, 0)
            points.append(lower)
            points.append(upper)
            for dim in range(1, self.n_dim):
                point = defaults.copy()
                point[dim] = self._clip_value(defaults[dim] + offset, dim)
                points.append(point)

        return points

    def _clip_value(self, value: float, dim: int) -> float:
        return float(np.clip(value, self.bounds[dim, 0], self.bounds[dim, 1]))

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        if self.n_dim < 1:
            raise ValueError("nelder_mead requires at least one parameter.")
        if self._ALPHA <= 0:
            raise ValueError(f"options.alpha must be > 0, got {self._ALPHA!r}.")
        if self._GAMMA <= 1:
            raise ValueError(f"options.gamma must be > 1, got {self._GAMMA!r}.")
        if not 0 < self._RHO < 1:
            raise ValueError(f"options.rho must be in (0, 1), got {self._RHO!r}.")
        if not 0 < self._SIGMA < 1:
            raise ValueError(f"options.sigma must be in (0, 1), got {self._SIGMA!r}.")
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")


# --------------------------------------------------------------------- #
# Finite Difference Optimizer                                           #
# --------------------------------------------------------------------- #

class FiniteDifferenceOptimizer:
    """Gradient-based optimizer using finite differences.

    Each cycle: probe the objective at base + delta*e_i for each dimension i
    (forward finite difference), compute the gradient, take a descent step,
    and unconditionally accept the stepped point as the base for the next
    cycle.  There is no keep-best acceptance check — see the 'step' branch of
    _process_result for the rationale.

    Bounds are NOT enforced on probes or gradient steps (same policy as
    Nelder-Mead): the walk may leave the parameter bounds, the objective is
    evaluated there, and a poor score steers it back naturally.  Clipping to
    the bounds caused boundary limit cycles — a step pinned to the bound
    became the next base and every subsequent cycle re-evaluated (nearly) the
    same boundary points.  Optimizer comparisons are meant to expose this
    kind of boundary behavior, not mask it.
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        options: dict[str, Any],
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.parameters = parameters
        self.options = options or {}
        self.epsilon = epsilon
        self.window = window

        self.bounds = np.array([[p.low, p.high] for p in parameters])
        self.n_dim = len(parameters)

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

        self._step_size     = float(self.options.get("step_size",     0.05))
        self._learning_rate = float(self.options.get("learning_rate", 0.2))
        self._min_step      = float(self.options.get("min_step",      1e-5))
        self._max_step      = float(self.options.get("max_step",      0.25))

        # State machine
        self._pending_x:  np.ndarray | None = None
        self._pending_op: str | None = None   # 'probe' | 'step'
        self._base:    np.ndarray | None = None
        self._base_y:  float | None = None
        self._probe_dim:    int = 0
        self._probe_deltas: list[float] = []  # actual delta used per dim
        self._probe_y:      list[float] = []  # f(base + delta*e_i) per dim

    def ask(self) -> list[float]:
        if self._pending_x is None:
            defaults = defaults_for_parameters(self.parameters)
            self._pending_x = np.clip(
                [defaults[p.name] for p in self.parameters],
                self.bounds[:, 0], self.bounds[:, 1],
            )
        return list(self._pending_x)

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))
        self._process_result(np.array(x, dtype=float), float(y))

    # ------------------------------------------------------------------ #
    # State machine                                                       #
    # ------------------------------------------------------------------ #

    def _process_result(self, x: np.ndarray, y: float) -> None:
        if self._base is None:
            # Received the initial point — start first probing cycle.
            self._base   = x.copy()
            self._base_y = y
            self._start_probing()

        elif self._pending_op == 'probe':
            self._probe_y.append(y)
            self._probe_dim += 1
            if self._probe_dim < self.n_dim:
                self._pending_x = self._next_probe_point()
            else:
                # All dims probed — propose gradient step.
                self._pending_x  = self._gradient_step_point()
                self._pending_op = 'step'

        elif self._pending_op == 'step':
            # Gradient step evaluated — unconditionally accept it as the next
            # base (classic FD gradient descent, no "keep best" acceptance
            # check). Accepting the best-of-history point instead would make
            # the base "stick" whenever a cycle fails to improve, and since
            # ask() is a pure function of self._base, every later cycle would
            # then probe the exact same points forever.
            self._base   = x.copy()
            self._base_y = y
            self._start_probing()

    def _start_probing(self) -> None:
        self._probe_dim    = 0
        self._probe_deltas = []
        self._probe_y      = []
        self._pending_op   = 'probe'
        self._pending_x    = self._next_probe_point()

    def _next_probe_point(self) -> np.ndarray:
        dim  = self._probe_dim
        low, high = self.bounds[dim]
        raw  = (high - low) * self._step_size
        delta = float(np.clip(raw, self._min_step, self._max_step))

        # No bounds enforcement: the probe may leave the parameter bounds
        # (see the class docstring).  The bounds are only used to scale the
        # perturbation to the parameter's range.
        self._probe_deltas.append(delta)
        x = self._base.copy()
        x[dim] += delta
        return x

    def _gradient_step_point(self) -> np.ndarray:
        grad = np.array([
            (f_probe - self._base_y) / delta if abs(delta) > 1e-12 else 0.0
            for f_probe, delta in zip(self._probe_y, self._probe_deltas)
        ])

        # No bounds enforcement on the step (see the class docstring).
        return self._base - self._learning_rate * grad

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        if self.n_dim < 1:
            raise ValueError("finite_difference requires at least one parameter.")
        # Validate parameter names at build time: ask() resolves the starting
        # base from the canonical GEKO defaults, and hybrid_bayes_fd injects
        # its own base without ever taking that ask() branch.
        defaults_for_parameters(self.parameters)
        if self._step_size <= 0:
            raise ValueError(f"options.step_size must be > 0, got {self._step_size!r}.")
        if self._learning_rate <= 0:
            raise ValueError(f"options.learning_rate must be > 0, got {self._learning_rate!r}.")
        if self._min_step <= 0:
            raise ValueError(f"options.min_step must be > 0, got {self._min_step!r}.")
        if self._max_step < self._min_step:
            raise ValueError(
                f"options.max_step ({self._max_step}) must be >= min_step ({self._min_step})."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")



# --------------------------------------------------------------------- #
# Particle Swarm Optimizer                                              #
# --------------------------------------------------------------------- #

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization with a sequential ask/tell interface.

    The swarm is evaluated one particle at a time.  Each swarm iteration
    consists of ``n_particles`` ask/tell cycles; once all results are in,
    personal bests, the global best, and velocities are updated and the
    inertia weight is decayed.

    Initial positions are drawn from a Sobol' sequence (same
    ``skopt.sampler.Sobol`` mechanism, seeded from ``random_state`` the same
    way as the BO optimizers) rather than uniform random sampling, for better
    coverage of the parameter space.  Initial velocities are zero — the first
    swarm move is driven entirely by the cognitive/social attraction terms
    once personal/global bests are known from the Sobol evaluations.

    The number of swarm iterations is derived from the evaluation budget:
    ``max_iter = n_calls // n_particles - 1`` (the init sweep costs
    ``n_particles`` evaluations, each swarm iteration costs ``n_particles``
    more), so the linear inertia decay exactly spans the run.  ``test_config``
    requires ``n_calls`` to be divisible by ``n_particles``.

    Optional options (with defaults):
        n_particles : int   – swarm size (default 10)
        w_start     : float – initial inertia weight (default 0.7298)
        w_end       : float – final inertia weight   (default 0.7298)
        c1          : float – cognitive coefficient  (default 1.49618)
        c2          : float – social coefficient     (default 1.49618)
        v_max_frac  : float – v_max as fraction of each dimension's range
                              (default 1.0)
        random_state: int   – RNG seed (default 42)

    The defaults are the constricted-swarm Standard PSO values (Clerc &
    Kennedy 2002): constant inertia w = chi = 0.7298, c1 = c2 =
    chi * 2.05, and a full-range velocity cap as a safeguard only.  The
    comparison study overrides them in its configs with a more
    conservative set (decaying inertia, tight velocity cap).

    Boundary handling: absorption — particles that overshoot a bound are
    placed exactly on it and their velocity in that dimension is zeroed.
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        options: dict[str, Any],
        n_calls: int,
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.parameters = parameters
        self.epsilon = epsilon
        self.window = window
        opts = options or {}

        self.bounds  = np.array([[p.low, p.high] for p in parameters])
        self.n_dim   = len(parameters)

        self.n_particles = int(opts.get("n_particles", 10))
        # Defaults follow Standard PSO (Clerc & Kennedy constriction):
        # constant inertia chi = 0.7298, c1 = c2 = chi * 2.05 = 1.49618,
        # full-range velocity cap as a safeguard only.
        self.w_start     = float(opts.get("w_start",   0.7298))
        self.w_end       = float(opts.get("w_end",     0.7298))
        self.c1          = float(opts.get("c1",        1.49618))
        self.c2          = float(opts.get("c2",        1.49618))
        self.v_max_frac  = float(opts.get("v_max_frac", 1.0))
        # max_iter is derived from the evaluation budget so the inertia decay
        # exactly spans the run: the init sweep costs n_particles evaluations
        # and each swarm iteration costs n_particles more.  test_config checks
        # that the budget divides evenly into swarm iterations.
        self.n_calls  = int(n_calls)
        self.max_iter = self.n_calls // self.n_particles - 1
        self._random_state = opts.get("random_state", 42)

        self._rng    = np.random.default_rng(int(self._random_state))
        self._v_max  = self.v_max_frac * (self.bounds[:, 1] - self.bounds[:, 0])

        # Full history
        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

        # Swarm state — None until the init phase completes.
        self._positions:  np.ndarray | None = None   # (n_particles, n_dim)
        self._velocities: np.ndarray | None = None   # (n_particles, n_dim)
        self._pbest_x:    np.ndarray | None = None   # (n_particles, n_dim)
        self._pbest_y:    np.ndarray | None = None   # (n_particles,)
        self._gbest_x:    np.ndarray | None = None   # (n_dim,)
        self._gbest_y:    float = float("inf")

        # State machine
        self._phase        = "init"
        self._particle_idx = 0
        self._swarm_iter   = 0

        # Positions pre-computed for the current swarm iteration.
        self._pending_positions: list[np.ndarray] = []
        # Results collected within the current swarm iteration.
        self._iter_results_x: list[np.ndarray] = []
        self._iter_results_y: list[float] = []

        # Draw initial particle positions once so ask() is pure.
        self._init_positions: list[np.ndarray] = self._sample_initial_positions()

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def _sample_initial_positions(self) -> list[np.ndarray]:
        """Draw initial particle positions from a Sobol' sequence.

        Mirrors skopt.Optimizer's own Sobol initialisation exactly: the
        ``random_state`` option is passed through ``sklearn.utils.
        check_random_state`` and a single ``randint`` draw from the
        resulting RandomState seeds ``skopt.sampler.Sobol.generate``
        (see ``skopt.optimizer.optimizer.Optimizer.__init__``).
        """
        from sklearn.utils import check_random_state
        from skopt.sampler import Sobol
        from skopt.space import Real

        dimensions = [Real(p.low, p.high) for p in self.parameters]
        rng = check_random_state(self._random_state)
        seed = rng.randint(0, np.iinfo(np.int32).max)
        points = Sobol().generate(dimensions, self.n_particles, random_state=seed)
        return [np.array(p, dtype=float) for p in points]

    def _finalize_init(self) -> None:
        """Transition from init phase to iterate: set up swarm state."""
        self._positions  = np.array(self._iter_results_x, dtype=float)
        self._pbest_x    = self._positions.copy()
        self._pbest_y    = np.array(self._iter_results_y, dtype=float)
        self._velocities = np.zeros((self.n_particles, self.n_dim))
        best_idx       = int(np.argmin(self._pbest_y))
        self._gbest_x  = self._pbest_x[best_idx].copy()
        self._gbest_y  = float(self._pbest_y[best_idx])

        self._iter_results_x = []
        self._iter_results_y = []
        self._particle_idx   = 0
        self._compute_next_positions()
        self._phase = "iterate"

    # ------------------------------------------------------------------ #
    # Velocity / position update                                          #
    # ------------------------------------------------------------------ #

    def _current_w(self) -> float:
        t = min(self._swarm_iter, self.max_iter)
        return float(self.w_start - (self.w_start - self.w_end) * t / self.max_iter)

    def _compute_next_positions(self) -> None:
        """Pre-compute proposed positions for every particle in the next iteration."""
        w = self._current_w()
        self._pending_positions = []
        for i in range(self.n_particles):
            r1 = self._rng.uniform(0.0, 1.0, self.n_dim)
            r2 = self._rng.uniform(0.0, 1.0, self.n_dim)

            v_new = (w * self._velocities[i]
                     + self.c1 * r1 * (self._pbest_x[i] - self._positions[i])
                     + self.c2 * r2 * (self._gbest_x    - self._positions[i]))
            v_new = np.clip(v_new, -self._v_max, self._v_max)

            x_new = self._positions[i] + v_new

            # Absorption: overshoot → pin to bound, zero that velocity component.
            for d in range(self.n_dim):
                if x_new[d] < self.bounds[d, 0]:
                    x_new[d]  = self.bounds[d, 0]
                    v_new[d]  = 0.0
                elif x_new[d] > self.bounds[d, 1]:
                    x_new[d]  = self.bounds[d, 1]
                    v_new[d]  = 0.0

            self._velocities[i] = v_new
            self._pending_positions.append(x_new)

    def _finalize_iteration(self) -> None:
        """Update positions, pbest, gbest after all particles have been evaluated."""
        for i in range(self.n_particles):
            self._positions[i] = self._iter_results_x[i]
            y = self._iter_results_y[i]
            if y < self._pbest_y[i]:
                self._pbest_y[i] = y
                self._pbest_x[i] = self._iter_results_x[i].copy()
            if y < self._gbest_y:
                self._gbest_y = y
                self._gbest_x = self._iter_results_x[i].copy()

        self._swarm_iter   += 1
        self._iter_results_x = []
        self._iter_results_y = []
        self._particle_idx   = 0
        self._compute_next_positions()

    # ------------------------------------------------------------------ #
    # ask / tell                                                          #
    # ------------------------------------------------------------------ #

    def ask(self) -> list[float]:
        if self._phase == "init":
            return list(self._init_positions[self._particle_idx])
        return list(self._pending_positions[self._particle_idx])

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        self._iter_results_x.append(np.array(x, dtype=float))
        self._iter_results_y.append(float(y))
        self._particle_idx += 1

        if self._particle_idx >= self.n_particles:
            if self._phase == "init":
                self._finalize_init()
            else:
                self._finalize_iteration()

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def test_config(self) -> None:
        if self.n_dim < 1:
            raise ValueError("pso requires at least one parameter.")
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be >= 2, got {self.n_particles}.")
        if self.n_calls % self.n_particles != 0:
            raise ValueError(
                f"stopping_criteria.n_calls ({self.n_calls}) must be divisible "
                f"by n_particles ({self.n_particles}) so swarm iterations use "
                f"the full evaluation budget."
            )
        if self.max_iter < 1:
            raise ValueError(
                f"n_calls ({self.n_calls}) must be at least 2 x n_particles "
                f"({2 * self.n_particles}) to fit the init sweep plus one "
                f"swarm iteration."
            )
        if not (0 < self.w_end <= self.w_start <= 1):
            raise ValueError(
                f"Need 0 < w_end <= w_start <= 1, "
                f"got w_start={self.w_start}, w_end={self.w_end}."
            )
        if self.c1 <= 0 or self.c2 <= 0:
            raise ValueError(
                f"c1 and c2 must be > 0, got c1={self.c1}, c2={self.c2}."
            )
        if not (0 < self.v_max_frac <= 1):
            raise ValueError(
                f"v_max_frac must be in (0, 1], got {self.v_max_frac}."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")


# --------------------------------------------------------------------- #
# Hybrid Nelder-Mead → Bayesian Optimizer                              #
# --------------------------------------------------------------------- #

class HybridNelderMeadBayesOptimizer:
    """Two-phase optimizer: Nelder-Mead warmup, then Bayesian Optimisation.

    The first ``nelder_mead_iterations`` evaluations use ``NelderMeadOptimizer``.
    Subsequent calls switch to a skopt GP (warm-started with the NM history).
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        n_initial: int,
        bo_options: dict[str, Any],
        nm_options: dict[str, Any],
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.nelder_mead_iterations = n_initial
        self.epsilon = epsilon
        self.window = window
        self._parameters = parameters

        # Get BO config
        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        self.bo_options = bo_options or {}
        self.bo_dimensions = [Real(p.low, p.high, name=p.name) for p in self._parameters]
        # Resolve "GP" to the explicitly-built estimator so the BO phase uses
        # the same kernel/noise/n_restarts settings as every other GP-based
        # optimizer in this module (see _resolve_bo_base_estimator).
        self.bo_base_estimator = _resolve_bo_base_estimator(
            self.bo_options.get("bayesian_kind", "GP"),
            random_state=self.bo_options.get("random_state", 42),
        )
        self.bo_random_state = self.bo_options.get("random_state", 42)

        # Get BO optimizer
        self._bo_optimizer = SkoptOptimizer(
                        dimensions = self.bo_dimensions,
                        base_estimator = self.bo_base_estimator,
                        n_initial_points = 0,  # warm-start with NM history
                        random_state = self.bo_random_state,
        )
        # Get NM optimizer
        self._nm_options = nm_options or {}
        self._nm_optimizer = NelderMeadOptimizer(
                        parameters = self._parameters,
                        options = self._nm_options,
                        epsilon = epsilon,
                        window = window,
        )

        # Change after n_initial iterations
        self._phase = "nelder_mead"

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

    def ask(self) -> list[float]:
        if len(self._history_y) < self.nelder_mead_iterations:
            self._phase = "nelder_mead"
            return self._nm_optimizer.ask()

        else:
            self._phase = "bayesian"
            return self._bo_optimizer.ask()  # type: ignore[union-attr]

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        # Route by evaluation index, not by self._phase: a resumed run replays
        # its history through tell() without ever calling ask(), so _phase
        # would still hold its constructor value for every replayed point.
        if len(self._history_y) <= self.nelder_mead_iterations:
            self._nm_optimizer.tell(x, y)
            # NM deliberately proposes points outside the bounds (no clipping,
            # see NelderMeadOptimizer), but skopt raises when told a point
            # outside its space.  Out-of-bounds NM evaluations are therefore
            # excluded from the BO warm-start history.
            if all(p.low <= xi <= p.high for xi, p in zip(x, self._parameters)):
                self._bo_optimizer.tell(x, y)
        else:
            self._bo_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        """Raise ValueError if the configuration is invalid."""
        if self.nelder_mead_iterations < 1:
            raise ValueError(
                f"n_initial must be >= 1 for the NM phase, "
                f"got {self.nelder_mead_iterations}."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")
        # Validate NM sub-optimizer options by delegating.
        self._nm_optimizer.test_config()


# --------------------------------------------------------------------- #
# Hybrid Finite-Difference → Bayesian Optimizer                        #
# --------------------------------------------------------------------- #

class HybridFiniteDifferenceBayesOptimizer:
    """Two-phase optimizer: finite-difference gradient warmup, then BO.

    The first ``finite_difference_iterations`` evaluations use
    ``FiniteDifferenceOptimizer``.  Subsequent calls switch to a skopt GP
    (warm-started with the FD history).
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        n_initial: int,
        bo_options: dict[str, Any],
        fd_options: dict[str, Any],
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.finite_difference_iterations = n_initial
        self.epsilon = epsilon
        self.window = window
        self._parameters = parameters

        # Get BO config
        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        self.bo_options = bo_options or {}
        self.bo_dimensions = [Real(p.low, p.high, name=p.name) for p in self._parameters]
        # Resolve "GP" to the explicitly-built estimator so the BO phase uses
        # the same kernel/noise/n_restarts settings as every other GP-based
        # optimizer in this module (see _resolve_bo_base_estimator).
        self.bo_base_estimator = _resolve_bo_base_estimator(
            self.bo_options.get("bayesian_kind", "GP"),
            random_state=self.bo_options.get("random_state", 42),
        )
        self.bo_random_state = self.bo_options.get("random_state", 42)

        # Get BO optimizer
        self._bo_optimizer = SkoptOptimizer(
            dimensions=self.bo_dimensions,
            base_estimator=self.bo_base_estimator,
            n_initial_points=0,  # warm-start with FD history
            random_state=self.bo_random_state,
        )
        # Get FD optimizer
        self._fd_optimizer = FiniteDifferenceOptimizer(
            parameters=self._parameters,
            options=fd_options,
            epsilon=epsilon,
            window=window,
        )

        # Change after n_initial iterations
        self._phase = "finite_difference"

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

    def ask(self) -> list[float]:
        if len(self._history_y) < self.finite_difference_iterations:
            self._phase = "finite_difference"
            return self._fd_optimizer.ask()

        self._phase = "bayesian"
        return self._bo_optimizer.ask()

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        # Route by evaluation index, not by self._phase (see
        # HybridNelderMeadBayesOptimizer.tell for the replay rationale).
        if len(self._history_y) <= self.finite_difference_iterations:
            self._fd_optimizer.tell(x, y)
            # FD does not enforce bounds, but skopt raises when told a point
            # outside its space.  Out-of-bounds FD evaluations are therefore
            # excluded from the BO warm-start history (same guard as NM→BO).
            if all(p.low <= xi <= p.high for xi, p in zip(x, self._parameters)):
                self._bo_optimizer.tell(x, y)  # warm-starts the BO phase
        else:
            self._bo_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        """Raise ValueError if the configuration is invalid."""
        if self.finite_difference_iterations < 1:
            raise ValueError(
                f"n_initial must be >= 1 for the FD phase, "
                f"got {self.finite_difference_iterations}."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")
        # Validate FD sub-optimizer options by delegating.
        self._fd_optimizer.test_config()


# --------------------------------------------------------------------- #
# Hybrid Bayesian → Nelder-Mead Optimizer                              #
# --------------------------------------------------------------------- #

class HybridBayesNelderMeadOptimizer:
    """Two-phase optimizer: Bayesian Optimisation (Sobol + GP), then Nelder-Mead.

    The first ``n_initial`` evaluations use a skopt GP with Sobol initial
    sampling followed by standard GP-based BO.  Once ``n_initial`` results
    have been collected the optimizer switches to ``NelderMeadOptimizer``
    whose startup simplex is re-centred on the current BO best point
    (instead of the GEKO defaults), so the NM phase refines from the best
    region found by BO.

    Configuration knobs (inside ``bo_options``):
        n_initial_sobol : int  – Sobol points before GP takes over (default 5).
        bayesian_kind   : str  – skopt base estimator (default "GP").
        random_state    : int  – RNG seed (default 42).
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        n_initial: int,
        bo_options: dict[str, Any],
        nm_options: dict[str, Any],
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.bo_iterations = n_initial
        self.epsilon = epsilon
        self.window = window
        self._parameters = parameters

        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        bo_opts = bo_options or {}
        bo_dimensions = [Real(p.low, p.high, name=p.name) for p in self._parameters]
        n_sobol = int(bo_opts.get("n_initial_sobol", min(5, n_initial)))

        self._bo_optimizer = SkoptOptimizer(
            dimensions=bo_dimensions,
            base_estimator=_resolve_bo_base_estimator(
                bo_opts.get("bayesian_kind", "GP"),
                random_state=bo_opts.get("random_state", 42),
            ),
            n_initial_points=n_sobol,
            initial_point_generator="sobol",
            random_state=bo_opts.get("random_state", 42),
        )

        self._nm_optimizer = NelderMeadOptimizer(
            parameters=self._parameters,
            options=nm_options or {},
            epsilon=epsilon,
            window=window,
        )

        self._phase = "bayesian"
        self._nm_seeded = False

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

    # ------------------------------------------------------------------ #
    # Transition helper                                                   #
    # ------------------------------------------------------------------ #

    def _seed_nm_from_bo_best(self) -> None:
        """Re-centre NM's initial simplex on the BO best point."""
        if self._nm_seeded:
            return
        # Consider only the BO-phase entries so the seed point is identical
        # whether this runs at the live transition (history == BO phase) or
        # during a resume replay (history may already contain NM-phase points).
        best_idx = int(np.argmin(self._history_y[: self.bo_iterations]))
        best_x = list(self._history_x[best_idx])
        self._nm_optimizer._initial_points = self._build_nm_simplex(best_x)
        self._nm_seeded = True

    def _build_nm_simplex(self, center: list[float]) -> list[list[float]]:
        """Build an n_dim+1 simplex around *center* using the same perturbation
        offsets as NelderMeadOptimizer._build_initial_simplex.

        Pass ``simplex_scale`` in nm_options to shrink or expand the initial
        simplex (default 1.0).  Values < 1 increase local exploitation;
        values > 1 increase exploration from the BO best point.
        """
        scale = float(self._nm_optimizer.options.get("simplex_scale", 1.0))
        n_dim = len(center)
        bounds = self._nm_optimizer.bounds

        # Uniform startup offset of 0.1 in every dimension (matching
        # NelderMeadOptimizer._build_initial_simplex), scaled by simplex_scale.
        offset = 0.1 * scale

        def clip(val: float, dim: int) -> float:
            return float(np.clip(val, bounds[dim, 0], bounds[dim, 1]))

        if n_dim == 1:
            return [
                [clip(center[0] - offset, 0)],
                [clip(center[0] + offset, 0)],
            ]
        lower = list(center)
        upper = list(center)
        lower[0] = clip(center[0] - offset, 0)
        upper[0] = clip(center[0] + offset, 0)
        points = [lower, upper]
        for dim in range(1, n_dim):
            pt = list(center)
            pt[dim] = clip(center[dim] + offset, dim)
            points.append(pt)
        return points

    # ------------------------------------------------------------------ #
    # ask / tell                                                          #
    # ------------------------------------------------------------------ #

    def ask(self) -> list[float]:
        if len(self._history_y) < self.bo_iterations:
            self._phase = "bayesian"
            return self._bo_optimizer.ask()

        self._phase = "nelder_mead"
        self._seed_nm_from_bo_best()
        return self._nm_optimizer.ask()

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        # Route by evaluation index, not by self._phase: a resumed run replays
        # its history through tell() without ever calling ask(), so _phase
        # would still read "bayesian" and NM-phase points — which may lie
        # outside the bounds — would be fed to skopt, which rejects
        # out-of-space points with a ValueError.
        if len(self._history_y) <= self.bo_iterations:
            self._bo_optimizer.tell(x, y)
        else:
            # No-op if ask() already seeded NM at the live transition;
            # required when the NM phase is reached during a replay.
            self._seed_nm_from_bo_best()
            self._nm_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def test_config(self) -> None:
        if self.bo_iterations < 1:
            raise ValueError(
                f"n_initial must be >= 1 for the BO phase, got {self.bo_iterations}."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")
        # Validate NM sub-optimizer options by delegating.
        self._nm_optimizer.test_config()


# --------------------------------------------------------------------- #
# Hybrid Bayesian → Finite-Difference Optimizer                        #
# --------------------------------------------------------------------- #

class HybridBayesFiniteDifferenceOptimizer:
    """Two-phase optimizer: Bayesian Optimisation (Sobol + GP), then finite differences.

    The first ``n_initial`` evaluations use a skopt GP with Sobol initial
    sampling followed by standard GP-based BO.  Once ``n_initial`` results
    have been collected the optimizer switches to ``FiniteDifferenceOptimizer``
    warm-started from the current best point found by BO — the best point is
    injected directly as the FD base so it is never re-evaluated.

    Configuration knobs (all inside ``bo_options``):
        n_initial_sobol : int   – how many of the BO iterations use Sobol
                                  sampling before GP takes over (default 5).
        bayesian_kind   : str   – skopt base estimator (default "GP").
        random_state    : int   – RNG seed (default 42).
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        n_initial: int,
        bo_options: dict[str, Any],
        fd_options: dict[str, Any],
        epsilon: float | None = None,
        window: int = 3,
    ):
        self.bo_iterations = n_initial
        self.epsilon = epsilon
        self.window = window
        self._parameters = parameters

        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        bo_opts = bo_options or {}
        bo_dimensions = [Real(p.low, p.high, name=p.name) for p in self._parameters]
        n_sobol = int(bo_opts.get("n_initial_sobol", min(5, n_initial)))

        self._bo_optimizer = SkoptOptimizer(
            dimensions=bo_dimensions,
            base_estimator=_resolve_bo_base_estimator(
                bo_opts.get("bayesian_kind", "GP"),
                random_state=bo_opts.get("random_state", 42),
            ),
            n_initial_points=n_sobol,
            initial_point_generator="sobol",
            random_state=bo_opts.get("random_state", 42),
        )

        self._fd_optimizer = FiniteDifferenceOptimizer(
            parameters=self._parameters,
            options=fd_options or {},
            epsilon=epsilon,
            window=window,
        )

        self._phase = "bayesian"
        self._fd_seeded = False

        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

    # ------------------------------------------------------------------ #
    # Transition helper                                                   #
    # ------------------------------------------------------------------ #

    def _seed_fd_from_bo_best(self) -> None:
        """Inject BO best into FD optimizer as starting base, skipping re-evaluation."""
        if self._fd_seeded:
            return
        # Consider only the BO-phase entries so the seed point is identical
        # whether this runs at the live transition (history == BO phase) or
        # during a resume replay (history may already contain FD-phase points).
        best_idx = int(np.argmin(self._history_y[: self.bo_iterations]))
        best_x = np.array(self._history_x[best_idx], dtype=float)
        best_y = float(self._history_y[best_idx])
        # Directly set FD internal state so the state machine skips the
        # "evaluate initial point" step and begins probing immediately.
        self._fd_optimizer._base   = best_x
        self._fd_optimizer._base_y = best_y
        # Also record it in the FD optimizer's own history so the history
        # reflects the point its state was seeded from.  This is bookkeeping
        # only: FD's decision logic reads _base/_base_y;
        # _history_x/_history_y are just recorded and persisted.
        self._fd_optimizer._history_x.append(list(best_x))
        self._fd_optimizer._history_y.append(best_y)
        self._fd_optimizer._start_probing()
        self._fd_seeded = True

    # ------------------------------------------------------------------ #
    # ask / tell                                                          #
    # ------------------------------------------------------------------ #

    def ask(self) -> list[float]:
        if len(self._history_y) < self.bo_iterations:
            self._phase = "bayesian"
            return self._bo_optimizer.ask()

        self._phase = "finite_difference"
        self._seed_fd_from_bo_best()
        return self._fd_optimizer.ask()

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        # Route by evaluation index, not by self._phase: a resumed run replays
        # its history through tell() without ever calling ask(), so _phase
        # would still read "bayesian" for every replayed point.
        if len(self._history_y) <= self.bo_iterations:
            self._bo_optimizer.tell(x, y)
        else:
            # No-op if ask() already seeded FD at the live transition;
            # required when the FD phase is reached during a replay.
            self._seed_fd_from_bo_best()
            self._fd_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def test_config(self) -> None:
        if self.bo_iterations < 1:
            raise ValueError(
                f"n_initial must be >= 1 for the BO phase, got {self.bo_iterations}."
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")
        self._fd_optimizer.test_config()


# --------------------------------------------------------------------- #
# skopt GP wrapper                                                      #
# --------------------------------------------------------------------- #

class SkoptGPOptimizer:
    """Thin wrapper around skopt.Optimizer adding epsilon stopping support.

    The raw skopt object has no should_stop() and no history tracking.
    This wrapper records every tell() call and delegates ask/tell to skopt.
    """

    def __init__(self, skopt_opt, *, epsilon: float | None = None, window: int = 3):
        self._opt = skopt_opt
        self.epsilon = epsilon
        self.window = window
        self._history_y: list[float] = []

    def ask(self) -> list[float]:
        return self._opt.ask()

    def tell(self, x: list[float], y: float) -> Any:
        self._history_y.append(float(y))
        return self._opt.tell(x, y)

    def should_stop(self) -> bool:
        return _should_stop(self._history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}.")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}.")


# --------------------------------------------------------------------- #
# Builders                                                              #
# --------------------------------------------------------------------- #

def build_optimizer(
    optimizer_section: OptimizerSection,
    parameters: list[ParameterSpec],
) -> Optimizer:
    """Construct the optimizer for an experiment.

    Parameters
    ----------
    optimizer_section : OptimizerSection
        The ``optimizer`` block from the experiment JSON.
    parameters : list[ParameterSpec]
        The ``parameters`` block from the experiment JSON. 

    Returns
    -------
    Optimizer
        An object exposing ``ask()`` / ``tell()``.
    """

    kind   = optimizer_section.kind
    opts   = optimizer_section.kind_specific_options
    eps    = optimizer_section.stopping_criteria.get("epsilon")
    window = int(optimizer_section.stopping_criteria.get("window", 3))

    if kind == "skopt_gp":
        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        dimensions = [Real(p.low, p.high, name=p.name) for p in parameters]
        n_initial = int(opts.get("n_initial", 8))
        if n_initial < 1:
            raise ValueError(
                f"options.n_initial must be >= 1 for skopt_gp, got {n_initial}."
            )
        random_state = opts.get("random_state", 42)
        skopt_opt = SkoptOptimizer(
            dimensions=dimensions,
            base_estimator=_resolve_bo_base_estimator("GP", random_state=random_state),
            n_initial_points=n_initial,
            initial_point_generator="sobol",
            random_state=random_state,
        )
        opt = SkoptGPOptimizer(skopt_opt, epsilon=eps, window=window)

    elif kind == "pso":
        opt = ParticleSwarmOptimizer(
            parameters,
            options=opts,
            n_calls=int(optimizer_section.stopping_criteria.get("n_calls", 32)),
            epsilon=eps,
            window=window,
        )

    elif kind == "nelder_mead":
        opt = NelderMeadOptimizer(
            parameters,
            epsilon=eps,
            window=window,
            options=opts,
        )

    elif kind == "finite_differences":
        opt = FiniteDifferenceOptimizer(
            parameters,
            epsilon=eps,
            window=window,
            options=opts,
        )

    elif kind == "hybrid_nm_bayes":
        opt = HybridNelderMeadBayesOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            nm_options=opts.get("nm_options", {}),
            epsilon=eps,
            window=window,
        )

    elif kind == "hybrid_fd_bayes":
        opt = HybridFiniteDifferenceBayesOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            fd_options=opts.get("fd_options", {}),
            epsilon=eps,
            window=window,
        )

    elif kind == "hybrid_bayes_nm":
        opt = HybridBayesNelderMeadOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            nm_options=opts.get("nm_options", {}),
            epsilon=eps,
            window=window,
        )

    elif kind == "hybrid_bayes_fd":
        opt = HybridBayesFiniteDifferenceOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            fd_options=opts.get("fd_options", {}),
            epsilon=eps,
            window=window,
        )

    else:
        raise ValueError(f"Unknown optimizer kind: {kind!r}")

    # Fail fast on invalid options instead of surfacing them mid-run.
    opt.test_config()
    return opt


def vector_to_params(
    x: list[float], parameters: list[ParameterSpec]
) -> dict[str, float]:
    """Convert a parameter vector (from optimizer.ask) into a named dict."""
    if len(x) != len(parameters):
        raise ValueError(
            f"Parameter vector length {len(x)} != number of parameters {len(parameters)}"
        )
    return {p.name: float(v) for p, v in zip(parameters, x)}


def params_to_vector(
    params: dict[str, float], parameters: list[ParameterSpec]
) -> list[float]:
    """Convert a named-dict (e.g. loaded from CSV) back to a vector."""
    return [float(params[p.name]) for p in parameters]
