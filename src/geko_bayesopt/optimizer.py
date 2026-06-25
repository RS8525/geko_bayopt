"""
Optimizer dispatcher.

Each optimizer kind builds an object exposing ``ask()`` / ``tell()`` /
state save/load. ``skopt.Optimizer`` already implements this interface,
so the BO loop is identical regardless of which optimizer is in use.

Add a new optimizer by:
    1. Implementing a small adapter class with the four methods used
       in ``experiment.py`` (``ask``, ``tell``, ``get_state``, ``set_state``).
    2. Adding a builder branch in ``build_optimizer``.
    3. Extending ``Literal`` in ``config.OptimizerSection``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .config import OptimizerSection, ParameterSpec


class Optimizer(Protocol):
    """Structural type for what the experiment loop needs.

    skopt.Optimizer satisfies this directly. Custom optimizers (e.g.
    a future BoTorch wrapper) just need methods with the same names
    and signatures.
    """

    def ask(self) -> list[float]: ...
    def tell(self, x: list[float], y: float) -> Any: ...


# --------------------------------------------------------------------- #
# Stopping Criteria Helper                                              #
# --------------------------------------------------------------------- #

def _should_stop(
    history_y: list[float],
    epsilon: float | None,
    *,
    window: int = 3,
) -> bool:
    """Return True when relative improvement over the last ``window`` steps is below epsilon.

    Compares the best value in the last ``window`` entries against the best
    value in all preceding entries.  Requires at least ``2 * window`` entries
    to avoid spurious early stops at the very beginning.
    """
    if epsilon is None or len(history_y) < 2 * window:
        return False
    recent_best = min(history_y[-window:])
    prior_best  = min(history_y[:-window])
    denom = max(abs(prior_best), 1e-10)
    return abs(recent_best - prior_best) / denom < epsilon


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

        # α = 0.8 (textbook default 1.0): less aggressive reflection reduces
        # overshoots in the smooth quadratic bowl that the CFD cost approximates.
        self._ALPHA = float(self.options.get("alpha", 0.8))
        # γ = 1.5 (textbook default 2.0): less aggressive expansion for the
        # same reason — exploitation matters more than exploration here.
        self._GAMMA = float(self.options.get("gamma", 1.5))
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
        self._iter_best_y: list[float] = []

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

        # Phase 2: initialize the simplex on the first entry after startup.
        if self._simplex_x is None:
            self._init_simplex()
            self._prepare_reflect()

        return list(self._pending_x)

    def tell(self, x: list[float], y: float) -> None:
        self._history_x.append(list(x))
        self._history_y.append(float(y))

        # Still collecting startup evaluations.
        if len(self._history_x) <= len(self._initial_points):
            return

        # Simplex not yet initialized — next ask() will do it.
        if self._simplex_x is None:
            return

        self._process_result(np.array(x, dtype=float), float(y))

    # ------------------------------------------------------------------ #
    # Simplex initialization                                              #
    # ------------------------------------------------------------------ #

    def _init_simplex(self) -> None:
        n = len(self._initial_points)
        self._simplex_x = [np.array(self._history_x[i], dtype=float) for i in range(n)]
        self._simplex_y = [float(self._history_y[i]) for i in range(n)]

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
        self._iter_best_y.append(self._simplex_y[0])
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

        all_defaults = { "geko_csep": 1.75, "geko_cnw": 0.5, "geko_cmix": 0.0, "geko_cwall": 0.9 }

        defaults = [all_defaults[parameter.name] for parameter in self.parameters]

        points: list[list[float]] = []
        if self.n_dim == 1:
            points = [
                [self._clip_value(defaults[0] - 0.25, 0)],
                [self._clip_value(defaults[0] + 0.25, 0)],
            ]
        else:
            lower = defaults.copy()
            upper = defaults.copy()
            lower[0] = self._clip_value(defaults[0] - 0.25, 0)
            upper[0] = self._clip_value(defaults[0] + 0.25, 0)
            points.append(lower)
            points.append(upper)
            for dim in range(1, self.n_dim):
                point = defaults.copy()
                point[dim] = self._clip_value(defaults[dim] + 0.10, dim)
                points.append(point)

        return points

    def _clip_value(self, value: float, dim: int) -> float:
        return float(np.clip(value, self.bounds[dim, 0], self.bounds[dim, 1]))

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        return {
            "history_x":     self._history_x,
            "history_y":     self._history_y,
            "iter_best_y":   self._iter_best_y,
            "simplex_x":     [x.tolist() for x in self._simplex_x] if self._simplex_x is not None else None,
            "simplex_y":     self._simplex_y,
            "pending_op":    self._pending_op,
            "pending_x":     self._pending_x.tolist() if self._pending_x is not None else None,
            "x0":            self._x0.tolist() if self._x0 is not None else None,
            "x_r":           self._x_r.tolist() if self._x_r is not None else None,
            "f_r":           self._f_r,
            "x_e":           self._x_e.tolist() if self._x_e is not None else None,
            "x_c":           self._x_c.tolist() if self._x_c is not None else None,
            "contract_type": self._contract_type,
            "shrink_idx":    self._shrink_idx,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._history_x    = state["history_x"]
        self._history_y    = state["history_y"]
        self._iter_best_y  = state.get("iter_best_y", [])
        sx = state.get("simplex_x")
        self._simplex_x    = [np.array(x, dtype=float) for x in sx] if sx is not None else None
        self._simplex_y    = state.get("simplex_y")
        self._pending_op   = state.get("pending_op")
        px = state.get("pending_x")
        self._pending_x    = np.array(px, dtype=float) if px is not None else None
        x0 = state.get("x0")
        self._x0           = np.array(x0, dtype=float) if x0 is not None else None
        x_r = state.get("x_r")
        self._x_r          = np.array(x_r, dtype=float) if x_r is not None else None
        self._f_r          = state.get("f_r")
        x_e = state.get("x_e")
        self._x_e          = np.array(x_e, dtype=float) if x_e is not None else None
        x_c = state.get("x_c")
        self._x_c          = np.array(x_c, dtype=float) if x_c is not None else None
        self._contract_type = state.get("contract_type")
        self._shrink_idx   = state.get("shrink_idx", 0)

    def should_stop(self) -> bool:
        return _should_stop(self._iter_best_y, self.epsilon, window=self.window)


# --------------------------------------------------------------------- #
# Finite Difference Optimizer                                           #
# --------------------------------------------------------------------- #

class FiniteDifferenceOptimizer:
    """Gradient-based optimizer using finite differences.

    Each cycle: probe the objective at base + delta*e_i for each dimension i
    (forward finite difference), compute the gradient, take a descent step,
    then pick the best observed point as the base for the next cycle.
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
        self._step_history_y: list[float] = []

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

    _GEKO_DEFAULTS = {"geko_csep": 1.75, "geko_cnw": 0.5, "geko_cmix": 0.0, "geko_cwall": 0.9}

    def ask(self) -> list[float]:
        if self._pending_x is None:
            self._pending_x = np.clip(
                [self._GEKO_DEFAULTS[p.name] for p in self.parameters],
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
            # Gradient step evaluated — start a new cycle from the best seen.
            best_idx     = int(np.argmin(self._history_y))
            self._base   = np.array(self._history_x[best_idx], dtype=float)
            self._base_y = float(self._history_y[best_idx])
            self._step_history_y.append(self._base_y)
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

        # Flip direction if the positive step would leave the upper bound.
        if self._base[dim] + delta > high:
            delta = -delta
        # Clamp so we never leave either bound.
        delta = float(np.clip(delta, low - self._base[dim], high - self._base[dim]))

        self._probe_deltas.append(delta)
        x = self._base.copy()
        x[dim] += delta
        return x

    def _gradient_step_point(self) -> np.ndarray:
        grad = np.array([
            (f_probe - self._base_y) / delta if abs(delta) > 1e-12 else 0.0
            for f_probe, delta in zip(self._probe_y, self._probe_deltas)
        ])

        return np.clip(self._base - self._learning_rate * grad, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        return {
            "history_x":      self._history_x,
            "history_y":      self._history_y,
            "step_history_y": self._step_history_y,
            "pending_x":      self._pending_x.tolist() if self._pending_x is not None else None,
            "pending_op":   self._pending_op,
            "base":         self._base.tolist() if self._base is not None else None,
            "base_y":       self._base_y,
            "probe_dim":    self._probe_dim,
            "probe_deltas": self._probe_deltas,
            "probe_y":      self._probe_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._history_x      = state["history_x"]
        self._history_y      = state["history_y"]
        self._step_history_y = state.get("step_history_y", [])
        px = state.get("pending_x")
        self._pending_x    = np.array(px, dtype=float) if px is not None else None
        self._pending_op   = state.get("pending_op")
        base = state.get("base")
        self._base         = np.array(base, dtype=float) if base is not None else None
        self._base_y       = state.get("base_y")
        self._probe_dim    = state.get("probe_dim", 0)
        self._probe_deltas = state.get("probe_deltas", [])
        self._probe_y      = state.get("probe_y", [])

    def should_stop(self) -> bool:
        return _should_stop(self._step_history_y, self.epsilon, window=self.window)

    def test_config(self) -> None:
        if self.n_dim < 1:
            raise ValueError("finite_difference requires at least one parameter.")
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



# --------------------------------------------------------------------- #
# Particle Swarm Optimizer                                              #
# --------------------------------------------------------------------- #

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization with a sequential ask/tell interface.

    The swarm is evaluated one particle at a time.  Each swarm iteration
    consists of ``n_particles`` ask/tell cycles; once all results are in,
    personal bests, the global best, and velocities are updated and the
    inertia weight is decayed.

    Required option:
        max_iter    : int   – total swarm iterations for the linear inertia
                              decay schedule (typically n_calls // n_particles).

    Optional options (with defaults):
        n_particles : int   – swarm size (default 10)
        w_start     : float – initial inertia weight (default 0.9)
        w_end       : float – final inertia weight   (default 0.4)
        c1          : float – cognitive coefficient  (default 1.5)
        c2          : float – social coefficient     (default 1.5)
        v_max_frac  : float – v_max as fraction of each dimension's range
                              (default 0.2)
        random_state: int   – RNG seed (default 42)

    Boundary handling: absorption — particles that overshoot a bound are
    placed exactly on it and their velocity in that dimension is zeroed.
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
        self.epsilon = epsilon
        self.window = window
        opts = options or {}

        self.bounds  = np.array([[p.low, p.high] for p in parameters])
        self.n_dim   = len(parameters)

        self.n_particles = int(opts.get("n_particles", 10))
        self.w_start     = float(opts.get("w_start",   0.9))
        self.w_end       = float(opts.get("w_end",     0.4))
        self.c1          = float(opts.get("c1",        1.5))
        self.c2          = float(opts.get("c2",        1.5))
        self.v_max_frac  = float(opts.get("v_max_frac", 0.2))
        self.max_iter    = int(opts["max_iter"])

        self._rng    = np.random.default_rng(int(opts.get("random_state", 42)))
        self._v_max  = self.v_max_frac * (self.bounds[:, 1] - self.bounds[:, 0])

        # Full history
        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []
        # One entry per completed swarm iteration, for the stopping criterion.
        self._gbest_history: list[float] = []

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
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return [self._rng.uniform(low, high) for _ in range(self.n_particles)]

    def _finalize_init(self) -> None:
        """Transition from init phase to iterate: set up swarm state."""
        self._positions  = np.array(self._iter_results_x, dtype=float)
        self._pbest_x    = self._positions.copy()
        self._pbest_y    = np.array(self._iter_results_y, dtype=float)
        self._velocities = self._rng.uniform(
            -self._v_max, self._v_max, (self.n_particles, self.n_dim)
        )
        best_idx       = int(np.argmin(self._pbest_y))
        self._gbest_x  = self._pbest_x[best_idx].copy()
        self._gbest_y  = float(self._pbest_y[best_idx])
        self._gbest_history.append(self._gbest_y)

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

        self._gbest_history.append(self._gbest_y)
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
        return _should_stop(self._gbest_history, self.epsilon, window=self.window)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def test_config(self) -> None:
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be >= 2, got {self.n_particles}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
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

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        return {
            "phase":            self._phase,
            "particle_idx":     self._particle_idx,
            "swarm_iter":       self._swarm_iter,
            "history_x":        self._history_x,
            "history_y":        self._history_y,
            "gbest_history":    self._gbest_history,
            "positions":        self._positions.tolist()  if self._positions  is not None else None,
            "velocities":       self._velocities.tolist() if self._velocities is not None else None,
            "pbest_x":          self._pbest_x.tolist()   if self._pbest_x    is not None else None,
            "pbest_y":          self._pbest_y.tolist()   if self._pbest_y    is not None else None,
            "gbest_x":          self._gbest_x.tolist()   if self._gbest_x    is not None else None,
            "gbest_y":          self._gbest_y,
            "pending_positions": [p.tolist() for p in self._pending_positions],
            "iter_results_x":   [x.tolist() for x in self._iter_results_x],
            "iter_results_y":   self._iter_results_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._phase        = state.get("phase", "init")
        self._particle_idx = state.get("particle_idx", 0)
        self._swarm_iter   = state.get("swarm_iter", 0)
        self._history_x    = state.get("history_x", [])
        self._history_y    = state.get("history_y", [])
        self._gbest_history = state.get("gbest_history", [])

        def _arr(key):
            v = state.get(key)
            return np.array(v, dtype=float) if v is not None else None

        self._positions  = _arr("positions")
        self._velocities = _arr("velocities")
        self._pbest_x    = _arr("pbest_x")
        self._pbest_y    = _arr("pbest_y")
        self._gbest_x    = _arr("gbest_x")
        self._gbest_y    = state.get("gbest_y", float("inf"))

        self._pending_positions = [
            np.array(p, dtype=float) for p in state.get("pending_positions", [])
        ]
        self._iter_results_x = [
            np.array(x, dtype=float) for x in state.get("iter_results_x", [])
        ]
        self._iter_results_y = state.get("iter_results_y", [])


# --------------------------------------------------------------------- #
# Hybrid Nelder-Mead → Bayesian Optimizer                              #
# --------------------------------------------------------------------- #

class HybridNelderMeadBayesOptimizer:
    """Two-phase optimizer: Nelder-Mead warmup, then Bayesian Optimisation.

    The first ``nelder_mead_iterations`` evaluations use ``NelderMeadOptimizer``.
    Subsequent calls switch to a skopt GP (warm-started with the NM history).
    The epsilon stopping check in the BO phase compares the two best observed
    values (not consecutive iterates).
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
        self.bo_base_estimator = self.bo_options.get("bayesian_kind", "GP")
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

        if self._phase == "nelder_mead":
            self._nm_optimizer.tell(x, y)
            self._bo_optimizer.tell(x, y) # Also tell to bo optimizer; was not initialized
        elif self._bo_optimizer is not None:
            self._bo_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        if self._phase == "nelder_mead":
            return self._nm_optimizer.should_stop()
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

    def get_state(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "history_x": self._history_x,
            "history_y": self._history_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._phase = state.get("phase", "nelder_mead")
        self._history_x = state.get("history_x", [])
        self._history_y = state.get("history_y", [])


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
        self.bo_base_estimator = self.bo_options.get("bayesian_kind", "GP")
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

        if self._phase == "finite_difference":
            self._fd_optimizer.tell(x, y)
            self._bo_optimizer.tell(x, y)  # Also tell BO optimizer; warm-starts it
        elif self._bo_optimizer is not None:
            self._bo_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        if self._phase == "finite_difference":
            return self._fd_optimizer.should_stop()
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
        # Validate FD sub-optimizer options by delegating.
        self._fd_optimizer.test_config()

    def get_state(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "history_x": self._history_x,
            "history_y": self._history_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._phase = state.get("phase", "finite_difference")
        self._history_x = state.get("history_x", [])
        self._history_y = state.get("history_y", [])


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
            base_estimator=bo_opts.get("bayesian_kind", "GP"),
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
        best_idx = int(np.argmin(self._history_y))
        best_x = list(self._history_x[best_idx])
        self._nm_optimizer._initial_points = self._build_nm_simplex(best_x)
        self._nm_seeded = True

    def _build_nm_simplex(self, center: list[float]) -> list[list[float]]:
        """Build an n_dim+1 simplex around *center* using the same perturbation
        offsets as NelderMeadOptimizer._build_initial_simplex."""
        n_dim = len(center)
        bounds = self._nm_optimizer.bounds

        def clip(val: float, dim: int) -> float:
            return float(np.clip(val, bounds[dim, 0], bounds[dim, 1]))

        if n_dim == 1:
            return [
                [clip(center[0] - 0.25, 0)],
                [clip(center[0] + 0.25, 0)],
            ]
        lower = list(center)
        upper = list(center)
        lower[0] = clip(center[0] - 0.25, 0)
        upper[0] = clip(center[0] + 0.25, 0)
        points = [lower, upper]
        for dim in range(1, n_dim):
            pt = list(center)
            pt[dim] = clip(center[dim] + 0.10, dim)
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

        if self._phase == "bayesian":
            self._bo_optimizer.tell(x, y)
        else:
            self._nm_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        if self._phase == "bayesian":
            return _should_stop(self._history_y, self.epsilon, window=self.window)
        return self._nm_optimizer.should_stop()

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

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        return {
            "phase":     self._phase,
            "nm_seeded": self._nm_seeded,
            "history_x": self._history_x,
            "history_y": self._history_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._phase     = state.get("phase", "bayesian")
        self._nm_seeded = state.get("nm_seeded", False)
        self._history_x = state.get("history_x", [])
        self._history_y = state.get("history_y", [])


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
            base_estimator=bo_opts.get("bayesian_kind", "GP"),
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
        best_idx = int(np.argmin(self._history_y))
        best_x = np.array(self._history_x[best_idx], dtype=float)
        best_y = float(self._history_y[best_idx])
        # Directly set FD internal state so the state machine skips the
        # "evaluate initial point" step and begins probing immediately.
        self._fd_optimizer._base   = best_x
        self._fd_optimizer._base_y = best_y
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

        if self._phase == "bayesian":
            self._bo_optimizer.tell(x, y)
        else:
            self._fd_optimizer.tell(x, y)

    def should_stop(self) -> bool:
        if self._phase == "bayesian":
            return _should_stop(self._history_y, self.epsilon, window=self.window)
        return self._fd_optimizer.should_stop()

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
        self._fd_optimizer.test_config()

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        return {
            "phase":     self._phase,
            "fd_seeded": self._fd_seeded,
            "history_x": self._history_x,
            "history_y": self._history_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._phase     = state.get("phase", "bayesian")
        self._fd_seeded = state.get("fd_seeded", False)
        self._history_x = state.get("history_x", [])
        self._history_y = state.get("history_y", [])


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

    def get_state(self) -> dict[str, Any]:
        return {"history_y": self._history_y}

    def set_state(self, state: dict[str, Any]) -> None:
        self._history_y = state.get("history_y", [])


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
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern
        dimensions = [Real(p.low, p.high, name=p.name) for p in parameters]
        skopt_opt = SkoptOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10,),
            n_initial_points=opts.get("n_initial", 8),
            initial_point_generator="sobol",
            random_state=opts.get("random_state", 42),
        )
        opt = SkoptGPOptimizer(skopt_opt, epsilon=eps, window=window)

    elif kind == "pso":
        opt = ParticleSwarmOptimizer(
            parameters,
            options=opts,
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
