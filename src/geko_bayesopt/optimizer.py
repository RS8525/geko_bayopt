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
    use_best_two: bool = False,
) -> bool:
    """Return True when the epsilon convergence condition is met.

    Parameters
    ----------
    history_y : list[float]
        All observed objective values so far.
    epsilon : float | None
        Stop when the relative change in objective falls below this.
    use_best_two : bool
        When True (BO mode) compare the two *best* observed values.
        When False (sequential mode) compare the last two iterates.
    """
    if epsilon is not None and len(history_y) >= 2:

        # For BO because it's a non-decreasing sequence
        if use_best_two:
            s = sorted(history_y)
            ref, other = s[0], s[1]
        else:
            ref, other = history_y[-2], history_y[-1]
        denom = max(abs(ref), 1e-10)
        if abs(other - ref) / denom < epsilon:
            return True

    return False


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
        options: dict[str, float] | None = None,
    ):
        self.parameters = parameters
        self.epsilon = epsilon
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
        """Return True when the epsilon convergence condition is met."""
        return _should_stop(self._history_y, self.epsilon)


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
    ):
        self.parameters = parameters
        self.options = options or {}
        self.epsilon = epsilon

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
            "history_x":    self._history_x,
            "history_y":    self._history_y,
            "pending_x":    self._pending_x.tolist() if self._pending_x is not None else None,
            "pending_op":   self._pending_op,
            "base":         self._base.tolist() if self._base is not None else None,
            "base_y":       self._base_y,
            "probe_dim":    self._probe_dim,
            "probe_deltas": self._probe_deltas,
            "probe_y":      self._probe_y,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._history_x    = state["history_x"]
        self._history_y    = state["history_y"]
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
        return _should_stop(self._history_y, self.epsilon)

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
    ):
        self.nelder_mead_iterations = n_initial
        self.epsilon = epsilon
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
                        epsilon = None,  # NM phase doesn't use epsilon stopping
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
        use_best_two = self._phase == "bayesian"
        return _should_stop(self._history_y, self.epsilon, use_best_two=use_best_two)

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
    ):
        self.finite_difference_iterations = n_initial
        self.epsilon = epsilon
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
            epsilon=None,  # FD phase doesn't use epsilon stopping
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
        use_best_two = self._phase == "bayesian"
        return _should_stop(self._history_y, self.epsilon, use_best_two=use_best_two)

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

    kind = optimizer_section.kind
    opts = optimizer_section.kind_specific_options
    eps  = optimizer_section.stopping_criteria.get("epsilon")

    if kind == "skopt_gp":
        from skopt import Optimizer as SkoptOptimizer
        from skopt.space import Real
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern
        dimensions = [Real(p.low, p.high, name=p.name) for p in parameters]
        opt = SkoptOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10,),
            n_initial_points=opts.get("n_initial", 8),
            initial_point_generator="sobol",
            random_state=opts.get("random_state", 42),
        )

    elif kind == "nelder_mead":
        opt = NelderMeadOptimizer(
            parameters,
            epsilon=eps,
            options=opts,
        )

    elif kind == "finite_differences":
        opt = FiniteDifferenceOptimizer(
            parameters,
            epsilon=eps,
            options=opts,
        )

    elif kind == "hybrid_nm_bayes":
        opt = HybridNelderMeadBayesOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            nm_options=opts.get("nm_options", {}),
            epsilon=eps,
        )

    elif kind == "hybrid_fd_bayes":
        opt = HybridFiniteDifferenceBayesOptimizer(
            parameters,
            n_initial=opts.get("n_initial", 8),
            bo_options=opts.get("bo_options", {}),
            fd_options=opts.get("fd_options", {}),
            epsilon=eps,
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
