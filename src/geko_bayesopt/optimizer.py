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

import random
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
# Nelder-Mead Optimizer                                                 #
# --------------------------------------------------------------------- #

class NelderMeadOptimizer:
    """Adapter wrapping scipy's Nelder-Mead method via iterative minimize calls.

    During initialization (first n_initial points), returns random samples
    within the parameter bounds. Once enough points are collected, uses
    scipy's Nelder-Mead method to propose new parameter vectors by running
    the optimizer for a few iterations at a time.

    The history of evaluated points is stored for persistence and crash recovery.
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        *,
        n_initial: int,
        random_state: int | None,
        options: dict[str, Any],
    ):
        """Initialize the Nelder-Mead optimizer.

        Parameters
        ----------
        parameters : list[ParameterSpec]
            Parameter specifications with bounds.
        n_initial : int
            Number of random initial evaluations before using Nelder-Mead.
        random_state : int | None
            Random seed for reproducibility.
        options : dict[str, Any]
            Additional options:
            - maxiter: max iterations per minimize call (default: 50)
            - polishing: whether to polish the final result (default: False)
        """
        self.parameters = parameters
        self.n_initial = n_initial
        self.random_state = random_state
        self.options = options or {}

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Initialize bounds
        self.bounds = np.array([[p.low, p.high] for p in parameters])
        self.n_dim = len(parameters)

        # Build the Nelder-Mead startup simplex around defaults for 1-4 dims.
        self._initial_points = self._build_initial_simplex()
        if len(self._initial_points) > self.n_initial:
            self.n_initial = len(self._initial_points)

        # History of all evaluated points and their scores
        self._history_x: list[list[float]] = []
        self._history_y: list[float] = []

        # Nelder-Mead state
        self._next_x_to_ask = None  # Cache the next point to propose
        self._nm_active = False

    def ask(self) -> list[float]:
        """Propose a new point to evaluate.

        Returns
        -------
        list[float]
            A parameter vector within the specified bounds.
        """
        # Emit the configured Nelder-Mead startup simplex first.
        if len(self._history_x) < len(self._initial_points):
            return self._initial_points[len(self._history_x)]

        # During initial warmup, return random points.
        if len(self._history_x) < self.n_initial:
            point = [
                random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                for i in range(self.n_dim)
            ]
            return point

        # Activate Nelder-Mead after initial phase
        if not self._nm_active:
            self._start_nelder_mead()

        # If we have a cached point from the previous optimization, return it
        if self._next_x_to_ask is not None:
            result = self._next_x_to_ask
            self._next_x_to_ask = None
            return result

        # Otherwise run the optimizer for a few iterations and get the next point
        self._run_nelder_mead_step()

        if self._next_x_to_ask is None:
            # Fallback: return the current best with small perturbation
            best_idx = np.argmin(self._history_y)
            best_x = np.array(self._history_x[best_idx])
            perturbation = np.random.normal(0, 0.1, self.n_dim)
            self._next_x_to_ask = list(np.clip(
                best_x + perturbation,
                self.bounds[:, 0],
                self.bounds[:, 1],
            ))

        result = self._next_x_to_ask
        self._next_x_to_ask = None
        return result

    def tell(self, x: list[float], y: float) -> None:
        """Record an evaluated point and its loss.

        Parameters
        ----------
        x : list[float]
            The parameter vector that was evaluated.
        y : float
            The loss (objective function value) at x.
        """
        self._history_x.append(list(x))
        self._history_y.append(float(y))

    def _start_nelder_mead(self) -> None:
        """Initialize Nelder-Mead after the initial random phase."""
        if len(self._history_x) < self.n_initial:
            raise RuntimeError(
                f"Cannot start Nelder-Mead: only {len(self._history_x)} "
                f"points, but {self.n_initial} needed."
            )
        self._nm_active = True

    def _build_initial_simplex(self) -> list[list[float]]:
        """Build the Nelder-Mead startup simplex around GEKO defaults."""
        defaults = self._default_coefficients()
        if defaults is None:
            return []

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

    def _default_coefficients(self) -> list[float] | None:
        """Return the default GEKO coefficient vector for 1-4 dimensions."""
        defaults = [1.75, 0.5, 0.0, 0.9]
        if 1 <= self.n_dim <= len(defaults):
            return defaults[: self.n_dim]
        return None

    def _clip_value(self, value: float, dim: int) -> float:
        return float(np.clip(value, self.bounds[dim, 0], self.bounds[dim, 1]))

    def _run_nelder_mead_step(self) -> None:
        """Run Nelder-Mead for a few iterations to get the next point."""
        from scipy.optimize import minimize

        # Find the best point so far
        best_idx = np.argmin(self._history_y)
        best_x = np.array(self._history_x[best_idx])

        # Create a function that returns cached values for known points,
        # interpolates for unknowns
        history_dict = {
            tuple(np.round(x, 8)): y 
            for x, y in zip(self._history_x, self._history_y)
        }

        def objective(x):
            """Return cached value or interpolated estimate."""
            x_rounded = tuple(np.round(x, 8))
            if x_rounded in history_dict:
                return history_dict[x_rounded]
            # For unseen points, return best so far (Nelder-Mead will
            # explore from here)
            return self._history_y[best_idx]

        # Set up Nelder-Mead options
        nm_options = self.options.copy()
        nm_options.setdefault("maxiter", 50)
        nm_options.setdefault("xatol", 1e-4)
        nm_options.setdefault("fatol", 1e-4)

        # Run optimization for a few steps
        result = minimize(
            objective,
            best_x,
            method="Nelder-Mead",
            options=nm_options,
        )

        # Extract the proposed next point
        next_x = result.x

        # Clip to bounds
        next_x = np.clip(next_x, self.bounds[:, 0], self.bounds[:, 1])

        # Ensure it's not already in history (perturb slightly if needed)
        attempts = 0
        while attempts < 5:
            x_rounded = tuple(np.round(next_x, 8))
            if x_rounded not in history_dict:
                break
            # Add small noise to escape repeated point
            next_x = next_x + np.random.normal(0, 0.01, self.n_dim)
            next_x = np.clip(next_x, self.bounds[:, 0], self.bounds[:, 1])
            attempts += 1

        self._next_x_to_ask = list(next_x)

    def get_state(self) -> dict[str, Any]:
        """Serialize the optimizer state for checkpointing."""
        return {
            "history_x": self._history_x,
            "history_y": self._history_y,
            "n_initial": self.n_initial,
            "random_state": self.random_state,
            "nm_active": self._nm_active,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the optimizer state from a checkpoint."""
        self._history_x = state["history_x"]
        self._history_y = state["history_y"]
        self._nm_active = state.get("nm_active", False)


# --------------------------------------------------------------------- #
# Builders                                                              #
# --------------------------------------------------------------------- #

def _make_skopt(
    parameters: list[ParameterSpec],
    *,
    base_estimator: str,
    n_initial: int,
    random_state: int | None,
    extra: dict[str, Any],
) -> Optimizer:
    """Build a skopt.Optimizer with the given parameter bounds."""
    # Imported lazily so skopt isn't required for users who don't use it.
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real

    dimensions = [Real(p.low, p.high, name=p.name) for p in parameters]

    return SkoptOptimizer(
        dimensions=dimensions,
        base_estimator=base_estimator,
        n_initial_points=n_initial,
        random_state=random_state,
        **extra,
    )


def _make_nelder_mead(
    parameters: list[ParameterSpec],
    *,
    n_initial: int,
    random_state: int | None,
    extra: dict[str, Any],
) -> Optimizer:
    """Build a Nelder-Mead optimizer with the given parameter bounds."""
    return NelderMeadOptimizer(
        parameters,
        n_initial=n_initial,
        random_state=random_state,
        options=extra,
    )


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
    if kind == "skopt_gp":
        return _make_skopt(
            parameters,
            base_estimator="GP",
            n_initial=optimizer_section.n_initial,
            random_state=optimizer_section.random_state,
            extra=optimizer_section.options,
        )
    elif kind == "skopt_rf":
        return _make_skopt(
            parameters,
            base_estimator="RF",
            n_initial=optimizer_section.n_initial,
            random_state=optimizer_section.random_state,
            extra=optimizer_section.options,
        )
    elif kind == "random":
        return _make_skopt(
            parameters,
            base_estimator="dummy",
            n_initial=optimizer_section.n_initial,
            random_state=optimizer_section.random_state,
            extra=optimizer_section.options,
        )
    elif kind == "nelder_mead":
        return _make_nelder_mead(
            parameters,
            n_initial=optimizer_section.n_initial,
            random_state=optimizer_section.random_state,
            extra=optimizer_section.options,
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {kind!r}")


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
