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
