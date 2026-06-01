"""
Loss-function registry.

Three building blocks live in separate modules:
    - ``field_error.py`` -> ``FieldErrorCalculator`` (interpolated MSE)
    - ``integral_error.py`` -> ``IntegralErrorCalculator`` + extractors
    - ``GEDCP.py`` -> pure scalar math (gedcp, coefficient_preference)

This file composes them into loss functions selectable from JSON. Each
factory returns a ``LossFn`` (callable: RunResult -> float) the BO loop
can use without knowing the internals.

Registered loss kinds:
    - "mse" -- mean squared error between selected Fluent and DNS fields normalized by field variance
    - "mae" -- mean absolute error between selected Fluent and DNS fieldsnormalized by field variance
    - "mape"-- mean absolute percentage error between selected Fluent and DNS fields

Add a new loss by:
    1. Writing a factory ``my_loss(dns_coords, dns_fields, *, ...) -> LossFn``.
    2. Importing it here.
    3. Adding it to ``_REGISTRY``.
    4. Adding its string to the Literal in ``config.ObjectiveSection``.
"""

from __future__ import annotations

import numpy as np

from ..config import ObjectiveSection
from ..types import RunResult
from .field_error import FieldErrorCalculator
from .GEDCP import GEKO_DEFAULTS, coefficient_preference, gedcp
from .integral_error import (
    _BUILTIN_EXTRACTORS,
    IntegralErrorCalculator,
    integral_error_total,
)
from .mse import mse_cp, mse_field
from .types import LossFn
from .weighted import weighted_multi_field


# --------------------------------------------------------------------- #
# GEDCP factory -- composes field, integral, and preference terms       #
# --------------------------------------------------------------------- #

def objective_geko(
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
    *,
    field_error_kind: str = "mae",
    field_names: list[str] | None = None,
    field_weights: dict[str, float] | None = None,
    integral_weights: dict[str, float] | None = None,
    lambda_field: float = 1.0,
    lambda_integral: float = 1.0,
    lambda_preference: float = 0.0,
    defaults: dict[str, float] | None = None,
) -> LossFn:
    """Build the full GEDCP objective for the BO loop.

    Per trial this:

    1. Computes a field error E_F = sum over ``field_names`` of MSE
       between simulation and DNS (interpolated to the DNS grid).
    2. Computes an integral error E_I = weighted relative-squared-error
       sum over ``integral_weights``. Zero if no weights given.
    3. Computes a coefficient-preference penalty p from the GEKO
       coefficients the optimizer proposed (stored on the RunResult).
    4. Combines via ``gedcp(lambda_field, lambda_integral, lambda_preference)``.

    Mirrors the prototype ``objective_geko`` but as a factory that
    returns a closure -- so the BO loop can call it many times without
    re-loading DNS data each iteration.

    Parameters
    ----------
    dns_coords, dns_fields
        DNS reference data.
    field_names
        Which fields contribute to E_F. Default: ``["cp"]``.
        Each name must exist in both DNS and simulation outputs.
    integral_weights
        ``{integral_name: weight}`` for E_I. Default: empty (no integral
        term). Integrals are computed from the run's coords/fields using
        the extractors registered in ``integral_error._BUILTIN_EXTRACTORS``.
    lambda_field, lambda_integral, lambda_preference
        GEDCP combination weights.
    defaults
        Override Fluent's default GEKO coefficients for the preference
        penalty. Defaults to ``GEDCP.GEKO_DEFAULTS``.

    Example JSON::

        "objective": {
            "kind": "gedcp",
            "options": {
                "field_names": ["cp", "Ux"],
                "integral_weights": {"mean_pressure": 1.0},
                "lambda_field": 1.0,
                "lambda_integral": 0.5,
                "lambda_preference": 0.1
            }
        }
    """
        # Default weights
    if field_weights is None:
        field_weights = {
            "Ux": 1.0,
            "Uy": 1.0,
            "cp": 1.0,
            "turb-kinetic-energy": 1.0,
            "production-of-k": 1.0,
        }
    if field_names is None:
        field_names = ["cp"]

    field_names = [
    fname for fname in field_names
    if not np.isnan(np.asarray(dns_fields[fname], dtype=float)).all()
]

    field_calc = FieldErrorCalculator(dns_coords, dns_fields, kind=field_error_kind, field_weights=field_weights)
    pref_defaults = defaults if defaults is not None else GEKO_DEFAULTS

    # Pre-compute reference integrals once (if integral term is enabled).
    ref_integrals: dict[str, float] = {}
    if integral_weights:
        for name in integral_weights:
            if name not in _BUILTIN_EXTRACTORS:
                raise KeyError(
                    f"objective_geko: no extractor registered for integral "
                    f"'{name}'. Available: {sorted(_BUILTIN_EXTRACTORS)}"
                )
            ref_integrals[name] = _BUILTIN_EXTRACTORS[name](dns_coords, dns_fields)

    def loss(run: RunResult) -> float:
        # 1. Field error: sum of MSE over each named field.
        e_field = 0.0
        for fname in field_names:
            e_field += field_calc.calculate_error(
                sim_coords=run.grid_coords,
                sim_fields=run.fields,
                field_name=fname,
            )

        # 2. Integral error (optional).
        if integral_weights:
            sim_integrals = {
                name: _BUILTIN_EXTRACTORS[name](run.grid_coords, run.fields)
                for name in integral_weights
            }
            e_integral = integral_error_total(
                sim_integrals=sim_integrals,
                ref_integrals=ref_integrals,
                integral_weights=integral_weights,
            )
        else:
            e_integral = 0.0

        # 3. Default coefficient preference penalty.
        if lambda_preference > 0.0:
            p = coefficient_preference(
                coef_dict=run.parameters,
                coef_default_dict=pref_defaults,
            )
        else:
            p = 0.0

        # 4. Combine via GEDCP scalar math.
        return gedcp(
            field_error=e_field,
            integral_error=e_integral,
            coefficient_preference=p,
            lambda_field=lambda_field,
            lambda_integral=lambda_integral,
            lambda_preference=lambda_preference,
        )

    return loss


# --------------------------------------------------------------------- #
# Registry                                                              #
# --------------------------------------------------------------------- #

_REGISTRY = {
    "mae": objective_geko,
    "mse": objective_geko,
    "mape": objective_geko,
}


def build_loss_fn(
    objective_section: ObjectiveSection,
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
) -> LossFn:
    "Construct the loss function for an experiment."

    kind = objective_section.kind

    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown objective kind: {kind!r}. "
            f"Valid kinds: {sorted(_REGISTRY.keys())}"
        )

    return objective_geko(
        dns_coords,
        dns_fields,
        field_error_kind=kind,
        **objective_section.options,
    )


__all__ = [
    "FieldErrorCalculator",
    "IntegralErrorCalculator",
    "LossFn",
    "build_loss_fn",
    "coefficient_preference",
    "gedcp",
    "integral_error_total",
    "mse_cp",
    "mse_field",
    "objective_geko",
    "weighted_multi_field",
]
