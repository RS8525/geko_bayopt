"""
Single-field normalized field-error loss.

Computes an area-weighted, DNS-std-normalized error between simulation
and DNS for one field (default: cp), optionally scaled by a global weight.
"""

from __future__ import annotations

import numpy as np

from .field_error import FieldErrorCalculator
from .types import LossFn


def mse_field(
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
    *,
    field: str = "cp",
    weight: float = 1.0,
    field_error_norm: str = "l2",
) -> LossFn:
    """Build a normalized field-error loss on a single field.

    Parameters
    ----------
    dns_coords, dns_fields
        DNS reference data, as returned by ``FlowCase.load_dns``.
    field
        Which field to compare. Must be present in both DNS
        and simulation outputs.
    weight
        Scalar multiplier applied to the field error. Default 1.0.
    field_error_norm
        Field-error norm: ``"l2"`` (default) or ``"l1"``.

    Returns
    -------
    LossFn
        Callable ``run_result -> float``.
    """
    calc = FieldErrorCalculator(
        dns_coords,
        dns_fields,
        field_error_norm=field_error_norm,
    )

    def loss(run) -> float:
        err = calc.calculate_error(run.grid_coords, run.fields, field_name=field)
        return weight * err

    return loss


# Convenience alias: a cp-specific factory for the common case.
def mse_cp(
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
    *,
    weight: float = 1.0,
    field_error_norm: str = "l2",
) -> LossFn:
    """Pressure-coefficient field error. Shorthand for ``mse_field(field='cp')``."""
    return mse_field(
        dns_coords,
        dns_fields,
        field="cp",
        weight=weight,
        field_error_norm=field_error_norm,
    )
