"""
Single-field MSE loss.

Computes mean squared error between simulation and DNS for one field
(default: cp), optionally scaled by a global weight.
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
) -> LossFn:
    """Build an MSE loss on a single field.

    Parameters
    ----------
    dns_coords, dns_fields
        DNS reference data, as returned by ``FlowCase.load_dns``.
    field
        Which field to compute MSE on. Must be present in both DNS
        and simulation outputs.
    weight
        Scalar multiplier applied to the MSE. Default 1.0.

    Returns
    -------
    LossFn
        Callable ``run_result -> float``.
    """
    calc = FieldErrorCalculator(dns_coords, dns_fields)

    def loss(run) -> float:
        mse = calc.calculate_error(run.grid_coords, run.fields, field_name=field)
        return weight * mse

    return loss


# Convenience alias: a cp-specific factory for the common case.
def mse_cp(
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
    *,
    weight: float = 1.0,
) -> LossFn:
    """MSE on pressure coefficient. Shorthand for ``mse_field(field='cp')``."""
    return mse_field(dns_coords, dns_fields, field="cp", weight=weight)
