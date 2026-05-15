"""
Weighted multi-field MSE loss.

Sums per-field MSEs with user-supplied weights. Lets you tune the trade-off
between matching different quantities (pressure vs velocity, x vs y) without
changing code.
"""

from __future__ import annotations

import numpy as np

from .field_error import FieldErrorCalculator
from .types import LossFn


def weighted_multi_field(
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
    *,
    field_weights: dict[str, float],
) -> LossFn:
    """Build a weighted sum of per-field MSEs.

    Parameters
    ----------
    dns_coords, dns_fields
        DNS reference data.
    field_weights
        Mapping ``{field_name: weight}``. Each named field must exist in
        both DNS and simulation outputs. Example::

            {"cp": 1.0, "Ux": 0.5, "Uy": 0.5}

    Returns
    -------
    LossFn
        Callable ``run_result -> float`` returning the weighted sum of
        per-field MSEs.

    Raises
    ------
    ValueError
        If ``field_weights`` is empty.
    KeyError
        If a requested field is missing from DNS data at construction
        time, or from simulation data at evaluation time.
    """
    if not field_weights:
        raise ValueError(
            "field_weights must contain at least one (field, weight) pair."
        )

    # Validate up-front that DNS has every requested field.
    missing = [f for f in field_weights if f not in dns_fields]
    if missing:
        raise KeyError(
            f"DNS data is missing field(s): {missing}. "
            f"Available: {sorted(dns_fields.keys())}"
        )

    calc = FieldErrorCalculator(dns_coords, dns_fields)

    def loss(run) -> float:
        total = 0.0
        for field, weight in field_weights.items():
            mse = calc.calculate_error(run.grid_coords, run.fields, field_name=field)
            total += weight * mse
        return total

    return loss
