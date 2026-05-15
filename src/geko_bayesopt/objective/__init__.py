"""
Loss-function registry.

Each loss is a small factory function in its own module:
    - ``mse.py`` -> ``mse_field``, ``mse_cp``
    - ``weighted.py`` -> ``weighted_multi_field``

Add a new loss by:
    1. Creating ``my_loss.py`` with a factory that returns a ``LossFn``.
    2. Importing it here.
    3. Adding it to ``_REGISTRY`` keyed by the string used in JSON.
    4. Adding that string to the ``Literal`` in ``config.ObjectiveSection``.

The factory signature is::

    def my_loss(dns_coords, dns_fields, *, ...) -> LossFn:
        ...

Where ``...`` is whatever keyword arguments come from the JSON
``objective.options`` block.
"""

from __future__ import annotations

import numpy as np

from ..config import ObjectiveSection
from .field_error import FieldErrorCalculator
from .mse import mse_cp, mse_field
from .types import LossFn
from .weighted import weighted_multi_field


#: Registry: JSON kind string -> factory function.
_REGISTRY = {
    "mse_cp": mse_cp,
    "mse_field": mse_field,
    "weighted_multi_field": weighted_multi_field,
    # Add new losses here:
    # "rmse_field": rmse_field,
    # "wall_shear_at_stations": wall_shear_at_stations,
}


def build_loss_fn(
    objective_section: ObjectiveSection,
    dns_coords: np.ndarray,
    dns_fields: dict[str, np.ndarray],
) -> LossFn:
    """Construct the loss function for an experiment.

    Looks up ``objective_section.kind`` in the registry and calls the
    matching factory with ``objective_section.options`` as keyword
    arguments.

    Parameters
    ----------
    objective_section : ObjectiveSection
        The ``objective`` block from the experiment JSON.
    dns_coords, dns_fields
        DNS reference data from the flow case.

    Returns
    -------
    LossFn
        Ready-to-call ``run_result -> float``.
    """
    factory = _REGISTRY.get(objective_section.kind)
    if factory is None:
        valid = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown objective kind: {objective_section.kind!r}. "
            f"Valid kinds: {valid}"
        )
    return factory(dns_coords, dns_fields, **objective_section.options)


__all__ = [
    "FieldErrorCalculator",
    "LossFn",
    "mse_cp",
    "mse_field",
    "weighted_multi_field",
    "build_loss_fn",
]
