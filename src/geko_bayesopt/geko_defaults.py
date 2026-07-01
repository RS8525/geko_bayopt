"""Canonical Fluent GEKO coefficient defaults.

These values describe the baseline model configuration. Optimizers remain
responsible for their own startup behavior and do not consume baseline runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ParameterSpec


GEKO_DEFAULTS: dict[str, float] = {
    "geko_csep": 1.75,
    "geko_cnw": 0.5,
    "geko_cmix": 0.0,
    "geko_cjet": 1.0,
    "geko_ccorner": 1.0,
    "geko_cturb": 2.0,
}


def defaults_for_parameters(
    parameters: list[ParameterSpec],
) -> dict[str, float]:
    """Return canonical defaults for the configured optimization parameters."""
    unknown = [
        parameter.name
        for parameter in parameters
        if parameter.name not in GEKO_DEFAULTS
    ]
    if unknown:
        raise ValueError(
            "No canonical GEKO default is defined for parameter(s): "
            + ", ".join(sorted(unknown))
        )
    return {parameter.name: GEKO_DEFAULTS[parameter.name] for parameter in parameters}
