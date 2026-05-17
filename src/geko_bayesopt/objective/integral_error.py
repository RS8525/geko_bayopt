"""
Integral-quantity error.

Two pieces:
    - ``IntegralErrorCalculator``: relative-squared-error math for one
      reference value vs one simulation value (your original class).
    - ``integral_error_total``: simple scalar function that loops over
      named integrals and returns the weighted total. Mirrors the style
      of ``coefficient_preference`` and ``gedcp`` -- takes pre-computed
      numbers, returns a number.

Built-in extractors are at the top so the orchestrator can use them to
compute sim/DNS integrals before calling ``integral_error_total``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# An "integral extractor" takes coords + fields, returns a single scalar.
IntegralExtractor = Callable[[np.ndarray, dict[str, np.ndarray]], float]


# --------------------------------------------------------------------- #
# Built-in extractors -- minimal set, expand as needed                  #
# --------------------------------------------------------------------- #

def mean_pressure(coords: np.ndarray, fields: dict[str, np.ndarray]) -> float:
    """Domain-mean pressure coefficient."""
    return float(np.mean(fields["cp"]))


def mean_kinetic_energy(coords: np.ndarray, fields: dict[str, np.ndarray]) -> float:
    """Domain-mean kinetic energy 1/2 (u^2 + v^2)."""
    u = fields["Ux"]
    v = fields["Uy"]
    return float(0.5 * np.mean(u * u + v * v))


_BUILTIN_EXTRACTORS: dict[str, IntegralExtractor] = {
    "mean_pressure": mean_pressure,
    "mean_kinetic_energy": mean_kinetic_energy,
}


# --------------------------------------------------------------------- #
# Core calculator -- your class, unchanged                               #
# --------------------------------------------------------------------- #

class IntegralErrorCalculator:
    """
    Computes the error between reference integral quantities and RANS
    simulation integral quantities.
    """

    def __init__(self, ref_integrals: dict[str, float]):
        self.ref_integrals = ref_integrals

    def calculate_error(
        self,
        sim_integrals: dict[str, float],
        integral_name: str,
    ) -> float:
        """
        Calculates relative squared error for one integral quantity.

        error = ((sim - ref) / ref)^2
        """
        if integral_name not in self.ref_integrals or integral_name not in sim_integrals:
            raise KeyError(
                f"Integral quantity '{integral_name}' must be present in both "
                "reference and simulation dictionaries."
            )

        ref_val = float(self.ref_integrals[integral_name])
        sim_val = float(sim_integrals[integral_name])

        if ref_val == 0.0:
            raise ValueError(
                f"Cannot compute relative error for '{integral_name}' because "
                "the reference value is zero."
            )

        error = ((sim_val - ref_val) / ref_val) ** 2

        return float(error)


# --------------------------------------------------------------------- #
# Convenience: weighted sum over multiple integrals                     #
# --------------------------------------------------------------------- #

def integral_error_total(
    sim_integrals: dict[str, float],
    ref_integrals: dict[str, float],
    integral_weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of per-integral relative squared errors::

        E_I = sum_i w_i * ((sim_i - ref_i) / ref_i)^2

    Parameters
    ----------
    sim_integrals
        ``{name: value}`` computed from the current simulation.
    ref_integrals
        ``{name: value}`` from the DNS reference.
    integral_weights
        ``{name: weight}``. Defaults to equal weight 1.0 on every named
        integral present in both dicts.

    Returns 0.0 if no shared names are found.
    """
    if integral_weights is None:
        shared = set(sim_integrals) & set(ref_integrals)
        integral_weights = {name: 1.0 for name in shared}

    if not integral_weights:
        return 0.0

    calc = IntegralErrorCalculator(ref_integrals)
    total = 0.0
    for name, weight in integral_weights.items():
        total += weight * calc.calculate_error(sim_integrals, name)
    return float(total)
