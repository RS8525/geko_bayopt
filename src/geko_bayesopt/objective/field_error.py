"""
Field-level error between DNS reference and RANS simulation.

Computes a scale-aware error norm so different fields (cp, Ux, Uy, ...)
can be summed without one dominating just because it lives on a larger
numerical scale. The norm is::

    error = mean( |dns - sim| ) / std(dns)

where ``std(dns)`` is computed once per field at construction time. This
keeps the loss dimensionless and roughly comparable across fields, while
avoiding the per-point near-zero blow-up of true relative error.

If ``std(dns) == 0`` for some field (a constant DNS field, unusual but
possible), the denominator falls back to ``eps`` to avoid division by
zero. In that case the error degenerates to the mean absolute error.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


class FieldErrorCalculator:
    """Compute scaled error between simulation and DNS for one field.

    Interpolates the simulation values onto the DNS grid (DNS grid is the
    common comparison space) and returns the mean absolute error divided
    by the DNS field's standard deviation.

    Parameters
    ----------
    dns_coords : np.ndarray, shape (N, 2)
        DNS sample points.
    dns_fields : dict[str, np.ndarray]
        DNS reference values. Each array has shape (N,).
    field_weights : dict[str, float], optional
        Per-field multiplicative weights. Default: 1.0 for any field.
    """

    def __init__(
        self,
        dns_coords: np.ndarray,
        dns_fields: dict[str, np.ndarray],
        field_weights: dict[str, float] | None = None,
    ):
        self.dns_coords = dns_coords
        self.dns_fields = dns_fields
        self.field_weights = field_weights or {}

        # Pre-compute std per DNS field so we don't recompute it every
        # trial. Falls back to eps when DNS is constant (degenerate case).
        eps = 1e-8
        self._dns_std: dict[str, float] = {}
        for name, vals in dns_fields.items():
            s = float(np.std(vals))
            self._dns_std[name] = s if s > eps else eps

    def calculate_error(
        self,
        sim_coords: np.ndarray,
        sim_fields: dict[str, np.ndarray],
        field_name: str = "cp",
    ) -> float:
        """Mean absolute error normalized by std(DNS) for one field.

            error = mean( |dns - sim_interp| ) / std(dns)

        Simulation values are interpolated onto the DNS grid using linear
        scipy.griddata. Points outside the simulation's convex hull are
        masked out before computing the mean.
        """
        if field_name not in self.dns_fields or field_name not in sim_fields:
            raise KeyError(
                f"Field '{field_name}' must be present in both DNS and "
                "simulation fields."
            )

        dns_vals = self.dns_fields[field_name]
        sim_vals = sim_fields[field_name]

        # Interpolate simulation onto DNS grid.
        sim_interp = griddata(
            sim_coords,
            sim_vals,
            self.dns_coords,
            method="linear",
        )

        # Drop NaN points (outside the simulation convex hull).
        valid = ~np.isnan(sim_interp)
        if not valid.any():
            raise ValueError(
                "Interpolation resulted in entirely NaN values. "
                "Check that the simulation and DNS coordinate systems match."
            )

        sim_valid = sim_interp[valid]
        dns_valid = dns_vals[valid]

        # Normalized mean absolute error: scale-invariant across fields,
        # no per-point denominator blow-up.
        mae = float(np.mean(np.abs(dns_valid - sim_valid)))
        normalized = mae / self._dns_std[field_name]

        weight = self.field_weights.get(field_name, 1.0)
        return float(weight * normalized)