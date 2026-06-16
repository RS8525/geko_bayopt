"""
Field-level error between DNS reference and RANS simulation.

Computes a scale-aware, area-weighted error norm so different fields
(cp, Ux, Uy, ...) can be summed without one dominating by scale, and
without dense-grid regions dominating by node count.

Two refinements over a naive point-mean MSE:

1. **Geometric masking.** DNS points lying below the analytic
   Mellen/Breuer hill surface are excluded, so the comparison covers
   only the shared fluid region. This removes contamination from points
   that fall inside (or right at) the hill where the DNS and simulation
   discretized surfaces disagree by a fraction of a cell.

2. **Area weighting.** Every comparison statistic (the cp gauge, the
   error mean, the normalizing std) is weighted by each DNS point's
   cell area, computed from the structured DNS grid spacing. This makes
   the metric grid-independent: a densely sampled region contributes in
   proportion to its physical area, not its node count. Because the
   whole comparison happens on the DNS grid, the simulation's boundary-
   layer clustering never biases the result.

The per-field error is::

    error = wmean( |dns - sim| ) / wstd(dns)

where ``wmean`` / ``wstd`` are area-weighted. For cp, both DNS and the
interpolated simulation are re-gauged to an area-weighted zero mean on
the masked common grid before the error is taken, so the pressure datum
is identical on both sides.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


# --------------------------------------------------------------------- #
# Mellen / Brefuer / Froehlich periodic-hill surface                     #
# Standard ERCOFTAC definition: x in mm, hill height h = 28 mm.          #
# 6 polynomial segments for the left hill (x_mm in [0, 54]); the right   #
# hill mirrors about the domain centre. Normalised here to H = 1.        #
# --------------------------------------------------------------------- #

_HILL_H_MM = 28.0          # reference hill height in the polynomial definition
_HILL_HALF_MM = 54.0       # left hill extends from x=0 to x=54 mm (= 1.929 H)


def _hill_left_mm(x_mm: float) -> float:
    """Left-hill surface height (mm) for x_mm in [0, 54]."""
    x = x_mm
    if x <= 9.0:
        y = 2.8e1 + 6.775070969851e-3 * x**2 - 2.124527775283e-3 * x**3
        return min(28.0, y)
    if x <= 14.0:
        return (2.507355893131e1 + 9.754803562315e-1 * x
                - 1.016116352781e-1 * x**2 + 1.889794677828e-3 * x**3)
    if x <= 20.0:
        return (2.579601052357e1 + 8.206693007457e-1 * x
                - 9.055370274339e-2 * x**2 + 1.626510569859e-3 * x**3)
    if x <= 30.0:
        return (4.046435022819e1 - 1.379581654948e0 * x
                + 1.945884504128e-2 * x**2 - 2.070318932190e-4 * x**3)
    if x <= 40.0:
        return (1.792461334664e1 + 8.743920332081e-1 * x
                - 5.567361123058e-2 * x**2 + 6.277731764683e-4 * x**3)
    if x <= 54.0:
        y = (5.639011190988e1 - 2.010520359035e0 * x
             + 1.644919857549e-2 * x**2 + 2.674976141766e-5 * x**3)
        return max(0.0, y)
    return 0.0


def hill_surface(x_star: np.ndarray, domain_length: float = 9.0) -> np.ndarray:
    """Hill surface height (in H units) at each x_star (also in H units).

    Vectorised. Points beyond either hill (flat channel floor) return 0.
    """
    x_star = np.asarray(x_star, dtype=float)
    x_mm = x_star * _HILL_H_MM
    domain_mm = domain_length * _HILL_H_MM

    out = np.zeros_like(x_mm)
    left = x_mm <= _HILL_HALF_MM
    right = x_mm >= (domain_mm - _HILL_HALF_MM)

    # Left hill
    for i in np.nonzero(left)[0]:
        out[i] = _hill_left_mm(x_mm[i]) / _HILL_H_MM
    # Right hill (mirror)
    for i in np.nonzero(right)[0]:
        out[i] = _hill_left_mm(domain_mm - x_mm[i]) / _HILL_H_MM
    return out


def _structured_area_weights(coords: np.ndarray) -> np.ndarray:
    """Per-point cell area for a structured (x-level x y-level) grid.

    Each point's weight is dx(x) * dy(y), where the per-level widths use
    midpoint spacing. Works directly on the full grid; masking afterwards
    just drops weights for excluded points (their neighbours' weights are
    unaffected to first order).
    """
    x = coords[:, 0]
    y = coords[:, 1]

    xu, x_inv = np.unique(x, return_inverse=True)
    yu, y_inv = np.unique(y, return_inverse=True)

    def midpoint_widths(u: np.ndarray) -> np.ndarray:
        w = np.empty_like(u)
        if len(u) == 1:
            return np.ones_like(u)
        w[1:-1] = (u[2:] - u[:-2]) / 2.0
        w[0] = u[1] - u[0]
        w[-1] = u[-1] - u[-2]
        return w

    wx = midpoint_widths(xu)
    wy = midpoint_widths(yu)
    return wx[x_inv] * wy[y_inv]


def _wmean(vals: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(vals * w) / np.sum(w))


def _wstd(vals: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(vals, w)
    var = np.sum(w * (vals - m) ** 2) / np.sum(w)
    return float(np.sqrt(var))


class FieldErrorCalculator:
    """Area-weighted, hill-masked field error between simulation and DNS.

    Parameters
    ----------
    dns_coords : np.ndarray, shape (N, 2)
        DNS sample points (x, y) in H units.
    dns_fields : dict[str, np.ndarray]
        DNS reference values; each array has shape (N,).
    field_weights : dict[str, float], optional
        Per-field multiplicative weights. Default 1.0 for any field.
    mask_hill : bool
        If True (default is False), exclude DNS points below the analytic hill
        surface from all statistics in the periodic hills example.
    domain_length : float
        Streamwise domain length in H units (default 9.0).
    """

    def __init__(
        self,
        dns_coords: np.ndarray,
        dns_fields: dict[str, np.ndarray],
        field_weights: dict[str, float] | None = None,
        mask_hill: bool = False,
        domain_length: float = 9.0,
    ):
        self.dns_coords = dns_coords
        self.dns_fields = dns_fields
        self.field_weights = field_weights or {}
        self.domain_length = domain_length

        # Area weights on the full DNS grid.
        w = _structured_area_weights(dns_coords)

        # Geometric mask: keep points at or above the hill surface.
        if mask_hill:
            y_surf = hill_surface(dns_coords[:, 0], domain_length)
            # Small tolerance so points exactly on the crest aren't dropped.
            keep = dns_coords[:, 1] >= (y_surf - 1e-6)
        else:
            keep = np.ones(len(dns_coords), dtype=bool)

        self._mask = keep
        self._weights = w[keep]
        self._masked_coords = dns_coords[keep]

        # Pre-compute area-weighted std per DNS field on the masked grid.
        # For cp, gauge to area-weighted zero mean first so the datum
        # matches what we apply to the simulation at evaluation time.
        eps = 1e-8
        self._dns_masked: dict[str, np.ndarray] = {}
        self._dns_std: dict[str, float] = {}
        for name, vals in dns_fields.items():
            v = vals[keep]
            if name == "cp":
                v = v - _wmean(v, self._weights)   # area-weighted re-gauge
            self._dns_masked[name] = v
            s = _wstd(v, self._weights)
            self._dns_std[name] = s if s > eps else eps

    def calculate_error(
        self,
        sim_coords: np.ndarray,
        sim_fields: dict[str, np.ndarray],
        field_name: str = "cp",
    ) -> float:
        """Area-weighted normalized error for one field on the masked grid."""
        if field_name not in self.dns_fields or field_name not in sim_fields:
            raise KeyError(
                f"Field '{field_name}' must be present in both DNS and "
                "simulation fields."
            )

        dns_vals = self._dns_masked[field_name]

        # Interpolate the simulation onto the masked DNS points.
        sim_interp = griddata(
            sim_coords,
            sim_fields[field_name],
            self._masked_coords,
            method="linear",
        )

        valid = ~np.isnan(sim_interp)
        if not valid.any():
            raise ValueError(
                "Interpolation produced entirely NaN values. Check that the "
                "simulation and DNS coordinate systems match."
            )

        w = self._weights[valid]
        sim_v = sim_interp[valid]
        dns_v = dns_vals[valid]

        # For cp, re-gauge the interpolated simulation to area-weighted
        # zero mean on the SAME masked grid, so DNS and sim share a datum.
        if field_name == "cp":
            sim_v = sim_v - _wmean(sim_v, w)

        """Uncomment this part if you want to use the 1-norm"""
        # mae = _wmean(np.abs(dns_v - sim_v), w)
        # normalized = mae / self._dns_std[field_name]

        """Comment the lines to use another norm (weighted 2-norm is used bellow)"""
        # 1. Square the differences instead of taking the absolute value
        squared_diff = (dns_v - sim_v) ** 2
        
        # 2. Compute the weighted Mean Squared Error Weighted (MSEW)
        mse = _wmean(squared_diff, w)
        
        # 3. Take the square root to yield the 2-norm (RMSEW)
        normalized = np.sqrt(mse)/ self._dns_std[field_name]

        return float(self.field_weights.get(field_name, 1.0) * normalized)
