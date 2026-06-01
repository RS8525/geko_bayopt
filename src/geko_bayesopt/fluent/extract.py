"""
Parse Fluent ASCII exports into the package's ``RunResult`` data contract.

The Fluent ASCII export is whitespace-separated, with one header line and
one row per node:

    nodenumber     x-coordinate     y-coordinate       x-velocity       y-velocity         pressure
             1  0.000000000E+00  2.800090000E-02  1.120101077E-03 -6.660770135E-07 -1.127335036E-01
             ...

Coordinates and velocities are rescaled here to non-dimensional H / U_bulk
units so the loss function can compare directly against DNS data without
the experiment loop having to know about unit systems.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..types import RunResult


# Column names as they appear in Fluent's ASCII header. Keep in sync with
# ``solver.PeriodicHillSolver.EXPORT_VARIABLES``.
_FLUENT_COLUMNS = [
    "nodenumber",
    "x-coordinate",
    "y-coordinate",
    "x-velocity",
    "y-velocity",
    "pressure",
    "turb-kinetic-energy", #k
]
#     "k",
#     "omega",
#     "vorticity-mag",
#     "wall-shear-stress",
# ]


def parse_fluent_ascii(
    ascii_path: str | Path,
    *,
    hill_height: float,
    u_bulk: float,
    fluid_density: float,
    cp_reference_index: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a Fluent ASCII export and rescale to non-dimensional units.

    Returns ``(coords, fields)`` matching the DNS loader's convention:
        coords: (N, 2), columns (x/H, y/H)
        fields: {"Ux": Ux/U_b, "Uy": Uy/U_b, "p": p/(rho U_b^2), "cp": cp}
                plus any additional Fluent fields present (k, omega, ...)
                also non-dimensionalized.

    Parameters
    ----------
    ascii_path : str or Path
        Path to the Fluent ASCII export.
    hill_height : float
        Reference length H, in the same units as the Fluent export.
    u_bulk : float
        Reference velocity U_b, in the same units as the Fluent export.
    fluid_density : float
        Fluid density rho, used for non-dimensionalizing pressure.
    cp_reference_index : int, optional
        Index of the reference point used to gauge cp. If None (default),
        uses the same convention as the existing DNS loader (last row).
        WARNING: this is grid-order dependent. For rigorous DNS comparison,
        pass an explicit index matching the DNS gauge point.
    """
    df = pd.read_csv(
        ascii_path,
        sep=r"\s+",
        engine="python",
        skipinitialspace=True,
    )

    # Strip whitespace from column names — Fluent's header has extra spaces.
    df.columns = [c.strip() for c in df.columns]

    # Drop the nodenumber column — it's grid-internal and not useful.
    if "nodenumber" in df.columns:
        df = df.drop(columns=["nodenumber"])

    # Non-dimensionalize coordinates: x/H, y/H
    # x = df["x-coordinate"].to_numpy() / hill_height
    x = df["x-coordinate"].to_numpy()
    # y = df["y-coordinate"].to_numpy() / hill_height
    y = df["y-coordinate"].to_numpy()
    coords = np.column_stack([x, y])

    # Non-dimensionalize velocity: u/U_b, v/U_b
    # fields: dict[str, np.ndarray] = {
    #     "Ux": df["x-velocity"].to_numpy() / u_bulk,
    #     "Uy": df["y-velocity"].to_numpy() / u_bulk,
    # }
    fields: dict[str, np.ndarray] = {
        "Ux": df["x-velocity"].to_numpy(),
        "Uy": df["y-velocity"].to_numpy(),
    }
    # CHANGE WHEN FINISHED WITH THE TEST CASE
    # Non-dimensionalize pressure: p/(rho U_b^2)
    p_dim = df["pressure"].to_numpy()
    p_scale = fluid_density * u_bulk * u_bulk
    # fields["p"] = p_dim / p_scale
    fields["p"] = p_dim

    # Pressure coefficient. Reference convention should match the DNS
    # loader so MSE between them is meaningful. The existing DNS loader
    # uses ``p - p[-1]``; we match that by default but allow override.
    ref_idx = cp_reference_index if cp_reference_index is not None else -1
    fields["cp"] = fields["p"] - fields["p"][ref_idx]

    # Optional fields, also non-dimensionalized where physically meaningful.
    if "turb-kinetic-energy" in df.columns:
        fields["turb-kinetic-energy"] = df["turb-kinetic-energy"].to_numpy() / (u_bulk * u_bulk)
    if "omega" in df.columns:
        # Omega has units of 1/time. Non-dim by H/U_b: omega * H / U_b.
        fields["omega"] = df["omega"].to_numpy() * (hill_height / u_bulk)
    if "vorticity-mag" in df.columns:
        fields["vorticity_mag"] = df["vorticity-mag"].to_numpy() * (hill_height / u_bulk)
    if "wall-shear-stress" in df.columns:
        fields["wall_shear_stress"] = df["wall-shear-stress"].to_numpy() / p_scale
    if "production-of-k" in df.columns:
        # k production has units of energy/time. Non-dim by (U_b^3 / H): prod_k * H / (U_b^3).
        fields["production-of-k"] = df["production-of-k"].to_numpy() * (hill_height / (u_bulk * u_bulk * u_bulk))

    return coords, fields


def build_run_result(
    *,
    run_id: str,
    parameters: dict[str, float],
    ascii_path: str | Path,
    hill_height: float,
    u_bulk: float,
    fluid_density: float,
    cost_seconds: float = 0.0,
    converged: bool = True,
    cp_reference_index: int | None = None,
) -> RunResult:
    """Convenience wrapper: parse an ASCII file and return a ``RunResult``."""
    coords, fields = parse_fluent_ascii(
        ascii_path,
        hill_height=hill_height,
        u_bulk=u_bulk,
        fluid_density=fluid_density,
        cp_reference_index=cp_reference_index,
    )
    return RunResult(
        run_id=run_id,
        parameters=parameters,
        grid_coords=coords,
        fields=fields,
        converged=converged,
        cost_seconds=cost_seconds,
        ascii_path=Path(ascii_path),
    )
