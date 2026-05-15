"""
Shared data types passed between components.

The central type is ``RunResult``: produced by the Fluent extractor,
consumed by loss functions. Every component that handles simulation
output speaks this currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunResult:
    """One simulation run, normalized to non-dimensional units for DNS comparison.

    Coordinates are in units of the case's reference length (H for periodic
    hill). Velocity fields are in units of the case's reference velocity
    (U_bulk for periodic hill). Pressure is in units of rho * U_bulk^2.

    Attributes
    ----------
    run_id
        Unique identifier for this run, used in filenames and the store.
        Typically derived from case.case_id (e.g. "alpha1.0_Re5600_Csep1.85").
    parameters
        The optimizer-controlled parameters that produced this run, e.g.
        ``{"geko_csep": 1.85, "geko_cnw": 0.6}``.
    grid_coords
        Shape (N, 2). Columns are (x, y) in non-dimensional H units.
    fields
        Mapping from field name to numpy array of shape (N,).
        Standard keys: "Ux", "Uy", "p", "cp". Cases may add their own.
    converged
        True if the solver reported convergence at the residual targets.
        Currently always True since we iterate to a fixed count; reserved
        for future criteria.
    cost_seconds
        Wall-clock time for the trial. Used in metadata, not in the loss.
    ascii_path
        Optional path to the raw Fluent ASCII export, kept for debugging.
        Not part of the data contract — consumers should not depend on it.
    """

    run_id: str
    parameters: dict[str, float]
    grid_coords: np.ndarray
    fields: dict[str, np.ndarray]
    converged: bool = True
    cost_seconds: float = 0.0
    ascii_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)
