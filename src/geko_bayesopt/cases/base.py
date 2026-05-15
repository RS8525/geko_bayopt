"""
Base class for flow cases.

A ``FlowCase`` knows three things that vary between flow geometries:

1. How to set boundary conditions on a freshly-loaded mesh.
2. How to load the case's DNS reference data.
3. How to construct the ``CaseConfig`` and ``MeshConfig`` from the
   experiment JSON.

Subclasses override the abstract methods. Shared helpers (e.g. running a
single trial through the solver, packaging the result) live here so each
new case doesn't have to re-implement them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..fluent.case_config import CaseConfig
from ..fluent.mesh_config import MeshConfig
from ..fluent.extract import build_run_result
from ..types import RunResult


class FlowCase(ABC):
    """Base class. Subclass once per flow geometry.

    Subclasses must implement:
        - ``case_id`` (class attribute)
        - ``build_case_config(options)``
        - ``apply_boundary_conditions(solver, case_config)``
        - ``load_dns(dns_path)``

    Subclasses may also expose ``geometry_path`` after construction so
    the mesh generator knows where to read the CAD file from. The base
    class reads it from ``options["geometry_path"]`` if present; cases
    that need different handling can override.
    """

    #: Short identifier matching the ``case.kind`` field in the JSON config.
    case_id: str = ""

    def __init__(self, options: dict[str, Any], mesh_config: MeshConfig):
        self.options = options
        self.mesh_config = mesh_config
        self.case_config = self.build_case_config(options)

        # Explicit geometry path (separate from output directory).
        # Subclasses may override this in their build_case_config if they
        # derive it from other fields.
        gp = options.get("geometry_path")
        self.geometry_path: Path | None = Path(gp) if gp else None

    # ------------------------------------------------------------------ #
    # Abstract methods -- subclasses implement                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_case_config(self, options: dict[str, Any]) -> CaseConfig:
        """Translate the case-section JSON options into a ``CaseConfig``.

        Called once at construction. Subclasses know which option keys
        are required vs optional for their flow.
        """

    @abstractmethod
    def apply_boundary_conditions(self, solver) -> None:
        """Set boundary conditions on a live Fluent solver session.

        Called by the solver after mesh-load and before any iteration.
        For periodic hill this creates the periodic interface and
        applies mass-flow forcing; for forward-facing-step it would
        configure velocity-inlet and pressure-outlet.

        Parameters
        ----------
        solver : pyfluent solver session
            The live session whose boundary conditions need setting.
        """

    @abstractmethod
    def load_dns(
        self, dns_path: str | Path
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load DNS reference data for this case.

        Returns
        -------
        coords : np.ndarray, shape (M, 2)
            DNS coordinates in non-dimensional H units (x/H, y/H).
        fields : dict[str, np.ndarray]
            Field values at those coordinates, e.g.
            ``{"cp": ..., "Ux": ..., "Uy": ...}``. Field names must
            match the keys the loss function expects.
        """

    # ------------------------------------------------------------------ #
    # Shared helpers -- subclasses get these for free                    #
    # ------------------------------------------------------------------ #

    def build_run_result(
        self,
        *,
        run_id: str,
        parameters: dict[str, float],
        ascii_path: str | Path,
        cost_seconds: float = 0.0,
    ) -> RunResult:
        """Convert a Fluent ASCII export into a non-dimensional ``RunResult``.

        Uses the case's reference length and bulk velocity for scaling.
        Subclasses can override if they need a different convention.
        """
        return build_run_result(
            run_id=run_id,
            parameters=parameters,
            ascii_path=ascii_path,
            hill_height=self.case_config.hill_height,
            u_bulk=self.case_config.u_bulk,
            fluid_density=self.case_config.fluid_density,
            cost_seconds=cost_seconds,
        )

    def make_trial_case(self, parameters: dict[str, float]) -> CaseConfig:
        """Build a per-trial ``CaseConfig`` from base + GEKO overrides."""
        return replace(self.case_config, **parameters)
