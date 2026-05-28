"""
Periodic-hill flow case (Laizet 2021 / Breuer 2009 canonical configuration).

This module defines:
    - ``PeriodicHillsCase``: the FlowCase implementation
    - The boundary conditions: streamwise periodic + mass-flow forcing
    - DNS loading from Laizet's ``mean_files.dat`` format
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...fluent.case_config import CaseConfig
from ..base import FlowCase


class PeriodicHillsCase(FlowCase):
    """2D periodic-hill at Re_h = 5600 (typical) using GEKO turbulence.

    Boundary conditions:
        - inlet + outlet: streamwise periodic interface (auto-detected
          translation vector)
        - top wall: no-slip
        - bottom wall (hill): no-slip
        - flow driven by a target mass-flow forcing through the periodic
          pair, with relaxation factor 0.5

    DNS data:
        Laizet 2021 ``mean_files.dat`` format. Already non-dimensional
        in H, U_b. Columns: x, y, u, v, w, p.
    """

    case_id = "periodic_hills"

    def build_case_config(self, options: dict[str, Any]) -> CaseConfig:
        """Construct a CaseConfig from the JSON ``case.options`` block.

        Required keys: hill_height, re_h.
        Optional: alpha, ly_over_h, fluid_density, fluid_viscosity,
        iter_count, zone_*.

        Note: ``geometry_basename`` is derived from ``alpha`` and is NOT
        accepted here (it's a computed property on CaseConfig).
        """
        return CaseConfig(
            alpha=options.get("alpha", 1.0),
            hill_height=options["hill_height"],
            ly_over_h=options.get("ly_over_h", 3.036),
            fluid_density=options.get("fluid_density", 1.0),
            fluid_viscosity=options.get("fluid_viscosity", 1.0e-5),
            re_h=options["re_h"],
            iter_count=options.get("iter_count", 2000),
            zone_inlet=options.get("zone_inlet", "inlet"),
            zone_outlet=options.get("zone_outlet", "outlet"),
            zone_top=options.get("zone_top", "wall"),
            zone_bottom=options.get("zone_bottom", "wall_lower"),
        )

    def apply_boundary_conditions(self, solver) -> None:
        """Create periodic interface + apply mass-flow forcing.

        Two TUI calls:

        1. ``create-periodic-interface`` converts the inlet/outlet pair
           into a translational periodic boundary with auto-computed
           translation vector.

        2. ``massflow-rate-specification`` sets the target mass flow,
           initial pressure-gradient guess, relaxation factor, and
           flow direction.

        Uses raw TUI strings rather than the structured API because
        these command paths have moved between Fluent versions and
        TUI is the most stable interface.
        """
        cc = self.case_config

        solver.execute_tui(
            "/mesh/modify-zones/create-periodic-interface "
            "auto "                  # creation method (auto/conformal/non-conformal)
            f"{cc.case_id} "         # interface name (unique per run)
            f"{cc.zone_inlet} "
            f"{cc.zone_outlet} "
            "no "                    # rotational? no = translational
            "yes "                   # auto-compute offset
            "yes "                   # create periodic zones
        )

        solver.execute_tui(
            "/define/periodic-conditions/massflow-rate-specification? "
            f"{cc.target_mass_flow} "  # mass flow rate
            "1 "                       # initial pressure-gradient guess
            "0.5 "                     # relaxation factor
            "1 "                       # flow direction x
            "0 "                       # flow direction y
        )

    def load_dns(
        self, dns_path: str | Path
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load Laizet ``mean_files.dat`` for this case.

        File format: whitespace-separated, columns (x, y, u, v, w, p).
        Already in non-dimensional H, U_b units.

        The cp gauge convention here matches the existing utility loader
        (``cp = p - p[-1]``). This is grid-order dependent and should
        be revisited for rigorous comparison.
        """
        dns_path = Path(dns_path)
        if not dns_path.is_file():
            raise FileNotFoundError(
                f"DNS file not found: {dns_path}\n"
                "Expected Laizet 2021 'mean_files.dat' format."
            )

        data = np.genfromtxt(dns_path, dtype=float)
        coords = data[1:, 0:2]                  # x, y
        u = data[1:, 2]                        # u
        v = data[1:, 3]                        # v
        # data[:, 4] is w (spanwise), unused in 2D RANS comparison
        p = data[1:, 5]                        # p

        cp = p - p[-1]                        # match existing convention

        fields = {
            "Ux": u,
            "Uy": v,
            "p": p,
            "cp": cp,
        }
        return coords, fields
