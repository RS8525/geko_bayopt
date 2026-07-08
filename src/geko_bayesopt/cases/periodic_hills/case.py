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
import pandas as pd

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
        """Load periodic-hill DNS reference data.

        Two formats are supported, auto-detected from the first line:

        * Combined CSV (``periodic_hill_dns.csv``, produced by
          ``fetch_periodic_hill_dns.py`` / ``build_periodic_hill_dns.py``)
          with a named header:
          ``x-coordinate, y-coordinate, x-velocity, y-velocity, TKE, pressure``.
          Exposes ``turb-kinetic-energy`` (= TKE) so turbulent kinetic energy
          can be used as a calibration field. This is the format used by the
          29-case Re=5600 study.

        * Legacy columnwise ASCII (numeric, single header line) with columns
          ``nodenumber, x, y, Ux, Uy, pressure, density, production-of-k,
          dissipation, prod-over-diss, k, shear`` (used by the Re=2800 study).

        All quantities are in non-dimensional H, U_b units.
        """
        dns_path = Path(dns_path)
        if not dns_path.is_file():
            raise FileNotFoundError(f"DNS file not found: {dns_path}")

        # Detect whether the file is a combined CSV (comma-separated header)
        # or the legacy space-separated columnwise ASCII. Use file suffix or
        # presence of a comma on the header line as the signal.
        with open(dns_path, encoding="utf-8", errors="ignore") as fh:
            first_line = fh.readline()
        if dns_path.suffix.lower() == ".csv" or "," in first_line:
            return self._load_combined_csv(dns_path)
        return self._load_columnwise_ascii(dns_path)

    @staticmethod
    def _load_combined_csv(
        dns_path: Path,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read the combined CSV (x, y, u, v, TKE, p) by column name."""
        df = pd.read_csv(dns_path)
        df.columns = [c.strip() for c in df.columns]

        # allow case-insensitive column names
        colmap = {c.lower(): c for c in df.columns}

        def choose(*names: str) -> str:
            for n in names:
                key = n.lower()
                if key in colmap:
                    return colmap[key]
            raise KeyError(f"None of columns found: {names}")

        xcol = choose("x-coordinate", "x")
        ycol = choose("y-coordinate", "y")
        uxcol = choose("x-velocity", "x-vel", "ux", "u")
        uycol = choose("y-velocity", "y-vel", "uy", "v")
        pcol = choose("pressure", "p")

        coords = df[[xcol, ycol]].to_numpy()
        p = df[pcol].to_numpy()
        fields = {
            "Ux": df[uxcol].to_numpy(),
            "Uy": df[uycol].to_numpy(),
            "p": p,
            "cp": p,
        }

        # optional TKE column
        if "tke" in colmap:
            fields["turb-kinetic-energy"] = df[colmap["tke"]].to_numpy()
        elif "k" in colmap:
            fields["turb-kinetic-energy"] = df[colmap["k"]].to_numpy()

        return coords, fields

    def _load_columnwise_ascii(
        self, dns_path: Path
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Legacy loader for the 12-column columnwise-organized ASCII.

        Columns: nodenumber, x, y, Ux, Uy, pressure, density,
        production-of-k, dissipation, prod-over-diss, k, shear.
        """
        data = np.genfromtxt(dns_path, dtype=float, skip_header=1)
        coords = data[:, 1:3]
        fields = {
            "Ux": data[:, 3],
            "Uy": data[:, 4],
            "turb-kinetic-energy": data[:, 9],
            "production-of-k": data[:, 6],
            "turb-diss-rate": data[:, 7],
        }
        return coords, fields
