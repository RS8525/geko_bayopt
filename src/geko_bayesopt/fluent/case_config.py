"""
Case configuration: physics, GEKO coefficients, zone names, solver settings.

Mesh parameters live separately in MeshConfig.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseConfig:
    """All physics/solver parameters for one periodic-hill simulation.

    Designed to be created once and used for both meshing (only the
    `geometry_basename` is needed there) and solving. Use ``dataclasses.replace``
    to create variants for parameter sweeps:

        from dataclasses import replace
        case_csep15 = replace(base, geko_csep=1.5)
    """

    # ---- Identity / files ---------------------------------------------------
    # Hill-shape parameter from the Laizet 2021 database.
    alpha: float = 1.0

    # ---- Geometry (in mesh units, mm by default) ----------------------------
    hill_height: float = 28.0   # H, the canonical reference length
    ly_over_h: float = 3.036    # domain height / hill height (2.024, 3.036, 4.048)

    # ---- Working fluid (incompressible, constant properties) ----------------
    fluid_density: float = 1.0       # [kg/m^3]
    fluid_viscosity: float = 1.0e-5  # [kg/(m s)]

    # ---- Reynolds number (Re_h = rho * U_b * H / mu) ------------------------
    re_h: int = 5600

    # ---- GEKO tunable coefficients (None -> Fluent default) -----------------
    # Defaults: Csep=1.75, Cnw=0.5, Cmix=0.0, Cjet=0.9, Ccorner=1.0
    geko_csep: float | None = None
    geko_cnw: float | None = None
    geko_cmix: float | None = None
    geko_cjet: float | None = None
    geko_ccorner: float | None = None

    # ---- Solver controls ----------------------------------------------------
    iter_count: int = 2000

    # ---- Zone names (must match the mesh) -----------------------------------
    zone_inlet: str = "inlet"
    zone_outlet: str = "outlet"
    zone_top: str = "wall"
    zone_bottom: str = "wall_lower"

    # ---- Derived quantities -------------------------------------------------
    # In case_config.py:
    @property
    def geometry_basename(self) -> str:
        return f"periodic_hill_2d_alpha_{self.alpha}"

    @property
    def u_bulk(self) -> float:
        """Bulk velocity at the hill crest, derived from Re_h."""
        return self.re_h * self.fluid_viscosity / (self.fluid_density * self.hill_height)

    @property
    def h_channel(self) -> float:
        """Wall-normal distance from the crest to the top of the domain."""
        return (self.ly_over_h - 1.0) * self.hill_height

    @property
    def target_mass_flow(self) -> float:
        """Target mass flow per unit depth for periodic forcing."""
        return self.fluid_density * self.u_bulk * self.h_channel

    @property
    def case_id(self) -> str:
        """Filename-safe identifier that disambiguates parameter sweeps.

        Includes only knobs that differ from defaults, so a baseline run with
        no GEKO overrides produces a short ID.
        """
        parts = [f"alpha{self.alpha}", f"Re{self.re_h}"]
        if self.geko_csep is not None:
            parts.append(f"Csep{self.geko_csep}")
        if self.geko_cnw is not None:
            parts.append(f"Cnw{self.geko_cnw}")
        if self.geko_cmix is not None:
            parts.append(f"Cmix{self.geko_cmix}")
        if self.geko_cjet is not None:
            parts.append(f"Cjet{self.geko_cjet}")
        if self.geko_ccorner is not None:
            parts.append(f"Ccorner{self.geko_ccorner}")
        return "_".join(parts)
