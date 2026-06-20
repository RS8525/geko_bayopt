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

        data = np.genfromtxt(dns_path, dtype=float,skip_header=1)
        coords = data[:, 1:3]                  # x, y
        u = data[:, 6]                        # u
        v = data[:, 7]                        # v
        # data[:, 5] is w (spanwise), unused in 2D RANS comparison
        p = data[:, 8]                        # p
        cp = p - p[-1]                        # match existing convention
        k = data[:, 5] 
        prod_k = data[:, 4]

  

        fields = {
            "Ux": u,
            "Uy": v,
            "p": p,
            "cp": cp,
            "turb-kinetic-energy": k,
            "production-of-k": prod_k
        }
        return coords, fields


#FLUENT UNITS
# quantity                             units                            factor  offset
# -----------------------------------  -------------------------  ------------  ------
# acceleration                         m/s^2                                 1        
# angle                                deg                         0.017453292        
# angular-velocity                     rad/s                                 1        
# area                                 m^2                                   1        
# area-inverse                         m^-2                                  1        
# area-per-time                        m^2/s                                 1        
# area-temperature                     m^2 K                                 1        
# collision-rate                       m^-3 s^-1                             1        
# nucleation-rate                      m^-3 s^-1                             1        
# concentration                        kmol/m^3                              1        
# contact-resistance                   (m^2 K)/W                             1        
# contact-resistance-vol               Ohm m^3                               1        
# crank-angle                          deg                                   1        
# crank-angular-velocity               rev/min                               1        
# current                              A                                     1        
# current-density                      A/m^2                                 1        
# current-vol-density                  A/m^3                                 1        
# density                              kg/m^3                                1        
# density-gradient                     kg/m^4                                1        
# density-inverse                      m^3/kg                                1        
# density*specific-energy              J/m^3                                 1        
# density*specific-heat                J/(m^3 K)                             1        
# density*time                         (kg s)/m^3                            1        
# density*velocity                     kg/(m^2 s)                            1        
# depth                                m                                     1        
# elec-capacity                        Ah                                    1        
# elec-charge                          A h                                3600        
# elec-charge-density                  (A s)/m^3                             1        
# elec-charge-flux-density             Ah/m^2                                1        
# elec-concentration                   mol/m^3                               1        
# elec-conductivity                    S/m                                   1        
# elec-contact-resistance              Ohm m^2                               1        
# elec-field                           V/m                                   1        
# elec-permittivity                    farad/m                               1        
# elec-resistivity                     Ohm m                                 1        
# elec-resistance                      Ohm                                   1        
# energy                               J                                     1        
# energy-density                       J/m2                                  1        
# force                                N                                     1        
# force-per-area                       N/m^2                                 1        
# force-per-volume                     N/m^3                                 1        
# force*time-per-volume                (N s)/m^3                             1        
# frequency                            Hz                                    1        
# gas-constant                         J/(kg K)                              1        
# heat-flux                            W/m^2                                 1        
# heat-flux-resolved                   (m K)/s                               1        
# heat-generation-rate                 W/m^3                                 1        
# heat-transfer-coefficient            W/(m^2 K)                             1        
# ignition-energy                      J/mol                                 1        
# kinematic-viscosity                  m^2/s                                 1        
# length                               m                                     1        
# length-inverse                       m^-1                                  1        
# length-temperature                   m K                                   1        
# length-time-inverse                  m^-1 s^-1                             1        
# length4-per-time                     m^4/s                                 1        
# mag-permeability                     H/m                                   1        
# mass                                 kg                                    1        
# mass-diffusivity                     m^2/s                                 1        
# mass-flow                            kg/s                                  1        
# mass-flow-per-depth                  kg/(m s)                              1        
# mass-flow-per-time                   kg/s^2                                1        
# mass-flux                            kg/(m^2 s)                            1        
# mass-transfer-rate                   kg/(m^3 s)                            1        
# mole-transfer-rate                   kgmol/(m^3 s)                         1        
# surface-mole-transfer-rate           kgmol/(m^2 s)                         1        
# mole-specific-energy                 J/kgmol                               1        
# mole-specific-entropy                J/(kgmol K)                           1        
# molec-wt                             kg/kmol                               1        
# moment                               N m                                   1        
# moment-of-inertia                    kg m^2                                1        
# no-unit                                                                    1        
# number-density                       m^-3                                  1        
# particles-conc                       1.e15-particles/kg                    1        
# particles-rate                       1.e15 m^-3 s^-1                       1        
# percentage                           %                                  0.01        
# power                                W                                     1        
# power-per-time                       W/s                                   1        
# pressure                             Pa                                    1        
# mole-con-henry-const                 (Pa m^3)/kgmol                        1        
# pressure-gradient                    Pa/m                                  1        
# pressure-time-derivative             Pa/s                                  1        
# pressure-time-deriv-sqr              Pa^2/s^2                              1        
# pressure-2nd-time-derivative         Pa/s^2                                1        
# resistance                           m^-1                                  1        
# site-density                         kgmol/m^2                             1        
# soot-formation-constant-unit         kg/(N m s)                            1        
# soot-limiting-nuclei-rate            1e+15-particles/m3-s                  1        
# soot-linear-termination              m^3/s                                 1        
# soot-oxidation-constant              kg m kgmol^-1 K^-0.5 s^-1             1        
# soot-pre-exponential-constant        1.e15 kg^-1 s^-1                      1        
# soot-surface-growth-scale-factor     (kg m)/(kgmol s)                      1        
# soot-sitespecies-concentration       kmol/m^3                              1        
# source-elliptic-relaxation-function  kg/(m^3 s^2)                          1        
# source-energy                        W/m^3                                 1        
# source-kinetic-energy                kg/(m s^3)                            1        
# source-mass                          kg/(m^3 s)                            1        
# source-momentum                      N/m^3                                 1        
# source-specific-dissipation-rate     kg/(m^3 s^2)                          1        
# source-temperature-variance          K^2/(m^3 s)                           1        
# source-turbulent-dissipation-rate    kg/(m s^4)                            1        
# source-turbulent-viscosity           kg/(m s^2)                            1        
# specific-area                        m^2/kg                                1        
# specific-energy                      J/kg                                  1        
# specific-heat                        J/(kg K)                              1        
# spring-constant                      N/m                                   1        
# spring-constant-angular              (N m)/rad                             1        
# stefan-boltzmann-constant            W/(m^2 K^4)                           1        
# surface-density                      kg/m^2                                1        
# surface-tension                      N/m                                   1        
# surface-tension-gradient             N/(m K)                               1        
# temperature                          K                                     1        
# temperature-difference               K                                     1        
# temperature-gradient                 K/m                                   1        
# temperature-inverse                  K^-1                                  1        
# temperature-variance                 K^2                                   1        
# thermal-conductivity                 W/(m K)                               1        
# thermal-resistivity                  (m K)/W                               1        
# thermal-resistance                   (m^2 K)/W                             1        
# thermophoretic-diffusivity           (kg m^2)/s^2                          1        
# time                                 s                                     1        
# time-inverse                         s^-1                                  1        
# time-inverse-squared                 s^-2                                  1        
# time-inverse-cubed                   s^-3                                  1        
# turb-kinetic-energy-production       kg/(m s^3)                            1        
# turbulent-energy-diss-rate           m^2/s^3                               1        
# turbulent-energy-diss-rate-gradient  m/s^3                                 1        
# turbulent-kinetic-energy             m^2/s^2                               1        
# turbulent-kinetic-energy-gradient    m/s^2                                 1        
# univ-gas-constant                    J/(K kgmol)                           1        
# velocity                             m/s                                   1        
# viscosity                            kg/(m s)                              1        
# viscosity-consistency-index          kg s^n-2 m^-1                         1        
# volume                               m^3                                   1        
# volume-flow-rate                     m^3/s                                 1        
# volume-flow-rate-per-depth           m^3/(s m)                             1        
# volume-inverse                       m^-3                                  1        
# volume-temperature                   m^3/K                                 1        
# voltage                              V                                     1        
# wave-length                          Angstrom                              1        
# youngs-modulus                       N/m^2                                 1        

