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
    "x-velocity",
    "y-velocity",
    "pressure",
    "turb-kinetic-energy", #k
    "production-of-k",
    "turb-diss-rate", #epsilon
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

    # Pressure coefficient. Reference convention MUST match the DNS
    # loader so the field error between them is meaningful. Both now use
    # a mean-zero gauge (cp = p - mean(p)), which is order-independent
    # and references both fields to their own domain mean -- unlike
    # ``p - p[-1]``, where DNS and sim have different last points and the
    # comparison picks up a spurious constant offset.
    if cp_reference_index is not None:
        # Explicit override: gauge to a specific point.
        fields["cp"] = fields["p"] - fields["p"][cp_reference_index]
    else:
        fields["cp"] = fields["p"]

    # Pressure coefficient. Reference convention should match the DNS
    # loader so MSE between them is meaningful. The existing DNS loader
    # uses ``p - p[-1]``; we match that by default but allow override.
    # ref_idx = cp_reference_index if cp_reference_index is not None else -1
    # fields["cp"] = fields["p"] - fields["p"][ref_idx]

    # Optional fields, also non-dimensionalized where physically meaningful.
    if "turb-kinetic-energy" in df.columns:
        # Turbulent kinetic energy has units of m^2/s^2
        fields["turb-kinetic-energy"] = df["turb-kinetic-energy"].to_numpy() / (u_bulk * u_bulk)
    if "production-of-k" in df.columns:
        # Production of k has units of k/time = m^2/s^3
        fields["production-of-k"] = df["production-of-k"].to_numpy() * (hill_height / (u_bulk * u_bulk * u_bulk))
    if "turb-diss-rate" in df.columns:
        # Omega has units of 1/time. Non-dim by H/U_b: omega * H / U_b.
        fields["diss"] = df["turb-diss-rate"].to_numpy() * (hill_height / u_bulk)
    if "vorticity-mag" in df.columns:
        fields["vor"] = df["vorticity-mag"].to_numpy() * (hill_height / u_bulk)
    if "wall-shear-stress" in df.columns:
        fields["wall_shear_stress"] = df["wall-shear-stress"].to_numpy() / p_scale
    if "turb-diss-rate" in df.columns:
        # Turbulent dissipation rate has units of m^2/s^3
        fields["turb-diss-rate"] = df["turb-diss-rate"].to_numpy() * (hill_height / (u_bulk * u_bulk * u_bulk))

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



#table of fluent units, might be useful for now, delete later
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
