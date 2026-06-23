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
    length_ref: float,
    velocity_ref: float,
    fluid_density: float,
    cp_reference_index: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a Fluent ASCII export and rescale to non-dimensional units.

    Returns ``(coords, fields)`` matching the DNS loader's convention:
        coords: (N, 2), columns (x/L_ref, y/L_ref)
        fields: {"Ux": Ux/U_ref, "Uy": Uy/U_ref, "p": p/(rho U_ref^2), "cp": cp}
                plus any additional Fluent fields present (k, diss, ...)
                also non-dimensionalized.

    The reference scales are supplied by the caller (the flow case), not
    hardcoded here, because the correct velocity scale is case-dependent:
    periodic hills uses the Re-derived bulk velocity, whereas a
    velocity-inlet case (e.g. FFS) uses its inlet velocity. Keeping the
    convention in the flow case is what makes ``RunResult`` directly
    comparable to the (already non-dimensional) DNS reference, regardless
    of what physical units the mesh/CAD happen to be in.

    Parameters
    ----------
    ascii_path : str or Path
        Path to the Fluent ASCII export.
    length_ref : float
        Reference length L_ref (e.g. hill/step height), in the same units
        as the Fluent export. Coordinates are divided by this.
    velocity_ref : float
        Reference velocity U_ref, in the same units as the Fluent export.
        Velocities are divided by this; pressure by ``rho * U_ref^2``.
    fluid_density : float
        Fluid density rho, used for non-dimensionalizing pressure.
    cp_reference_index : int, optional
        Index of a reference point used to gauge cp. If None (default),
        cp is left equal to the non-dimensional pressure; the field-error
        calculator re-gauges cp to an area-weighted zero mean on both DNS
        and sim before comparing, so the absolute datum cancels.
        WARNING: an explicit index is grid-order dependent.
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

    # Non-dimensionalize coordinates: x/L_ref, y/L_ref
    x = df["x-coordinate"].to_numpy() / length_ref
    y = df["y-coordinate"].to_numpy() / length_ref
    coords = np.column_stack([x, y])

    # Non-dimensionalize velocity: u/U_ref, v/U_ref
    fields: dict[str, np.ndarray] = {
        "Ux": df["x-velocity"].to_numpy() / velocity_ref,
        "Uy": df["y-velocity"].to_numpy() / velocity_ref,
    }

    # Non-dimensionalize pressure: p / (rho U_ref^2)
    p_scale = fluid_density * velocity_ref * velocity_ref
    fields["p"] = df["pressure"].to_numpy() / p_scale

    # Pressure coefficient. The field-error calculator re-gauges cp to an
    # area-weighted zero mean on both DNS and sim before comparing, so the
    # absolute datum cancels and the comparison is order-independent. By
    # default cp == non-dimensional p; an explicit reference index gauges
    # to a specific point instead.
    if cp_reference_index is not None:
        fields["cp"] = fields["p"] - fields["p"][cp_reference_index]
    else:
        fields["cp"] = fields["p"]

    # Optional fields, also non-dimensionalized where physically meaningful.
    if "turb-kinetic-energy" in df.columns:
        # Turbulent kinetic energy has units of m^2/s^2
        fields["turb-kinetic-energy"] = df["turb-kinetic-energy"].to_numpy() / (velocity_ref * velocity_ref)
    if "production-of-k" in df.columns:
        # Production of k has units of k/time = m^2/s^3
        fields["production-of-k"] = df["production-of-k"].to_numpy() * (length_ref / (velocity_ref * velocity_ref * velocity_ref))
    if "k" in df.columns:
        fields["k"] = df["k"].to_numpy() / (velocity_ref * velocity_ref)
    if "turb-diss-rate" in df.columns:
        # Specific dissipation has units 1/time. Non-dim by L_ref/U_ref.
        fields["diss"] = df["turb-diss-rate"].to_numpy() * (length_ref / velocity_ref**3)
    if "vorticity-mag" in df.columns:
        fields["vor"] = df["vorticity-mag"].to_numpy() * (length_ref / velocity_ref)
    if "wall-shear-stress" in df.columns:
        fields["wall_shear_stress"] = df["wall-shear-stress"].to_numpy() / p_scale
    if "turb-diss-rate" in df.columns:
        # Turbulent dissipation rate has units of m^2/s^3
        fields["turb-diss-rate"] = df["turb-diss-rate"].to_numpy() * (length_ref / (velocity_ref * velocity_ref * velocity_ref))

    return coords, fields


def build_run_result(
    *,
    run_id: str,
    parameters: dict[str, float],
    ascii_path: str | Path,
    length_ref: float,
    velocity_ref: float,
    fluid_density: float,
    cost_seconds: float = 0.0,
    converged: bool = True,
    cp_reference_index: int | None = None,
) -> RunResult:
    """Convenience wrapper: parse an ASCII file and return a ``RunResult``."""
    coords, fields = parse_fluent_ascii(
        ascii_path,
        length_ref=length_ref,
        velocity_ref=velocity_ref,
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
