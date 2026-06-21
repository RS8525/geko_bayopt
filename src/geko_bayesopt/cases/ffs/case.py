"""
2D Forward-Facing Step flow case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

import numpy as np

from ...fluent.case_config import CaseConfig
from ..base import FlowCase


class ForwardFacingStepCase(FlowCase):
    """2D Forward-Facing Step using GEKO turbulence.

    Boundary conditions:
        - inlet: velocity-inlet
        - outlet: pressure-outlet
        - top: symmetry
        - bottom: wall
    """

    case_id = "ffs"

    def build_case_config(self, options: dict[str, Any]) -> CaseConfig:
        """Construct a CaseConfig from the JSON options block.

        Required keys: step_height
        """
        return CaseConfig(
            base_case_name="ffs",
            
            # Using 'step_height' as the reference length for generic scaling
            hill_height=options["step_height"],
            re_h=options.get("re_h", 0),  # Required by CaseConfig, might be 0 if unspecified for FFS
            fluid_density=options.get("fluid_density", 854.5648),
            fluid_viscosity=options.get("fluid_viscosity", 0.01424275),
            
            # FFS specific configurations
            inlet_velocity=options.get("inlet_velocity", 10.0),
            turb_intensity=options.get("turb_intensity", 3.25),
            turb_viscosity_ratio=options.get("turb_viscosity_ratio", 12.0),
            
            iter_count=options.get("iter_count", 2000),
            zone_inlet=options.get("zone_inlet", "inlet"),
            zone_outlet=options.get("zone_outlet", "outlet"),
            zone_top=options.get("zone_top", "ceiling"),
            zone_bottom=options.get("zone_bottom", "step"),
        )

    def apply_boundary_conditions(self, solver) -> None:
        """Apply Velocity Inlet and Pressure Outlet BCs."""
        cc = self.case_config

        # 1. Velocity Inlet
        # Set velocity magnitude, turbulent specification method (Intensity and Viscosity Ratio)
        #solver.execute_tui(
            # Use Profile for gauge total pressure? (dependent on Fluent version prompts) -- wait, let's keep it simple and sequence-agnostic if possible, or adapt to the known sequence.
            # TUI prompts for velocity-inlet usually go:
            # Velocity Specification Method: Magnitude, Normal to Boundary
            # Reference Frame: Absolute
            # Velocity Magnitude: x
            # Supersonic/Initial Gauge Pressure: y
            # Turbulent Specification Method: Intensity and Viscosity Ratio
            # Turbulent Intensity: z
            # Turbulent Viscosity Ratio: w
        
        
        # NOTE: A more robust way to set velocity inlet in TUI when prompts vary is to provide exactly what it asks, BUT often using the structured `solver.setup...` API is safer for standard BCs. For now, since the periodic case used TUI, let's do the standard TUI string for velocity inlet carefully. 
        # A simpler TUI sequence for setting just the magnitude and turbulence:
        #solver.execute_tui(f"/define/boundary-conditions/velocity-inlet {cc.zone_inlet} no no yes no {cc.inlet_velocity} no yes no yes intensity-and-viscosity-ratio {cc.turb_intensity} {cc.turb_viscosity_ratio}")

        
        solver.execute_tui(f"/define/boundary-conditions/velocity-inlet {cc.zone_inlet} "
            f"yes "  # Velocity Specification Method: Magnitude and Direction
            f"yes "  #Reference Frame: Absolute
            f"no "   # Use Profile for Velocity Magnitude?
            f"{cc.inlet_velocity} "
            f"yes "  #Use Profile for Supersonic/Initial Gauge Pressure?
            f"no "   #Use UDF Profile for Supersonic/Initial Gauge Pressure?
            f"yes "  #Use Profile for X-Component of Flow Direction?
            f"no "   #Use UDF Profile for X-Component of Flow Direction?
            f"yes "  #Use Profile for Y-Component of Flow Direction?
            f"no "   #Use UDF Profile for Y-Component of Flow Direction?
            f"no "   #Turbulence Specification Method: K and Omega
            f"no "   #Turbulence Specification Method: Intensity and Length Scale
            f"yes "  #Turbulence Specification Method: Intensity and Viscosity Ratio
            f"{cc.turb_intensity} " #Turbulent Intensity
            f"{cc.turb_viscosity_ratio} ") #Turbulent Viscosity Ratio
                            


        # 2. Pressure Outlet
        # Gauge pressure = 0, with backflow turbulence definitions
        solver.execute_tui(
            f"/define/boundary-conditions/pressure-outlet {cc.zone_outlet} "
            f"yes "  #Backflow Reference Frame: Absolute
            f"no "   #Use Profile for Gauge Pressure?
            f"0 "    #Gauge Pressure (in Pa)
            f"yes "  #Backflow Direction Specification Method: Direction Vector
            f"yes "  #Coordinate System: Cartesian (X, Y, Z)
            f"yes "  #Use Profile for X-Component of Flow Direction?
            f"no "   #Use UDF Profile for X-Component of Flow Direction?
            f"yes "  #Use Profile for Y-Component of Flow Direction?
            f"no "   #Use UDF Profile for Y-Component of Flow Direction?
            f"yes "  #Turbulence Specification Method: K and Omega
            f"yes "  #Use Profile for Backflow Turbulent Kinetic Energy?
            f"no "   #Use UDF Profile for Backflow Turbulent Kinetic Energy?
            f"yes "  #Use Profile for Backflow Specific Dissipation Rate?
            f"no "   #Use UDF Profile for Backflow Specific Dissipation Rate?
            f"yes "  #Backflow Pressure Specification: Total Pressure
            f"no "   #Backflow Pressure Specification: Static Pressure
            f"yes "  #Average Pressure Specification?
            f"yes "  #Specify targeted mass flow rate
            f"yes "  #Use Profile for Targeted mass flow?
            f"no "   #Use UDF Profile for Targeted mass flow?
            f"yes "  #Use Profile for Upper Limit of Absolute Pressure Value?
            f"no "   #Use UDF Profile for Upper Limit of Absolute Pressure Value?
            f"yes "  #Use Profile for Lower Limit of Absolute Pressure Value?
            f"no "   #Use UDF Profile for Lower Limit of Absolute Pressure Value?
        )


    def load_dns(self, dns_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load FFS DNS reference data.

        File format: CSV with header.
        Columns used: x-coordinate, y-coordinate, mean-x-velocity, mean-y-velocity, mean-pressure.
        Currently loads dimensional data as-is to match extract.py setup.
        """

        
        dns_path = Path(dns_path)
        if not dns_path.is_file():
            raise FileNotFoundError(f"DNS file not found: {dns_path}")

        df = pd.read_csv(dns_path)

        x = df["x-coordinate"].to_numpy()
        y = df["y-coordinate"].to_numpy()
        coords = np.column_stack([x, y])

        u = df["mean-x-velocity"].to_numpy()
        v = df["mean-y-velocity"].to_numpy()
        p = df["mean-pressure"].to_numpy()

        fields = {
            "Ux": u,
            "Uy": v,
            "p": p,
            "cp": p,  # Aligning with current extract.py behavior
        }
        
        return coords, fields
