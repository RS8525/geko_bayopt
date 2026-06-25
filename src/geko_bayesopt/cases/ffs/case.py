"""
2D Forward-Facing Step flow case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

import numpy as np

from ...fluent.case_config import CaseConfig
from ...types import RunResult
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

    _DEFAULT_DNS_COLUMNS = {
        "Ux": "mean-x-velocity",
        "Uy": "mean-y-velocity",
        "p": "mean-pressure",
        "total-turbulent-kinetic-energy": "total-turbulent-kinetic-energy",
    }

    def build_case_config(self, options: dict[str, Any]) -> CaseConfig:
        """Construct a CaseConfig from the JSON options block.

        Required keys: step_height
        """
        return CaseConfig(
            base_case_name=options.get("base_case_name", "ffs"),
            
            # Using 'step_height' as the reference length for generic scaling
            hill_height=options["step_height"],
            re_h=options.get("re_h", 0),  # Required by CaseConfig, might be 0 if unspecified for FFS
            fluid_density=options.get("fluid_density", 854.5648),
            fluid_viscosity=options.get("fluid_viscosity", 0.01424275),
            
            # FFS specific configurations
            inlet_velocity=options.get("inlet_velocity", 10.0),
            turb_intensity=options.get("turb_intensity", 3.25),
            turb_viscosity_ratio=options.get("turb_viscosity_ratio", 12.0),
            outlet_static_pressure=options.get("outlet_static_pressure", 0.0),
            
            iter_count=options.get("iter_count", 2000),
            zone_inlet=options.get("zone_inlet", "inlet"),
            zone_outlet=options.get("zone_outlet", "outlet"),
            zone_top=options.get("zone_top", "ceiling"),
            zone_bottom=options.get("zone_bottom", "step"),
        )

    def apply_boundary_conditions(self, solver) -> None:
        """Apply Velocity Inlet and Pressure Outlet BCs."""
        cc = self.case_config

        # Named selections can arrive from meshing as wall zones. Convert
        # the FFS-specific boundary types before applying numeric BC values.
        self._set_zone_type(solver, cc.zone_top, "symmetry")
        self._set_zone_type(solver, cc.zone_inlet, "velocity-inlet")
        self._set_zone_type(solver, cc.zone_outlet, "pressure-outlet")

        self._set_velocity_inlet(
            solver,
            cc.zone_inlet,
            velocity=cc.inlet_velocity,
            turbulent_intensity=cc.turb_intensity,
            turbulent_viscosity_ratio=cc.turb_viscosity_ratio,
        )
        self._set_pressure_outlet_static_pressure(
            solver,
            cc.zone_outlet,
            static_pressure=cc.outlet_static_pressure,
            turbulent_intensity=cc.turb_intensity,
            turbulent_viscosity_ratio=cc.turb_viscosity_ratio,
        )

    @staticmethod
    def _set_velocity_inlet(
        solver,
        zone_name: str,
        *,
        velocity: float,
        turbulent_intensity: float,
        turbulent_viscosity_ratio: float,
    ) -> None:
        """Configure a value-based inlet without empty profile references."""

        try:
            inlet = solver.settings.setup.boundary_conditions.velocity_inlet[
                zone_name
            ]
            inlet.momentum.velocity_specification_method = (
                "Magnitude, Normal to Boundary"
            )
            inlet.momentum.reference_frame = "Absolute"
            inlet.momentum.velocity_magnitude.option = "value"
            inlet.momentum.velocity_magnitude.value = velocity
            inlet.momentum.initial_gauge_pressure.option = "value"
            inlet.momentum.initial_gauge_pressure.value = 0.0

            inlet.turbulence.turbulence_specification = (
                "Intensity and Viscosity Ratio"
            )
            inlet.turbulence.turbulent_intensity = turbulent_intensity / 100.0
            inlet.turbulence.turbulent_viscosity_ratio = (
                turbulent_viscosity_ratio
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to configure FFS velocity inlet '{zone_name}' through "
                "PyFluent settings."
            ) from exc
        print(
            f"[ffs] Boundary '{zone_name}' configured as velocity-inlet "
            f"with velocity {velocity} and turbulence intensity "
            f"{turbulent_intensity}%."
        )

    @staticmethod
    def _set_zone_type(solver, zone_name: str, zone_type: str) -> None:
        """Set a boundary zone type through PyFluent with a TUI fallback."""

        try:
            solver.settings.setup.boundary_conditions.set_zone_type(
                zone_list=[zone_name],
                new_type=zone_type,
            )
        except Exception:
            solver.execute_tui(
                f"/define/boundary-conditions/zone-type {zone_name} {zone_type}"
            )
        print(f"[ffs] Boundary '{zone_name}' configured as {zone_type}.")

    @staticmethod
    def _set_pressure_outlet_static_pressure(
        solver,
        zone_name: str,
        *,
        static_pressure: float,
        turbulent_intensity: float,
        turbulent_viscosity_ratio: float,
    ) -> None:
        """Set value-based pressure-outlet and backflow conditions.

        The FFS reference uses a static-pressure outlet. Use the structured
        PyFluent settings tree instead of TUI prompt streams, because the
        pressure-outlet TUI prompts drift between Fluent releases.
        """

        try:
            outlet = solver.settings.setup.boundary_conditions.pressure_outlet[
                zone_name
            ]
            outlet.momentum.gauge_pressure.option = "value"
            outlet.momentum.gauge_pressure.value = static_pressure
            outlet.momentum.backflow_pressure_spec = "Static Pressure"
            outlet.momentum.backflow_dir_spec_method = "Normal to Boundary"
            outlet.momentum.target_mass_flow_rate = False

            outlet.turbulence.turbulence_specification = (
                "Intensity and Viscosity Ratio"
            )
            outlet.turbulence.backflow_turbulent_intensity = (
                turbulent_intensity / 100.0
            )
            outlet.turbulence.backflow_turbulent_viscosity_ratio = (
                turbulent_viscosity_ratio
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to configure FFS pressure outlet "
                f"'{zone_name}' with static pressure {static_pressure} Pa "
                "through PyFluent settings."
            ) from exc
        print(
            f"[ffs] Boundary '{zone_name}' configured as pressure-outlet "
            f"with static pressure {static_pressure} Pa."
        )


    def load_dns(self, dns_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load FFS DNS reference data.

        File format: CSV with header.
        Column names can be overridden with ``case.options.dns_columns``.
        Known fields are loaded when present; Ux, Uy, and pressure remain
        required for backward compatibility with existing FFS objectives.

        Values remain dimensional to match the current FFS extraction:
        velocity in m/s, pressure in Pa, and total TKE in m^2/s^2.
        """
        dns_path = Path(dns_path)
        if not dns_path.is_file():
            raise FileNotFoundError(f"DNS file not found: {dns_path}")

        df = pd.read_csv(dns_path)
        df.columns = df.columns.str.strip()

        coordinate_columns = {
            "x": "x-coordinate",
            "y": "y-coordinate",
        }
        coordinate_columns.update(self.options.get("dns_coordinate_columns", {}))

        field_columns = dict(self._DEFAULT_DNS_COLUMNS)
        configured_fields = self.options.get("dns_columns", {})
        field_columns.update(configured_fields)

        required_columns = {
            coordinate_columns["x"],
            coordinate_columns["y"],
            field_columns["Ux"],
            field_columns["Uy"],
            field_columns["p"],
        }
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise KeyError(
                f"FFS DNS file {dns_path} is missing required column(s): {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

        x = df[coordinate_columns["x"]].to_numpy()
        y = df[coordinate_columns["y"]].to_numpy()
        coords = np.column_stack([x, y])

        fields: dict[str, np.ndarray] = {}
        for field_name, column_name in field_columns.items():
            if column_name in df.columns:
                fields[field_name] = df[column_name].to_numpy()
            elif field_name in configured_fields:
                raise KeyError(
                    f"Configured FFS DNS field {field_name!r} uses missing "
                    f"column {column_name!r}."
                )

        fields["cp"] = fields["p"]  # Preserve current pressure convention.
        return coords, fields

    def build_run_result(
        self,
        *,
        run_id: str,
        parameters: dict[str, float],
        ascii_path: str | Path,
        cost_seconds: float = 0.0,
    ) -> RunResult:
        """Build dimensional FFS output and expose RANS k as total TKE.

        In RANS all turbulent kinetic energy is modeled, so Fluent's
        ``turb-kinetic-energy`` is directly comparable to the SBES total:
        resolved fluctuations plus modeled k.
        """

        result = super().build_run_result(
            run_id=run_id,
            parameters=parameters,
            ascii_path=ascii_path,
            cost_seconds=cost_seconds,
        )
        if "turb-kinetic-energy" in result.fields:
            result.fields["total-turbulent-kinetic-energy"] = (
                result.fields["turb-kinetic-energy"]
                * self.case_config.u_bulk**2
            )
        return result
