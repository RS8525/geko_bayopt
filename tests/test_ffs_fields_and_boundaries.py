from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from geko_bayesopt.cases.ffs.case import ForwardFacingStepCase
from geko_bayesopt.fluent.extract import parse_fluent_ascii
from geko_bayesopt.fluent.mesh_config import MeshConfig


def _ffs_case(**options) -> ForwardFacingStepCase:
    return ForwardFacingStepCase(
        {
            "step_height": 0.01,
            "re_h": 6000,
            **options,
        },
        MeshConfig(),
    )


def test_ffs_dns_loader_includes_total_tke(tmp_path: Path) -> None:
    path = tmp_path / "ffs.csv"
    pd.DataFrame(
        {
            "x-coordinate": [0.0, 1.0],
            "y-coordinate": [0.0, 0.5],
            "mean-x-velocity": [10.0, 9.0],
            "mean-y-velocity": [0.0, 0.1],
            "mean-pressure": [2.0, 1.0],
            "total-turbulent-kinetic-energy": [1.2, 1.4],
        }
    ).to_csv(path, index=False)

    coords, fields = _ffs_case().load_dns(path)

    assert coords.shape == (2, 2)
    assert np.allclose(
        fields["total-turbulent-kinetic-energy"],
        [1.2, 1.4],
    )
    assert np.array_equal(fields["cp"], fields["p"])


def test_ffs_dns_loader_accepts_column_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ffs-custom.csv"
    pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "u": [10.0],
            "v": [0.0],
            "pressure": [1.0],
            "k_total": [1.2],
        }
    ).to_csv(path, index=False)
    flow_case = _ffs_case(
        dns_coordinate_columns={"x": "x", "y": "y"},
        dns_columns={
            "Ux": "u",
            "Uy": "v",
            "p": "pressure",
            "total-turbulent-kinetic-energy": "k_total",
        },
    )

    _, fields = flow_case.load_dns(path)

    assert fields["total-turbulent-kinetic-energy"][0] == 1.2


def test_extract_keeps_old_ascii_compatible(tmp_path: Path) -> None:
    path = tmp_path / "legacy.ascii"
    path.write_text(
        "nodenumber x-coordinate y-coordinate x-velocity y-velocity pressure\n"
        "1 0.0 0.0 1.0 0.0 2.0\n",
        encoding="utf-8",
    )

    _, fields = parse_fluent_ascii(
        path,
        hill_height=1.0,
        u_bulk=1.0,
        fluid_density=1.0,
    )

    assert {"Ux", "Uy", "p", "cp"}.issubset(fields)


def test_extract_keeps_periodic_hills_tke_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "periodic-k.ascii"
    path.write_text(
        "nodenumber x-coordinate y-coordinate x-velocity y-velocity pressure "
        "turb-kinetic-energy\n"
        "1 0.0 0.0 1.0 0.0 2.0 4.0\n",
        encoding="utf-8",
    )

    _, fields = parse_fluent_ascii(
        path,
        hill_height=1.0,
        u_bulk=2.0,
        fluid_density=1.0,
    )

    assert np.allclose(fields["turb-kinetic-energy"], [1.0])


def test_ffs_run_result_exposes_dimensional_rans_k_as_total(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fluent-k.ascii"
    path.write_text(
        "nodenumber x-coordinate y-coordinate x-velocity y-velocity pressure "
        "turb-kinetic-energy\n"
        "1 0.0 0.0 10.0 0.0 100.0 4.0\n",
        encoding="utf-8",
    )
    flow_case = _ffs_case(
        fluid_density=1.0,
        fluid_viscosity=0.01,
        re_h=10,
    )

    result = flow_case.build_run_result(
        run_id="ffs-test",
        parameters={},
        ascii_path=path,
    )

    assert np.allclose(result.fields["turb-kinetic-energy"], [0.04])
    assert np.allclose(
        result.fields["total-turbulent-kinetic-energy"],
        [4.0],
    )


def test_ffs_boundary_conditions_use_values_not_profiles() -> None:
    value_setting = lambda: SimpleNamespace(option="profile", value=None)
    inlet = SimpleNamespace(
        momentum=SimpleNamespace(
            velocity_specification_method=None,
            reference_frame=None,
            velocity_magnitude=value_setting(),
            initial_gauge_pressure=value_setting(),
        ),
        turbulence=SimpleNamespace(
            turbulence_specification=None,
            turbulent_intensity=None,
            turbulent_viscosity_ratio=None,
        ),
    )
    outlet = SimpleNamespace(
        momentum=SimpleNamespace(
            gauge_pressure=value_setting(),
            backflow_pressure_spec=None,
            backflow_dir_spec_method=None,
            target_mass_flow_rate=True,
        ),
        turbulence=SimpleNamespace(
            turbulence_specification=None,
            backflow_turbulent_intensity=None,
            backflow_turbulent_viscosity_ratio=None,
        ),
    )
    zone_type_calls = []
    boundary_conditions = SimpleNamespace(
        velocity_inlet={"inlet": inlet},
        pressure_outlet={"outlet": outlet},
        set_zone_type=lambda **kwargs: zone_type_calls.append(kwargs),
    )
    solver = SimpleNamespace(
        settings=SimpleNamespace(
            setup=SimpleNamespace(boundary_conditions=boundary_conditions)
        )
    )

    _ffs_case(
        inlet_velocity=10.0,
        turb_intensity=3.25,
        turb_viscosity_ratio=12.0,
        zone_inlet="inlet",
        zone_outlet="outlet",
        zone_top="ceiling",
    ).apply_boundary_conditions(solver)

    assert inlet.momentum.velocity_specification_method == (
        "Magnitude, Normal to Boundary"
    )
    assert inlet.momentum.velocity_magnitude.option == "value"
    assert inlet.momentum.velocity_magnitude.value == 10.0
    assert inlet.momentum.initial_gauge_pressure.option == "value"
    assert inlet.turbulence.turbulent_intensity == 0.0325
    assert inlet.turbulence.turbulent_viscosity_ratio == 12.0

    assert outlet.momentum.gauge_pressure.option == "value"
    assert outlet.momentum.backflow_pressure_spec == "Static Pressure"
    assert outlet.momentum.backflow_dir_spec_method == "Normal to Boundary"
    assert outlet.momentum.target_mass_flow_rate is False
    assert outlet.turbulence.turbulence_specification == (
        "Intensity and Viscosity Ratio"
    )
    assert outlet.turbulence.backflow_turbulent_intensity == 0.0325
    assert outlet.turbulence.backflow_turbulent_viscosity_ratio == 12.0
    assert [call["new_type"] for call in zone_type_calls] == [
        "symmetry",
        "velocity-inlet",
        "pressure-outlet",
    ]
