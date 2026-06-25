from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dns"
    / "ffs"
    / "average_z_dns_ffs.py"
)
SPEC = importlib.util.spec_from_file_location("average_z_dns_ffs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_total_tke_is_derived_before_spanwise_averaging() -> None:
    chunk = pd.DataFrame(
        {
            "mean-turbulent-kinetic-energy--k-dataset": [1.0, 2.0],
            "rmse-x-velocity": [2.0, 3.0],
            "rmse-y-velocity": [4.0, 0.0],
            "rmse-z-velocity": [0.0, 4.0],
        }
    )

    MODULE._add_total_turbulent_kinetic_energy(chunk)

    expected = np.array(
        [
            1.0 + 0.5 * (2.0**2 + 4.0**2),
            2.0 + 0.5 * (3.0**2 + 4.0**2),
        ]
    )
    assert np.allclose(chunk[MODULE.TOTAL_TKE_COLUMN], expected)


def test_exported_total_tke_is_preferred_when_available() -> None:
    chunk = pd.DataFrame(
        {
            "mean-tke_tot-dataset": [3.0],
            "mean-turbulent-kinetic-energy--k-dataset": [1.0],
            "rmse-x-velocity": [10.0],
            "rmse-y-velocity": [10.0],
            "rmse-z-velocity": [10.0],
        }
    )

    MODULE._add_total_turbulent_kinetic_energy(chunk)

    assert np.allclose(chunk[MODULE.TOTAL_TKE_COLUMN], [3.0])
