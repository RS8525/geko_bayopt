"""
Pydantic schemas for experiment JSON configs.

One JSON file per experiment. Validated at load time so typos are caught
before any expensive CFD runs.

Example minimal config::

    {
        "experiment_id": "csep_sweep_v1",
        "case": {
            "kind": "periodic_hills",
            "options": {
                "geometry_basename": "periodic_hill_2d_alpha_1.0",
                "alpha": 1.0,
                "hill_height": 28.0,
                "re_h": 5600,
                "dns_case_name": "alph10-9-3036"
            }
        },
        "parameters": [
            {"name": "geko_csep", "low": 0.5, "high": 2.0}
        ],
        "objective": {
            "kind": "mse_cp",
            "options": {"field": "cp", "weight": 1.0}
        },
        "optimizer": {
            "kind": "skopt_gp",
            "n_initial": 8,
            "n_iterations": 32
        },
        "session_strategy": "live"
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ParameterSpec(BaseModel):
    """One GEKO coefficient that the optimizer will vary.

    The ``name`` must match a field on ``fluent.case_config.CaseConfig``,
    e.g. ``geko_csep``, ``geko_cnw``, ``geko_cmix``, ``geko_cjet``,
    ``geko_ccorner``.
    """
    name: str
    low: float
    high: float

    @field_validator("high")
    @classmethod
    def _high_greater_than_low(cls, v: float, info) -> float:
        low = info.data.get("low")
        if low is not None and v <= low:
            raise ValueError(f"high ({v}) must be greater than low ({low})")
        return v


class CaseSection(BaseModel):
    """Selects which flow case to run and its case-specific options.

    ``kind`` matches an entry in ``cases/__init__.py``'s dispatcher.
    ``options`` is a free-form dict passed to the case's constructor.
    """
    kind: Literal["periodic_hills"]
    options: dict[str, Any] = Field(default_factory=dict)


class ObjectiveSection(BaseModel):
    """Selects the loss function and its options.

    ``kind`` matches an entry in ``objective/__init__.py``'s dispatcher.
    ``options`` is passed to the loss factory as keyword arguments.
    """
    kind: Literal[
        "mse_cp",
        "mse_field",
        "weighted_multi_field",
        "gedcp",
    ]
    options: dict[str, Any] = Field(default_factory=dict)


class ResidualCriteria(BaseModel):
    """Convergence thresholds for Fluent's residual monitors.

    Fluent stops iterating early once every residual drops below its
    threshold. Values are absolute (not normalized). Defaults are
    Fluent's stock 1e-3 for each.

    Use ``None`` for any field to leave Fluent's default in place.
    """
    continuity: float = 1.0e-3
    x_velocity: float = 1.0e-3
    y_velocity: float = 1.0e-3
    k: float = 1.0e-3
    omega: float = 1.0e-3


class OptimizerSection(BaseModel):
    """Selects the optimizer and its general settings."""
    kind: Literal["skopt_gp", "skopt_rf", "random"]
    n_initial: int = 8
    n_iterations: int = 32
    random_state: int | None = 42
    options: dict[str, Any] = Field(default_factory=dict)


class MeshSection(BaseModel):
    """Optional mesh-sizing overrides. Anything not specified uses the
    defaults in ``fluent.mesh_config.MeshConfig``."""
    length_unit: str = "mm"
    cad_route: str = "Workbench"
    cad_extension: str = "pmdb"
    min_size: float = 0.02
    max_size: float = 0.5
    growth_rate: float = 1.2
    curvature_normal_angle: int = 12
    bl_first_layer_height: float = 0.0009
    bl_number_of_layers: int = 22
    bl_growth_rate: float = 1.15
    generate_quads: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from JSON."""
    experiment_id: str
    case: CaseSection
    parameters: list[ParameterSpec]
    objective: ObjectiveSection
    optimizer: OptimizerSection
    mesh: MeshSection = Field(default_factory=MeshSection)

    # Convergence criteria for Fluent residuals. Omit to keep Fluent defaults.
    residual_criteria: ResidualCriteria | None = None

    # Where Fluent writes raw outputs (mesh, .cas, .dat, ASCII).
    # Defaults to ``<repo>/results/fluent/<experiment_id>/`` when not set.
    fluent_work_dir: str | None = None

    # Where the store writes metadata.csv and the optimizer pickle.
    # Defaults to ``<repo>/results/experiments/<experiment_id>/``.
    results_dir: str | None = None

    # "live": one Fluent session reused for all trials (faster, but
    # one crash kills the sweep).
    # "per_trial": launch + exit Fluent per trial (safer on Student license).
    session_strategy: Literal["live", "per_trial"] = "live"

    # If True, after each trial delete any .cas/.dat files that don't
    # belong to the current-best trial. The .ascii is always preserved.
    keep_only_best_case_files: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
