"""Config-driven plotting helper for forward-facing-step DNS and simulation data.

Run from the repository root and pass the plotting config explicitly:

    python scripts/ffs/plot_ffs_fields.py scripts/ffs/plots/<config-name>.json

Absolute config paths are also accepted:

    python scripts/ffs/plot_ffs_fields.py C:/path/to/my_ffs_plot.json

Configuration is stored outside this script. Each config file defines:
- the relative input paths for the simulation and DNS data
- which columns to plot in each dataset
- which simulation/DNS field pairs to compare
- optional ``plots.normalized_error`` common-grid settings and the subset of
  comparison aliases that contribute to the field-only objective
- optional root-relative ``output_dir``. Without it, the output folder name is
  used under ``scripts/ffs/plots/<name>/``

For comparison plots, DNS values are interpolated onto the simulation grid
before the difference is computed. Normalized-error plots instead interpolate
both datasets to the objective's common grid and show
``(simulation - DNS) / std(DNS)``. Their reported contribution can be either
the normalized L1 or L2 field error used by the field objective, excluding
parameter preference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from geko_bayesopt.objective.field_error import (
    _common_grid,
    _idw_interpolate,
    _normalized_field_error,
    _wstd,
)

BASE_DIR = Path(__file__).resolve().parents[2]

SIM_CMAP = "viridis"
DNS_CMAP = "viridis"
ERROR_CMAP = "coolwarm"
FIELD_LEVELS = 100


def load_ascii(path: Path) -> pd.DataFrame:
    """Load a Fluent ASCII export while preserving its header names."""

    frame = pd.read_csv(path, sep=r"\s+", engine="python")
    frame.columns = frame.columns.str.strip()
    return frame


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def _column_values(
    frame: pd.DataFrame,
    column: str | int,
    *,
    dataset_name: str,
) -> np.ndarray:
    """Resolve a configured column by header name or legacy numeric index."""

    if isinstance(column, bool):
        raise TypeError(f"{dataset_name} column selectors cannot be booleans.")
    if isinstance(column, int):
        try:
            return frame.iloc[:, column].to_numpy()
        except IndexError as exc:
            raise IndexError(
                f"{dataset_name} column index {column} is out of range for "
                f"{len(frame.columns)} columns."
            ) from exc
    if not isinstance(column, str):
        raise TypeError(
            f"{dataset_name} column selectors must be header strings or integer indices."
        )
    if column not in frame.columns:
        raise KeyError(
            f"{dataset_name} column {column!r} was not found. "
            f"Available columns: {frame.columns.tolist()}"
        )
    return frame[column].to_numpy()


def _ffs_floor(x: np.ndarray) -> np.ndarray:
    """Piecewise floor used to mask the solid step region."""

    return np.where(x < 0.0, -0.01, 0.0)


def _ffs_triangle_mask(tri: mtri.Triangulation, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask triangles that fall below the FFS solid boundary."""

    triangles = tri.triangles
    tri_x = x[triangles]
    tri_y = y[triangles]
    floor_y = _ffs_floor(tri_x)
    below_floor = np.any(tri_y < floor_y - 1e-12, axis=1)

    # A triangulation of a floor-masked uniform grid can bridge the vertical
    # step face: all vertices are valid, but the triangle interior crosses
    # the downstream solid. Reject triangles spanning x=0 with an upstream
    # vertex below the downstream floor.
    crosses_step_face = (
        (np.min(tri_x, axis=1) < 0.0)
        & (np.max(tri_x, axis=1) >= 0.0)
        & (np.min(tri_y, axis=1) < 0.0)
    )
    return below_floor | crosses_step_face


def _triangulation_with_mask(
    x: np.ndarray,
    y: np.ndarray,
    extra_point_mask: np.ndarray | None = None,
) -> mtri.Triangulation:
    """Build a simulation-grid triangulation and apply the FFS solid mask."""

    tri = mtri.Triangulation(x, y)
    triangle_mask = _ffs_triangle_mask(tri, x, y)
    if extra_point_mask is not None:
        triangle_mask = triangle_mask | np.any(extra_point_mask[tri.triangles], axis=1)
    tri.set_mask(triangle_mask)
    return tri


def _apply_transform(
    values: np.ndarray,
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]],
    field_name: str,
) -> np.ndarray:
    transform = transforms.get(field_name)
    return transform(values) if transform is not None else values


def _plot_scatter_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=4, marker=".")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_tri_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap: str,
) -> None:
    point_mask = ~np.isfinite(values)
    tri = _triangulation_with_mask(x, y, point_mask)

    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.tricontourf(tri, np.ma.masked_invalid(values), levels=FIELD_LEVELS, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    fig.colorbar(contour, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _interpolate_dns_to_sim(
    dns_x: np.ndarray,
    dns_y: np.ndarray,
    dns_values: np.ndarray,
    sim_x: np.ndarray,
    sim_y: np.ndarray,
) -> np.ndarray:
    """Interpolate DNS values onto the simulation points.

    Use inverse-distance weighting over a KD-tree. This is robust to irregular
    point clouds, duplicate coordinates, and large DNS sets without requiring a
    Delaunay triangulation.
    """

    dns_points = np.column_stack((dns_x, dns_y))
    unique_points, inverse = np.unique(dns_points, axis=0, return_inverse=True)

    if len(unique_points) != len(dns_points):
        summed = np.zeros(len(unique_points), dtype=float)
        counts = np.zeros(len(unique_points), dtype=float)
        np.add.at(summed, inverse, dns_values)
        np.add.at(counts, inverse, 1.0)
        dns_values = summed / counts
    else:
        dns_values = np.asarray(dns_values, dtype=float)

    sim_points = np.column_stack((sim_x, sim_y))
    tree = cKDTree(unique_points)

    k = min(8, len(unique_points))
    distances, indices = tree.query(sim_points, k=k)

    if k == 1:
        return np.asarray(dns_values[indices], dtype=float)

    distances = np.asarray(distances, dtype=float)
    indices = np.asarray(indices, dtype=int)

    result = np.empty(len(sim_points), dtype=float)
    exact_match = np.any(distances == 0.0, axis=1)
    if np.any(exact_match):
        result[exact_match] = dns_values[indices[exact_match, np.argmin(distances[exact_match], axis=1)]]

    need_interp = ~exact_match
    if np.any(need_interp):
        d = distances[need_interp]
        idx = indices[need_interp]
        weights = 1.0 / np.maximum(d, 1e-12) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        result[need_interp] = np.sum(weights * dns_values[idx], axis=1)

    return result


def _plot_comparison(
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    sim_values: np.ndarray,
    dns_on_sim: np.ndarray,
    sim_name: str,
    dns_name: str,
    output_path: Path,
) -> None:
    """Plot simulation, interpolated DNS, and their error on one figure."""

    error = sim_values - dns_on_sim
    point_mask = ~np.isfinite(dns_on_sim)
    tri = _triangulation_with_mask(sim_x, sim_y, point_mask)

    valid_error = np.abs(error[np.isfinite(error)])
    if valid_error.size == 0:
        raise ValueError(f"No valid overlap between simulation and DNS for {sim_name} vs {dns_name}.")

    field_values = np.concatenate(
        (
            sim_values[np.isfinite(sim_values)],
            dns_on_sim[np.isfinite(dns_on_sim)],
        )
    )
    if field_values.size == 0:
        raise ValueError(f"No finite simulation/DNS values for {sim_name} vs {dns_name}.")

    field_vmin = float(np.min(field_values))
    field_vmax = float(np.max(field_values))
    if field_vmin == field_vmax:
        pad = max(abs(field_vmin), 1.0) * 1e-6
        field_vmin -= pad
        field_vmax += pad
    field_levels = np.linspace(field_vmin, field_vmax, FIELD_LEVELS)
    field_norm = mcolors.Normalize(vmin=field_vmin, vmax=field_vmax)

    error_limit = float(np.max(valid_error))
    error_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-error_limit, vmax=error_limit)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    sim_plot = axs[0].tricontourf(
        tri,
        np.ma.masked_invalid(sim_values),
        levels=field_levels,
        cmap=SIM_CMAP,
        norm=field_norm,
    )
    axs[0].set_title(f"Simulation {sim_name}")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sim_plot, ax=axs[0], label=sim_name)

    dns_plot = axs[1].tricontourf(
        tri,
        np.ma.masked_invalid(dns_on_sim),
        levels=field_levels,
        cmap=DNS_CMAP,
        norm=field_norm,
    )
    axs[1].set_title(f"DNS interpolated to sim grid {dns_name}")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect("equal", adjustable="box")
    fig.colorbar(dns_plot, ax=axs[1], label=dns_name)

    error_plot = axs[2].tricontourf(
        tri,
        np.ma.masked_invalid(error),
        levels=FIELD_LEVELS,
        cmap=ERROR_CMAP,
        norm=error_norm,
    )
    axs[2].set_title(f"Error {sim_name} - {dns_name}")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_aspect("equal", adjustable="box")
    fig.colorbar(error_plot, ax=axs[2], label="Difference")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    valid_error = error[np.isfinite(error)]
    rmse = float(np.sqrt(np.mean(valid_error**2)))
    mae = float(np.mean(np.abs(valid_error)))
    print(f"{sim_name} vs {dns_name}: RMSE={rmse:.6g}, MAE={mae:.6g}")


def _fixed_limit_from_spec(
    spec: object,
    *,
    sim_name: str,
    dns_name: str,
) -> tuple[float, float] | None:
    """Resolve a configured fixed color scale for an error plot."""

    if spec is None:
        return None

    if isinstance(spec, (int, float)):
        limit = abs(float(spec))
        return (-limit, limit)

    if isinstance(spec, list):
        if len(spec) != 2:
            raise ValueError("Error-limit lists must contain exactly [vmin, vmax].")
        return (float(spec[0]), float(spec[1]))

    if not isinstance(spec, dict):
        raise TypeError("Error limits must be a number, [vmin, vmax], or an object.")

    pair_key = f"{sim_name}_vs_{dns_name}"
    for key in (pair_key, sim_name, dns_name, "default"):
        if key in spec:
            return _fixed_limit_from_spec(
                spec[key],
                sim_name=sim_name,
                dns_name=dns_name,
            )

    return None


def _auto_symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    limit = float(np.max(np.abs(values[np.isfinite(values)])))
    limit = max(limit, 1e-12)
    return (-limit, limit)


def _validate_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Error color limits must be finite.")
    if vmin >= vmax:
        raise ValueError("Error color limits must satisfy vmin < vmax.")
    if not (vmin < 0.0 < vmax):
        raise ValueError("Error color limits must include zero.")
    return (vmin, vmax)


def _plot_normalized_error(
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    sim_values: np.ndarray,
    dns_x: np.ndarray,
    dns_y: np.ndarray,
    dns_values: np.ndarray,
    sim_name: str,
    dns_name: str,
    output_path: Path,
    settings: dict,
    norm: str,
) -> float:
    """Plot the signed pointwise error used by the common-grid objective."""

    dns_coords = np.column_stack((dns_x, dns_y))
    sim_coords = np.column_stack((sim_x, sim_y))
    grid = _common_grid(
        dns_coords,
        nx=settings.get("common_grid_nx", 360),
        ny=settings.get("common_grid_ny", 120),
        floor_mode=settings.get("common_grid_floor"),
    )
    dns_grid = _idw_interpolate(dns_coords, dns_values, grid)
    sim_grid = _idw_interpolate(sim_coords, sim_values, grid)
    weights = np.ones(len(grid), dtype=float)

    dns_std = max(_wstd(dns_grid, weights), 1e-8)
    normalized_error = (sim_grid - dns_grid) / dns_std
    field_score = _normalized_field_error(
        dns_grid - sim_grid,
        weights,
        dns_std,
        norm=norm,
    )

    limits = _fixed_limit_from_spec(
        settings.get("error_limits"),
        sim_name=sim_name,
        dns_name=dns_name,
    )
    if limits is None:
        limits = _auto_symmetric_limits(normalized_error)
    vmin, vmax = _validate_limits(*limits)
    levels = np.linspace(vmin, vmax, FIELD_LEVELS)
    tri = _triangulation_with_mask(grid[:, 0], grid[:, 1])

    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.tricontourf(
        tri,
        normalized_error,
        levels=levels,
        extend="both",
        cmap=ERROR_CMAP,
        norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
    )
    ax.set_title(
        f"Normalized error {sim_name} - {dns_name} ({norm.upper()})\n"
        f"field objective contribution = {field_score:.6g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    fig.colorbar(contour, ax=ax, label="(simulation - DNS) / std(DNS)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return field_score


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _resolve_output_dir(cfg: dict, config_name: str) -> Path:
    output_dir = cfg.get("output_dir")
    if output_dir:
        return _resolve_path(output_dir)
    return (Path(__file__).resolve().parent / "plots" / config_name).resolve()


def _normalize_transform_spec(spec: dict[str, str]) -> Callable[[np.ndarray], np.ndarray]:
    kind = spec.get("kind")
    if kind == "subtract_min":
        return lambda values: values - np.min(values)
    raise ValueError(f"Unsupported transform kind: {kind}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the JSON plotting configuration.",
    )
    return parser.parse_args()


def main() -> None:
    config_path = _parse_args().config
    config_path = config_path if config_path.is_absolute() else (Path.cwd() / config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_config(config_path)

    config_name = cfg["name"]
    output_dir = _resolve_output_dir(cfg, config_name)

    sim_path = _resolve_path(cfg["simulation"]["path"])
    dns_path = _resolve_path(cfg["dns"]["path"])

    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")
    if not dns_path.is_file():
        raise FileNotFoundError(f"DNS file not found: {dns_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sim_df = load_ascii(sim_path)
    dns_df = load_csv(dns_path)

    sim_x = _column_values(
        sim_df,
        cfg["simulation"]["x_column"],
        dataset_name="Simulation",
    )
    sim_y = _column_values(
        sim_df,
        cfg["simulation"]["y_column"],
        dataset_name="Simulation",
    )
    dns_x = _column_values(dns_df, cfg["dns"]["x_column"], dataset_name="DNS")
    dns_y = _column_values(dns_df, cfg["dns"]["y_column"], dataset_name="DNS")

    sim_field_columns = cfg["simulation"]["fields"]
    dns_field_columns = {name: column for name, column in cfg["dns"]["fields"].items()}

    sim_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for field_name, transform_spec in cfg.get("simulation", {}).get("transforms", {}).items():
        sim_transforms[field_name] = _normalize_transform_spec(transform_spec)

    dns_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for field_name, transform_spec in cfg.get("dns", {}).get("transforms", {}).items():
        dns_transforms[field_name] = _normalize_transform_spec(transform_spec)

    sim_fields: dict[str, np.ndarray] = {}
    for field_name, column in sim_field_columns.items():
        values = _column_values(sim_df, column, dataset_name="Simulation")
        sim_fields[field_name] = _apply_transform(values, sim_transforms, field_name)

    dns_fields: dict[str, np.ndarray] = {}
    for field_name, column in dns_field_columns.items():
        values = _column_values(dns_df, column, dataset_name="DNS")
        dns_fields[field_name] = _apply_transform(values, dns_transforms, field_name)

    for field_name in cfg.get("plots", {}).get("simulation", []):
        output_path = output_dir / f"sim_{field_name}.png"
        _plot_tri_field(
            sim_x,
            sim_y,
            sim_fields[field_name],
            title=f"Simulation {field_name}",
            cbar_label=field_name,
            output_path=output_path,
            cmap=SIM_CMAP,
        )
        print(f"Saved {output_path}")

    for field_name in cfg.get("plots", {}).get("dns", []):
        output_path = output_dir / f"dns_{field_name}.png"
        _plot_scatter_field(
            dns_x,
            dns_y,
            dns_fields[field_name],
            title=f"DNS {field_name}",
            cbar_label=field_name,
            output_path=output_path,
            cmap=DNS_CMAP,
        )
        print(f"Saved {output_path}")

    normalized_settings = cfg.get("plots", {}).get("normalized_error")
    normalized_norms = ["l2"]
    if normalized_settings:
        configured_norms = normalized_settings.get(
            "norms",
            [normalized_settings.get("field_error_norm", "l2")],
        )
        if isinstance(configured_norms, str):
            configured_norms = [configured_norms]
        normalized_norms = list(configured_norms)
        invalid_norms = [norm for norm in normalized_norms if norm not in {"l1", "l2"}]
        if invalid_norms:
            raise ValueError(
                "plots.normalized_error.norms entries must be 'l1' or 'l2'. "
                f"Invalid: {invalid_norms}"
            )

    objective_fields = set(
        normalized_settings.get("objective_fields", [])
        if normalized_settings
        else []
    )
    objective_field_totals = {norm: 0.0 for norm in normalized_norms}

    for pair in cfg.get("plots", {}).get("comparison", []):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each comparison entry must be a 2-item list: [simulation_name, dns_name]")

        sim_name, dns_name = pair
        if sim_name not in sim_fields:
            raise KeyError(f"Simulation comparison field not loaded: {sim_name}")
        if dns_name not in dns_fields:
            raise KeyError(f"DNS comparison field not loaded: {dns_name}")

        dns_on_sim = _interpolate_dns_to_sim(dns_x, dns_y, dns_fields[dns_name], sim_x, sim_y)
        output_path = output_dir / f"compare_{sim_name}_vs_{dns_name}.png"
        _plot_comparison(
            sim_x=sim_x,
            sim_y=sim_y,
            sim_values=sim_fields[sim_name],
            dns_on_sim=dns_on_sim,
            sim_name=sim_name,
            dns_name=dns_name,
            output_path=output_path,
        )
        print(f"Saved {output_path}")

        if normalized_settings:
            for norm in normalized_norms:
                normalized_path = (
                    output_dir / f"normalized_error_{norm}_{sim_name}_vs_{dns_name}.png"
                )
                field_score = _plot_normalized_error(
                    sim_x=sim_x,
                    sim_y=sim_y,
                    sim_values=sim_fields[sim_name],
                    dns_x=dns_x,
                    dns_y=dns_y,
                    dns_values=dns_fields[dns_name],
                    sim_name=sim_name,
                    dns_name=dns_name,
                    output_path=normalized_path,
                    settings=normalized_settings,
                    norm=norm,
                )
                print(
                    f"{sim_name} normalized {norm.upper()} field contribution: "
                    f"{field_score:.6g}"
                )
                print(f"Saved {normalized_path}")
                if sim_name in objective_fields:
                    objective_field_totals[norm] += field_score

    if normalized_settings and objective_fields:
        for norm, total in objective_field_totals.items():
            print(
                "Field-only objective "
                f"({', '.join(sorted(objective_fields))}, {norm.upper()}): "
                f"{total:.6g}"
            )

    print(f"All plots written to {output_dir}")


if __name__ == "__main__":
    main()
