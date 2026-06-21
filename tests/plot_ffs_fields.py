"""Config-driven plotting helper for forward-facing-step DNS and simulation data.

Usage:
    python tests/plot_ffs_fields.py tests/plots/<config-name>.json

Configuration is stored outside this script. Each config file defines:
- the relative input paths for the simulation and DNS data
- which columns to plot in each dataset
- which simulation/DNS field pairs to compare
- the output folder name, which is used under ``tests/plots/<name>/``

For comparison plots, DNS values are interpolated onto the simulation grid
before the difference is computed.
"""

from __future__ import annotations

import json
import sys
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

# To use the working, copy default and rename to working, edit as wanted changes will be ignored on the working config by git.
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "plots" / "ffs_working.json"

SIM_CMAP = "viridis"
DNS_CMAP = "viridis"
ERROR_CMAP = "coolwarm"
FIELD_LEVELS = 100


def load_ascii(path: Path) -> np.ndarray:
    return np.genfromtxt(path, skip_header=1)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _ffs_floor(x: np.ndarray) -> np.ndarray:
    """Piecewise floor used to mask the solid step region."""

    return np.where(x < 0.0, -0.01, 0.0)


def _ffs_triangle_mask(tri: mtri.Triangulation, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask triangles that fall below the FFS solid boundary."""

    triangles = tri.triangles
    tri_x = x[triangles]
    tri_y = y[triangles]
    floor_y = _ffs_floor(tri_x)
    return np.any(tri_y < floor_y - 1e-12, axis=1)


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

    error_limit = float(np.max(valid_error))
    error_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-error_limit, vmax=error_limit)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    sim_plot = axs[0].tricontourf(
        tri,
        np.ma.masked_invalid(sim_values),
        levels=FIELD_LEVELS,
        cmap=SIM_CMAP,
    )
    axs[0].set_title(f"Simulation {sim_name}")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sim_plot, ax=axs[0], label=sim_name)

    dns_plot = axs[1].tricontourf(
        tri,
        np.ma.masked_invalid(dns_on_sim),
        levels=FIELD_LEVELS,
        cmap=DNS_CMAP,
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


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _normalize_transform_spec(spec: dict[str, str]) -> Callable[[np.ndarray], np.ndarray]:
    kind = spec.get("kind")
    if kind == "subtract_min":
        return lambda values: values - np.min(values)
    raise ValueError(f"Unsupported transform kind: {kind}")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config_path = config_path if config_path.is_absolute() else (Path.cwd() / config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_config(config_path)

    config_name = cfg["name"]
    output_dir = (Path(__file__).resolve().parent / "plots" / config_name).resolve()

    sim_path = _resolve_path(cfg["simulation"]["path"])
    dns_path = _resolve_path(cfg["dns"]["path"])

    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")
    if not dns_path.is_file():
        raise FileNotFoundError(f"DNS file not found: {dns_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sim_data = load_ascii(sim_path)
    dns_df = load_csv(dns_path)

    sim_x = sim_data[:, int(cfg["simulation"]["x_column"])]
    sim_y = sim_data[:, int(cfg["simulation"]["y_column"])]
    dns_x = dns_df[cfg["dns"]["x_column"]].to_numpy()
    dns_y = dns_df[cfg["dns"]["y_column"]].to_numpy()

    sim_field_columns = {name: int(column) for name, column in cfg["simulation"]["fields"].items()}
    dns_field_columns = {name: column for name, column in cfg["dns"]["fields"].items()}

    sim_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for field_name, transform_spec in cfg.get("simulation", {}).get("transforms", {}).items():
        sim_transforms[field_name] = _normalize_transform_spec(transform_spec)

    dns_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for field_name, transform_spec in cfg.get("dns", {}).get("transforms", {}).items():
        dns_transforms[field_name] = _normalize_transform_spec(transform_spec)

    sim_fields: dict[str, np.ndarray] = {}
    for field_name, column in sim_field_columns.items():
        sim_fields[field_name] = _apply_transform(sim_data[:, column], sim_transforms, field_name)

    dns_fields: dict[str, np.ndarray] = {}
    for field_name, column in dns_field_columns.items():
        dns_fields[field_name] = _apply_transform(dns_df[column].to_numpy(), dns_transforms, field_name)

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

    print(f"All plots written to {output_dir}")


if __name__ == "__main__":
    main()
