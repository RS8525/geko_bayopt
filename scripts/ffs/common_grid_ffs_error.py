"""Evaluate FFS DNS/RANS errors on a common physical grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
DNS_PATH = ROOT / "data" / "dns" / "ffs" / "FFS_Reh6000_SBES_Node_2D.csv"
FLUENT_DIR = ROOT / "results" / "fluent" / "ffs_csep_v3_density_validation"
RUN_IDS = [
    "alpha1.0_Re6000_Csep1.937",
    "alpha1.0_Re6000_Csep0.8957",
    "alpha1.0_Re6000_Csep0.7667",
    "alpha1.0_Re6000_Csep1.7805",
    "alpha1.0_Re6000_Csep1.5386",
]


def _ffs_floor(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0.0, -0.01, 0.0)


def _idw(source_coords: np.ndarray, source_values: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    tree = cKDTree(source_coords)
    distances, indices = tree.query(target_coords, k=8)
    exact = np.any(distances == 0.0, axis=1)
    out = np.empty(len(target_coords), dtype=float)
    if np.any(exact):
        out[exact] = source_values[indices[exact, np.argmin(distances[exact], axis=1)]]
    need = ~exact
    if np.any(need):
        weights = 1.0 / np.maximum(distances[need], 1e-12) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        out[need] = np.sum(weights * source_values[indices[need]], axis=1)
    return out


def _load_dns() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(
        DNS_PATH,
        usecols=[
            "x-coordinate",
            "y-coordinate",
            "mean-x-velocity",
            "mean-y-velocity",
            "mean-pressure",
        ],
    )
    coords = df[["x-coordinate", "y-coordinate"]].to_numpy()
    fields = {
        "Ux": df["mean-x-velocity"].to_numpy(),
        "Uy": df["mean-y-velocity"].to_numpy(),
        "cp": df["mean-pressure"].to_numpy(),
    }
    return coords, fields


def _load_sim(run_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(
        FLUENT_DIR / f"{run_id}.ascii",
        sep=r"\s+",
        engine="python",
        skipinitialspace=True,
    )
    df.columns = [c.strip() for c in df.columns]
    coords = df[["x-coordinate", "y-coordinate"]].to_numpy()
    fields = {
        "Ux": df["x-velocity"].to_numpy(),
        "Uy": df["y-velocity"].to_numpy(),
        "cp": df["pressure"].to_numpy(),
    }
    return coords, fields


def _common_grid(coords: np.ndarray, nx: int = 360, ny: int = 120) -> np.ndarray:
    x = np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), nx)
    y = np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), ny)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    keep = grid[:, 1] >= (_ffs_floor(grid[:, 0]) - 1e-12)
    return grid[keep]


def _normalized_error(dns_values: np.ndarray, sim_values: np.ndarray, field_name: str) -> float:
    if field_name == "cp":
        dns_values = dns_values - np.mean(dns_values)
        sim_values = sim_values - np.mean(sim_values)
    denom = max(float(np.std(dns_values)), 1e-8)
    return float(np.sqrt(np.mean((dns_values - sim_values) ** 2)) / denom)


def main() -> None:
    dns_coords, dns_fields = _load_dns()
    grid = _common_grid(dns_coords)
    dns_on_grid = {name: _idw(dns_coords, values, grid) for name, values in dns_fields.items()}

    print(f"Common grid points: {len(grid):,}")
    print("run_id, cp, Ux, Uy, sum")
    for run_id in RUN_IDS:
        sim_coords, sim_fields = _load_sim(run_id)
        contributions = {}
        for name, dns_values in dns_on_grid.items():
            sim_values = _idw(sim_coords, sim_fields[name], grid)
            contributions[name] = _normalized_error(dns_values, sim_values, name)
        print(
            f"{run_id}, {contributions['cp']:.6g}, {contributions['Ux']:.6g}, "
            f"{contributions['Uy']:.6g}, {sum(contributions.values()):.6g}"
        )


if __name__ == "__main__":
    main()
