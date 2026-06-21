"""Localize FFS field error by coarse spatial bins for one Fluent run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DNS_PATH = ROOT / "data" / "dns" / "ffs" / "FFS_Reh6000_SBES_Node_2D.csv"
ASCII_PATH = (
    ROOT
    / "results"
    / "fluent"
    / "ffs_csep_v3_density_validation"
    / "alpha1.0_Re6000_Csep0.8957.ascii"
)


def _density_weights(coords: np.ndarray, k: int = 8) -> np.ndarray:
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=min(k, len(coords)))
    local_radius = np.mean(distances[:, 1:], axis=1)
    weights = np.maximum(local_radius, 1e-12) ** 2
    return weights / np.mean(weights)


def _load_sim(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    coords = df[["x-coordinate", "y-coordinate"]].to_numpy()
    values = df["x-velocity"].to_numpy()
    return coords, values


def _print_bins(label: str, axis_values: np.ndarray, diff: np.ndarray, weights: np.ndarray) -> None:
    edges = np.linspace(np.min(axis_values), np.max(axis_values), 11)
    print(label)
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (axis_values >= low) & (axis_values < high)
        if not np.any(keep):
            continue
        local_w = weights[keep]
        local_diff = diff[keep]
        rmse = np.sqrt(np.sum(local_w * local_diff**2) / np.sum(local_w))
        mean_abs = np.sum(local_w * np.abs(local_diff)) / np.sum(local_w)
        weight_fraction = np.sum(local_w) / np.sum(weights)
        print(
            f"  [{low:.4g}, {high:.4g}): rmse={rmse:.6g}, "
            f"mae={mean_abs:.6g}, weight={weight_fraction:.2%}"
        )


def main() -> None:
    dns_df = pd.read_csv(
        DNS_PATH,
        usecols=["x-coordinate", "y-coordinate", "mean-x-velocity"],
    )
    dns_coords = dns_df[["x-coordinate", "y-coordinate"]].to_numpy()
    dns_ux = dns_df["mean-x-velocity"].to_numpy()
    sim_coords, sim_ux = _load_sim(ASCII_PATH)

    sim_interp = griddata(sim_coords, sim_ux, dns_coords, method="linear")
    valid = np.isfinite(sim_interp)
    coords = dns_coords[valid]
    diff = dns_ux[valid] - sim_interp[valid]
    weights = _density_weights(dns_coords)[valid]

    print(f"Run: {ASCII_PATH.name}")
    print(f"valid interpolation: {np.mean(valid):.3%}")
    print(f"global density-weighted Ux RMSE: {np.sqrt(np.sum(weights * diff**2) / np.sum(weights)):.6g}")
    _print_bins("Ux error by x", coords[:, 0], diff, weights)
    _print_bins("Ux error by y", coords[:, 1], diff, weights)


if __name__ == "__main__":
    main()
