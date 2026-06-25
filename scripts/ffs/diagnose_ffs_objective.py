"""Diagnostics for FFS DNS/RANS objective alignment.

This script is intentionally read-only. It compares the DNS CSV and one Fluent
ASCII export using the same field names as the optimization objective, then
prints coordinate ranges, field ranges, interpolation coverage, and per-field
errors under a few pressure-gauge conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "ffs_retired" / "ffs_csep_v3.json"
ASCII_PATH = ROOT / "results" / "fluent" / "ffs_csep_v3" / "alpha1.0_Re6000_Csep0.9421.ascii"


def _stats(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    return (
        f"min={np.nanmin(values):.6g}, max={np.nanmax(values):.6g}, "
        f"mean={np.nanmean(values):.6g}, std={np.nanstd(values):.6g}"
    )


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_dns(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    columns = [
        "x-coordinate",
        "y-coordinate",
        "mean-x-velocity",
        "mean-y-velocity",
        "mean-pressure",
    ]
    df = pd.read_csv(path, usecols=columns)
    coords = df[["x-coordinate", "y-coordinate"]].to_numpy()
    fields = {
        "Ux": df["mean-x-velocity"].to_numpy(),
        "Uy": df["mean-y-velocity"].to_numpy(),
        "p": df["mean-pressure"].to_numpy(),
    }
    return coords, fields


def _load_sim(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    coords = df[["x-coordinate", "y-coordinate"]].to_numpy()
    fields = {
        "Ux": df["x-velocity"].to_numpy(),
        "Uy": df["y-velocity"].to_numpy(),
        "p": df["pressure"].to_numpy(),
    }
    return coords, fields


def _structured_area_weights(coords: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    xu, x_inv = np.unique(x, return_inverse=True)
    yu, y_inv = np.unique(y, return_inverse=True)

    def widths(u: np.ndarray) -> np.ndarray:
        w = np.empty_like(u)
        if len(u) == 1:
            return np.ones_like(u)
        w[1:-1] = (u[2:] - u[:-2]) / 2.0
        w[0] = u[1] - u[0]
        w[-1] = u[-1] - u[-2]
        return w

    return widths(xu)[x_inv] * widths(yu)[y_inv]


def _density_area_weights(coords: np.ndarray, k: int = 8) -> np.ndarray:
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=min(k, len(coords)))
    if distances.ndim == 1:
        return np.ones(len(coords), dtype=float)
    local_radius = np.mean(distances[:, 1:], axis=1)
    weights = np.maximum(local_radius, 1e-12) ** 2
    return weights / np.mean(weights)


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _wstd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _wmean(values, weights)
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))


def _field_error(
    dns_coords: np.ndarray,
    dns_values: np.ndarray,
    sim_coords: np.ndarray,
    sim_values: np.ndarray,
    *,
    gauge: str | None = None,
    weight_mode: str = "structured",
) -> tuple[float, float, dict[str, float]]:
    sim_interp = griddata(sim_coords, sim_values, dns_coords, method="linear")
    valid = np.isfinite(sim_interp)
    if weight_mode == "structured":
        all_weights = _structured_area_weights(dns_coords)
    elif weight_mode == "density":
        all_weights = _density_area_weights(dns_coords)
    elif weight_mode == "uniform":
        all_weights = np.ones(len(dns_coords), dtype=float)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    weights = all_weights[valid]
    dns_v = dns_values[valid]
    sim_v = sim_interp[valid]

    if gauge == "mean":
        dns_v = dns_v - _wmean(dns_v, weights)
        sim_v = sim_v - _wmean(sim_v, weights)
    elif gauge == "min":
        dns_v = dns_v - np.min(dns_v)
        sim_v = sim_v - np.min(sim_v)

    denom = max(_wstd(dns_v, weights), 1e-8)
    rmse = np.sqrt(_wmean((dns_v - sim_v) ** 2, weights))
    diff = dns_v - sim_v
    extra = {
        "weighted_rmse": float(rmse),
        "weighted_dns_std": float(denom),
        "plain_rmse": float(np.sqrt(np.mean(diff**2))),
        "plain_dns_std": float(np.std(dns_v)),
        "plain_normalized": float(np.sqrt(np.mean(diff**2)) / max(np.std(dns_v), 1e-8)),
        "mean_diff": float(np.mean(diff)),
    }
    return float(rmse / denom), float(np.mean(valid)), extra


def _ffs_floor(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0.0, -0.01, 0.0)


def main() -> None:
    cfg = _load_config()
    dns_path = ROOT / cfg["case"]["options"]["dns_path"]
    dns_coords, dns_fields = _load_dns(dns_path)
    sim_coords, sim_fields = _load_sim(ASCII_PATH)

    print(f"Config: {CONFIG_PATH}")
    print(f"DNS:    {dns_path}")
    print(f"Sim:    {ASCII_PATH}")
    print()

    print("Coordinate ranges")
    print(f"DNS x: {_stats(dns_coords[:, 0])}")
    print(f"DNS y: {_stats(dns_coords[:, 1])}")
    print(f"Sim x: {_stats(sim_coords[:, 0])}")
    print(f"Sim y: {_stats(sim_coords[:, 1])}")
    print()

    unique_dns = np.unique(dns_coords, axis=0)
    unique_x = len(np.unique(dns_coords[:, 0]))
    unique_y = len(np.unique(dns_coords[:, 1]))
    below_floor_dns = dns_coords[:, 1] < (_ffs_floor(dns_coords[:, 0]) - 1e-12)
    below_floor_sim = sim_coords[:, 1] < (_ffs_floor(sim_coords[:, 0]) - 1e-12)
    print("Mesh/topology checks")
    print(f"DNS rows={len(dns_coords):,}, unique xy={len(unique_dns):,}")
    print(f"DNS unique x={unique_x:,}, unique y={unique_y:,}")
    print(f"DNS duplicate xy rows={len(dns_coords) - len(unique_dns):,}")
    print(f"DNS below FFS floor={np.mean(below_floor_dns):.3%}")
    print(f"Sim below FFS floor={np.mean(below_floor_sim):.3%}")
    print()

    print("Field ranges")
    for name in ["Ux", "Uy", "p"]:
        print(f"DNS {name}: {_stats(dns_fields[name])}")
        print(f"Sim {name}: {_stats(sim_fields[name])}")
    print()

    print("Per-field normalized errors on DNS grid")
    for name in ["Ux", "Uy"]:
        for weight_mode in ["structured", "density", "uniform"]:
            err, coverage, extra = _field_error(
                dns_coords,
                dns_fields[name],
                sim_coords,
                sim_fields[name],
                weight_mode=weight_mode,
            )
            print(
                f"{name} ({weight_mode} weights): error={err:.6g}, "
                f"rmse={extra['weighted_rmse']:.6g}, mean_diff={extra['mean_diff']:.6g}, "
                f"interpolation_coverage={coverage:.3%}"
            )

    for gauge in [None, "mean", "min"]:
        for weight_mode in ["structured", "density", "uniform"]:
            err, coverage, extra = _field_error(
                dns_coords,
                dns_fields["p"],
                sim_coords,
                sim_fields["p"],
                gauge=gauge,
                weight_mode=weight_mode,
            )
            label = "raw" if gauge is None else f"{gauge}-gauged"
            print(
                f"p {label} ({weight_mode} weights): error={err:.6g}, "
                f"mean_diff={extra['mean_diff']:.6g}, interpolation_coverage={coverage:.3%}"
            )


if __name__ == "__main__":
    main()
