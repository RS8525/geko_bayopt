"""
Profile plots: DNS vs Simulation vertical profiles at given x locations.

Usage:
    python profiles_plot_ph.py
    python profiles_plot_ph.py ../per_hill/plots/plot_config.json

JSON contains:
    name
    simulation: path, x, y, fields
    dns:        path, x, y, fields
    plots:      fields, x_locations, x_tol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "per_hill" / "plots" / "plot_config.json"


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(config_path: str | Path) -> dict:
    with open(_repo_path(config_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)


def _col(df: pd.DataFrame, col):
    if isinstance(col, int):
        return df.iloc[:, col]
    return df[col]


def load_data(sim_cfg: dict, dns_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = _read_table(_repo_path(sim_cfg["path"]))
    dns = _read_table(_repo_path(dns_cfg["path"]))

    # Keep exactly the old derived DNS turbulent-viscosity calculation when
    # the required columns exist.
    if {"k", "dissipation", "density"}.issubset(dns.columns):
        k = dns["k"].to_numpy()
        eps = dns["dissipation"].to_numpy()
        rho = dns["density"].to_numpy()

        eps_safe = np.where(eps > 1e-12, eps, np.nan)

        dns["viscosity-turb"] = rho * 0.09 * k**2 / eps_safe

    return sim, dns


def plot_profiles(
    x_locations: list[float],
    fields: list[str],
    sim: pd.DataFrame,
    dns: pd.DataFrame,
    sim_cfg: dict,
    dns_cfg: dict,
    output_dir: Path,
    tol: float,
) -> None:
    """
    Plot vertical profiles (field vs y) at each x location for each field.
    Saves one image per field.
    """
    unknown_sim = [f for f in fields if f not in sim_cfg["fields"]]
    unknown_dns = [f for f in fields if f not in dns_cfg["fields"]]
    if unknown_sim:
        raise ValueError(f"Unknown simulation fields: {unknown_sim}. Choose from {list(sim_cfg['fields'])}")
    if unknown_dns:
        raise ValueError(f"Unknown DNS fields: {unknown_dns}. Choose from {list(dns_cfg['fields'])}")

    n_x = len(x_locations)

    sim_x_col = sim_cfg["x"]
    sim_y_col = sim_cfg["y"]
    dns_x_col = dns_cfg["x"]
    dns_y_col = dns_cfg["y"]

    for field in fields:
        sim_col = sim_cfg["fields"][field]
        dns_col = dns_cfg["fields"][field]

        fig, axes = plt.subplots(1, n_x, figsize=(5 * n_x, 4), squeeze=False)

        for col, x_val in enumerate(x_locations):
            ax = axes[0][col]

            sim_slice = (
                sim[np.abs(_col(sim, sim_x_col) - x_val) < tol]
                .sort_values(by=sim_y_col if isinstance(sim_y_col, str) else sim.columns[sim_y_col])
            )
            dns_slice = (
                dns[np.abs(_col(dns, dns_x_col) - x_val) < tol]
                .sort_values(by=dns_y_col if isinstance(dns_y_col, str) else dns.columns[dns_y_col])
            )

            ax.plot(
                _col(sim_slice, sim_col),
                _col(sim_slice, sim_y_col),
                color="tab:red",
                linewidth=1.5,
                alpha=0.7,
                label="Simulation",
            )
            ax.plot(
                _col(dns_slice, dns_col),
                _col(dns_slice, dns_y_col),
                color="tab:blue",
                linewidth=1.5,
                alpha=0.7,
                label="DNS",
            )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_xlabel(field)
            ax.set_ylabel("y")
            ax.set_title(f"{field} — x = {x_val}  (tol = {tol})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"{field}_profiles.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"Saved: {out_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point — configure in JSON
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to JSON config. Default: scripts/per_hill/plots/plot_config.json",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    sim_cfg = cfg["simulation"]
    dns_cfg = cfg["dns"]
    plots_cfg = cfg["plots"]

    fields = plots_cfg["fields"]
    x_locations = plots_cfg["x_locations"]
    x_tol = plots_cfg["x_tol"]

    sim, dns = load_data(sim_cfg, dns_cfg)

    output_dir = REPO_ROOT / "results" / "experiments" / cfg["name"] / "plots"
    plot_profiles(x_locations, fields, sim, dns, sim_cfg, dns_cfg, output_dir, x_tol)
