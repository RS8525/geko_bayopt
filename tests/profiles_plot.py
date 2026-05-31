"""
Profile plots: DNS vs Simulation vertical profiles at given x locations.

Usage:
    python profile_plots.py

Configure the X_LOCATIONS and FIELDS lists at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

SIM_PATH = (
    REPO_ROOT
    / "results/fluent/periodic_hills_2800_v1"
    / "alpha1.0_Re2800_Csep0.8557517256760887_Cnw0.500973460097605.ascii"
)
DNS_PATH = (
    REPO_ROOT
    / "data/dns/periodic_hills/pehill-29-cases-DNS/alph10-9-3036"
    / "dns_avg_Re2800_columnwise_organized.ascii"
)

# ---------------------------------------------------------------------------
# Column name mapping: field label -> (dns_col, sim_col)
# ---------------------------------------------------------------------------
FIELD_MAP = {
    "Ux": ("x-velocity", "x-velocity"),
    "Uy": ("y-velocity", "y-velocity"),
    "turb-kinetic-energy": ("k", "turb-kinetic-energy"),
}

X_TOL = 0.1  # tolerance for selecting points near a given x


def load_data(sim_path: Path, dns_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = pd.read_csv(sim_path, sep=r"\s+")
    dns = pd.read_csv(dns_path, sep=r"\s+")
    return sim, dns


def plot_profiles(
    x_locations: list[float],
    fields: list[str],
    sim: pd.DataFrame,
    dns: pd.DataFrame,
    tol: float = X_TOL,
) -> None:
    """
    Plot vertical profiles (field vs y) at each x location for each field.
    Saves one image per field.

    Parameters
    ----------
    x_locations : list of float
        x positions at which to extract profiles.
    fields : list of str
        Physical fields to plot. Each must be a key in FIELD_MAP.
        Accepted values: 'Ux', 'Uy', 'turb-kinetic-energy'.
    sim : pd.DataFrame
        Simulation data.
    dns : pd.DataFrame
        DNS reference data.
    tol : float
        Half-width of the x-window used to select points near each x location.
    """
    unknown = [f for f in fields if f not in FIELD_MAP]
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}. Choose from {list(FIELD_MAP)}")

    n_x = len(x_locations)

    for field in fields:
        dns_col, sim_col = FIELD_MAP[field]

        fig, axes = plt.subplots(1, n_x, figsize=(5 * n_x, 4), squeeze=False)

        for col, x_val in enumerate(x_locations):
            ax = axes[0][col]

            sim_slice = (
                sim[np.abs(sim["x-coordinate"] - x_val) < tol]
                .sort_values("y-coordinate")
            )
            dns_slice = (
                dns[np.abs(dns["x-coordinate"] - x_val) < tol]
                .sort_values("y-coordinate")
            )

            ax.plot(
                sim_slice[sim_col],
                sim_slice["y-coordinate"],
                color="tab:red",
                linewidth=1.5,
                label="Simulation",
            )
            ax.plot(
                dns_slice[dns_col],
                dns_slice["y-coordinate"],
                color="tab:blue",
                linewidth=1.5,
                label="DNS",
            )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_xlabel(field)
            ax.set_ylabel("y")
            ax.set_title(f"{field} — x = {x_val}  (tol = {tol})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = (
            REPO_ROOT
            / f"results/experiments/periodic_hills_2800_v1/plots/{field}_profiles.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"Saved: {out_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point — configure here
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_LOCATIONS = [2.0, 4.5, 7.0]
    FIELDS = ["Ux", "Uy", "turb-kinetic-energy"]

    sim, dns = load_data(SIM_PATH, DNS_PATH)
    plot_profiles(X_LOCATIONS, FIELDS, sim, dns)