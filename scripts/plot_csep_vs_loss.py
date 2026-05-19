"""
Plot BO sweep results for a two-parameter (Csep, Cnw) experiment.

Shows three views:
    1. Csep vs loss   (projection onto the Csep axis)
    2. Cnw vs loss    (projection onto the Cnw axis)
    3. 3D scatter     (Csep, Cnw, loss together)

Usage::

    python scripts/plot_csep_vs_loss.py results/experiments/periodic_hills_csep_v1/metadata.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # interactive backend (Windows-friendly)
import matplotlib.pyplot as plt
import pandas as pd

# Importing the 3D toolkit registers the "3d" projection with matplotlib.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _scatter_1d(ax, df, param_col, best, label):
    """One 2D scatter: parameter vs loss, coloured by trial index."""
    sc = ax.scatter(
        df[param_col], df["score"],
        c=df.index, cmap="viridis", s=50, alpha=0.8,
        edgecolors="black", linewidths=0.5,
    )
    ax.scatter(
        best[param_col], best["score"],
        marker="*", s=300, color="red", edgecolors="black",
        linewidths=1.0, zorder=10,
        label=f"Best: {label}={best[param_col]:.4f}, score={best['score']:.4g}",
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return sc


def plot_bo_results(csv_path: Path) -> None:
    """Read metadata.csv and show three views of the sweep."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["geko_csep", "geko_cnw", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns missing in {csv_path}: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    best = df.loc[df["score"].idxmin()]
    title = f"BO sweep: {csv_path.parent.name}"

    fig = plt.figure(figsize=(16, 5))

    # --- 1. Csep vs loss ---
    ax1 = fig.add_subplot(1, 3, 1)
    _scatter_1d(ax1, df, "geko_csep", best, r"$C_\mathrm{sep}$")
    ax1.set_title("Csep vs loss")

    # --- 2. Cnw vs loss ---
    ax2 = fig.add_subplot(1, 3, 2)
    _scatter_1d(ax2, df, "geko_cnw", best, r"$C_\mathrm{nw}$")
    ax2.set_title("Cnw vs loss")

    # --- 3. 3D scatter: Csep, Cnw, loss ---
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc3 = ax3.scatter(
        df["geko_csep"], df["geko_cnw"], df["score"],
        c=df.index, cmap="viridis", s=50, alpha=0.8,
        edgecolors="black", linewidths=0.3,
    )
    ax3.scatter(
        [best["geko_csep"]], [best["geko_cnw"]], [best["score"]],
        marker="*", s=300, color="red", edgecolors="black",
        linewidths=1.0,
        label=(
            f"Best: Csep={best['geko_csep']:.3f}, "
            f"Cnw={best['geko_cnw']:.3f}, "
            f"score={best['score']:.4g}"
        ),
    )
    ax3.set_xlabel(r"$C_\mathrm{sep}$")
    ax3.set_ylabel(r"$C_\mathrm{nw}$")
    ax3.set_zlabel("Loss")
    ax3.set_title("Csep, Cnw vs loss")
    ax3.legend(loc="best", fontsize=8)

    # Shared colourbar (trial index) on the right of the 3D plot.
    cbar = fig.colorbar(sc3, ax=ax3, shrink=0.7, pad=0.12)
    cbar.set_label("Trial index")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    csv_path = Path(sys.argv[1]).resolve()
    plot_bo_results(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())