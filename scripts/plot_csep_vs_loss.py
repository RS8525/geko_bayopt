"""
Plot Csep vs loss for a BO experiment (interactive window).

Usage::

    python scripts/plot_csep_vs_loss.py results/experiments/periodic_hills_csep_v1/metadata.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_csep_vs_loss(csv_path: Path) -> None:
    """Read metadata.csv and show a Csep-vs-score scatter plot."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "geko_csep" not in df.columns:
        raise KeyError(
            f"Column 'geko_csep' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    best = df.loc[df["score"].idxmin()]

    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        df["geko_csep"], df["score"],
        c=df.index, cmap="viridis", s=50, alpha=0.8,
        edgecolors="black", linewidths=0.5,
    )

    ax.scatter(
        best["geko_csep"], best["score"],
        marker="*", s=300, color="red", edgecolors="black",
        linewidths=1.0, zorder=10,
        label=f"Best: Csep={best['geko_csep']:.4f}, score={best['score']:.4g}",
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Trial index")

    ax.set_xlabel(r"$C_\mathrm{sep}$")
    ax.set_ylabel("Loss (MSE on cp)")
    ax.set_title(f"BO sweep: {csv_path.parent.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    csv_path = Path(sys.argv[1]).resolve()
    plot_csep_vs_loss(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())