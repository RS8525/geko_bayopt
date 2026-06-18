"""
Adapter: produce plots from a BO experiment's ``metadata.csv``
using a Plotting-class-compatible interface.

Differences from the original Plotting class:
    - Reads ``metadata.csv`` directly (no need to pass ``history`` manually).
    - Uses scattered-data interpolation (griddata) for the 2D plot, since
      BO samples are NOT on a regular grid.
    - Plots a running MINIMUM (BO minimizes here, not maximizes).
    - The 1D interpolation method uses ``score`` only; the original
      multi-curve view (Ux_score, Uy_score, cp_score, field_score) cannot
      be reproduced because those per-field scores are not currently
      persisted to metadata.csv.

Usage::

    python scripts/plot_from_metadata.py results/experiments/periodic_hills_csep_v1/metadata.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # interactive on Windows; swap to "Agg" if no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, griddata


class BOMetadataPlotting:
    """Plotting wrapper that reads a BO experiment's metadata.csv.

    Parameters
    ----------
    csv_path : str | Path
        Path to ``metadata.csv``.
    output_dir : str | Path, optional
        Where to save PNGs. Defaults to ``<csv_dir>/plots/``.
    n_initial : int, optional
        Number of initial random / Sobol points (vertical line on history
        plot). Pulled from your config's ``optimizer.n_initial``.
    figsize : tuple
        Default figure size.
    """

    def __init__(
        self,
        csv_path: str | Path,
        output_dir: str | Path | None = None,
        n_initial: int | None = None,
        figsize: tuple = (10, 6),
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"metadata.csv not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)
        self.df["trial_index"] = self.df.index + 1

        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_initial = n_initial
        self.figsize = figsize

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _save(self, filename: str) -> Path:
        path = self.output_dir / filename
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {path}")
        return path

    def _check_columns(self, *needed: str) -> None:
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in {self.csv_path.name}: {missing}. "
                f"Available: {self.df.columns.tolist()}"
            )

    # ------------------------------------------------------------------ #
    # 1. Convergence history (running minimum)                           #
    # ------------------------------------------------------------------ #

    def plot_running_minimum(
        self,
        reference_score: float | None = None,
        show: bool = True,
    ) -> None:
        """Plot the running-minimum score vs. trial index.

        Replaces the original ``plot_RunningMaximum``. We minimize here.
        """
        self._check_columns("score")

        scores = self.df["score"].to_numpy()
        running_min = np.minimum.accumulate(scores)
        trials = self.df["trial_index"].to_numpy()

        plt.figure(figsize=self.figsize)
        plt.plot(trials, scores, "o", alpha=0.4, color="gray", label="Per-trial score")
        plt.plot(trials, running_min, "s--", color="black",
                 label="Running minimum (best so far)")

        if reference_score is not None:
            plt.axhline(reference_score, color="blue", linestyle="-",
                        label=f"Reference: {reference_score:.4g}")

        if self.n_initial is not None:
            plt.axvline(self.n_initial, color="red", linestyle="--",
                        label="End of initial sampling")

        plt.xlabel("Trial")
        plt.ylabel("Score (lower is better)")
        plt.title(f"BO history: {self.csv_path.parent.name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            self._save("bo_history.png")

    # ------------------------------------------------------------------ #
    # 2. 1D parameter vs. score with monotone interpolation              #
    # ------------------------------------------------------------------ #

    def plot_1d(
        self,
        param: str,
        reference: float | None = None,
        show: bool = True,
    ) -> None:
        """Plot ``score`` against one parameter, with a PCHIP interpolant
        through the (sorted, deduplicated) samples.

        BO often revisits values; we keep the minimum score at each
        repeated point so the interpolation through them is well-defined.
        """
        self._check_columns(param, "score")

        # If BO sampled the same x twice, keep the lower score so the
        # spline can be built (PCHIP needs strictly-increasing x).
        agg = (self.df[[param, "score"]]
               .groupby(param, as_index=False)
               .min()
               .sort_values(param))
        x = agg[param].to_numpy()
        y = agg["score"].to_numpy()

        plt.figure(figsize=self.figsize)
        plt.scatter(self.df[param], self.df["score"],
                    c=self.df["trial_index"], cmap="viridis", s=40,
                    edgecolors="black", linewidths=0.4, label="Trials")

        if len(x) >= 2:
            x_smooth = np.linspace(x.min(), x.max(), 400)
            interp = PchipInterpolator(x, y)
            plt.plot(x_smooth, interp(x_smooth), "k--", linewidth=1.4,
                     alpha=0.7, label="PCHIP through per-x minimum")

        best = self.df.loc[self.df["score"].idxmin()]
        plt.scatter(best[param], best["score"], marker="*", s=300,
                    color="red", edgecolors="black", zorder=10,
                    label=f"Best: {param}={best[param]:.4f}, score={best['score']:.4g}")

        if reference is not None:
            plt.axvline(reference, linestyle="--", color="blue",
                        label=f"Reference {param}={reference:.4f}")

        cbar = plt.colorbar(label="Trial index")

        plt.xlabel(param)
        plt.ylabel("Score")
        plt.title(f"{param} vs score")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            self._save(f"score_vs_{param}.png")

    # ------------------------------------------------------------------ #
    # 3. 2D parameter sweep with scattered-data interpolation            #
    # ------------------------------------------------------------------ #

    def plot_2d(
        self,
        x_param: str,
        y_param: str,
        x_ref: float | None = None,
        y_ref: float | None = None,
        n_smooth: int = 100,
        show: bool = True,
    ) -> None:
        """Plot a 2D contour of ``score`` over two parameters.

        Uses ``scipy.interpolate.griddata`` (scattered-data interpolation)
        because BO samples are not on a regular grid. Falls back to
        ``nearest`` for points outside the convex hull of the samples.
        """
        self._check_columns(x_param, y_param, "score")

        x = self.df[x_param].to_numpy()
        y = self.df[y_param].to_numpy()
        z = self.df["score"].to_numpy()

        if len(self.df) < 4:
            raise ValueError(
                "Need at least 4 trials to interpolate a 2D surface; "
                f"only {len(self.df)} found."
            )

        # Build a target grid over the sample envelope
        xg = np.linspace(x.min(), x.max(), n_smooth)
        yg = np.linspace(y.min(), y.max(), n_smooth)
        Xg, Yg = np.meshgrid(xg, yg, indexing="xy")

        # Linear interp inside convex hull, nearest outside (no NaN gaps)
        Z_lin = griddata((x, y), z, (Xg, Yg), method="linear")
        Z_near = griddata((x, y), z, (Xg, Yg), method="nearest")
        Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

        fig, ax = plt.subplots(figsize=self.figsize)
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="Score")

        # Sample overlay coloured by score
        ax.scatter(x, y, c=z, cmap="viridis", edgecolors="white",
                   linewidths=0.5, s=40, zorder=5, label="Trials")

        best = self.df.loc[self.df["score"].idxmin()]
        ax.scatter(best[x_param], best[y_param], marker="*", s=320,
                   color="red", edgecolors="black", linewidths=1.0,
                   zorder=10,
                   label=f"Best: ({best[x_param]:.3f}, {best[y_param]:.3f})")

        if x_ref is not None:
            ax.axvline(x_ref, linestyle="--", color="red", linewidth=1.2,
                       alpha=0.6, label=f"Reference {x_param}")
        if y_ref is not None:
            ax.axhline(y_ref, linestyle="--", color="orange", linewidth=1.2,
                       alpha=0.6, label=f"Reference {y_param}")

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"Score over ({x_param}, {y_param})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            self._save(f"score_2d_{x_param}_vs_{y_param}.png")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    csv_path = Path(sys.argv[1]).resolve()
    plotter = BOMetadataPlotting(csv_path, n_initial=8)

    # Detect which parameter columns exist
    df = plotter.df
    param_cols = [c for c in df.columns if c.startswith("geko_")]
    print(f"[plot] detected parameters: {param_cols}")

    # 1. Convergence history
    plotter.plot_running_minimum()

    # 2. 1D plot per parameter
    for p in param_cols:
        plotter.plot_1d(p)

    # 3. 2D plot if we have at least two parameters
    if len(param_cols) >= 2:
        plotter.plot_2d(param_cols[0], param_cols[1])

    return 0


if __name__ == "__main__":
    sys.exit(main())