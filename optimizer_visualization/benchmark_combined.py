"""
benchmark_combined.py

Produces a single figure with all eight optimizers arranged in rows —
1-D trajectory on the left, 2-D contour trajectory on the right.

Output: optimizer_visualization/plots/benchmark_combined.png

Run from the project root:
    .venv/Scripts/python.exe optimizer_visualization/benchmark_combined.py
"""

from __future__ import annotations

from pathlib import Path
from matplotlib.cm import ScalarMappable

from _benchmark_core import (
    plt,
    OPTIMIZERS,
    run_1d, run_2d,
    plot_1d_ax, plot_2d_ax,
    N_1D, N_2D,
    X1D_STAR, X2D_STAR,
    _CMAP_TRAJ, _NORM_TRAJ,
)

_HERE  = Path(__file__).resolve().parent
_PLOTS = _HERE / "plots"
_PLOTS.mkdir(exist_ok=True)


def main() -> None:
    n_opt = len(OPTIMIZERS)

    fig, axes = plt.subplots(n_opt, 2, figsize=(15.5, n_opt * 4.4))

    fig.suptitle(
        "Optimizer Benchmark  —  Convergence on Synthetic Test Functions\n"
        r"$f_{\mathrm{1D}}(x) = -2.5x\,\sin(2.5x)$"
        r"$\qquad\qquad$"
        r"$f_{\mathrm{2D}}$ : scaled Himmelblau  (4 global optima,  $f^* = 0$)",
        fontsize=11.5, fontweight="bold", y=1.004,
    )

    for row, (name, slug, ps1, ps2, sec1d, sec2d) in enumerate(OPTIMIZERS):
        label_safe = name.split("  (")[0].replace("→", "->").replace("–", "-")
        print(f"  [{row + 1}/{n_opt}]  {label_safe} ...", flush=True)

        xs1, ys1 = run_1d(sec1d)
        xs2, ys2 = run_2d(sec2d)

        plot_1d_ax(axes[row, 0], xs1, ys1, title=name, phase_split=ps1)
        plot_2d_ax(axes[row, 1], xs2, ys2, phase_split=ps2)

    # Column headers above the first row
    for col_idx, header in enumerate([
        f"1-D  ·  {N_1D} evaluations   ·   global optimum  x* ≈ {X1D_STAR:.3f}",
        f"2-D  ·  {N_2D} evaluations   ·   4 global optima at  f* = 0",
    ]):
        axes[0, col_idx].annotate(
            header,
            xy=(0.5, 1.13), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9, color="#1a1a6e",
            fontweight="bold",
        )

    # Layout and shared colorbar
    plt.subplots_adjust(
        left=0.06, right=0.88,
        top=0.96,  bottom=0.03,
        hspace=0.62, wspace=0.30,
    )

    cax  = fig.add_axes([0.905, 0.055, 0.016, 0.875])
    sm   = ScalarMappable(cmap=_CMAP_TRAJ, norm=_NORM_TRAJ)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Evaluation order", fontsize=9, labelpad=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["1st", "mid", "last"])
    cbar.ax.tick_params(labelsize=8)

    # Figure-level legend key
    fig.text(
        0.50, 0.008,
        "★  red = true optimum     ★  gold = best found by optimizer"
        "     ●  circle = Phase 1     ▲  triangle = Phase 2"
        "  (hybrid optimizers only)",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )

    out = _PLOTS / "benchmark_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    print("benchmark_combined.py -- running all optimizers ...\n")
    main()
