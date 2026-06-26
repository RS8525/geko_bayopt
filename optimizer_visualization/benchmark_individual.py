"""
benchmark_individual.py

Produces one PNG per optimizer — a side-by-side pair of 1-D trajectory
and 2-D contour plots.  Intended for documentation where each optimizer
needs its own standalone figure.

Output: optimizer_visualization/plots/<slug>.png  (8 files total)

Run from the project root:
    .venv/Scripts/python.exe optimizer_visualization/benchmark_individual.py
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



def _add_colorbar(fig) -> None:
    """Attach the shared evaluation-order colorbar to the right edge."""
    cax  = fig.add_axes([0.913, 0.12, 0.018, 0.72])
    sm   = ScalarMappable(cmap=_CMAP_TRAJ, norm=_NORM_TRAJ)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Evaluation order", fontsize=9, labelpad=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["1st", "mid", "last"])
    cbar.ax.tick_params(labelsize=8)


def make_individual_figure(
    name: str,
    slug: str,
    phase_split_1d: int | None,
    phase_split_2d: int | None,
    xs1: list[float],
    ys1: list[float],
    xs2: list[tuple[float, float]],
    ys2: list[float],
) -> Path:
    """Create and save one side-by-side figure; return the output path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- super-title: optimizer name only -----------------------------------
    fig.suptitle(name, fontsize=12, fontweight="bold", y=1.01)

    # --- 1-D subplot ---------------------------------------------------------
    plot_1d_ax(ax1, xs1, ys1, phase_split=phase_split_1d)

    # --- 2-D subplot ---------------------------------------------------------
    plot_2d_ax(ax2, xs2, ys2, phase_split=phase_split_2d)

    # --- shared colorbar and layout -----------------------------------------
    plt.subplots_adjust(left=0.07, right=0.90, wspace=0.30,
                        top=0.92, bottom=0.13)
    _add_colorbar(fig)

    # --- legend key at the bottom -------------------------------------------
    fig.text(
        0.50, 0.01,
        "★  red = true optimum     ★  gold = best found"
        "     ●  circle = Phase 1     ▲  triangle = Phase 2  (hybrid only)",
        ha="center", va="bottom", fontsize=7.5, color="#444444",
    )

    out = _PLOTS / f"{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    n_opt = len(OPTIMIZERS)

    for idx, (name, slug, ps1, ps2, sec1d, sec2d) in enumerate(OPTIMIZERS):
        label_safe = name.split("  (")[0].replace("→", "->").replace("–", "-")
        print(f"  [{idx + 1}/{n_opt}]  {label_safe} ...", flush=True)

        xs1, ys1 = run_1d(sec1d)
        xs2, ys2 = run_2d(sec2d)

        out = make_individual_figure(name, slug, ps1, ps2, xs1, ys1, xs2, ys2)
        print(f"          saved -> {out.name}", flush=True)

    print(f"\nAll {n_opt} figures saved in  {_PLOTS}")


if __name__ == "__main__":
    print("benchmark_individual.py -- running all optimizers ...\n")
    main()
