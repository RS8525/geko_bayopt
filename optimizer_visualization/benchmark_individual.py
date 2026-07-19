"""
benchmark_individual.py

Produces one PNG per optimizer: a side-by-side pair of 1-D trajectory
and 2-D contour plots.  Intended for documentation where each optimizer
needs its own standalone figure.

Output: optimizer_visualization/plots/individual/<slug>.png  (8 files total)

Run from the project root:
    python optimizer_visualization/benchmark_individual.py
"""

from __future__ import annotations

from pathlib import Path
from matplotlib.cm import ScalarMappable

import numpy as np
from scipy.optimize import minimize

from _benchmark_core import (
    plt,
    OPTIMIZERS,
    f1d, f2d,
    run_1d, run_2d,
    plot_1d_ax, plot_2d_ax,
    N_1D, N_2D,
    X1D_STAR, Y1D_STAR, X2D_STAR,
    _CMAP_TRAJ, _NORM_TRAJ,
)

_HERE  = Path(__file__).resolve().parent
_PLOTS = _HERE / "plots" / "individual"
_PLOTS.mkdir(parents=True, exist_ok=True)


def make_test_function_figure() -> Path:
    """Render the two synthetic test functions side by side (no trajectories).

    Marks the global minimum (red star), the local minimum (open circle),
    and the GEKO-default starting point (black square) on both panels.
    Output: plots/individual/00_test_functions.png
    """
    # Local minimum of f1d (the trap), found from the left basin.
    r_loc = minimize(lambda v: f1d(v[0]), x0=[0.7],
                     bounds=[(0.5, 3.5)], method="L-BFGS-B")
    x_loc, y_loc = float(r_loc.x[0]), float(r_loc.fun)
    x_start = 1.75          # GEKO default for csep-like x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle("Synthetic test functions", fontsize=12,
                 fontweight="bold", y=1.01)

    # --- 1-D panel -----------------------------------------------------------
    xs = np.linspace(0.5, 3.5, 600)
    ax1.plot(xs, np.vectorize(f1d)(xs), color="#555555", lw=1.6, zorder=2)
    ax1.scatter([X1D_STAR], [Y1D_STAR], marker="*", s=200, color="#e63946",
                edgecolors="k", linewidths=0.7, zorder=5,
                label="global minimum")
    ax1.scatter([x_loc], [y_loc], marker="o", s=70, facecolors="none",
                edgecolors="#1d3557", linewidths=1.6, zorder=5,
                label="local minimum")
    ax1.scatter([x_start], [f1d(x_start)], marker="s", s=60, color="black",
                zorder=5, label="start (GEKO default)")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$f_{1D}(x)$")
    ax1.set_title(r"$f_{1D}(x) = -2.5x\,\sin(2.5x)$", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # --- 2-D panel -----------------------------------------------------------
    gx1 = np.linspace(0.5, 3.5, 250)
    gx2 = np.linspace(0.1, 0.9, 160)
    GX1, GX2 = np.meshgrid(gx1, gx2)
    GZ = np.vectorize(f2d)(GX1, GX2)
    cf = ax2.contourf(GX1, GX2, GZ, levels=26, cmap="Blues_r", alpha=0.88)
    ax2.contour(GX1, GX2, GZ, levels=10, colors="white",
                linewidths=0.5, alpha=0.55)
    ax2.scatter([X1D_STAR], [0.5], marker="*", s=200, color="#e63946",
                edgecolors="k", linewidths=0.7, zorder=5,
                label="global minimum")
    ax2.scatter([x_loc], [0.5], marker="o", s=70, facecolors="none",
                edgecolors="#1d3557", linewidths=1.6, zorder=5,
                label="local minimum")
    ax2.scatter([x_start], [0.5], marker="s", s=60, color="black",
                zorder=5, label="start (GEKO default)")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_title(r"$f_{2D}(x_1, x_2) = f_{1D}(x_1) + 3\,|x_2 - 0.5|$",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    cbar = fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label("$f_{2D}$", fontsize=9)

    plt.subplots_adjust(left=0.07, right=0.97, wspace=0.26,
                        top=0.90, bottom=0.13)
    out = _PLOTS / "00_test_functions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out



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
    mask1: list[bool] | None = None,
    mask2: list[bool] | None = None,
    n_particles: int | None = None,
) -> Path:
    """Create and save one side-by-side figure; return the output path.

    Pass *n_particles* for PSO: dots are coloured by particle, and the
    sequential evaluation-order colorbar is omitted.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- super-title: optimizer name only -----------------------------------
    fig.suptitle(name, fontsize=12, fontweight="bold", y=1.01)

    # --- 1-D subplot ---------------------------------------------------------
    plot_1d_ax(ax1, xs1, ys1, phase_split=phase_split_1d, step_mask=mask1,
               n_particles=n_particles)

    # --- 2-D subplot ---------------------------------------------------------
    plot_2d_ax(ax2, xs2, ys2, phase_split=phase_split_2d, step_mask=mask2,
               n_particles=n_particles)

    # --- layout + optional colorbar -----------------------------------------
    if n_particles is None:
        plt.subplots_adjust(left=0.07, right=0.90, wspace=0.30,
                            top=0.92, bottom=0.13)
        _add_colorbar(fig)
    else:
        plt.subplots_adjust(left=0.07, right=0.96, wspace=0.30,
                            top=0.92, bottom=0.13)

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

    out = make_test_function_figure()
    print(f"  [0/{n_opt}]  test functions ...\n          saved -> {out.name}",
          flush=True)

    for idx, (name, slug, ps1, ps2, sec1d, sec2d) in enumerate(OPTIMIZERS):
        label_safe = name.split("  (")[0].replace("→", "->").replace("–", "-")
        print(f"  [{idx + 1}/{n_opt}]  {label_safe} ...", flush=True)

        xs1, ys1, mask1 = run_1d(sec1d)
        xs2, ys2, mask2 = run_2d(sec2d)

        # PSO: colour dots by particle index instead of evaluation order
        n_particles = (
            sec1d.kind_specific_options.get("n_particles")
            if sec1d.kind == "pso" else None
        )

        out = make_individual_figure(name, slug, ps1, ps2, xs1, ys1, xs2, ys2,
                                     mask1, mask2, n_particles=n_particles)
        print(f"          saved -> {out.name}", flush=True)

    print(f"\nAll {n_opt} figures saved in  {_PLOTS}")


if __name__ == "__main__":
    print("benchmark_individual.py -- running all optimizers ...\n")
    main()
