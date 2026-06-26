"""
optimizer_benchmark.py

Heuristic / visual proof that all eight optimizers converge on standard
1-D and 2-D test functions.  Produces optimizer_benchmark.png next to
this file.

Run from anywhere:
    python src/geko_bayesopt/optimizer_benchmark.py
"""

from __future__ import annotations

import sys
import os as _os

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any non-builtin stdlib import.
#
# When CPython runs a script it inserts the script's directory at sys.path[0].
# Here that directory is  src/geko_bayesopt/, which contains types.py.
# That file shadows the stdlib `types` module and crashes pathlib / functools
# before a single line of our code can execute.  Removing it here — before
# importing pathlib — avoids the collision.
# ---------------------------------------------------------------------------
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)

# Now add the *parent* (i.e. src/) so `import geko_bayesopt` resolves.
_src_dir = _os.path.dirname(_pkg_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

del _pkg_dir, _src_dir, _os

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent

from geko_bayesopt.config import ParameterSpec, OptimizerSection  # noqa: E402
from geko_bayesopt.optimizer import build_optimizer                # noqa: E402

# ===========================================================================
# Global style
# ===========================================================================
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.8,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ===========================================================================
# Test functions
# ===========================================================================

def f1d(x: float) -> float:
    """Shifted quadratic + mild sinusoidal perturbation.

    Domain: geko_csep ∈ [0.5, 3.5].  Default value 1.75 sits 0.85 to the
    left of the minimum, giving all optimizers a clear convergence task.
    Sinusoidal amplitude (0.08) is small enough that no spurious local
    minima are created, but large enough to make the landscape non-trivial.
    """
    return (x - 2.6) ** 2 + 0.08 * np.sin(5.0 * x)


def f2d(x1: float, x2: float) -> float:
    """Asymmetric quadratic bowl.

    Domain: geko_csep ∈ [0.5, 3.5],  geko_cnw ∈ [0.1, 0.9].
    Aspect ratio 3:1 makes the landscape anisotropic and thus non-trivial
    for gradient-based methods.  GEKO defaults (1.75, 0.5) are offset
    from the minimum (2.4, 0.65), producing a visible optimization path.
    """
    return (x1 - 2.4) ** 2 + 3.0 * (x2 - 0.65) ** 2


# Compute true minima once via numerical optimisation
_r1d = minimize(lambda v: f1d(v[0]), x0=[2.6],
                bounds=[(0.5, 3.5)], method="L-BFGS-B")
X1D_STAR: float = float(_r1d.x[0])
Y1D_STAR: float = float(_r1d.fun)

_r2d = minimize(lambda v: f2d(v[0], v[1]), x0=[2.4, 0.65],
                bounds=[(0.5, 3.5), (0.1, 0.9)], method="L-BFGS-B")
X2D_STAR: tuple[float, float] = (float(_r2d.x[0]), float(_r2d.x[1]))
Y2D_STAR: float = float(_r2d.fun)

# ===========================================================================
# Parameter specs
# NM and FD look up initial values by GEKO parameter name, so the specs
# must use the canonical GEKO names.
# ===========================================================================

PARAMS_1D = [ParameterSpec(name="geko_csep", low=0.5, high=3.5)]
PARAMS_2D = [
    ParameterSpec(name="geko_csep", low=0.5, high=3.5),
    ParameterSpec(name="geko_cnw",  low=0.1, high=0.9),
]

N_1D = 15   # total function evaluations for 1-D benchmark
N_2D = 25   # total function evaluations for 2-D benchmark

# ===========================================================================
# Optimizer catalogue
#
# Tuple layout:
#   (display_name, phase_split_1d, phase_split_2d, section_1d, section_2d)
#
# phase_split_*  – eval index where Phase 2 begins (None = single-phase).
#                  Chosen to align with the natural cycle lengths of NM / FD
#                  so that no cycle is left half-finished at the transition.
#
# PSO accounting (n_particles × (1 init + k swarm iterations)):
#   1-D: 3 × (1+4) = 15 evals   →  n_particles=3, max_iter=4
#   2-D: 5 × (1+4) = 25 evals   →  n_particles=5, max_iter=4
#
# FD cycle lengths per dimension:
#   1-D: 1 (init) + 2 × k  →  k=3 gives 7 evals for phase-1 FD
#   2-D: 1 (init) + 3 × k  →  k=4 gives 13 evals for phase-1 FD
# ===========================================================================

_S1 = {"n_calls": N_1D}
_S2 = {"n_calls": N_2D}

OPTIMIZERS: list[tuple[str, int | None, int | None, OptimizerSection, OptimizerSection]] = [

    (
        "Bayesian Opt.  (GP + Sobol)",
        None, None,
        OptimizerSection(kind="skopt_gp", stopping_criteria=_S1,
                         kind_specific_options={"n_initial": 5, "random_state": 42}),
        OptimizerSection(kind="skopt_gp", stopping_criteria=_S2,
                         kind_specific_options={"n_initial": 8, "random_state": 42}),
    ),

    (
        "Nelder–Mead",
        None, None,
        OptimizerSection(kind="nelder_mead", stopping_criteria=_S1,
                         kind_specific_options={}),
        OptimizerSection(kind="nelder_mead", stopping_criteria=_S2,
                         kind_specific_options={}),
    ),

    (
        "Finite Differences",
        None, None,
        OptimizerSection(kind="finite_differences", stopping_criteria=_S1,
                         kind_specific_options={"step_size": 0.05, "learning_rate": 0.2}),
        OptimizerSection(kind="finite_differences", stopping_criteria=_S2,
                         kind_specific_options={"step_size": 0.05, "learning_rate": 0.2}),
    ),

    (
        "Particle Swarm  (PSO)",
        None, None,
        OptimizerSection(kind="pso", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_particles": 3, "max_iter": 4, "random_state": 42}),
        OptimizerSection(kind="pso", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_particles": 5, "max_iter": 4, "random_state": 42}),
    ),

    (
        "Hybrid  NM → BO  (8 NM + 7 BO  |  12 NM + 13 BO)",
        8, 12,
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 8,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 12,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
    ),

    (
        "Hybrid  FD → BO  (7 FD + 8 BO  |  13 FD + 12 BO)",
        7, 13,
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 7,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.2},
                         }),
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 13,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.2},
                         }),
    ),

    (
        "Hybrid  BO → NM  (8 BO + 7 NM  |  12 BO + 13 NM)",
        8, 12,
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 8,
                             "bo_options": {"n_initial_sobol": 4, "random_state": 42},
                             "nm_options": {},
                         }),
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 12,
                             "bo_options": {"n_initial_sobol": 6, "random_state": 42},
                             "nm_options": {},
                         }),
    ),

    (
        "Hybrid  BO → FD  (8 BO + 7 FD  |  12 BO + 13 FD)",
        8, 12,
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 8,
                             "bo_options": {"n_initial_sobol": 4, "random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.2},
                         }),
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 12,
                             "bo_options": {"n_initial_sobol": 6, "random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.2},
                         }),
    ),
]

# ===========================================================================
# Run helpers
# ===========================================================================

def run_1d(section: OptimizerSection) -> tuple[list[float], list[float]]:
    opt = build_optimizer(section, PARAMS_1D)
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(N_1D):
        x = opt.ask()
        y = float(f1d(x[0]))
        opt.tell(x, y)
        xs.append(float(x[0]))
        ys.append(y)
    return xs, ys


def run_2d(
    section: OptimizerSection,
) -> tuple[list[tuple[float, float]], list[float]]:
    opt = build_optimizer(section, PARAMS_2D)
    xs: list[tuple[float, float]] = []
    ys: list[float] = []
    for _ in range(N_2D):
        x = opt.ask()
        y = float(f2d(x[0], x[1]))
        opt.tell(x, y)
        xs.append((float(x[0]), float(x[1])))
        ys.append(y)
    return xs, ys

# ===========================================================================
# Pre-computed evaluation grids (computed once, reused for every subplot)
# ===========================================================================

_x1d_grid = np.linspace(0.5, 3.5, 600)
_y1d_grid = np.vectorize(f1d)(_x1d_grid)

_gx1 = np.linspace(0.5, 3.5, 250)
_gx2 = np.linspace(0.1, 0.9, 160)
_GX1, _GX2 = np.meshgrid(_gx1, _gx2)
_GZ = np.vectorize(f2d)(_GX1, _GX2)

# ===========================================================================
# Visual constants
# ===========================================================================

_CMAP_TRAJ = "plasma"    # dark purple → bright yellow (early → late)
_CMAP_BG   = "Blues_r"  # dark blue = low cost (minimum), pale = high cost
_NORM_TRAJ = Normalize(0, 1)

_SCATTER_KW = dict(cmap=_CMAP_TRAJ, norm=_NORM_TRAJ,
                   edgecolors="k", linewidths=0.45, zorder=4)
_TRUE_MIN_KW = dict(marker="*", s=170, zorder=6,
                    edgecolors="k", linewidths=0.7)
_BEST_KW     = dict(marker="*", s=230, color="gold",
                    edgecolors="k", linewidths=0.8, zorder=7)


def _c(n: int) -> np.ndarray:
    """Evaluation-order colours normalized to [0, 1]."""
    return np.linspace(0, 1, n)

# ===========================================================================
# Scatter helpers with optional two-phase markers
# ===========================================================================

def _scatter_1d(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    n: int,
    phase_split: int | None,
) -> None:
    c = _c(n)
    if phase_split is None:
        ax.scatter(xs, ys, c=c, s=62, marker="o", **_SCATTER_KW)
    else:
        i = phase_split
        if i > 0:
            ax.scatter(xs[:i], ys[:i], c=c[:i], s=62, marker="o",
                       label="Phase 1", **_SCATTER_KW)
        if i < n:
            ax.scatter(xs[i:], ys[i:], c=c[i:], s=68, marker="^",
                       label="Phase 2", **_SCATTER_KW)


def _scatter_2d(
    ax: plt.Axes,
    x1s: list[float],
    x2s: list[float],
    n: int,
    phase_split: int | None,
) -> None:
    c = _c(n)
    if phase_split is None:
        ax.scatter(x1s, x2s, c=c, s=57, marker="o", **_SCATTER_KW)
    else:
        i = phase_split
        if i > 0:
            ax.scatter(x1s[:i], x2s[:i], c=c[:i], s=57, marker="o",
                       label="Phase 1", **_SCATTER_KW)
        if i < n:
            ax.scatter(x1s[i:], x2s[i:], c=c[i:], s=63, marker="^",
                       label="Phase 2", **_SCATTER_KW)

# ===========================================================================
# Per-subplot drawing functions
# ===========================================================================

def plot_1d_ax(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    name: str,
    phase_split: int | None = None,
) -> None:
    n = len(xs)

    # ---- background function curve ----------------------------------------
    ax.plot(_x1d_grid, _y1d_grid, color="#c8c8c8", lw=2.0, zorder=1)

    # ---- true minimum -------------------------------------------------------
    ax.axvline(X1D_STAR, color="#e63946", lw=0.9, ls="--", alpha=0.65, zorder=2)
    ax.scatter([X1D_STAR], [Y1D_STAR], color="#e63946", **_TRUE_MIN_KW)

    # ---- trajectory connectors (thin gray lines) ---------------------------
    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color="#777777", lw=0.55, alpha=0.38, zorder=3)

    # ---- evaluation dots ----------------------------------------------------
    _scatter_1d(ax, xs, ys, n, phase_split)

    # ---- best found ---------------------------------------------------------
    best = int(np.argmin(ys))
    ax.scatter([xs[best]], [ys[best]], **_BEST_KW)

    # ---- annotation (best value, top-right) ---------------------------------
    ax.text(
        0.977, 0.975,
        f"best  x = {xs[best]:.3f}\nf  = {ys[best]:.5f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbbbbb"),
    )

    # ---- axes ---------------------------------------------------------------
    ax.set_xlim(0.42, 3.58)
    ax.set_xlabel(r"$x$  (geko\_csep)", fontsize=8)
    ax.set_ylabel(r"$f(x)$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(name, fontsize=9, fontweight="bold", loc="left", pad=5)

    # phase legend (hybrid only)
    if phase_split is not None:
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                  borderpad=0.4, handlelength=1.2, handletextpad=0.5)


def plot_2d_ax(
    ax: plt.Axes,
    xs: list[tuple[float, float]],
    ys: list[float],
    phase_split: int | None = None,
) -> None:
    n   = len(xs)
    x1s = [p[0] for p in xs]
    x2s = [p[1] for p in xs]

    # ---- contour background -------------------------------------------------
    ax.grid(False)
    ax.contourf(_GX1, _GX2, _GZ, levels=26, cmap=_CMAP_BG, alpha=0.88)
    ax.contour(_GX1, _GX2, _GZ, levels=10,
               colors="white", linewidths=0.35, alpha=0.55)

    # ---- true minimum -------------------------------------------------------
    ax.scatter([X2D_STAR[0]], [X2D_STAR[1]], color="white", **_TRUE_MIN_KW)

    # ---- trajectory path line -----------------------------------------------
    ax.plot(x1s, x2s, color="white", lw=0.9, alpha=0.45, zorder=3)

    # ---- evaluation dots ----------------------------------------------------
    _scatter_2d(ax, x1s, x2s, n, phase_split)

    # ---- best found ---------------------------------------------------------
    best = int(np.argmin(ys))
    ax.scatter([x1s[best]], [x2s[best]], **_BEST_KW)

    # ---- annotation (best value, top-right) ---------------------------------
    ax.text(
        0.977, 0.975,
        f"best  ({x1s[best]:.2f},  {x2s[best]:.2f})\nf  = {ys[best]:.5f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbbbbb"),
    )

    # ---- axes ---------------------------------------------------------------
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel(r"$x_1$  (geko\_csep)", fontsize=8)
    ax.set_ylabel(r"$x_2$  (geko\_cnw)", fontsize=8)
    ax.tick_params(labelsize=7)

    # phase legend (hybrid only)
    if phase_split is not None:
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                  borderpad=0.4, handlelength=1.2, handletextpad=0.5)

# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    n_opt   = len(OPTIMIZERS)
    row_h   = 4.4           # inches per row
    fig_w   = 15.5          # total figure width

    fig, axes = plt.subplots(n_opt, 2, figsize=(fig_w, n_opt * row_h))

    # --- super-title --------------------------------------------------------
    fig.suptitle(
        "Optimizer Benchmark  –  Convergence on Synthetic Test Functions\n"
        r"$f_{\mathrm{1D}}(x) = (x - 2.6)^2 + 0.08\,\sin(5x)$"
        r"$\qquad\qquad$"
        r"$f_{\mathrm{2D}}(x_1,x_2) = (x_1 - 2.4)^2 + 3\,(x_2 - 0.65)^2$",
        fontsize=11.5, fontweight="bold", y=1.004,
    )

    # --- run and plot each optimizer ----------------------------------------
    for row, (name, ps1, ps2, sec1d, sec2d) in enumerate(OPTIMIZERS):
        label = name.split("  (")[0]                          # strip phase-count parenthetical
        label_safe = label.replace("→", "->").replace("–", "-")
        print(f"  [{row + 1}/{n_opt}]  {label_safe} ...", flush=True)

        xs1, ys1 = run_1d(sec1d)
        xs2, ys2 = run_2d(sec2d)

        plot_1d_ax(axes[row, 0], xs1, ys1, name, phase_split=ps1)
        plot_2d_ax(axes[row, 1], xs2, ys2, phase_split=ps2)

    # --- column headers (placed above first-row axes via annotation) ---------
    for col_idx, header in enumerate([
        f"1-D  ·  {N_1D} evaluations"
        f"   ·   true optimum  x* ≈ {X1D_STAR:.3f}",
        f"2-D  ·  {N_2D} evaluations"
        f"   ·   true optimum  (x₁*, x₂*) ≈ ({X2D_STAR[0]:.2f}, {X2D_STAR[1]:.2f})",
    ]):
        axes[0, col_idx].annotate(
            header,
            xy=(0.5, 1.13), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9, color="#1a1a6e",
            fontweight="bold",
        )

    # --- layout & colorbar --------------------------------------------------
    plt.subplots_adjust(
        left=0.06, right=0.88,
        top=0.96,  bottom=0.03,
        hspace=0.62, wspace=0.30,
    )

    cax = fig.add_axes([0.905, 0.055, 0.016, 0.875])
    sm  = ScalarMappable(cmap=_CMAP_TRAJ, norm=_NORM_TRAJ)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Evaluation order", fontsize=9, labelpad=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["1st", "mid", "last"])
    cbar.ax.tick_params(labelsize=8)

    # --- figure-level legend key --------------------------------------------
    fig.text(
        0.50, 0.008,
        "★  red = true optimum     ★  gold = best found by optimizer"
        "     ●  circle = Phase 1     ▲  triangle = Phase 2  (hybrid optimizers only)",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )

    # --- save ---------------------------------------------------------------
    out = _HERE / "optimizer_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    print("Running optimizer benchmark …\n")
    main()
