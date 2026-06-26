"""
_benchmark_core.py

Shared code imported by benchmark_combined.py and benchmark_individual.py.
Do not run this file directly.

Exports
-------
plt, ScalarMappable, Normalize
OPTIMIZERS        list of (name, slug, phase_split_1d, phase_split_2d, sec_1d, sec_2d)
run_1d, run_2d    functions that execute an optimizer and collect (xs, ys)
plot_1d_ax        draws a 1-D trajectory subplot
plot_2d_ax        draws a 2-D contour trajectory subplot
N_1D, N_2D        evaluation budgets
X1D_STAR, Y1D_STAR   true global minimum of the 1-D test function
X2D_GLOBALS, Y2D_STAR  all global minima of the 2-D test function (Himmelblau)
X2D_STAR          representative single global minimum (for header text)
_CMAP_TRAJ, _NORM_TRAJ   shared colormap / normalisation for the colorbar
"""

from __future__ import annotations

import sys
import os as _os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# Adds  <project_root>/src/  to sys.path so  import geko_bayesopt  resolves.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # optimizer_visualization/
_src  = _HERE.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize

from geko_bayesopt.config import ParameterSpec, OptimizerSection
from geko_bayesopt.optimizer import build_optimizer

# ===========================================================================
# Global matplotlib style
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
    """Scaled  -x·sin(x)  with a local and a global minimum.

    Input is scaled by 2.5 so that both critical points fall within
    geko_csep ∈ [0.5, 3.5]:
      local  minimum near x ≈ 0.81   (trap for gradient-based methods)
      global minimum near x ≈ 3.19
    GEKO default (x=1.75) lands near the local maximum — a deliberate trap.
    """
    return -(2.5 * x) * np.sin(2.5 * x)


def f2d(x1: float, x2: float) -> float:
    """Scaled Himmelblau function with 4 global minima at f = 0.

    The standard Himmelblau domain [-5, 5]^2 is mapped onto
    geko_csep × geko_cnw ∈ [0.5, 3.5] × [0.1, 0.9].
    All 4 global minima lie within the GEKO parameter bounds.
    GEKO defaults (1.75, 0.5) sit at f ≈ 168, far from any minimum.
    """
    u = -5.0 + (x1 - 0.5) * (10.0 / 3.0)   # [0.5, 3.5] -> [-5, 5]
    v = -5.0 + (x2 - 0.1) * 12.5            # [0.1, 0.9] -> [-5, 5]
    return (u ** 2 + v - 11.0) ** 2 + (u + v ** 2 - 7.0) ** 2


# ---------------------------------------------------------------------------
# True minima (computed at import time)
# ---------------------------------------------------------------------------

# 1-D: use three starts to distinguish local from global minimum
_cands_1d = [
    minimize(lambda v: f1d(v[0]), x0=[xi],
             bounds=[(0.5, 3.5)], method="L-BFGS-B")
    for xi in [0.7, 1.6, 3.2]
]
_r1d = min(_cands_1d, key=lambda r: r.fun)
X1D_STAR: float = float(_r1d.x[0])
Y1D_STAR: float = float(_r1d.fun)

# 2-D Himmelblau: 4 known global minima in (u, v) space, all at f = 0
_HIMMELBLAU_UV = [
    ( 3.0,       2.0      ),
    (-2.805118,  3.131312 ),
    (-3.779310, -3.283186 ),
    ( 3.584428, -1.848126 ),
]
X2D_GLOBALS: list[tuple[float, float]] = [
    (0.5 + (u + 5.0) * 0.3, 0.1 + (v + 5.0) * 0.08)
    for u, v in _HIMMELBLAU_UV
]
Y2D_STAR: float = 0.0
X2D_STAR: tuple[float, float] = X2D_GLOBALS[0]   # (≈2.90, ≈0.66) — for header text

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

N_1D = 20   # total function evaluations for 1-D benchmark
N_2D = 35   # total function evaluations for 2-D benchmark

# ===========================================================================
# Optimizer catalogue
#
# Tuple layout:
#   (name, slug, phase_split_1d, phase_split_2d, section_1d, section_2d)
#
# slug          – filesystem-safe identifier used as the output filename.
# phase_split_* – eval index where Phase 2 begins (None = single-phase).
#                 Chosen to align with the natural cycle lengths of NM / FD.
#
# PSO accounting  n_particles × (1 init + max_iter swarm iterations):
#   1-D:  4 × (1+4) = 20 evals   →  n_particles=4, max_iter=4
#   2-D:  5 × (1+6) = 35 evals   →  n_particles=5, max_iter=6
#
# FD cycle lengths (1 init + D probes + 1 step  per cycle):
#   1-D: 1 (init) + 2 × k  →  k=4 gives  9 evals  (4 complete gradient cycles)
#   2-D: 1 (init) + 3 × k  →  k=4 gives 13 evals  (4 complete gradient cycles)
#
# BO→FD: FD phase starts from the BO best (no re-evaluation of the base),
#   so FD uses its full budget in pairs:  2×5=10 (1-D),  3×5=15 (2-D).
# ===========================================================================

_S1 = {"n_calls": N_1D}
_S2 = {"n_calls": N_2D}

OPTIMIZERS: list[tuple[
    str, str, int | None, int | None,
    OptimizerSection, OptimizerSection,
]] = [

    (
        "Bayesian Opt.  (GP + Sobol)",
        "01_bayesian_opt",
        None, None,
        OptimizerSection(kind="skopt_gp", stopping_criteria=_S1,
                         kind_specific_options={"n_initial": 8, "random_state": 42}),
        OptimizerSection(kind="skopt_gp", stopping_criteria=_S2,
                         kind_specific_options={"n_initial": 8, "random_state": 42}),
    ),

    (
        "Nelder–Mead",
        "02_nelder_mead",
        None, None,
        OptimizerSection(kind="nelder_mead", stopping_criteria=_S1,
                         kind_specific_options={}),
        OptimizerSection(kind="nelder_mead", stopping_criteria=_S2,
                         kind_specific_options={}),
    ),

    (
        "Finite Differences",
        "03_finite_differences",
        None, None,
        OptimizerSection(kind="finite_differences", stopping_criteria=_S1,
                         kind_specific_options={"step_size": 0.05, "learning_rate": 0.015}),
        OptimizerSection(kind="finite_differences", stopping_criteria=_S2,
                         kind_specific_options={"step_size": 0.05, "learning_rate": 0.001}),
    ),

    (
        "Particle Swarm  (PSO)",
        "04_particle_swarm",
        None, None,
        # 4 particles × (1 init + 4 swarm) = 20 evals
        OptimizerSection(kind="pso", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_particles": 4, "max_iter": 4, "random_state": 42}),
        # 5 particles × (1 init + 6 swarm) = 35 evals
        OptimizerSection(kind="pso", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_particles": 5, "max_iter": 6, "random_state": 42}),
    ),

    (
        "Hybrid  NM → BO  (10 NM + 10 BO  |  15 NM + 20 BO)",
        "05_hybrid_nm_bo",
        10, 15,
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 15,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
    ),

    (
        "Hybrid  FD → BO  (9 FD + 11 BO  |  13 FD + 22 BO)",
        "06_hybrid_fd_bo",
        9, 13,
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 9,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.015},
                         }),
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 13,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.001},
                         }),
    ),

    (
        "Hybrid  BO → NM  (10 BO + 10 NM  |  20 BO + 15 NM)",
        "07_hybrid_bo_nm",
        10, 20,
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "nm_options": {},
                         }),
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 20,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "nm_options": {},
                         }),
    ),

    (
        "Hybrid  BO → FD  (10 BO + 10 FD  |  20 BO + 15 FD)",
        "08_hybrid_bo_fd",
        10, 20,
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.015},
                         }),
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 20,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.001},
                         }),
    ),
]

# ===========================================================================
# Run helpers
# ===========================================================================

def _clip(x: list[float], params: list[ParameterSpec]) -> list[float]:
    """Clip a suggested point to parameter bounds.

    NM can suggest simplex vertices outside bounds (reflections near boundaries).
    Clipping before evaluation and tell is the standard bounded-optimisation
    convention: treat the boundary as the actual evaluation site.
    """
    return [float(np.clip(x[i], params[i].low, params[i].high))
            for i in range(len(params))]


def run_1d(section: OptimizerSection) -> tuple[list[float], list[float]]:
    opt = build_optimizer(section, PARAMS_1D)
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(N_1D):
        x = _clip(opt.ask(), PARAMS_1D)
        y = float(f1d(x[0]))
        opt.tell(x, y)
        xs.append(x[0])
        ys.append(y)
    return xs, ys


def run_2d(
    section: OptimizerSection,
) -> tuple[list[tuple[float, float]], list[float]]:
    opt = build_optimizer(section, PARAMS_2D)
    xs: list[tuple[float, float]] = []
    ys: list[float] = []
    for _ in range(N_2D):
        x = _clip(opt.ask(), PARAMS_2D)
        y = float(f2d(x[0], x[1]))
        opt.tell(x, y)
        xs.append((x[0], x[1]))
        ys.append(y)
    return xs, ys

# ===========================================================================
# Pre-computed evaluation grids (shared across all figures)
# ===========================================================================

_x1d_grid = np.linspace(0.5, 3.5, 600)
_y1d_grid = np.vectorize(f1d)(_x1d_grid)

_gx1 = np.linspace(0.5, 3.5, 250)
_gx2 = np.linspace(0.1, 0.9, 160)
_GX1, _GX2 = np.meshgrid(_gx1, _gx2)
_GZ      = np.vectorize(f2d)(_GX1, _GX2)
_GZ_PLOT = np.clip(_GZ, 0.0, 250.0)   # cap for contour visualisation

# ===========================================================================
# Visual constants
# ===========================================================================

_CMAP_TRAJ = "plasma"    # dark purple (early) -> bright yellow (late)
_CMAP_BG   = "Blues_r"  # dark blue = low cost / minimum, pale = high cost
_NORM_TRAJ = Normalize(0, 1)

_SCATTER_KW = dict(cmap=_CMAP_TRAJ, norm=_NORM_TRAJ,
                   edgecolors="k", linewidths=0.45, zorder=4)
_TRUE_MIN_KW = dict(marker="*", s=170, zorder=6,
                    edgecolors="k", linewidths=0.7)
_BEST_KW     = dict(marker="*", s=230, color="gold",
                    edgecolors="k", linewidths=0.8, zorder=7)


def _c(n: int) -> np.ndarray:
    """Evaluation-order colours normalised to [0, 1]."""
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
    title: str = "",
    phase_split: int | None = None,
) -> None:
    """Draw the 1-D benchmark trajectory onto *ax*."""
    n = len(xs)

    # Background function curve
    ax.plot(_x1d_grid, _y1d_grid, color="#c8c8c8", lw=2.0, zorder=1)

    # True minimum
    ax.axvline(X1D_STAR, color="#e63946", lw=0.9, ls="--", alpha=0.65, zorder=2)
    ax.scatter([X1D_STAR], [Y1D_STAR], color="#e63946", **_TRUE_MIN_KW)

    # Trajectory connectors
    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color="#777777", lw=0.55, alpha=0.38, zorder=3)

    # Evaluation dots (phase-aware)
    _scatter_1d(ax, xs, ys, n, phase_split)

    # Best point found
    best = int(np.argmin(ys))
    ax.scatter([xs[best]], [ys[best]], **_BEST_KW)

    # Best-value annotation
    ax.text(
        0.977, 0.975,
        f"best  x = {xs[best]:.3f}\nf  = {ys[best]:.5f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbbbbb"),
    )

    ax.set_xlim(0.42, 3.58)
    ax.set_xlabel(r"$x$", fontsize=8)
    ax.set_ylabel(r"$f(x)$", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=5)

    if phase_split is not None:
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                  borderpad=0.4, handlelength=1.2, handletextpad=0.5)


def plot_2d_ax(
    ax: plt.Axes,
    xs: list[tuple[float, float]],
    ys: list[float],
    phase_split: int | None = None,
) -> None:
    """Draw the 2-D benchmark trajectory onto *ax*."""
    n   = len(xs)
    x1s = [p[0] for p in xs]
    x2s = [p[1] for p in xs]

    # Contour background (clipped for visual clarity near minima)
    ax.grid(False)
    ax.contourf(_GX1, _GX2, _GZ_PLOT, levels=26, cmap=_CMAP_BG, alpha=0.88)
    ax.contour(_GX1, _GX2, _GZ_PLOT, levels=10,
               colors="white", linewidths=0.35, alpha=0.55)

    # All 4 global minima of the Himmelblau function
    gx = [pt[0] for pt in X2D_GLOBALS]
    gy = [pt[1] for pt in X2D_GLOBALS]
    ax.scatter(gx, gy, color="white", **_TRUE_MIN_KW)

    # Trajectory path line
    ax.plot(x1s, x2s, color="white", lw=0.9, alpha=0.45, zorder=3)

    # Evaluation dots (phase-aware)
    _scatter_2d(ax, x1s, x2s, n, phase_split)

    # Best point found
    best = int(np.argmin(ys))
    ax.scatter([x1s[best]], [x2s[best]], **_BEST_KW)

    # Best-value annotation
    ax.text(
        0.977, 0.975,
        f"best  ({x1s[best]:.2f},  {x2s[best]:.2f})\nf  = {ys[best]:.5f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbbbbb"),
    )

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel(r"$x_1$", fontsize=8)
    ax.set_ylabel(r"$x_2$", fontsize=8)
    ax.tick_params(labelsize=7)

    if phase_split is not None:
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                  borderpad=0.4, handlelength=1.2, handletextpad=0.5)
