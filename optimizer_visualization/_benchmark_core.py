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
    """2-D benchmark: 1-D objective along x1, linear deviation penalty along x2.

    f(x1, x2) = f1d(x1)  +  3 * |x2 - 0.5|

    The x2 term is non-smooth (absolute value) at x2 = 0.5, the ridge minimum.
    Global minimum at (X1D_STAR, 0.5);  local minimum at (~0.81, 0.5).
    GEKO defaults (1.75, 0.5) land near the local maximum of f1d, at the x2 valley floor.
    """
    return f1d(x1) + 3.0 * abs(x2 - 0.5)


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

# 2-D: global minimum shares the x1 location of the 1-D global minimum, at x2 = 0.5
X2D_GLOBALS: list[tuple[float, float]] = [(X1D_STAR, 0.5)]
Y2D_STAR: float = Y1D_STAR       # f2d(X1D_STAR, 0.5) = f1d(X1D_STAR) + 0
X2D_STAR: tuple[float, float] = (X1D_STAR, 0.5)

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
N_2D = 36   # total function evaluations for 2-D benchmark

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
#   max_iter is derived internally from n_calls // n_particles - 1, so
#   n_calls must divide evenly by n_particles:
#   1-D:  20 evals / 4 particles → max_iter = 4
#   2-D:  36 evals / 4 particles → max_iter = 8
#
# FD cycle lengths (base + D probes + 1 step per cycle):
#   1-D: total must be divisible by 2 (D+1).  20 / 2 = 10 cycles ✓
#   2-D: total must be divisible by 3 (D+1).  36 / 3 = 12 cycles ✓
#
# BO→FD: FD phase starts from the BO best (no re-evaluation of the base),
#   so FD uses its full budget in triples:  2×5=10 (1-D),  3×5=15 (2-D).
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
                         kind_specific_options={"step_size": 0.05, "learning_rate": 0.015}),
    ),

    (
        "Particle Swarm  (PSO)",
        "04_particle_swarm",
        None, None,
        # 4 particles, 20 evals → max_iter = 20/4 - 1 = 4 (derived internally)
        OptimizerSection(kind="pso", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_particles": 4, "random_state": 1}),
        # 4 particles, 36 evals → max_iter = 36/4 - 1 = 8 (derived internally)
        OptimizerSection(kind="pso", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_particles": 4, "random_state": 1}),
    ),

    (
        "Hybrid  NM → BO",
        "05_hybrid_nm_bo",
        10, 16,
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
        OptimizerSection(kind="hybrid_nm_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 16,
                             "bo_options": {"random_state": 42},
                             "nm_options": {},
                         }),
    ),

    (
        "Hybrid  FD → BO",
        "06_hybrid_fd_bo",
        9, 16,
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 9,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.015},
                         }),
        OptimizerSection(kind="hybrid_fd_bayes", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 16,
                             "bo_options": {"random_state": 42},
                             "fd_options": {"step_size": 0.05, "learning_rate": 0.015},
                         }),
    ),

    (
        "Hybrid  BO → NM",
        "07_hybrid_bo_nm",
        10, 20,
        # nm_options: simplex_scale=0.6 shrinks the NM startup simplex to 60% of
        # normal, tightening the search around the BO best point for exploitation.
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "nm_options": {"simplex_scale": 0.6},
                         }),
        OptimizerSection(kind="hybrid_bayes_nm", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 20,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "nm_options": {"simplex_scale": 0.6},
                         }),
    ),

    (
        "Hybrid  BO → FD",
        "08_hybrid_bo_fd",
        10, 20,
        # fd_options: step_size and learning_rate at 60% of standalone FD values
        # to tighten the gradient search around the BO best point for exploitation.
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S1,
                         kind_specific_options={
                             "n_initial": 10,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "fd_options": {"step_size": 0.03, "learning_rate": 0.009},
                         }),
        OptimizerSection(kind="hybrid_bayes_fd", stopping_criteria=_S2,
                         kind_specific_options={
                             "n_initial": 20,
                             "bo_options": {"n_initial_sobol": 5, "random_state": 42},
                             "fd_options": {"step_size": 0.03, "learning_rate": 0.009},
                         }),
    ),
]

# ===========================================================================
# Run helpers
# ===========================================================================

def _fd_show_mask(section: OptimizerSection, n: int, d: int) -> list[bool]:
    """Return True for evals to show; False for FD finite-difference probes to hide.

    For non-FD sections all entries are True.  For FD sections:
      - standalone FD:  eval 0 is the base (show); then D probes (hide) + 1
                        gradient step (show) repeat for every subsequent cycle.
      - hybrid FD->BO:  same pattern during the FD phase; BO phase all show.
      - hybrid BO->FD:  BO phase all show; FD phase begins immediately with a
                        probe (the base is injected, not re-evaluated), so the
                        step sits at fd_index % (D+1) == D.
    """
    kind = section.kind
    if kind not in ("finite_differences", "hybrid_fd_bayes", "hybrid_bayes_fd"):
        return [True] * n

    mask   = [True] * n
    period = d + 1  # D probes followed by 1 gradient step per cycle

    if kind == "finite_differences":
        for i in range(1, n):
            mask[i] = ((i - 1) % period == d)

    elif kind == "hybrid_fd_bayes":
        n_init = int(section.kind_specific_options.get("n_initial", 0))
        for i in range(1, min(n_init, n)):
            mask[i] = ((i - 1) % period == d)
        # BO phase (n_init … n-1) stays True

    elif kind == "hybrid_bayes_fd":
        n_init = int(section.kind_specific_options.get("n_initial", 0))
        # BO phase (0 … n_init-1) stays True
        for i in range(n_init, n):
            fd_i = i - n_init
            mask[i] = (fd_i % period == d)

    return mask


def run_1d(section: OptimizerSection) -> tuple[list[float], list[float], list[bool]]:
    n   = section.stopping_criteria.get("n_calls", N_1D)
    opt = build_optimizer(section, PARAMS_1D)
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(n):
        # Proposals are evaluated exactly where suggested — no clipping.
        # NM and FD may deliberately wander outside the bounds; comparisons
        # are meant to expose that boundary behavior, not mask it.
        x = [float(v) for v in opt.ask()]
        y = float(f1d(x[0]))
        opt.tell(x, y)
        xs.append(x[0])
        ys.append(y)
    return xs, ys, _fd_show_mask(section, n, len(PARAMS_1D))


def run_2d(
    section: OptimizerSection,
) -> tuple[list[tuple[float, float]], list[float], list[bool]]:
    n   = section.stopping_criteria.get("n_calls", N_2D)
    opt = build_optimizer(section, PARAMS_2D)
    xs: list[tuple[float, float]] = []
    ys: list[float] = []
    for _ in range(n):
        # No clipping — see run_1d.
        x = [float(v) for v in opt.ask()]
        y = float(f2d(x[0], x[1]))
        opt.tell(x, y)
        xs.append((x[0], x[1]))
        ys.append(y)
    return xs, ys, _fd_show_mask(section, n, len(PARAMS_2D))

# ===========================================================================
# Pre-computed evaluation grids (shared across all figures)
# ===========================================================================

_x1d_grid = np.linspace(0.5, 3.5, 600)
_y1d_grid = np.vectorize(f1d)(_x1d_grid)

_gx1 = np.linspace(0.5, 3.5, 250)
_gx2 = np.linspace(0.1, 0.9, 160)
_GX1, _GX2 = np.meshgrid(_gx1, _gx2)
_GZ      = np.vectorize(f2d)(_GX1, _GX2)
_GZ_PLOT = _GZ   # range is naturally bounded (~-8 to +6); no clipping needed

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

# PSO: one fixed colour per particle so each swarm member's trajectory is
# traceable. Avoids red (true min) and gold (best found); chosen for contrast
# on both white (1D) and blue-contour (2D) backgrounds.
_PARTICLE_COLORS = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]  # tab10 1/2/4/5
_PSO_SCATTER_KW  = dict(edgecolors="k", linewidths=0.45, zorder=4)


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
    label_p1: str = "",
    label_p2: str = "",
) -> None:
    c = _c(n)
    if phase_split is None:
        ax.scatter(xs, ys, c=c, s=62, marker="o", **_SCATTER_KW)
    else:
        i = phase_split
        if i > 0:
            ax.scatter(xs[:i], ys[:i], c=c[:i], s=62, marker="o",
                       label=label_p1, **_SCATTER_KW)
        if i < n:
            ax.scatter(xs[i:], ys[i:], c=c[i:], s=68, marker="^",
                       label=label_p2, **_SCATTER_KW)


def _scatter_2d(
    ax: plt.Axes,
    x1s: list[float],
    x2s: list[float],
    n: int,
    phase_split: int | None,
    label_p1: str = "",
    label_p2: str = "",
) -> None:
    c = _c(n)
    if phase_split is None:
        ax.scatter(x1s, x2s, c=c, s=57, marker="o", **_SCATTER_KW)
    else:
        i = phase_split
        if i > 0:
            ax.scatter(x1s[:i], x2s[:i], c=c[:i], s=57, marker="o",
                       label=label_p1, **_SCATTER_KW)
        if i < n:
            ax.scatter(x1s[i:], x2s[i:], c=c[i:], s=63, marker="^",
                       label=label_p2, **_SCATTER_KW)


def _scatter_1d_pso(ax: plt.Axes, xs, ys, n: int, n_particles: int) -> None:
    """PSO: each particle gets one fixed colour; legend shows Particle 1…k."""
    for p in range(n_particles):
        idxs = list(range(p, n, n_particles))
        ax.scatter([xs[i] for i in idxs], [ys[i] for i in idxs],
                   c=_PARTICLE_COLORS[p % len(_PARTICLE_COLORS)],
                   s=62, marker="o", label=f"Particle {p + 1}",
                   **_PSO_SCATTER_KW)


def _scatter_2d_pso(ax: plt.Axes, x1s, x2s, n: int, n_particles: int) -> None:
    """PSO: each particle gets one fixed colour; legend shows Particle 1…k."""
    for p in range(n_particles):
        idxs = list(range(p, n, n_particles))
        ax.scatter([x1s[i] for i in idxs], [x2s[i] for i in idxs],
                   c=_PARTICLE_COLORS[p % len(_PARTICLE_COLORS)],
                   s=57, marker="o", label=f"Particle {p + 1}",
                   **_PSO_SCATTER_KW)

# ===========================================================================
# Per-subplot drawing functions
# ===========================================================================

def plot_1d_ax(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    title: str = "",
    phase_split: int | None = None,
    step_mask: list[bool] | None = None,
    n_particles: int | None = None,
) -> None:
    """Draw the 1-D benchmark trajectory onto *ax*.

    Pass *n_particles* for PSO plots: dots are coloured by particle index
    instead of evaluation order, and the sequential colorbar is not needed.
    """
    # Capture totals BEFORE mask filtering — used for "function evals" annotations.
    n_total          = len(xs)
    phase_split_total = phase_split

    if step_mask is not None:
        orig = [i for i, m in enumerate(step_mask) if m]
        xs   = [xs[i] for i in orig]
        ys   = [ys[i] for i in orig]
        if phase_split is not None:
            phase_split = sum(1 for i in orig if i < phase_split)
    n = len(xs)

    # Background function curve
    ax.plot(_x1d_grid, _y1d_grid, color="#c8c8c8", lw=2.0, zorder=1)

    # True minimum
    ax.axvline(X1D_STAR, color="#e63946", lw=0.9, ls="--", alpha=0.65, zorder=2)
    ax.scatter([X1D_STAR], [Y1D_STAR], color="#e63946", **_TRUE_MIN_KW)

    # Trajectory connectors — per-particle for PSO, sequential otherwise
    if n_particles is not None:
        for p in range(n_particles):
            px = [xs[i] for i in range(p, n, n_particles)]
            py = [ys[i] for i in range(p, n, n_particles)]
            ax.plot(px, py, color=_PARTICLE_COLORS[p % len(_PARTICLE_COLORS)],
                    lw=0.7, alpha=0.45, zorder=3)
    else:
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color="#777777", lw=0.55, alpha=0.38, zorder=3)

    # Evaluation dots — phase-aware for hybrids, per-particle for PSO
    if phase_split is not None:
        lp1 = f"Phase 1  ({phase_split_total} function evals)"
        lp2 = f"Phase 2  ({n_total - phase_split_total} function evals)"
        _scatter_1d(ax, xs, ys, n, phase_split, lp1, lp2)
    elif n_particles is not None:
        _scatter_1d_pso(ax, xs, ys, n, n_particles)
    else:
        _scatter_1d(ax, xs, ys, n, None)

    # Best point found
    best = int(np.argmin(ys))
    ax.scatter([xs[best]], [ys[best]], **_BEST_KW)

    # Best-value annotation (N count embedded for PSO since there is no colorbar)
    if n_particles is not None:
        ann = f"N = {n_total} function evals\nbest  x = {xs[best]:.3f}\nf  = {ys[best]:.5f}"
    else:
        ann = f"best  x = {xs[best]:.3f}\nf  = {ys[best]:.5f}"
    ax.text(
        0.977, 0.975, ann,
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
        leg = ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                        borderpad=0.4, handlelength=1.2, handletextpad=0.5)
        leg.set_zorder(20)
    elif n_particles is not None:
        leg = ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                        borderpad=0.4, handlelength=1.2, handletextpad=0.5)
        leg.set_zorder(20)
    else:
        ax.text(
            0.02, 0.975, f"N = {n_total} function evals",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.80, ec="#bbbbbb"),
        )


def plot_2d_ax(
    ax: plt.Axes,
    xs: list[tuple[float, float]],
    ys: list[float],
    phase_split: int | None = None,
    step_mask: list[bool] | None = None,
    n_particles: int | None = None,
) -> None:
    """Draw the 2-D benchmark trajectory onto *ax*.

    Pass *n_particles* for PSO plots: dots are coloured by particle index
    and each particle's path is drawn in its own colour.
    """
    # Capture totals BEFORE mask filtering — used for "function evals" annotations.
    n_total           = len(xs)
    phase_split_total = phase_split

    if step_mask is not None:
        orig = [i for i, m in enumerate(step_mask) if m]
        xs   = [xs[i] for i in orig]
        ys   = [ys[i] for i in orig]
        if phase_split is not None:
            phase_split = sum(1 for i in orig if i < phase_split)
    n   = len(xs)
    x1s = [p[0] for p in xs]
    x2s = [p[1] for p in xs]

    # Contour background
    ax.grid(False)
    ax.contourf(_GX1, _GX2, _GZ_PLOT, levels=26, cmap=_CMAP_BG, alpha=0.88)
    ax.contour(_GX1, _GX2, _GZ_PLOT, levels=10,
               colors="white", linewidths=0.35, alpha=0.55)

    # Global minimum of the 2-D test function
    gx = [pt[0] for pt in X2D_GLOBALS]
    gy = [pt[1] for pt in X2D_GLOBALS]
    ax.scatter(gx, gy, color="#e63946", **_TRUE_MIN_KW)

    # Trajectory path — per-particle for PSO, single white line otherwise
    if n_particles is not None:
        for p in range(n_particles):
            px = [x1s[i] for i in range(p, n, n_particles)]
            py = [x2s[i] for i in range(p, n, n_particles)]
            ax.plot(px, py, color=_PARTICLE_COLORS[p % len(_PARTICLE_COLORS)],
                    lw=0.9, alpha=0.5, zorder=3)
    else:
        ax.plot(x1s, x2s, color="white", lw=0.9, alpha=0.45, zorder=3)

    # Evaluation dots — phase-aware for hybrids, per-particle for PSO
    if phase_split is not None:
        lp1 = f"Phase 1  ({phase_split_total} function evals)"
        lp2 = f"Phase 2  ({n_total - phase_split_total} function evals)"
        _scatter_2d(ax, x1s, x2s, n, phase_split, lp1, lp2)
    elif n_particles is not None:
        _scatter_2d_pso(ax, x1s, x2s, n, n_particles)
    else:
        _scatter_2d(ax, x1s, x2s, n, None)

    # Best point found
    best = int(np.argmin(ys))
    ax.scatter([x1s[best]], [x2s[best]], **_BEST_KW)

    # Best-value annotation (N count embedded for PSO)
    if n_particles is not None:
        ann = (f"N = {n_total} function evals\n"
               f"best  ({x1s[best]:.2f},  {x2s[best]:.2f})\nf  = {ys[best]:.5f}")
    else:
        ann = f"best  ({x1s[best]:.2f},  {x2s[best]:.2f})\nf  = {ys[best]:.5f}"
    ax.text(
        0.977, 0.975, ann,
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbbbbb"),
    )

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel(r"$x_1$", fontsize=8)
    ax.set_ylabel(r"$x_2$", fontsize=8)
    ax.tick_params(labelsize=7)

    if phase_split is not None:
        leg = ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                        borderpad=0.4, handlelength=1.2, handletextpad=0.5)
        leg.set_zorder(20)
    elif n_particles is not None:
        leg = ax.legend(fontsize=6.8, loc="upper left", framealpha=0.80,
                        borderpad=0.4, handlelength=1.2, handletextpad=0.5)
        leg.set_zorder(20)
    else:
        ax.text(
            0.02, 0.975, f"N = {n_total} function evals",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.80, ec="#bbbbbb"),
        )
