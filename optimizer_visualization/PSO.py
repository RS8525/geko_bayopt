"""Educational PSO visualization on 2-D benchmark functions.

Generates a sequence of presentation-quality figures that illustrate how a
Particle Swarm Optimizer (PSO) explores the search space: initial Sobol
placement, personal/global best tracking, velocity updates (drawn as arrows
of the *actual* step length) and convergence toward the optimum.

The update rule mirrors the project optimizer
(src/geko_bayesopt/optimizer.py, ParticleSwarmOptimizer): linearly decayed
inertia, cognitive + social attraction, velocity clamping and absorption at
the domain bounds.  The focus here is clarity, not optimization performance.

The benchmark is selectable via ``FUNCTION`` below (rosenbrock, beale or
goldstein_price).  Figures are written to
``optimizer_visualization/plots/PSO/<function>/`` as iteration_000.png
(Initialization), iteration_001.png (Iteration 1), ...

Libraries: numpy, matplotlib, scipy only.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy.stats import qmc


# --------------------------------------------------------------------------- #
# Benchmark functions                                                         #
# --------------------------------------------------------------------------- #

def rosenbrock(x, y):
    """f = (1 - x)^2 + 100 (y - x^2)^2.  Minimum 0 at (1, 1)."""
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def beale(x, y):
    """Beale function.  Minimum 0 at (3, 0.5)."""
    return ((1.5 - x + x * y) ** 2
            + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def goldstein_price(x, y):
    """Goldstein-Price function.  Minimum 3 at (0, -1)."""
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2
                                - 14 * y + 6 * x * y + 3 * y ** 2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2
                                     + 48 * y - 36 * x * y + 27 * y ** 2)
    return a * b


# name -> (function, x-bounds, y-bounds, global optimum)
BENCHMARKS = {
    "rosenbrock":      (rosenbrock,      (-2.0, 2.0), (-1.0, 3.0), (1.0,  1.0)),
    "beale":           (beale,           (-4.5, 4.5), (-4.5, 4.5), (3.0,  0.5)),
    "goldstein_price": (goldstein_price, (-2.0, 2.0), (-2.0, 2.0), (0.0, -1.0)),
}


# --------------------------------------------------------------------------- #
# Configuration (exposed near the top)                                        #
# --------------------------------------------------------------------------- #

# Select which benchmark to visualize: "rosenbrock", "beale" or "goldstein_price".
FUNCTION = "goldstein_price"
FUNCTION = os.environ.get("PSO_FUNCTION", FUNCTION)  # optional override for batch runs

N_PARTICLES  = 6    # swarm size
N_ITERATIONS = 20      # number of swarm moves after initialization
PLOT_EVERY   = 1      # save a figure every PLOT_EVERY-th frame (init + final always)
PBEST_LAST_ITER = 5   # show per-particle personal-best stars up to this iteration only

# PSO coefficients.  Tuned softer than the project defaults so the per-step
# moves (and therefore the arrows) stay short and readable for the demo.
W_START    = 0.7      # initial inertia weight
W_END      = 0.4      # final inertia weight
C1         = 1.5      # cognitive coefficient (pull toward personal best)
C2         = 1.5      # social coefficient    (pull toward global best)
V_MAX_FRAC = 0.08     # velocity clamp as a fraction of each dimension's range

# Presentation
PARTICLE_COLOR = "black"       # all particles share this colour
GBEST_COLOR    = "tab:orange"  # current global best marker
PBEST_COLOR    = "tab:green"   # per-particle personal best marker (green star)
ARROW_COLOR    = "black"
FIGSIZE        = (10, 8)

# Reproducibility (used but not surfaced as a knob)
SEED = 42

# --- Derived from the selected benchmark ----------------------------------- #
if FUNCTION not in BENCHMARKS:
    raise ValueError(f"Unknown FUNCTION {FUNCTION!r}; choose one of "
                     f"{list(BENCHMARKS)}.")
objective, X_BOUNDS, Y_BOUNDS, _OPTIMUM = BENCHMARKS[FUNCTION]
BOUNDS         = np.array([X_BOUNDS, Y_BOUNDS])     # [[xlo,xhi],[ylo,yhi]]
GLOBAL_OPTIMUM = np.array(_OPTIMUM)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plots", "PSO")

# Precomputed velocity clamp per dimension.
_V_MAX = V_MAX_FRAC * (BOUNDS[:, 1] - BOUNDS[:, 0])

# Larger fonts so the figures stay readable on a projector.
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   20,
    "axes.labelsize":   16,
    "legend.fontsize":  13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
})


# --------------------------------------------------------------------------- #
# Initialization                                                              #
# --------------------------------------------------------------------------- #

def sobol_initial_positions() -> np.ndarray:
    """Draw N_PARTICLES Sobol points and scale them into the domain."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=SEED)
    with warnings.catch_warnings():
        # A non-power-of-2 sample size sacrifices Sobol balance, which is
        # irrelevant for a small teaching demo; silence the notice.
        warnings.simplefilter("ignore", category=UserWarning)
        unit = sampler.random(N_PARTICLES)               # (N_PARTICLES, 2) in [0, 1)
    return qmc.scale(unit, BOUNDS[:, 0], BOUNDS[:, 1])   # scaled into the domain


# --------------------------------------------------------------------------- #
# PSO update                                                                  #
# --------------------------------------------------------------------------- #

def current_inertia(t: int) -> float:
    """Linearly decay inertia from W_START to W_END across N_ITERATIONS."""
    frac = min(t, N_ITERATIONS) / N_ITERATIONS
    return W_START - (W_START - W_END) * frac


def compute_next(positions, velocities, pbest_x, gbest_x, rng, t):
    """Return (next_positions, next_velocities) without mutating the inputs.

    Standard PSO velocity update with clamping and absorbing boundaries:
        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x <- x + v
    Overshooting a bound pins the particle to it and zeroes that velocity
    component (absorption), matching the project optimizer.
    """
    w = current_inertia(t)
    n = positions.shape[0]
    next_pos = np.empty_like(positions)
    next_vel = np.empty_like(velocities)

    for i in range(n):
        r1 = rng.uniform(0.0, 1.0, 2)
        r2 = rng.uniform(0.0, 1.0, 2)
        v_new = (w * velocities[i]
                 + C1 * r1 * (pbest_x[i] - positions[i])
                 + C2 * r2 * (gbest_x     - positions[i]))
        v_new = np.clip(v_new, -_V_MAX, _V_MAX)
        x_new = positions[i] + v_new

        for d in range(2):
            if x_new[d] < BOUNDS[d, 0]:
                x_new[d], v_new[d] = BOUNDS[d, 0], 0.0
            elif x_new[d] > BOUNDS[d, 1]:
                x_new[d], v_new[d] = BOUNDS[d, 1], 0.0

        next_pos[i] = x_new
        next_vel[i] = v_new

    return next_pos, next_vel


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

# Contour scaffolding, computed once for the selected function.
_gx = np.linspace(*X_BOUNDS, 400)
_gy = np.linspace(*Y_BOUNDS, 400)
_GX, _GY = np.meshgrid(_gx, _gy)
_GZ = objective(_GX, _GY)
# Log-spaced levels.  Floor the low end at 0.1 for functions whose minimum is
# 0 (rosenbrock, beale); functions with a positive minimum (goldstein_price)
# start their levels at that minimum.
_LO = max(_GZ.min(), 0.1)
_LEVELS = np.logspace(np.log10(_LO), np.log10(_GZ.max()), 30)


def _label_point(ax, xy, label):
    """Annotate a marker with its particle number, offset a touch up-right."""
    ax.annotate(str(label), xy=(float(xy[0]), float(xy[1])),
                xytext=(7, 7), textcoords="offset points",
                fontsize=12, fontweight="bold", color="black",
                ha="left", va="bottom", zorder=10,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def plot_frame(frame_idx, positions, gbest_x, steps, pbest_x=None,
               show_arrows=True, show_gbest=True, show_pbest=True):
    """Render one PSO frame and save it to OUT_DIR/iteration_%03d.png."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Objective landscape: filled log-spaced contours + faint isolines.
    ax.contourf(_GX, _GY, _GZ, levels=_LEVELS, norm=LogNorm(),
                cmap="Blues", alpha=0.75, extend="max")
    ax.contour(_GX, _GY, _GZ, levels=_LEVELS, norm=LogNorm(),
               colors="white", linewidths=0.3, alpha=0.5)

    # Velocity arrows: exact length and direction of the upcoming step.
    if show_arrows:
        ax.quiver(positions[:, 0], positions[:, 1], steps[:, 0], steps[:, 1],
                  angles="xy", scale_units="xy", scale=1.0,
                  color=ARROW_COLOR, width=0.0028, headwidth=5, headlength=6,
                  zorder=5, alpha=0.9)

    # --- Per-particle coincidence bookkeeping --------------------------- #
    # pos == pbest, and (when the global best is shown) pbest == gbest.  These
    # decide which of the overlapping markers/labels to keep:
    #   pbest == gbest -> personal best *is* the global best star: drop the
    #                     green star, keep the orange one (the black dot still
    #                     shows, on top of the orange star)
    #   pos == pbest   -> one shared number for the dot + green star
    n   = positions.shape[0]
    tol = 1e-9
    pb  = positions if pbest_x is None else pbest_x

    pos_eq_pb = np.array([np.allclose(positions[i], pb[i], atol=tol)
                          for i in range(n)])
    if show_gbest:
        pb_eq_gb = np.array([np.allclose(pb[i], gbest_x, atol=tol)
                             for i in range(n)])
    else:
        pb_eq_gb = np.zeros(n, dtype=bool)

    green_drawn = (~pb_eq_gb) if show_pbest else np.zeros(n, dtype=bool)

    # Personal bests: green stars, below the particle dots so a particle that
    # sits on its own best shows the black dot inside the green star.
    if green_drawn.any():
        ax.scatter(pb[green_drawn, 0], pb[green_drawn, 1], marker="*", s=500,
                   c=PBEST_COLOR, edgecolors="black", linewidths=1.0, zorder=3)

    # Particles (all the same colour), drawn above the global-best star so a
    # particle sitting on the global best stays visible on top of it.
    ax.scatter(positions[:, 0], positions[:, 1], s=75, c=PARTICLE_COLOR,
               edgecolors="black", linewidths=0.5, zorder=20)

    # Global optimum: large gold star.
    ax.scatter(GLOBAL_OPTIMUM[0], GLOBAL_OPTIMUM[1], marker="*", s=650,
               c="gold", edgecolors="black", linewidths=1.4, zorder=8)

    # Current global best: orange star.
    if show_gbest:
        ax.scatter(gbest_x[0], gbest_x[1], marker="*", s=700, c=GBEST_COLOR,
                   edgecolors="black", linewidths=1.4, zorder=4)

    # Particle numbers.  Each particle is numbered next to its current position;
    # its green personal-best star carries the same number.  When the position
    # and the personal best coincide, a single number is shown.
    for i in range(n):
        _label_point(ax, positions[i], i + 1)
        if green_drawn[i] and not pos_eq_pb[i]:
            _label_point(ax, pb[i], i + 1)

    ax.set_xlim(X_BOUNDS)
    ax.set_ylim(Y_BOUNDS)
    ax.set_xticks([])
    ax.set_yticks([])

    # Manual legend so proxy handles match the on-plot styling.
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=PARTICLE_COLOR,
               markeredgecolor="white", markersize=13, label="Particles"),
    ]
    if green_drawn.any():
        handles.append(Line2D([], [], marker="*", color="none",
                              markerfacecolor=PBEST_COLOR, markeredgecolor="black",
                              markersize=17, label="Personal best"))
    if show_gbest:
        handles.append(Line2D([], [], marker="*", color="none",
                              markerfacecolor=GBEST_COLOR, markeredgecolor="black",
                              markersize=20, label="Current global best"))
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="gold",
                          markeredgecolor="black", markersize=22,
                          label="Global optimum"))
    if show_arrows:
        handles.append(Line2D([], [], color=ARROW_COLOR, lw=2,
                              label="Next PSO step"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"iteration_{frame_idx:03d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- Initialization -------------------------------------------------- #
    positions  = sobol_initial_positions()
    velocities = np.zeros_like(positions)
    values     = objective(positions[:, 0], positions[:, 1])

    pbest_x = positions.copy()
    pbest_y = values.copy()
    gbest_i = int(np.argmin(pbest_y))
    gbest_x = pbest_x[gbest_i].copy()
    gbest_y = float(pbest_y[gbest_i])

    # --- Opening frames: the initial swarm, no direction vectors --------- #
    # First just the particles, then the same frame with the current global
    # best (orange star) revealed.
    plot_frame(0, positions, gbest_x, steps=None, pbest_x=pbest_x,
               show_arrows=False, show_gbest=False, show_pbest=False)
    plot_frame(1, positions, gbest_x, steps=None, pbest_x=pbest_x,
               show_arrows=False, show_gbest=True, show_pbest=True)

    # --- Frame loop ------------------------------------------------------ #
    # Frame t+2 shows the swarm state at step t together with arrows toward
    # step t+1.  Frames 0-1 (above) are the swarm without arrows; frames
    # 2..N+2 carry the velocity arrows.
    for t in range(N_ITERATIONS + 1):
        next_pos, next_vel = compute_next(positions, velocities,
                                          pbest_x, gbest_x, rng, t)
        steps = next_pos - positions

        if t % PLOT_EVERY == 0 or t == N_ITERATIONS:
            plot_frame(t + 2, positions, gbest_x, steps, pbest_x=pbest_x,
                       show_pbest=(t <= PBEST_LAST_ITER))

        if t == N_ITERATIONS:
            break  # final frame only illustrates the next move; do not apply it

        # Apply the exact move that was drawn as arrows.
        positions  = next_pos
        velocities = next_vel
        values     = objective(positions[:, 0], positions[:, 1])

        improved = values < pbest_y
        pbest_x[improved] = positions[improved]
        pbest_y[improved] = values[improved]

        best_i = int(np.argmin(pbest_y))
        if pbest_y[best_i] < gbest_y:
            gbest_y = float(pbest_y[best_i])
            gbest_x = pbest_x[best_i].copy()

    print(f"\nDone ({FUNCTION}). Global best value f = {gbest_y:.4g} at "
          f"({gbest_x[0]:.3f}, {gbest_x[1]:.3f}).")


if __name__ == "__main__":
    main()
