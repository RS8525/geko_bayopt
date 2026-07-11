"""Wall-clock time of one BO ask/tell cycle vs. surrogate memory size.

Benchmarks how the cost of a single ask/tell cycle of GP-based Bayesian
optimization grows as the surrogate model accumulates observations, up to
N_TOTAL = 500 points. Exact GP regression refits the model on every tell(),
and inverting the (n x n) kernel matrix costs O(n^3), so the per-cycle time
is expected to scale cubically in the number of stored observations.

Setup mirrors the production BO in src/geko_bayesopt/optimizer.py:
    - skopt.Optimizer with the same explicitly-built GaussianProcessRegressor
      (Matern nu=2.5 kernel, n_restarts_optimizer=10, fixed random_state) as
      _resolve_bo_base_estimator("GP") constructs,
    - the first N_INITIAL_SOBOL = 60 points drawn by skopt's Sobol
      initial-point generator (no GP fit during that phase),
    - one ask/tell cycle per iteration afterwards, each cycle refitting the
      GP and optimizing the acquisition function.

The objective is a cheap fixed degree-5 polynomial (separable, random
coefficients from a seeded RNG), NOT the CFD cost function; its evaluation
time is negligible so the measured cycle time is pure optimizer overhead.

Output:
    plots/BayOpt/ask_tell_timing.png   iteration vs. cycle wall-clock time,
                                       with a cubic fit over the GP phase
    plots/BayOpt/ask_tell_timing.csv   raw (iteration, seconds) pairs, so the
                                       plot can be restyled without rerunning

Note: with the production GP settings (10 kernel-hyperparameter restarts per
fit) the full 500-iteration run takes a while in 5-D; reduce N_TOTAL or
n_restarts_optimizer below for a quick smoke test.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from skopt import Optimizer as SkoptOptimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

SAVE_TO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plots", "BayOpt")

DIM             = 5      # search-space dimension (matches the 5D GEKO runs)
N_INITIAL_SOBOL = 60     # Sobol points before the GP takes over
N_TOTAL         = 500    # total ask/tell cycles (= final surrogate memory)
RANDOM_STATE    = 42
POLY_DEGREE     = 5      # degree of the stand-in objective polynomial

FIGSIZE = (10, 6)

MEASURED_COLOR = "tab:blue"
FIT_COLOR      = "tab:orange"
SOBOL_SHADE    = "0.92"

plt.rcParams["savefig.dpi"] = 150


# --------------------------------------------------------------------------- #
# Stand-in objective: fixed random degree-5 polynomial                        #
# --------------------------------------------------------------------------- #

# Coefficients are drawn once from a seeded RNG so the objective is a fixed,
# deterministic function of x. Separable form: f(x) = sum_j sum_k c[j,k] x_j^k.
_rng = np.random.default_rng(RANDOM_STATE)
_COEFFS = _rng.uniform(-1.0, 1.0, size=(DIM, POLY_DEGREE + 1))


def objective(x: list[float]) -> float:
    """Cheap degree-5 polynomial standing in for the CFD cost function."""
    xj = np.asarray(x)
    powers = xj[:, None] ** np.arange(POLY_DEGREE + 1)[None, :]
    return float(np.sum(_COEFFS * powers))


# --------------------------------------------------------------------------- #
# Timed BO loop                                                               #
# --------------------------------------------------------------------------- #

def build_optimizer() -> SkoptOptimizer:
    """skopt Optimizer configured like the production GP-based BO."""
    dimensions = [Real(0.0, 1.0, name=f"x{j}") for j in range(DIM)]
    base_estimator = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=10,
        random_state=RANDOM_STATE,
    )
    return SkoptOptimizer(
        dimensions=dimensions,
        base_estimator=base_estimator,
        n_initial_points=N_INITIAL_SOBOL,
        initial_point_generator="sobol",
        random_state=RANDOM_STATE,
    )


def run_timed_loop() -> np.ndarray:
    """Run N_TOTAL ask/tell cycles and return per-cycle wall-clock seconds."""
    opt = build_optimizer()
    seconds = np.empty(N_TOTAL)

    for i in range(N_TOTAL):
        t0 = time.perf_counter()
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
        seconds[i] = time.perf_counter() - t0

        if (i + 1) % 25 == 0:
            print(f"  iteration {i + 1:4d}/{N_TOTAL}: {seconds[i]:.3f} s")

    return seconds


# --------------------------------------------------------------------------- #
# Cubic fit and plot                                                          #
# --------------------------------------------------------------------------- #

def cubic_fit(iterations: np.ndarray, seconds: np.ndarray):
    """Least-squares cubic fit over the GP phase (Sobol cycles excluded)."""
    gp_phase = iterations > N_INITIAL_SOBOL
    coeffs = np.polyfit(iterations[gp_phase], seconds[gp_phase], deg=3)
    return coeffs, gp_phase


def make_plot(iterations: np.ndarray, seconds: np.ndarray) -> None:
    coeffs, gp_phase = cubic_fit(iterations, seconds)
    fit_x = iterations[gp_phase]
    fit_y = np.polyval(coeffs, fit_x)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Shade the Sobol warm-up phase: no GP fit happens there, so those
    # cycles are nearly free and excluded from the cubic fit.
    ax.axvspan(1, N_INITIAL_SOBOL, color=SOBOL_SHADE, zorder=0)
    ax.text(N_INITIAL_SOBOL / 2, 0.95, "Sobol\ninitialization",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="0.35")

    ax.plot(iterations, seconds, "o", markersize=4, color=MEASURED_COLOR,
            alpha=0.7, label="measured ask/tell cycle", zorder=2)
    ax.plot(fit_x, fit_y, "--", linewidth=2, color=FIT_COLOR,
            label="cubic fit (GP phase)", zorder=3)

    ax.set_xlabel("Iteration (= observations in surrogate memory)")
    ax.set_ylabel("Wall-clock time per ask/tell cycle [s]")
    ax.set_xlim(0, N_TOTAL)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_TO, "ask_tell_timing.png"))
    plt.close(fig)

    print("Cubic fit t(n) = a3*n^3 + a2*n^2 + a1*n + a0:")
    for name, c in zip(("a3", "a2", "a1", "a0"), coeffs):
        print(f"  {name} = {c:.3e}")


def save_csv(iterations: np.ndarray, seconds: np.ndarray) -> None:
    path = os.path.join(SAVE_TO, "ask_tell_timing.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "seconds"])
        writer.writerows(zip(iterations.tolist(), seconds.tolist()))


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    os.makedirs(SAVE_TO, exist_ok=True)

    print(f"Timing {N_TOTAL} ask/tell cycles "
          f"({N_INITIAL_SOBOL} Sobol + {N_TOTAL - N_INITIAL_SOBOL} GP) "
          f"in {DIM}-D ...")
    seconds = run_timed_loop()
    iterations = np.arange(1, N_TOTAL + 1)

    save_csv(iterations, seconds)
    make_plot(iterations, seconds)

    print(f"Done. Output written to {SAVE_TO}")


if __name__ == "__main__":
    main()
