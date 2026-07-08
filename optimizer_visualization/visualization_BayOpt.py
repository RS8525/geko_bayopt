"""Bayesian Optimization visualization.

Produces an ordered sequence of figures that walk through the BO scheme on a
1-D cost function, matching the presentation storyboard:

    01_objective.png            the (unknown) cost function we want to minimize
    02_initial_sampling.png     cost function + a few random initial samples
    03_observations_only.png    only the sampled points (no cost function)
    04_surrogate.png            samples + surrogate model (mean + uncertainty)
    05_surrogate_with_min.png   surrogate + gold star at the surrogate minimum
    06_first_evaluation.png     cost function evaluated at the star -> red star
    07_many_observations.png    the cost function after many more samples
    08_interpolation_vs_truth.png   surrogate now matches the true function
    legend.png                  standalone legend for the slides

The optimizer maximizes ``target_MAX``; every plot negates the values so the
figures are shown in the original minimization ("cost") space.

Plots are stripped of axes (ticks/labels), like the PSO figures, and share a
common y-range so the sequence does not jump around between slides.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

SAVE_FIGURE_TO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "plots", "BayOpt")

FIGSIZE   = (10, 8)
KAPPA     = 2.0          # UCB exploration/exploitation trade-off
RANDOM_STATE = 27
N_EXTRA   = 8            # BO iterations to skip forward for the "many points" plot

# Styling
COST_COLOR   = "tab:blue"
PRED_COLOR   = "black"
BAND_COLOR   = "lightsteelblue"
OBS_COLOR    = "red"
STAR_MIN     = "gold"    # surrogate minimum
STAR_NEXT    = "red"     # next evaluation

# Alternate surrogate kernel, for a side-by-side comparison against the
# bayes_opt default (Matern nu=2.5). Matern nu=0.5 (the exponential kernel) is
# far less smooth, so its posterior mean and confidence band look markedly
# different. Swap for RBF() or RationalQuadratic() to try other contrasts.
DEFAULT_KERNEL_NAME = "Matern 2.5 (default)"
ALT_KERNEL          = Matern(nu=0.5)
ALT_KERNEL_NAME     = "Matern 0.5 (exponential)"
ALT_BAND_COLOR      = "moccasin"
ALT_PRED_COLOR      = "tab:orange"

# Use LaTeX for the legend text only if a LaTeX install is present, otherwise
# matplotlib's mathtext is used so the script still runs.
plt.rcParams["text.usetex"] = shutil.which("latex") is not None
plt.rcParams["savefig.dpi"] = 150


# --------------------------------------------------------------------------- #
# Target function                                                             #
# --------------------------------------------------------------------------- #

def target(x):
    """Cost function (minimization space)."""
    return -(np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1))


def target_MAX(x):
    """Maximizing this is equivalent to minimizing ``target``."""
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def get_state(optimizer, x):
    """Return observations and the fitted surrogate, all in minimization space."""
    x_obs = np.array([res["params"]["x"] for res in optimizer.res])
    y_obs = -np.array([res["target"] for res in optimizer.res])

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = optimizer._gp.predict(x, return_std=True)
    mu = -mu
    return x_obs, y_obs, mu, sigma


def new_ax(ylim):
    """A figure with a plain frame: fixed limits, no ticks, no labels."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(-2, 10)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.015)
    return fig, ax


def save(fig, plotname):
    fig.savefig(os.path.join(SAVE_FIGURE_TO, plotname))
    plt.close(fig)


def draw_cost(ax, x, y, alpha=1.0):
    ax.plot(x, y, linewidth=5, color=COST_COLOR, alpha=alpha, zorder=2)


def draw_observations(ax, x_obs, y_obs, alpha=1.0):
    ax.plot(x_obs, y_obs, 'D', markersize=18, color=OBS_COLOR, alpha=alpha,
            zorder=5)


def draw_surrogate(ax, x, mu, sigma, band=True, band_color=BAND_COLOR,
                   pred_color=PRED_COLOR):
    if band:
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
                alpha=.6, fc=band_color, ec='None', zorder=1)
    ax.plot(x, mu, dashes=(2, 1), color=pred_color, linewidth=3, zorder=4)


def get_state_alt_kernel(optimizer, x, kernel):
    """Fit a fresh GP with ``kernel`` on the same observations as the optimizer.

    Mirrors the settings of the bayes_opt default GP but swaps the kernel, so
    the returned mean/std (in minimization space) are directly comparable to
    what :func:`get_state` produces for the Matern 2.5 default.
    """
    X = optimizer._space.params            # raw sample locations, shape (n, 1)
    y_max = optimizer._space.target        # targets in maximization space

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=RANDOM_STATE,
    )
    gp.fit(X, y_max)
    mu, sigma = gp.predict(x, return_std=True)
    mu = -mu
    return mu, sigma


# --------------------------------------------------------------------------- #
# Storyboard figures                                                          #
# --------------------------------------------------------------------------- #

def fig_objective(x, y, ylim):
    fig, ax = new_ax(ylim)
    draw_cost(ax, x, y)
    save(fig, "01_objective.png")


def fig_initial_sampling(x, y, ylim, x_obs, y_obs):
    fig, ax = new_ax(ylim)
    draw_cost(ax, x, y)
    draw_observations(ax, x_obs, y_obs)
    save(fig, "02_initial_sampling.png")


def fig_observations_only(ylim, x_obs, y_obs):
    fig, ax = new_ax(ylim)
    draw_observations(ax, x_obs, y_obs)
    save(fig, "03_observations_only.png")


def fig_surrogate(x, ylim, x_obs, y_obs, mu, sigma):
    fig, ax = new_ax(ylim)
    draw_surrogate(ax, x, mu, sigma)
    draw_observations(ax, x_obs, y_obs)
    save(fig, "04_surrogate.png")


def fig_surrogate_kernel_comparison(x, ylim, x_obs, y_obs, mu, sigma,
                                    mu_alt, sigma_alt):
    """Same observations, two surrogates: default Matern 2.5 vs ALT_KERNEL."""
    fig, ax = new_ax(ylim)
    draw_surrogate(ax, x, mu, sigma)                       # default colors
    draw_surrogate(ax, x, mu_alt, sigma_alt,
                   band_color=ALT_BAND_COLOR, pred_color=ALT_PRED_COLOR)
    draw_observations(ax, x_obs, y_obs)

    # Inline legend so the comparison is readable on its own.
    handles = [
        plt.Line2D([], [], color=PRED_COLOR, dashes=(2, 1)),
        plt.Line2D([], [], color=ALT_PRED_COLOR, dashes=(2, 1)),
    ]
    ax.legend(handles, [DEFAULT_KERNEL_NAME, ALT_KERNEL_NAME],
              loc='upper right', framealpha=0.9)
    save(fig, "04b_surrogate_kernel_comparison.png")


def fig_surrogate_with_min(x, ylim, x_obs, y_obs, mu, sigma, plotname,
                           highlight_last=False):
    fig, ax = new_ax(ylim)
    draw_surrogate(ax, x, mu, sigma)
    if highlight_last:
        # Keep the just-evaluated point as a red star, the rest as diamonds.
        draw_observations(ax, x_obs[:-1], y_obs[:-1])
        ax.plot(x_obs[-1], y_obs[-1], '*', markersize=32,
                markerfacecolor=STAR_NEXT, markeredgecolor='k',
                markeredgewidth=2, zorder=10)
    else:
        draw_observations(ax, x_obs, y_obs)

    lower_bound = mu - 1.96 * sigma
    min_idx = int(np.argmin(lower_bound))
    min_x = float(x[min_idx, 0])
    min_y = float(lower_bound[min_idx])
    ax.plot(min_x, min_y, '*', color=STAR_MIN, markersize=32,
            markeredgecolor='k', markeredgewidth=2, zorder=6)
    save(fig, plotname)
    return min_x


def fig_first_evaluation(x, y, ylim, x_obs, y_obs):
    fig, ax = new_ax(ylim)
    draw_cost(ax, x, y)
    # All but the newest sample as diamonds, the newest as a red star.
    draw_observations(ax, x_obs[:-1], y_obs[:-1])
    ax.plot(x_obs[-1], y_obs[-1], '*', markersize=32, markerfacecolor=STAR_NEXT,
            markeredgecolor='k', markeredgewidth=2, zorder=10)
    save(fig, "06_first_evaluation.png")


def fig_many_observations(x, y, ylim, x_obs, y_obs):
    fig, ax = new_ax(ylim)
    draw_cost(ax, x, y)
    draw_observations(ax, x_obs, y_obs)
    save(fig, "08_many_observations.png")


def fig_interpolation_vs_truth(x, y, ylim, x_obs, y_obs, mu, sigma):
    fig, ax = new_ax(ylim)
    draw_cost(ax, x, y, alpha=0.5)
    draw_surrogate(ax, x, mu, sigma, band=False)
    save(fig, "09_interpolation_vs_truth.png")


def make_legend():
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([], [], color=COST_COLOR, linewidth=3, label='Cost function')
    ax.plot([], [], color=PRED_COLOR, dashes=(2, 1), label='Surrogate model')
    ax.fill([], [], fc=BAND_COLOR, alpha=.6, label='95% confidence interval')
    ax.plot([], [], 'D', color=OBS_COLOR, label='Observations')
    ax.plot([], [], '*', markersize=12, color=STAR_MIN, markeredgecolor='k',
            markeredgewidth=1, label='Minimum of surrogate model')
    ax.plot([], [], '*', markersize=12, color=STAR_NEXT, markeredgecolor='k',
            markeredgewidth=1, label='Next evaluation')
    ax.axis('off')
    ax.legend(loc='center')
    save(fig, "legend.png")


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    os.makedirs(SAVE_FIGURE_TO, exist_ok=True)

    x = np.linspace(-2.0, 10, 10000).reshape(-1, 1)
    y = target(x)

    # Build the optimizer (maximizing target_MAX minimizes target).
    acquisition_function = acquisition.UpperConfidenceBound(kappa=KAPPA)
    optimizer = BayesianOptimization(target_MAX, {"x": (-2, 10)},
                                     acquisition_function=acquisition_function,
                                     random_state=RANDOM_STATE)

    # Initial user-defined random samples.
    points = [-1.7, 0.0, 4.0, 5.0, 6.0, 9.0]
    for p in points:
        optimizer.register(params=p, target=target_MAX(p))

    # Fix a common y-range from the widest (initial) surrogate band so the
    # sequence of slides shares the same framing.
    x_obs, y_obs, mu, sigma = get_state(optimizer, x)
    lo = min(y.min(), (mu - 1.96 * sigma).min())
    hi = max(y.max(), (mu + 1.96 * sigma).max())
    margin = 0.05 * (hi - lo)
    ylim = (lo - margin, hi + margin)

    # 1) The unknown objective.
    fig_objective(x, y, ylim)

    # 2) Initial random sampling.
    fig_initial_sampling(x, y, ylim, x_obs, y_obs)

    # 3) Only the observations.
    fig_observations_only(ylim, x_obs, y_obs)

    # 3.1) Observations + surrogate model.
    fig_surrogate(x, ylim, x_obs, y_obs, mu, sigma)

    # 3.1b) Same observations, but a very different kernel (ALT_KERNEL) for the
    #       surrogate, overlaid on the default Matern 2.5 band to compare.
    mu_alt, sigma_alt = get_state_alt_kernel(optimizer, x, ALT_KERNEL)
    fig_surrogate_kernel_comparison(x, ylim, x_obs, y_obs, mu, sigma,
                                    mu_alt, sigma_alt)

    # 3.2) Surrogate + its minimum (gold star), which is the next place to look.
    min_x = fig_surrogate_with_min(x, ylim, x_obs, y_obs, mu, sigma,
                                   "05_surrogate_with_min.png")

    # 4) Evaluate the true cost function at the surrogate minimum -> red star.
    optimizer.register(params=min_x, target=target_MAX(min_x))
    x_obs, y_obs, mu, sigma = get_state(optimizer, x)
    fig_first_evaluation(x, y, ylim, x_obs, y_obs)

    # 4.1) Refit the surrogate on the new point and show its updated minimum:
    #      one turn of the BO loop is complete, and it proposes the next spot.
    fig_surrogate_with_min(x, ylim, x_obs, y_obs, mu, sigma,
                           "07_surrogate_with_min.png", highlight_last=True)

    # 5) Skip several BO iterations forward.
    for _ in range(N_EXTRA):
        nxt = optimizer.suggest()
        optimizer.register(params=nxt, target=target_MAX(**nxt))
    x_obs, y_obs, mu, sigma = get_state(optimizer, x)
    fig_many_observations(x, y, ylim, x_obs, y_obs)

    # 5.1) The surrogate now interpolates the true function closely.
    fig_interpolation_vs_truth(x, y, ylim, x_obs, y_obs, mu, sigma)

    # Standalone legend for the slides.
    make_legend()

    print(f"Done. Figures written to {SAVE_FIGURE_TO}")


if __name__ == "__main__":
    main()
