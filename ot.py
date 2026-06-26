"""
Optimizer comparison for the periodic hills Re=2800 case.

Usage
-----
    python optimizers_test.py            # plot results from metadata.csv files
    python optimizers_test.py --fake     # overlay fake histories (no CFD needed)

The comparison plot is saved to optimizer_comparison.png in the repo root.
Missing experiment folders and early-stopped runs are silently skipped.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.resolve()

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

# Number of initial random-exploration samples in the BO run; a vertical
# line is drawn after this many iterations to mark where BO takes over.
n_initial = 8  # from BO_1D_ph2800.json: optimizer.kind_specific_options.n_initial

RUNS = [
    ("BO_1D_2800",  "Bayesian Opt (GP)"),
    ("FD_1D_2800",  "Finite Differences"),
    ("PSO_1D_2800", "Particle Swarm"),
]

_title = "Optimizer Comparison — Periodic Hills Re=2800, 1D (geko_csep)"

OUT_PATH_LINEAR = REPO_ROOT / "optimizer_comparison_linear.png"
OUT_PATH_LOG    = REPO_ROOT / "optimizer_comparison_log.png"
OUT_PATH_ZOOM   = REPO_ROOT / "optimizer_comparison_zoom.png"


# ------------------------------------------------------------------ #
# Loading results                                                     #
# ------------------------------------------------------------------ #

def _load_scores(experiment_id: str) -> np.ndarray | None:
    """Return the raw score sequence from metadata.csv, or None if absent."""
    csv_path = (
        REPO_ROOT / "Results" / "experiments" / experiment_id / "metadata.csv"
    )
    if not csv_path.exists():
        return None
    scores = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores.append(float(row["score"]))
    return np.array(scores) if scores else None


# ------------------------------------------------------------------ #
# Plotting                                                            #
# ------------------------------------------------------------------ #

def _fake_histories(n: int = 22) -> list[tuple[str, np.ndarray]]:
    """Synthetic convergence curves for visual testing of the plot."""
    rng = np.random.default_rng(0)
    x = np.arange(1, n + 1)

    def noisy_decay(start: float, end: float, rate: float, noise: float = 0.01):
        raw = end + (start - end) * np.exp(-rate * (x - 1))
        raw += rng.normal(0, noise, n)
        return np.minimum.accumulate(raw)

    return [
        ("Fake: fast converger",  noisy_decay(0.45, 0.07, 0.35, 0.008)),
        ("Fake: slow converger",  noisy_decay(0.45, 0.20, 0.08, 0.008)),
        ("Fake: stuck at plateau", np.full(n, 0.38)),
        ("Fake: late improvement", noisy_decay(0.45, 0.10, 0.06, 0.005)),
    ]


def plot_comparison(include_fake: bool = False, log: bool = False, zoom: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    loaded: list[tuple[str, np.ndarray]] = []
    for exp_id, label in RUNS:
        scores = _load_scores(exp_id)
        if scores is None:
            print(f"[plot] No results yet for '{label}' — skipping.")
        else:
            loaded.append((label, scores))

    if not loaded:
        print("[plot] No real results found — showing fake histories only.")

    max_iters = max(len(s) for _, s in loaded) if loaded else 0

    # Keep per-run data for inset reuse; ext_* holds the prolongation segment.
    plot_data = []
    any_extension = False
    for label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)
        (line,) = ax.plot(iters, best_so_far, linewidth=2, label=label)
        color = line.get_color()
        ax.scatter(iters, scores, color=color, s=18, alpha=0.35, zorder=3)

        ext_iters = ext_y = None
        if len(scores) < max_iters:
            # Overlap by one point for visual continuity.
            ext_iters = np.arange(len(scores), max_iters + 1)
            ext_y = np.full(len(ext_iters), best_so_far[-1])
            ax.plot(ext_iters, ext_y, color=color, linewidth=2,
                    alpha=0.35, linestyle="--")
            any_extension = True

        plot_data.append((iters, best_so_far, scores, color, ext_iters, ext_y))

    if include_fake:
        for label, curve in _fake_histories():
            ax.plot(range(1, len(curve) + 1), curve,
                    linewidth=1.5, linestyle="--", alpha=0.5, label=label)

    if log or zoom:
        ax.set_yscale("log")

    if zoom and plot_data:
        global_best_zoom = min(np.min(s) for _, s in loaded)
        ax.set_xlim(13.5, max_iters + 0.5)
        ax.set_ylim(global_best_zoom * 0.999,
                    max(bsf[-1] for _, bsf, _, _, _, _ in plot_data) * 1.005)

    ax.axvline(x=n_initial + 0.5, color="red", linestyle="--", linewidth=1.2,
               label=f"BO: exploration → optimisation (iter {n_initial + 1})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far" + (" (log scale)" if log or zoom else ""))
    ax.set_title(_title)

    handles, labels = ax.get_legend_handles_labels()
    if any_extension:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="gray", linewidth=2,
                               alpha=0.35, linestyle="--"))
        labels.append("prolonged (final best repeated)")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Inset zoom — linear, non-zoom plot only.
    if not log and not zoom and plot_data:
        global_best = min(np.min(s) for _, s in loaded)

        x0, x1 = 10.5, max_iters + 0.5
        y0 = global_best * 0.999
        # Tight upper bound: worst final best-so-far + small pad, so all three
        # convergence lines are spread across the inset rather than crushed at the bottom.
        y1 = max(bsf[-1] for _, bsf, _, _, _, _ in plot_data) * 1.005

        axins = ax.inset_axes([0.05, 0.52, 0.40, 0.44])
        for iters, best_so_far, scores, color, ext_iters, ext_y in plot_data:
            axins.plot(iters, best_so_far, linewidth=1.5, color=color)
            axins.scatter(iters, scores, color=color, s=12, alpha=0.35, zorder=3)
            if ext_iters is not None:
                axins.plot(ext_iters, ext_y, color=color, linewidth=1.5,
                           alpha=0.35, linestyle="--")

        axins.set_yscale("log")
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.tick_params(labelsize=7)
        axins.set_xlabel("Iteration", fontsize=7)
        axins.set_ylabel("Best cost", fontsize=7)
        axins.grid(True, alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor="0.5")

    fig.tight_layout()
    out_path = OUT_PATH_ZOOM if zoom else (OUT_PATH_LOG if log else OUT_PATH_LINEAR)
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot optimizer comparison.")
    parser.add_argument(
        "--fake", action="store_true",
        help="Overlay fake histories — useful when no CFD results are available.",
    )
    args = parser.parse_args()

    plot_comparison(include_fake=args.fake, log=False)
    plot_comparison(include_fake=args.fake, log=True)
    plot_comparison(include_fake=args.fake, zoom=True)
