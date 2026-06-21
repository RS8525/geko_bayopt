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

dimension = 2   # 1 or 2
DNS       = False

# Number of initial (non-BO) iterations; a vertical line is drawn after
# this many iterations to mark where the Bayesian Optimisation phase starts.
n_initial = 13  # 1D: 9  |  2D: 13

# (experiment_id, display label)
if dimension == 1 and not DNS:
    RUNS = [
        ("optimizer_test_bo",    "Bayesian Opt (GP)"),
        ("optimizer_test_nm",    "Nelder-Mead"),
        ("optimizer_test_fd",    "Finite Differences"),
        ("optimizer_test_nm_bo", "NM -> Bayesian"),
        ("optimizer_test_fd_bo", "FD -> Bayesian"),
    ]
    _title = "Optimizer Comparison — Periodic Hills Re=2800, 1D (geko_csep)"

elif dimension == 2 and not DNS:
    RUNS = [
        ("2D_optimizer_test_bo",    "Bayesian Opt (GP)"),
        ("2D_optimizer_test_nm",    "Nelder-Mead"),
        ("2D_optimizer_test_fd",    "Finite Differences"),
        ("2D_optimizer_test_nm_bo", "NM -> Bayesian"),
        ("2D_optimizer_test_fd_bo", "FD -> Bayesian"),
    ]
    _title = "Optimizer Comparison — Periodic Hills Re=2800, 2D (geko_csep, geko_cnw)"

elif dimension == 1 and DNS:
    RUNS = [
        ("DNS_optimizer_test_bo",    "Bayesian Opt (GP)"),
        ("DNS_optimizer_test_nm",    "Nelder-Mead"),
        ("DNS_optimizer_test_fd",    "Finite Differences"),
        ("DNS_optimizer_test_nm_bo", "NM -> Bayesian"),
        ("DNS_optimizer_test_fd_bo", "FD -> Bayesian"),
    ]
    _title = "Optimizer Comparison — Periodic Hills Re=2800, 1D DNS (geko_csep)"

else:  # dimension == 2 and DNS
    RUNS = [
        ("DNS_2D_optimizer_test_bo",    "Bayesian Opt (GP)"),
        ("DNS_2D_optimizer_test_nm",    "Nelder-Mead"),
        ("DNS_2D_optimizer_test_fd",    "Finite Differences"),
        ("DNS_2D_optimizer_test_nm_bo", "NM -> Bayesian"),
        ("DNS_2D_optimizer_test_fd_bo", "FD -> Bayesian"),
    ]
    _title = "Optimizer Comparison — Periodic Hills Re=2800, 2D DNS (geko_csep, geko_cnw)"

OUT_PATH = REPO_ROOT / "optimizer_comparison.png"


# ------------------------------------------------------------------ #
# Loading results                                                     #
# ------------------------------------------------------------------ #

def _load_scores(experiment_id: str) -> np.ndarray | None:
    """Return the raw score sequence from metadata.csv, or None if absent."""
    csv_path = (
        REPO_ROOT / "results" / "experiments" / experiment_id / "metadata.csv"
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


def plot_comparison(include_fake: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Real optimizer histories
    any_real = False
    for exp_id, label in RUNS:
        scores = _load_scores(exp_id)
        if scores is None:
            print(f"[plot] No results yet for '{label}' — skipping.")
            continue
        best_so_far = np.minimum.accumulate(scores)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, linewidth=2, label=label)
        any_real = True

    if not any_real:
        print("[plot] No real results found — showing fake histories only.")

    # Fake histories
    if include_fake:
        for label, curve in _fake_histories():
            ax.plot(
                range(1, len(curve) + 1), curve,
                linewidth=1.5, linestyle="--", alpha=0.5, label=label,
            )

    ax.axvline(x=n_initial + 0.5, color="red", linestyle="--", linewidth=1.2,
               label=f"BO phase start (iter {n_initial + 1})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(_title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUT_PATH, dpi=150)
    print(f"[plot] Saved -> {OUT_PATH}")
    plt.show()


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

    plot_comparison(include_fake=args.fake)
