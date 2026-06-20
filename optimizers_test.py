"""
Optimizer comparison for the periodic hills Re=2800 case.

Usage
-----
    python optimizers_test.py            # run all optimizers, then plot
    python optimizers_test.py --plot     # skip running, only plot
    python optimizers_test.py --fake     # plot fake histories only (no CFD needed)

Running is fully resumable: if a config's results/metadata.csv already contains
all n_calls trials, run_experiment() returns immediately and the next optimizer
starts. Re-run the script at any time to pick up where it left off.

The comparison plot is saved to optimizer_comparison.png in the repo root.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from geko_bayesopt.experiment import run_experiment

REPO_ROOT = Path(__file__).parent.resolve()

# (config file, experiment_id, display label)
RUNS = [
    (REPO_ROOT / "configs/tests/BO.json",    "optimizer_test_bo",    "Bayesian Opt (GP)"),
    (REPO_ROOT / "configs/tests/NM.json",    "optimizer_test_nm",    "Nelder-Mead"),
    (REPO_ROOT / "configs/tests/FD.json",    "optimizer_test_fd",    "Finite Differences"),
    (REPO_ROOT / "configs/tests/NM_BO.json", "optimizer_test_nm_bo", "NM -> Bayesian"),
    (REPO_ROOT / "configs/tests/FD_BO.json", "optimizer_test_fd_bo", "FD -> Bayesian"),
]

OUT_PATH = REPO_ROOT / "optimizer_comparison.png"


# ------------------------------------------------------------------ #
# Running                                                            #
# ------------------------------------------------------------------ #

def run_all() -> None:
    for config_path, _exp_id, label in RUNS:
        print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")
        run_experiment(config_path, repo_root=REPO_ROOT)


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
    for _cfg, exp_id, label in RUNS:
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

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title("Optimizer Comparison — Periodic Hills Re=2800, 1D (geko_csep)")
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
    parser = argparse.ArgumentParser(description="Run and compare optimizers.")
    parser.add_argument(
        "--plot", action="store_true",
        help="Skip running; only regenerate the comparison plot.",
    )
    parser.add_argument(
        "--fake", action="store_true",
        help="Plot fake histories only — no CFD results needed.",
    )
    args = parser.parse_args()

    if args.fake:
        plot_comparison(include_fake=True)
    else:
        if not args.plot:
            run_all()
        plot_comparison(include_fake=True)
