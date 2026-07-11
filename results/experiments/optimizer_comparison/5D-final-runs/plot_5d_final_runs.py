"""Plots for the 5D final runs living in this folder.

Combines two existing scripts, adapted to the runs stored next to this file:

1. Optimizer convergence comparison (from
   optimizer_visualization/optimizer_comparison.py, 5-D case): best-so-far
   cost over iterations for the five lambda_preference = 0.5 runs
   (BO, NM, FD, and the two PSO p15 variants), as one full plot plus
   after-cut plots at iterations 50 / 100 / 200.

2. Lambda-preference comparison (from
   scripts/lambda_comparison/lambda_comparison.py): the four BO runs at
   lambda_preference = 0.15 / 0.25 / 0.35 / 0.5. Picks each run's best
   optimizer trial and produces the coefficient table, the unweighted
   (pure field error) cost over lambda, and one optimized-coefficient-
   over-lambda plot per GEKO coefficient.

All final-run configs use lambda_integral = 0 and lambda_field = 1, so the
unweighted field error is recovered exactly from the stored weighted score:

    score = lambda_field * E_field * (1 + lambda_p * p(params))
    E_field = score / (lambda_field * (1 + lambda_p * p(params)))

with p computed by the same ``coefficient_preference`` implementation the
objective itself uses. No Fluent run is needed.

Run from anywhere:

    python results/experiments/optimizer_comparison/5D-final-runs/plot_5d_final_runs.py

Plots are written to a plots/ subfolder next to this script.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNS_DIR.parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geko_bayesopt.geko_defaults import GEKO_DEFAULTS  # noqa: E402
from geko_bayesopt.objective.GEDCP import coefficient_preference  # noqa: E402

OUTPUT_DIR = RUNS_DIR / "plots"

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

# Convergence comparison: every lambda_preference = 0.5 run in this folder.
# The lambda sweep runs (0.15 / 0.25 / 0.35) are excluded here because their
# scores carry a different preference weighting and are not comparable.
RUNS_CONVERGENCE = [
    ("BO_5D_ph2800",          "Bayesian Opt (GP)"),
    ("NM_5D_ph2800",          "Nelder-Mead"),
    ("FD_5D_ph2800",          "Finite Differences"),
    ("PSO_5D_ph2800_p15",     "Particle Swarm; 15 particles"),
    ("PSO_5D_ph2800_p15_std", "Particle Swarm; 15 particles (std hypers)"),
    # Valid runs still living in five-param-runs/; copy their folders here
    # to include them (missing runs are skipped with a notice).
    ("PSO_5D_ph2800_p30",     "Particle Swarm; 30 particles"),
    ("PSO_5D_ph2800_p50",     "Particle Swarm; 50 particles"),
]

# Iterations <= these values are excluded from the "after_iter<N>" plots.
_CUTOFFS = (50, 100, 200)

_TITLE = (
    "Optimizer Comparison - Periodic Hills Re=2800, 5D\n"
    "(geko_csep, geko_cnw, geko_cmix, geko_cturb, geko_cjet)"
)

PARAM_COLS = ["geko_csep", "geko_cnw", "geko_cmix", "geko_cturb", "geko_cjet"]
_PARAM_DISPLAY_NAMES = {
    "geko_csep": "Csep",
    "geko_cnw": "Cnw",
    "geko_cmix": "Cmix",
    "geko_cturb": "Cturb",
    "geko_cjet": "Cjet",
}

# Fixed label -> color mapping so each optimizer keeps the same colour across
# every plot. BO in dark blue, FD in black, NM in orange, the two PSO
# variants in shades of purple (orange is taken by NM here, unlike in the
# original BO-vs-PSO-only 5-D plots).
_RUN_COLORS = {
    "Bayesian Opt (GP)":                          "#08306b",
    "Nelder-Mead":                                "#e6550d",
    "Finite Differences":                         "#000000",
    "Particle Swarm; 15 particles":               "#756bb1",
    "Particle Swarm; 15 particles (std hypers)":  "#54278f",
    "Particle Swarm; 30 particles":               "#9e9ac8",
    "Particle Swarm; 50 particles":               "#3f007d",
}

# Lambda comparison: (run folder, lambda_preference). All final-run configs
# use lambda_field = 1 and lambda_integral = 0 (verified against
# configs/optimizer_comparison_configs/5D/final-runs/*.json), so only the
# preference weight differs per run.
LAMBDA_RUNS = [
    ("BO_5D_ph2800_lambda0.00", 0.00),  # pending run; skipped until results exist
    ("BO_5D_ph2800_lambda0.15", 0.15),
    ("BO_5D_ph2800_lambda0.25", 0.25),
    ("BO_5D_ph2800_lambda0.35", 0.35),
    ("BO_5D_ph2800",            0.50),
]
LAMBDA_FIELD = 1.0

COEFF_LABELS = {
    "geko_csep": r"$C_{sep}$",
    "geko_cnw": r"$C_{nw}$",
    "geko_cmix": r"$C_{mix}$",
    "geko_cturb": r"$C_{turb}$",
    "geko_cjet": r"$C_{jet}$",
}

SERIES_COLOR = "#1f77b4"
REFERENCE_COLOR = "#7f7f7f"


# ------------------------------------------------------------------ #
# Loading results                                                     #
# ------------------------------------------------------------------ #

def _load_scores(run_id: str) -> tuple[np.ndarray, list[dict[str, float]]] | None:
    """Return (scores, param_rows) from a run's metadata.csv, or None.

    Only trial_role == "optimizer" rows are returned: metadata.csv starts with
    a baseline row (GEKO-default run) that is not an optimizer iteration --
    including it would shift every curve by one.
    """
    csv_path = RUNS_DIR / run_id / "metadata.csv"
    if not csv_path.exists():
        return None
    scores: list[float] = []
    param_rows: list[dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["trial_role"] != "optimizer":
                continue
            scores.append(float(row["score"]))
            param_rows.append({c: float(row[c]) for c in PARAM_COLS})
    return (np.array(scores), param_rows) if scores else None


def _argmin_suffix(scores: np.ndarray, param_rows: list[dict[str, float]]) -> str:
    """'; Csep=0.886, ...; cost=1.234' -- parameters and cost of the best point."""
    best_idx = int(np.argmin(scores))
    best = param_rows[best_idx]
    parts = [f"{_PARAM_DISPLAY_NAMES[c]}={best[c]:.3f}" for c in PARAM_COLS]
    return "; " + ", ".join(parts) + f"; cost={scores[best_idx]:.4f}"


def _load_runs(runs: list[tuple[str, str]]) -> list[tuple[str, str, np.ndarray]]:
    """Load scores for each (run_id, label) pair, skipping missing runs."""
    loaded: list[tuple[str, str, np.ndarray]] = []
    for run_id, label in runs:
        result = _load_scores(run_id)
        if result is None:
            print(f"[plot] No results for '{label}' -- skipping.")
            continue
        scores, param_rows = result
        display_label = label + _argmin_suffix(scores, param_rows)
        loaded.append((label, display_label, scores))
    if not loaded:
        print("[plot] No results found -- nothing to plot.")
    return loaded


# ------------------------------------------------------------------ #
# Convergence comparison plots                                        #
# ------------------------------------------------------------------ #

def _set_adaptive_ylim(ax, bsf_arrays: list[np.ndarray], pad_frac: float = 0.05) -> None:
    """Zoom y-axis to the actual data range with proportional padding."""
    all_y = np.concatenate(bsf_arrays)
    y_min, y_max = all_y.min(), all_y.max()
    pad = (y_max - y_min) * pad_frac
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_comparison_full(runs: list[tuple[str, str]]) -> None:
    """Plot best-so-far cost over every iteration, uncut."""
    fig, ax = plt.subplots(figsize=(10, 5))
    loaded = _load_runs(runs)

    bsf_arrays: list[np.ndarray] = []
    for base_label, display_label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)
        ax.plot(iters, best_so_far, linewidth=2, label=display_label,
                color=_RUN_COLORS.get(base_label))
        bsf_arrays.append(best_so_far)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(_TITLE)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "optimizer_comparison_5d_full.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


def plot_comparison_cut(runs: list[tuple[str, str]], cut_iter: int) -> None:
    """Same as plot_comparison_full but omits all iterations <= cut_iter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    loaded = _load_runs(runs)

    max_iters = max(len(s) for _, _, s in loaded) if loaded else 0

    bsf_arrays: list[np.ndarray] = []
    for base_label, display_label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)

        mask = iters > cut_iter
        iters_cut = iters[mask]
        bsf_cut = best_so_far[mask]

        if len(iters_cut) == 0:
            continue

        ax.plot(iters_cut, bsf_cut, linewidth=2, label=display_label,
                color=_RUN_COLORS.get(base_label))
        bsf_arrays.append(bsf_cut)

    ax.set_xlim(cut_iter + 0.5, max_iters + 0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(_TITLE + f" (from iter {cut_iter + 1})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"optimizer_comparison_5d_after_iter{cut_iter}.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Lambda comparison                                                   #
# ------------------------------------------------------------------ #

def load_run_optimum(run_id: str, lambda_preference: float) -> dict[str, Any] | None:
    """Return the best trial of one lambda run, or None if unusable."""
    metadata_path = RUNS_DIR / run_id / "metadata.csv"
    if not metadata_path.is_file():
        print(f"[lambda-comparison] SKIP (no metadata): {metadata_path}")
        return None

    metadata = pd.read_csv(metadata_path)
    optimizer_rows = metadata[
        metadata.get("trial_role", pd.Series("optimizer", index=metadata.index))
        .fillna("optimizer")
        .eq("optimizer")
    ].copy()
    optimizer_rows["score"] = pd.to_numeric(optimizer_rows["score"], errors="coerce")
    optimizer_rows = optimizer_rows[np.isfinite(optimizer_rows["score"])]
    if optimizer_rows.empty:
        print(f"[lambda-comparison] SKIP (no scored optimizer trials): {metadata_path}")
        return None

    best = optimizer_rows.loc[optimizer_rows["score"].idxmin()]
    best_params = {name: float(best[name]) for name in PARAM_COLS}
    preference = coefficient_preference(best_params, GEKO_DEFAULTS)
    weighted_score = float(best["score"])
    unweighted_cost = weighted_score / (
        LAMBDA_FIELD * (1.0 + lambda_preference * preference)
    )

    return {
        "experiment_id": run_id,
        "lambda_preference": lambda_preference,
        "parameters": best_params,
        "weighted_score": weighted_score,
        "unweighted_cost": unweighted_cost,
        "n_trials": int(len(optimizer_rows)),
    }


def plot_coefficient_table(runs: list[dict[str, Any]]) -> None:
    row_labels = [f"{run['lambda_preference']:.2f}" for run in runs]
    col_labels = [COEFF_LABELS.get(name, name) for name in PARAM_COLS]
    cell_text = [
        [f"{run['parameters'][name]:.4f}" for name in PARAM_COLS]
        for run in runs
    ]

    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.45 * len(runs)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(
        r"Optimal GEKO coefficients per preference weight $\lambda_p$",
        pad=18,
    )
    fig.text(0.02, 0.02, r"rows: $\lambda_p$", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "table_optimal_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_unweighted_cost(runs: list[dict[str, Any]]) -> None:
    lambdas = [run["lambda_preference"] for run in runs]
    costs = [run["unweighted_cost"] for run in runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, costs, marker="o", markersize=7, linewidth=2.0, color=SERIES_COLOR)
    for x, y in zip(lambdas, costs):
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_xlabel(r"Preference weight $\lambda_p$")
    ax.set_ylabel("Unweighted cost at optimum (field error $E_F$)")
    ax.set_title(r"Unweighted field error of each $\lambda_p$ run's optimum")
    ax.set_xticks(lambdas)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unweighted_cost_vs_lambda.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients_vs_lambda(runs: list[dict[str, Any]]) -> None:
    lambdas = [run["lambda_preference"] for run in runs]
    for name in PARAM_COLS:
        values = [run["parameters"][name] for run in runs]
        default = GEKO_DEFAULTS[name]
        label = COEFF_LABELS.get(name, name)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            lambdas,
            values,
            marker="o",
            markersize=7,
            linewidth=2.0,
            color=SERIES_COLOR,
            label=f"optimized {label}",
        )
        ax.axhline(
            default,
            linestyle="--",
            linewidth=1.5,
            color=REFERENCE_COLOR,
            label=f"default {label} = {default:g}",
        )
        ax.set_xlabel(r"Preference weight $\lambda_p$")
        ax.set_ylabel(f"Optimized {label}")
        ax.set_title(f"Optimized {label} over $\\lambda_p$")
        ax.set_xticks(lambdas)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"optimal_{name}_vs_lambda.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)


def run_lambda_comparison() -> None:
    runs = []
    for run_id, lambda_preference in LAMBDA_RUNS:
        run = load_run_optimum(run_id, lambda_preference)
        if run is not None:
            runs.append(run)

    if not runs:
        print("[lambda-comparison] No usable runs found, nothing to plot.")
        return

    runs.sort(key=lambda run: run["lambda_preference"])

    summary = pd.DataFrame(
        [
            {
                "experiment_id": run["experiment_id"],
                "lambda_preference": run["lambda_preference"],
                "n_trials": run["n_trials"],
                "weighted_score": run["weighted_score"],
                "unweighted_cost": run["unweighted_cost"],
                **run["parameters"],
            }
            for run in runs
        ]
    )
    summary_path = OUTPUT_DIR / "lambda_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[lambda-comparison] Wrote {summary_path}")

    plot_coefficient_table(runs)
    plot_unweighted_cost(runs)
    plot_coefficients_vs_lambda(runs)
    print(f"[lambda-comparison] Wrote plots for {len(runs)} run(s) to {OUTPUT_DIR}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_comparison_full(RUNS_CONVERGENCE)
    for cut in _CUTOFFS:
        plot_comparison_cut(RUNS_CONVERGENCE, cut_iter=cut)

    run_lambda_comparison()
