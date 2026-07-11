"""Plots for the 5D final runs living in this folder.

Combines two existing scripts, adapted to the runs stored next to this file:

1. Optimizer convergence comparisons (from
   optimizer_visualization/optimizer_comparison.py, 5-D case): best-so-far
   cost over iterations. Each entry of CONVERGENCE_COMPARISONS below defines
   one comparison -- its own set of runs, title, and after-cut iterations --
   and produces one full plot plus one "after_iter<N>" plot per cutoff.

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

Plots are written to a plots/ subfolder next to this script, organized as

    plots/convergence/<comparison name>/   one folder per convergence comparison
    plots/lambda_comparison/               lambda sweep plots + summary CSV
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
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

@dataclass(frozen=True)
class ConvergenceComparison:
    """One configurable convergence comparison.

    name:    slug for the output folder (plots/convergence/<name>/) and the
             file names inside it.
    title:   plot title; the cut plots append " (from iter <N+1>)".
    runs:    (run folder next to this script, legend label) pairs; runs
             without a metadata.csv are skipped with a notice.
    cutoffs: one extra "after_iter<N>" plot per value, omitting all
             iterations <= N.
    """
    name: str
    title: str
    runs: list[tuple[str, str]]
    cutoffs: tuple[int, ...]


# Each entry produces its own set of plots -- edit runs / titles / cutoffs
# per comparison as needed. Only lambda_preference = 0.5 runs are comparable
# with each other; the lambda sweep runs (0.15 / 0.25 / 0.35) carry a
# different preference weighting and must not be mixed in.
CONVERGENCE_COMPARISONS = [
    ConvergenceComparison(
        name="BOvsNMvsFD",
        title=(
            "BO vs NM vs FD - Periodic Hills Re=2800, 5D\n"
            "(Csep, Cnw, Cmix, Cturb, Cjet)"
        ),
        runs=[
            ("BO_5D_ph2800",          "BO"),
            ("NM_5D_ph2800",          "Nelder-Mead"),
            ("FD_5D_ph2800",          "Finite Differences"),
        ],
        cutoffs=(100, 200, 250),
    ),
    # Second comparison -- placeholder selection, adjust to taste.
    ConvergenceComparison(
        name="BOvsPSO",
        title=(
            "BO vs PSO - Periodic Hills Re=2800, 5D\n"
            "(Csep, Cnw, Cmix, Cturb, Cjet)"
        ),
        runs=[
            ("BO_5D_ph2800",          "BO"),
            ("PSO_5D_ph2800_p15",     "PSO; 15 particles"),
            ("PSO_5D_ph2800_p15_std", "PSO; 15 particles (std hypers)"),
        ],
        cutoffs=(100, 200, 250),
    ),
    # Second comparison -- placeholder selection, adjust to taste.
    ConvergenceComparison(
        name="PSOvsPSO",
        title=(
            "PSO Swarm-Size Comparison - Periodic Hills Re=2800, 5D\n"
            "(Csep, Cnw, Cmix, Cturb, Cjet)"
        ),
        runs=[
            ("PSO_5D_ph2800_p15",     "PSO; 15 particles"),
            ("PSO_5D_ph2800_p30",     "PSO; 30 particles"),
            ("PSO_5D_ph2800_p50",     "PSO; 50 particles"),
        ],
        cutoffs=(100, 200, 250),
    ),
]

PARAM_COLS = ["geko_csep", "geko_cnw", "geko_cmix", "geko_cturb", "geko_cjet"]

# Fixed label -> color mapping so each optimizer keeps the same colour across
# every plot. Palette tuned for a beamer on a white background and validated
# for colour-vision deficiency: BO / NM / FD get maximally distinct anchor
# hues (blue / vermillion / green, all >= 3:1 contrast on white), the four
# PSO variants share one purple hue as an even lightness ramp so they read
# as a family. Adjacent ramp steps sit in the CVD floor band, which is why
# the PSO variants additionally carry distinct line styles (below).
_RUN_COLORS = {
    "BO":                          "#D55E00",
    "Nelder-Mead":                                "#0072B2",
    "Finite Differences":                         "#009E73",
    "PSO; 15 particles":                          "#A87EEB",
    "PSO; 15 particles (std hypers)":             "#9065D0",
    "PSO; 30 particles":                          "#794DB6",
    "PSO; 50 particles":                          "#62359C",
}

# Secondary encoding for the PSO family: the purple lightness steps alone are
# not colourblind-safe (and wash out on projectors), so each variant gets its
# own dash pattern. Runs not listed here are drawn solid.
_RUN_STYLES = {
    "PSO; 15 particles":               "solid",
    "PSO; 15 particles (std hypers)":  (0, (5, 2)),      # dashed
    "PSO; 30 particles":               (0, (5, 2, 1, 2)),  # dash-dot
    "PSO; 50 particles":               (0, (1, 1.5)),    # dotted
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

LAMBDA_DIR = OUTPUT_DIR / "lambda_comparison"

COEFF_LABELS = {
    "geko_csep": r"$C_{sep}$",
    "geko_cnw": r"$C_{nw}$",
    "geko_cmix": r"$C_{mix}$",
    "geko_cturb": r"$C_{turb}$",
    "geko_cjet": r"$C_{jet}$",
}

SERIES_COLOR = "#1f77b4"
REFERENCE_COLOR = "#7f7f7f"

# Combined per-slide figures: the single coefficient-over-lambda plots drawn
# side by side under one common title, as (file slug, title, coefficients).
# Grouped by whether the optimizer actually moves the coefficient away from
# its GEKO default (see lambda_comparison_summary.csv).
LAMBDA_COEFF_GROUPS = [
    ("cjet_cmix_cnw",
     r"Coefficients staying close to their defaults",
     ["geko_cjet", "geko_cmix", "geko_cnw"]),
    ("cturb_csep",
     r"Coefficients staying/driven away from their defaults",
     ["geko_cturb", "geko_csep"]),
]


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
    """'\ncost 1.9651 @ (0.886, ...)' -- best point as a compact second
    legend line; the coefficient order matches the title's parameter list."""
    best_idx = int(np.argmin(scores))
    best = param_rows[best_idx]
    coeffs = ", ".join(f"{best[c]:.3f}" for c in PARAM_COLS)
    return f"\ncost {scores[best_idx]:.4f} @ ({coeffs})"


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


def plot_comparison_full(cmp: ConvergenceComparison,
                         loaded: list[tuple[str, str, np.ndarray]],
                         out_dir: Path) -> None:
    """Plot best-so-far cost over every iteration, uncut."""
    fig, ax = plt.subplots(figsize=(10, 5))

    bsf_arrays: list[np.ndarray] = []
    for base_label, display_label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)
        ax.plot(iters, best_so_far, linewidth=2, label=display_label,
                color=_RUN_COLORS.get(base_label),
                linestyle=_RUN_STYLES.get(base_label, "solid"))
        bsf_arrays.append(best_so_far)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(cmp.title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = out_dir / f"{cmp.name}_full.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


def plot_comparison_cut(cmp: ConvergenceComparison,
                        loaded: list[tuple[str, str, np.ndarray]],
                        cut_iter: int, out_dir: Path) -> None:
    """Same as plot_comparison_full but omits all iterations <= cut_iter."""
    fig, ax = plt.subplots(figsize=(10, 5))

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
                color=_RUN_COLORS.get(base_label),
                linestyle=_RUN_STYLES.get(base_label, "solid"))
        bsf_arrays.append(bsf_cut)

    ax.set_xlim(cut_iter + 0.5, max_iters + 0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(cmp.title + f" (from iter {cut_iter + 1})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = out_dir / f"{cmp.name}_after_iter{cut_iter}.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


def run_convergence_comparison(cmp: ConvergenceComparison) -> None:
    """Produce all plots (full + one per cutoff) for one comparison."""
    loaded = _load_runs(cmp.runs)
    if not loaded:
        return
    out_dir = OUTPUT_DIR / "convergence" / cmp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison_full(cmp, loaded, out_dir)
    for cut in cmp.cutoffs:
        plot_comparison_cut(cmp, loaded, cut_iter=cut, out_dir=out_dir)


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
    fig.savefig(LAMBDA_DIR / "table_optimal_coefficients.png", dpi=200, bbox_inches="tight")
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
    fig.savefig(LAMBDA_DIR / "unweighted_cost_vs_lambda.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_coefficient_axis(ax, runs: list[dict[str, Any]], name: str) -> None:
    """Draw one optimized-coefficient-over-lambda curve onto an axis."""
    lambdas = [run["lambda_preference"] for run in runs]
    values = [run["parameters"][name] for run in runs]
    default = GEKO_DEFAULTS[name]
    label = COEFF_LABELS.get(name, name)

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
    ax.set_xticks(lambdas)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_coefficients_vs_lambda(runs: list[dict[str, Any]]) -> None:
    for name in PARAM_COLS:
        label = COEFF_LABELS.get(name, name)
        fig, ax = plt.subplots(figsize=(8, 5))
        _draw_coefficient_axis(ax, runs, name)
        ax.set_title(f"Optimized {label} over $\\lambda_p$")
        fig.tight_layout()
        fig.savefig(
            LAMBDA_DIR / f"optimal_{name}_vs_lambda.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)


def plot_coefficient_groups(runs: list[dict[str, Any]]) -> None:
    """One figure per LAMBDA_COEFF_GROUPS entry: the individual coefficient
    plots side by side under a common title."""
    for slug, title, names in LAMBDA_COEFF_GROUPS:
        fig, axes = plt.subplots(
            1, len(names), figsize=(5.2 * len(names), 4.4), layout="constrained"
        )
        for ax, name in zip(np.atleast_1d(axes), names):
            _draw_coefficient_axis(ax, runs, name)
            ax.set_title(f"Optimized {COEFF_LABELS.get(name, name)}")
        fig.suptitle(title, fontsize=14)
        fig.savefig(
            LAMBDA_DIR / f"optimal_{slug}_vs_lambda.png", dpi=200, bbox_inches="tight"
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
    LAMBDA_DIR.mkdir(parents=True, exist_ok=True)

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
    summary_path = LAMBDA_DIR / "lambda_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[lambda-comparison] Wrote {summary_path}")

    plot_coefficient_table(runs)
    plot_unweighted_cost(runs)
    plot_coefficients_vs_lambda(runs)
    plot_coefficient_groups(runs)
    print(f"[lambda-comparison] Wrote plots for {len(runs)} run(s) to {LAMBDA_DIR}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    for comparison in CONVERGENCE_COMPARISONS:
        run_convergence_comparison(comparison)

    run_lambda_comparison()
