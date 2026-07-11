"""Compare the 5D BO lambda-preference sweep runs (BO_5D_ph2800*).

Reads the metadata.csv iteration histories of the six lambda-sweep runs
(lambda_preference = 0.05 / 0.15 / 0.25 / 0.35 / 0.45 / 0.5), picks each
run's best trial (minimum weighted GEDCP score over optimizer trials), and
produces:

    1. A table figure: optimal coefficient per lambda (rows = lambdas,
       columns = the five optimized GEKO coefficients).
    2. Unweighted cost (pure field error, comparable across lambdas) at
       each run's optimum, plotted over lambda.
    3. One plot per coefficient: optimized value over lambda, with the
       canonical GEKO default as a horizontal reference line.

The unweighted cost is recovered exactly from the stored weighted score.
All sweep configs use lambda_integral = 0, so the GEDCP form reduces to

    score = lambda_field * E_field * (1 + lambda_p * p(params))

and therefore

    E_field = score / (lambda_field * (1 + lambda_p * p(params)))

with p computed by the same ``coefficient_preference`` implementation the
objective itself uses. No Fluent run is needed.

Run from the repository root:

    python scripts/lambda_comparison/lambda_comparison.py

Plots are written to scripts/lambda_comparison/plots/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geko_bayesopt.geko_defaults import GEKO_DEFAULTS  # noqa: E402
from geko_bayesopt.objective.GEDCP import coefficient_preference  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "optimizer_comparison_configs" / "5D"
CONFIG_NAMES = [
    "bo_5d_ph2800_lambda0.05.json",
    "bo_5d_ph2800_lambda0.15.json",
    "bo_5d_ph2800_lambda0.25.json",
    "bo_5d_ph2800_lambda0.35.json",
    "bo_5d_ph2800_lambda0.45.json",
    "bo_5d_ph2800.json",  # lambda_preference = 0.5
]

OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

# Display labels for the optimized coefficients (metadata/config order may
# differ per file; plotting always uses the config's parameter order).
COEFF_LABELS = {
    "geko_csep": r"$C_{sep}$",
    "geko_cnw": r"$C_{nw}$",
    "geko_cmix": r"$C_{mix}$",
    "geko_cturb": r"$C_{turb}$",
    "geko_cjet": r"$C_{jet}$",
}

SERIES_COLOR = "#1f77b4"
REFERENCE_COLOR = "#7f7f7f"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_run_optimum(config_path: Path) -> dict[str, Any] | None:
    """Return the best trial of one sweep run, or None if unusable."""
    cfg = _load_json(config_path)
    options = cfg["objective"]["options"]
    lambda_preference = float(options.get("lambda_preference", 0.0))
    lambda_field = float(options.get("lambda_field", 1.0))
    if float(options.get("lambda_integral", 1.0)) != 0.0:
        raise ValueError(
            f"{config_path.name}: lambda_integral != 0, cannot recover the "
            "unweighted field error from metadata scores."
        )

    parameter_names = [parameter["name"] for parameter in cfg["parameters"]]
    metadata_path = _resolve_path(cfg["results_dir"]) / "metadata.csv"
    if not metadata_path.is_file():
        print(f"[lambda-comparison] SKIP (no metadata yet): {metadata_path}")
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
    best_params = {name: float(best[name]) for name in parameter_names}
    preference = coefficient_preference(best_params, GEKO_DEFAULTS)
    weighted_score = float(best["score"])
    unweighted_cost = weighted_score / (
        lambda_field * (1.0 + lambda_preference * preference)
    )

    return {
        "experiment_id": cfg["experiment_id"],
        "lambda_preference": lambda_preference,
        "parameters": best_params,
        "parameter_names": parameter_names,
        "weighted_score": weighted_score,
        "unweighted_cost": unweighted_cost,
        "n_trials": int(len(optimizer_rows)),
    }


def plot_coefficient_table(runs: list[dict[str, Any]], output_dir: Path) -> None:
    parameter_names = runs[0]["parameter_names"]
    row_labels = [f"{run['lambda_preference']:.2f}" for run in runs]
    col_labels = [COEFF_LABELS.get(name, name) for name in parameter_names]
    cell_text = [
        [f"{run['parameters'][name]:.4f}" for name in parameter_names]
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
    fig.savefig(output_dir / "table_optimal_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_unweighted_cost(runs: list[dict[str, Any]], output_dir: Path) -> None:
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
    fig.savefig(output_dir / "unweighted_cost_vs_lambda.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients_vs_lambda(runs: list[dict[str, Any]], output_dir: Path) -> None:
    lambdas = [run["lambda_preference"] for run in runs]
    for name in runs[0]["parameter_names"]:
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
            output_dir / f"optimal_{name}_vs_lambda.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)


def main() -> int:
    runs = []
    for name in CONFIG_NAMES:
        config_path = CONFIG_DIR / name
        if not config_path.is_file():
            print(f"[lambda-comparison] SKIP (missing config): {config_path}")
            continue
        run = load_run_optimum(config_path)
        if run is not None:
            runs.append(run)

    if not runs:
        print("[lambda-comparison] No usable runs found, nothing to plot.")
        return 1

    runs.sort(key=lambda run: run["lambda_preference"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    plot_coefficient_table(runs, OUTPUT_DIR)
    plot_unweighted_cost(runs, OUTPUT_DIR)
    plot_coefficients_vs_lambda(runs, OUTPUT_DIR)
    print(f"[lambda-comparison] Wrote plots for {len(runs)} run(s) to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
