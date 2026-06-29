"""Plot FFS optimization score evolution from metadata.csv.

Run from the repository root and pass a JSON config:

    python scripts/ffs/plot_metadata_convergence.py scripts/ffs/plots/final_metadata/<config>.json

The helper can rescore a lower-dimensional GEKO run under a higher-dimensional
GEDCP preference convention using ``score_as``. This is exact for the final FFS
configs because their objectives use ``lambda_integral = 0``:

    field_error = score / (lambda_field * (1 + lambda_p * preference_source))
    score_as    = lambda_field_target * field_error * (1 + lambda_p_target * preference_target)

Missing target GEKO parameters are filled with canonical GEKO defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geko_bayesopt.objective.GEDCP import coefficient_preference  # noqa: E402
from geko_bayesopt.geko_defaults import GEKO_DEFAULTS  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _resolve_output_dir(cfg: dict[str, Any], config_name: str) -> Path:
    output_dir = cfg.get("output_dir")
    if output_dir:
        return _resolve_path(output_dir)
    return (Path(__file__).resolve().parent / "plots" / config_name).resolve()


def _parameter_names(experiment_cfg: dict[str, Any]) -> list[str]:
    return [parameter["name"] for parameter in experiment_cfg["parameters"]]


def _objective_options(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    objective = experiment_cfg.get("objective", {})
    if objective.get("kind") != "gedcp":
        raise ValueError("Metadata convergence rescoring currently supports only gedcp objectives.")
    return objective.get("options", {})


def _check_rescorable(source_cfg: dict[str, Any], target_cfg: dict[str, Any]) -> None:
    source_options = _objective_options(source_cfg)
    target_options = _objective_options(target_cfg)

    if float(source_options.get("lambda_integral", 1.0)) != 0.0:
        raise ValueError("Cannot recover field_error from metadata when source lambda_integral != 0.")
    if float(target_options.get("lambda_integral", 1.0)) != 0.0:
        raise ValueError("Cannot score_as target objectives with lambda_integral != 0.")

    source_norm = source_options.get("field_error_norm", "l2")
    target_norm = target_options.get("field_error_norm", "l2")
    if source_norm != target_norm:
        raise ValueError(
            "score_as requires matching field_error_norm values. "
            f"Got source={source_norm!r}, target={target_norm!r}."
        )

    source_fields = source_options.get("field_names", [])
    target_fields = target_options.get("field_names", [])
    if source_fields != target_fields:
        raise ValueError(
            "score_as requires matching field_names because metadata does not "
            "store per-field contributions."
        )


def _row_parameters(
    row: pd.Series,
    parameter_names: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in parameter_names:
        if name in row and pd.notna(row[name]):
            values[name] = float(row[name])
        else:
            values[name] = GEKO_DEFAULTS[name]
    return values


def _recover_field_error(
    *,
    score: float,
    row: pd.Series,
    source_cfg: dict[str, Any],
) -> float:
    options = _objective_options(source_cfg)
    lambda_field = float(options.get("lambda_field", 1.0))
    lambda_preference = float(options.get("lambda_preference", 0.0))
    source_params = _row_parameters(row, _parameter_names(source_cfg))
    preference = coefficient_preference(source_params, GEKO_DEFAULTS)
    denominator = lambda_field * (1.0 + lambda_preference * preference)
    if denominator <= 0.0:
        raise ValueError("Invalid non-positive GEDCP metadata conversion denominator.")
    return float(score / denominator)


def _score_from_field_error(
    *,
    field_error: float,
    row: pd.Series,
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
) -> float:
    options = _objective_options(target_cfg)
    lambda_field = float(options.get("lambda_field", 1.0))
    lambda_preference = float(options.get("lambda_preference", 0.0))
    target_params = _row_parameters(row, _parameter_names(target_cfg))

    # Values optimized by the source run override defaults; missing target
    # dimensions stay at their GEKO defaults and therefore contribute zero.
    for name in _parameter_names(source_cfg):
        if name in target_params and name in row and pd.notna(row[name]):
            target_params[name] = float(row[name])

    preference = coefficient_preference(target_params, GEKO_DEFAULTS)
    return float(lambda_field * field_error * (1.0 + lambda_preference * preference))


def _transformed_scores(
    metadata: pd.DataFrame,
    *,
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any] | None,
) -> np.ndarray:
    if target_cfg is None:
        return metadata["score"].astype(float).to_numpy()

    _check_rescorable(source_cfg, target_cfg)
    values = []
    for _, row in metadata.iterrows():
        field_error = _recover_field_error(
            score=float(row["score"]),
            row=row,
            source_cfg=source_cfg,
        )
        values.append(
            _score_from_field_error(
                field_error=field_error,
                row=row,
                source_cfg=source_cfg,
                target_cfg=target_cfg,
            )
        )
    return np.asarray(values, dtype=float)


def _load_run_series(run_cfg: dict[str, Any]) -> dict[str, Any]:
    metadata_path = _resolve_path(run_cfg["metadata"])
    experiment_config_path = _resolve_path(run_cfg["config"])
    source_cfg = _load_json(experiment_config_path)
    target_cfg = _load_json(_resolve_path(run_cfg["score_as"])) if run_cfg.get("score_as") else None

    metadata = pd.read_csv(metadata_path)
    if "score" not in metadata.columns:
        raise ValueError(f"metadata.csv has no score column: {metadata_path}")
    if "trial_role" not in metadata.columns:
        metadata["trial_role"] = "optimizer"

    scores = _transformed_scores(metadata, source_cfg=source_cfg, target_cfg=target_cfg)
    working = metadata.copy()
    working["plot_score"] = scores

    optimizer_rows = working[working["trial_role"].fillna("optimizer") == "optimizer"].copy()
    optimizer_rows["evaluation"] = np.arange(1, len(optimizer_rows) + 1)
    optimizer_rows["best_so_far"] = np.minimum.accumulate(optimizer_rows["plot_score"].to_numpy())

    baseline_rows = working[working["trial_role"] == "baseline"]
    baseline_score = None
    if not baseline_rows.empty:
        baseline_score = float(baseline_rows.iloc[0]["plot_score"])

    return {
        "label": run_cfg["label"],
        "metadata_path": metadata_path,
        "optimizer": optimizer_rows,
        "baseline_score": baseline_score,
        "plot_baseline": bool(run_cfg.get("plot_baseline", True)),
        "baseline_label": run_cfg.get("baseline_label"),
        "score_as": run_cfg.get("score_as"),
    }


def _positive_values_for_log(series: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for item in series:
        optimizer = item["optimizer"]
        values.extend(optimizer["plot_score"].astype(float).to_list())
        values.extend(optimizer["best_so_far"].astype(float).to_list())
        if item["baseline_score"] is not None:
            values.append(float(item["baseline_score"]))
    return np.asarray(values, dtype=float)


def _plot(cfg: dict[str, Any], output_path: Path) -> None:
    series = [_load_run_series(run_cfg) for run_cfg in cfg["runs"]]

    y_scale = cfg.get("y_scale", "linear")
    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be either 'linear' or 'log'.")

    if y_scale == "log":
        values = _positive_values_for_log(series)
        non_positive = values[values <= 0.0]
        if non_positive.size:
            raise ValueError("Log-scale convergence plots require strictly positive scores.")

    plot_raw = bool(cfg.get("plot_raw_scores", True))
    plot_best = bool(cfg.get("plot_best_so_far", True))
    plot_baseline = bool(cfg.get("plot_baseline", True))

    fig, ax = plt.subplots(figsize=(11, 7))

    for item in series:
        optimizer = item["optimizer"]
        label = item["label"]
        if optimizer.empty:
            continue

        if plot_raw:
            ax.plot(
                optimizer["evaluation"],
                optimizer["plot_score"],
                marker="o",
                linestyle="",
                alpha=0.35,
                markersize=3,
                label=f"{label} trials",
            )

        if plot_best:
            ax.plot(
                optimizer["evaluation"],
                optimizer["best_so_far"],
                linewidth=2.0,
                label=f"{label} best-so-far",
            )

        if plot_baseline and item["plot_baseline"] and item["baseline_score"] is not None:
            ax.axhline(
                item["baseline_score"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.55,
                label=item["baseline_label"] or f"{label} baseline",
            )

    ax.set_title(cfg.get("title", cfg["name"]))
    ax.set_xlabel(cfg.get("x_label", "Optimizer evaluation"))
    ax.set_ylabel(cfg.get("y_label", "GEDCP score"))
    ax.set_yscale(y_scale)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(cfg.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the JSON plotting configuration.")
    return parser.parse_args()


def main() -> None:
    config_path = _parse_args().config
    config_path = config_path if config_path.is_absolute() else (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_json(config_path)
    output_dir = _resolve_output_dir(cfg, cfg["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = cfg.get("output_name", "metadata_convergence.png")
    output_path = output_dir / output_name

    _plot(cfg, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
