"""Generate simple optimizer-performance and coefficient-vs-score plots from metadata.csv.

Run from the repository root, for example:

    python scripts/per_hill/plot_metadata_optimizer.py \
        scripts/per_hill/plots_metadata/periodic_hills_2800_l2_optimizer_plots.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, griddata


def _find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def _resolve_path(path: str | Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root / candidate)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_plot_inputs(plot_config_path: str | Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or _find_repo_root(Path(plot_config_path).resolve().parent)
    plot_config_path = Path(plot_config_path)
    plot_config = load_json(plot_config_path)

    experiment_config_path = _resolve_path(plot_config.get("experiment_config"), repo_root)
    if experiment_config_path is None or not experiment_config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {plot_config.get('experiment_config')}")
    experiment_config = load_json(experiment_config_path)

    metadata_path = _resolve_path(plot_config.get("metadata_path"), repo_root)
    if metadata_path is None or not metadata_path.is_file():
        candidate_results = experiment_config.get("results_dir")
        if candidate_results:
            metadata_path = _resolve_path(Path(candidate_results) / "metadata.csv", repo_root)
        if metadata_path is None or not metadata_path.is_file():
            experiment_id = experiment_config.get("experiment_id")
            if experiment_id:
                metadata_path = repo_root / "results" / "experiments" / experiment_id / "metadata.csv"
    if metadata_path is None or not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found for {experiment_config_path}")

    coefficients = plot_config.get("coefficients")
    if not coefficients:
        coefficients = [
            p["name"] for p in experiment_config.get("parameters", [])
            if isinstance(p, dict) and "name" in p
        ]

    output_dir = _resolve_path(plot_config.get("output_dir"), repo_root)
    if output_dir is None:
        output_dir = metadata_path.parent / "plots" / "metadata_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "repo_root": repo_root,
        "plot_config_path": plot_config_path,
        "experiment_config_path": experiment_config_path,
        "metadata_path": metadata_path,
        "coefficients": coefficients,
        "output_dir": output_dir,
        "output_folders": plot_config.get("output_folders", {}),
        "plots": plot_config.get("plots", {}),
    }


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "trial_role" in frame.columns:
        frame["trial_role"] = frame["trial_role"].fillna("optimizer")
    else:
        frame["trial_role"] = "optimizer"
    return frame.reset_index(drop=True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_performance(df: pd.DataFrame, output_dir: Path, metadata_name: str, folder_name: str | None = None) -> list[Path]:
    saved: list[Path] = []
    optimizer_df = df[df["trial_role"] != "baseline"].copy()
    baseline_df = df[df["trial_role"] == "baseline"].copy()

    if optimizer_df.empty:
        return saved

    optimizer_df = optimizer_df.reset_index(drop=True)
    optimizer_df["trial_index"] = np.arange(1, len(optimizer_df) + 1)
    scores = optimizer_df["score"].to_numpy(dtype=float)
    running_min = np.minimum.accumulate(scores)
    trials = optimizer_df["trial_index"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        trials,
        scores,
        s=12,
        alpha=0.22,
        color="gray",
        zorder=1,
        label="Per-trial score",
    )
    ax.plot(
        trials,
        running_min,
        color="black",
        linewidth=1.8,
        zorder=3,
        label="Best so far (running minimum)",
    )

    if not baseline_df.empty:
        baseline_score = float(baseline_df.iloc[0]["score"])
        ax.axhline(
            baseline_score,
            color="tab:orange",
            linestyle=":",
            linewidth=2,
            label=f"Default GEKO baseline: {baseline_score:.4g}",
        )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title(f"BO history: {metadata_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(trials) + 0.5)
    plt.tight_layout()
    plot_dir = output_dir / (folder_name or "Optz performance")
    saved.append(_save(fig, plot_dir / "bo_history.png"))
    return saved


def plot_1d(df: pd.DataFrame, coefficients: list[str], output_dir: Path, folder_name: str | None = None) -> list[Path]:
    saved: list[Path] = []
    optimizer_df = df[df["trial_role"] != "baseline"].copy()
    baseline_df = df[df["trial_role"] == "baseline"].copy()

    for coef in coefficients:
        if coef not in optimizer_df.columns:
            continue

        agg = (
            optimizer_df[[coef, "score"]]
            .groupby(coef, as_index=False)
            .min()
            .sort_values(coef)
        )
        x = agg[coef].to_numpy(dtype=float)
        y = agg["score"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            optimizer_df[coef],
            optimizer_df["score"],
            c=optimizer_df.index + 1,
            cmap="viridis",
            s=40,
            edgecolors="black",
            linewidths=0.4,
            label="Trials",
        )

        if len(x) >= 2:
            x_smooth = np.linspace(float(x.min()), float(x.max()), 400)
            interp = PchipInterpolator(x, y)
            ax.plot(
                x_smooth,
                interp(x_smooth),
                "k--",
                linewidth=1.4,
                alpha=0.7,
                label="PCHIP through per-x minimum",
            )

        best = optimizer_df.loc[optimizer_df["score"].idxmin()]
        ax.scatter(
            best[coef],
            best["score"],
            marker="*",
            s=300,
            color="red",
            edgecolors="black",
            zorder=10,
            label=f"Best: {coef}={best[coef]:.4f}, score={best['score']:.4g}",
        )

        if not baseline_df.empty and coef in baseline_df.columns:
            baseline_val = float(baseline_df.iloc[0][coef])
            baseline_score = float(baseline_df.iloc[0]["score"])
            ax.scatter(
                baseline_val,
                baseline_score,
                marker="D",
                s=110,
                color="tab:orange",
                edgecolors="black",
                zorder=11,
                label=(
                    f"Default GEKO: {coef}={baseline_val:.4g}, "
                    f"score={baseline_score:.4g}"
                ),
            )

        fig.colorbar(ax.collections[0], ax=ax, label="Trial index")

        ax.set_xlabel(coef)
        ax.set_ylabel("Score")
        ax.set_title(f"{coef} vs score")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_dir = output_dir / (folder_name or "1d Coefs vs Score")
        saved.append(_save(fig, plot_dir / f"score_vs_{coef}.png"))
    return saved


def plot_histogram(df: pd.DataFrame, output_dir: Path, metadata_name: str, folder_name: str | None = None) -> list[Path]:
    saved: list[Path] = []
    optimizer_df = df[df["trial_role"] != "baseline"].copy()
    baseline_df = df[df["trial_role"] == "baseline"].copy()

    if optimizer_df.empty:
        return saved

    scores = np.sort(optimizer_df["score"].to_numpy(dtype=float))
    n = len(scores)
    ecdf = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(scores, ecdf, where="post", color="steelblue", linewidth=2, label="ECDF")
    ax.scatter(scores, ecdf, color="black", s=24, alpha=0.7, zorder=4)

    if not baseline_df.empty:
        baseline_score = float(baseline_df.iloc[0]["score"])
        ax.axvline(
            baseline_score,
            color="tab:orange",
            linestyle=":",
            linewidth=2,
            label=f"Baseline: {baseline_score:.4g}",
        )

    best_score = float(optimizer_df["score"].min())
    ax.axvline(
        best_score,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Best: {best_score:.4g}",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"ECDF of optimizer scores: {metadata_name}")
    ax.set_xlim(scores.min(), scores.max())
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()

    plot_dir = output_dir / (folder_name or "Optz performance")
    saved.append(_save(fig, plot_dir / "score_ecdf.png"))
    return saved


def plot_2d(df: pd.DataFrame, coefficients: list[str], output_dir: Path, folder_name: str | None = None) -> list[Path]:
    saved: list[Path] = []
    optimizer_df = df[df["trial_role"] != "baseline"].copy()
    baseline_df = df[df["trial_role"] == "baseline"].copy()

    for x_coef, y_coef in combinations(coefficients, 2):
        if x_coef not in optimizer_df.columns or y_coef not in optimizer_df.columns:
            continue

        x = optimizer_df[x_coef].to_numpy(dtype=float)
        y = optimizer_df[y_coef].to_numpy(dtype=float)
        z = optimizer_df["score"].to_numpy(dtype=float)
        if len(optimizer_df) < 4:
            continue

        xg = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        yg = np.linspace(float(np.min(y)), float(np.max(y)), 100)
        Xg, Yg = np.meshgrid(xg, yg, indexing="xy")

        Z_lin = griddata((x, y), z, (Xg, Yg), method="linear")
        Z_near = griddata((x, y), z, (Xg, Yg), method="nearest")
        Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

        fig, ax = plt.subplots(figsize=(10, 6))
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="Score")
        ax.scatter(
            x,
            y,
            c=z,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.5,
            s=40,
            zorder=5,
            label="Trials",
        )

        best = optimizer_df.loc[optimizer_df["score"].idxmin()]
        ax.scatter(
            best[x_coef],
            best[y_coef],
            marker="*",
            s=320,
            color="red",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label=f"Best: ({best[x_coef]:.3f}, {best[y_coef]:.3f})",
        )

        if not baseline_df.empty and x_coef in baseline_df.columns and y_coef in baseline_df.columns:
            ax.scatter(
                float(baseline_df.iloc[0][x_coef]),
                float(baseline_df.iloc[0][y_coef]),
                marker="D",
                s=120,
                color="tab:orange",
                edgecolors="black",
                linewidths=0.8,
                zorder=11,
                label=f"Default GEKO (score={baseline_df.iloc[0]['score']:.4g})",
            )

        ax.set_xlabel(x_coef)
        ax.set_ylabel(y_coef)
        ax.set_title(f"Score over ({x_coef}, {y_coef})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plot_dir = output_dir / (folder_name or "2d Coefs vs Score")
        saved.append(_save(fig, plot_dir / f"score_2d_{x_coef}_vs_{y_coef}.png"))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot optimizer metadata from a periodic-hill experiment")
    parser.add_argument("config", help="Path to the plot config JSON in scripts/per_hill/plots")
    args = parser.parse_args()

    inputs = resolve_plot_inputs(args.config)
    metadata_path = inputs["metadata_path"]
    output_dir = inputs["output_dir"]
    coefficients = inputs["coefficients"]
    output_folders = inputs.get("output_folders", {})
    plots_cfg = inputs["plots"]

    df = pd.read_csv(metadata_path)
    df = _prepare_frame(df)

    if plots_cfg.get("performance", True):
        plot_performance(df, output_dir, metadata_path.parent.name, output_folders.get("performance"))
    if plots_cfg.get("one_d", True):
        plot_1d(df, coefficients, output_dir, output_folders.get("one_d"))
    if plots_cfg.get("two_d", True):
        plot_2d(df, coefficients, output_dir, output_folders.get("two_d"))
    if plots_cfg.get("histogram", False):
        plot_histogram(df, output_dir, metadata_path.parent.name, output_folders.get("histogram"))

    print(f"Plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
