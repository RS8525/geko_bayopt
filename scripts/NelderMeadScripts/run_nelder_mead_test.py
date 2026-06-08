"""
Run Nelder-Mead sanity checks on simple analytic polynomials and produce
plots + a table of iteration history.

Output is written to scripts/NelderMeadScripts/output/.

Usage: run from repo root with the active Python environment.

"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.geko_bayesopt.optimizer import NelderMeadOptimizer
from src.geko_bayesopt.config import ParameterSpec

OUT_DIR = Path("scripts/NelderMeadScripts/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def objective_1d(x: List[float]) -> float:
    # Simple 1D quadratic with minimum at 1.3
    return float((x[0] - 1.3) ** 2)


def objective_2d(x: List[float]) -> float:
    # Simple 2D polynomial with minimum near (1.7, 0.6)
    return float((x[0] - 1.7) ** 2 + (x[1] - 0.6) ** 2 + 0.2 * (x[0] * x[1]))


def load_parameters_from_config(config_path: str) -> List[ParameterSpec]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [ParameterSpec.model_validate(p) for p in cfg["parameters"]]


def sample_grid(params: List[ParameterSpec], obj: Callable[[List[float]], float], n_grid: int):
    if len(params) == 1:
        low, high = params[0].low, params[0].high
        xs = np.linspace(low, high, n_grid)
        zs = np.array([obj([float(x)]) for x in xs])
        return xs, zs
    elif len(params) == 2:
        x0 = np.linspace(params[0].low, params[0].high, n_grid)
        x1 = np.linspace(params[1].low, params[1].high, n_grid)
        X, Y = np.meshgrid(x0, x1)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = obj([float(X[i, j]), float(Y[i, j])])
        return (X, Y), Z
    else:
        raise ValueError("Only 1D and 2D supported for this test")


def run_nelder_mead(params: List[ParameterSpec], obj: Callable[[List[float]], float], n_iter: int = 20):
    nm = NelderMeadOptimizer(parameters=params, n_initial=8, random_state=42, options={})

    history = []
    best_cost = float("inf")
    best_x = None

    for it in range(n_iter):
        x = nm.ask()
        y = obj(x)
        nm.tell(x, y)
        if y < best_cost:
            best_cost = y
            best_x = x[:]
        history.append({
            "iteration": it,
            "point": x,
            "cost": float(y),
            "best_point": best_x,
            "best_cost": float(best_cost),
        })
    return history


def plot_1d(xs, zs, grid_points_x, nm_history, out_prefix: Path):
    # Fine interpolation curve
    x_fine = np.linspace(xs.min(), xs.max(), 400)
    z_fine = np.array([objective_1d([float(x)]) for x in x_fine])

    plt.figure(figsize=(8, 4))
    plt.plot(x_fine, z_fine, label="Objective (fine)")
    plt.scatter(xs, zs, color="C1", label="Grid samples")

    # Nelder-Mead iterates
    nm_x = [h["point"][0] for h in nm_history]
    nm_y = [h["cost"] for h in nm_history]
    plt.plot(nm_x, nm_y, marker="o", linestyle="-", color="C2", label="Nelder-Mead")
    # Draw arrows to indicate iteration order
    for i in range(len(nm_x) - 1):
        plt.annotate(
            "",
            xy=(nm_x[i + 1], nm_y[i + 1]),
            xytext=(nm_x[i], nm_y[i]),
            arrowprops=dict(arrowstyle="->", color="C2", lw=1),
        )

    plt.xlabel("x")
    plt.ylabel("cost")
    plt.legend()
    plt.title("Nelder-Mead 1D sanity check")
    plt.grid(True)
    plt.tight_layout()
    out_file = out_prefix / "nelder_mead_1d.png"
    plt.savefig(out_file)
    plt.close()

    # Save history table
    def fmt_point(p):
        return "(" + ", ".join(f"{float(v):.6f}" for v in p) + ")"

    df = pd.DataFrame([
        {
            "iteration": int(h["iteration"]),
            "point": fmt_point(h["point"]),
            "cost": f"{h['cost']:.6f}",
            "best_point": fmt_point(h["best_point"]) if h["best_point"] is not None else "",
            "best_cost": f"{h['best_cost']:.6f}",
        }
        for h in nm_history
    ])
    df.to_csv(out_prefix / "nm_history_1d.csv", index=False)

    # Also render a simple table image
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.25 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    tbl.scale(0.9, 1.0)
    plt.tight_layout()
    plt.savefig(out_prefix / "nm_history_1d_table.png")
    plt.close()


def plot_2d(grid, Z, nm_history, out_prefix: Path):
    X, Y = grid
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, Z, levels=40, cmap="viridis")
    plt.colorbar(cs, label="cost")

    nm_x = [h["point"][0] for h in nm_history]
    nm_y = [h["point"][1] for h in nm_history]
    plt.plot(nm_x, nm_y, marker="o", color="red", label="Nelder-Mead path")
    plt.scatter(nm_x[0], nm_y[0], color="white", edgecolor="black", label="start")
    # Draw arrows along the path to indicate order (annotate per segment for visibility)
    for i in range(len(nm_x) - 1):
        plt.annotate(
            "",
            xy=(nm_x[i + 1], nm_y[i + 1]),
            xytext=(nm_x[i], nm_y[i]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.title("Nelder-Mead 2D sanity check")
    plt.tight_layout()
    plt.savefig(out_prefix / "nelder_mead_2d.png")
    plt.close()

    def fmt_point(p):
        return "(" + ", ".join(f"{float(v):.6f}" for v in p) + ")"

    df = pd.DataFrame([
        {
            "iteration": int(h["iteration"]),
            "point": fmt_point(h["point"]),
            "cost": f"{h['cost']:.6f}",
            "best_point": fmt_point(h["best_point"]) if h["best_point"] is not None else "",
            "best_cost": f"{h['best_cost']:.6f}",
        }
        for h in nm_history
    ])
    df.to_csv(out_prefix / "nm_history_2d.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.25 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    tbl.scale(0.9, 1.0)
    plt.tight_layout()
    plt.savefig(out_prefix / "nm_history_2d_table.png")
    plt.close()


def main():
    # 1D test
    cfg1 = "configs/nelder_mead_test_1D.json"
    params1 = load_parameters_from_config(cfg1)
    xs, zs = sample_grid(params1, objective_1d, 20)
    history1 = run_nelder_mead(params1, objective_1d, n_iter=20)
    plot_1d(xs, zs, xs, history1, OUT_DIR)

    # 2D test
    cfg2 = "configs/nelder_mead_test_2D.json"
    params2 = load_parameters_from_config(cfg2)
    grid, Z = sample_grid(params2, objective_2d, 50)
    history2 = run_nelder_mead(params2, objective_2d, n_iter=20)
    plot_2d(grid, Z, history2, OUT_DIR)

    print(f"Outputs written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
