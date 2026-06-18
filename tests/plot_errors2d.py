"""
2D parameter sweep for GEDCP objective.

Runs ~N_X × N_Y Fluent simulations over a 2D grid of two GEKO coefficients
and produces an interpolated 2D heatmap of the GEDCP objective.

Configure SWEEP_2D, FIXED_PARAMS, and LAMBDAS at the top.

Usage:
    python plot_gedcp_2d_sweep.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.objective import build_loss_fn
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh


# ========================================================================
# CONFIGURATION
# ========================================================================

CONFIG_FILE = Path(__file__).parent.parent / "configs" / "periodic_hills_2800.json"

# Two coefficients to sweep: {name: (min, max, n_points)}
# n_x * n_y ≈ 60 simulations — e.g. 8×8=64
SWEEP_2D = {
    "geko_csep": (0.7, 2.5, 8),
    "geko_cnw":  (-2.0, 2.0, 8),
}

# Fixed values for all other coefficients (None = Fluent default)
FIXED_PARAMS: dict = {}

# Default values — marked on the plot with a cross
DEFAULTS = {
    "geko_csep": 1.75,
    "geko_cnw":  0.5,
}

# GEDCP lambda weights
LAMBDAS = {
    "lambda_field": 1.0,
    "lambda_integral": 0.0,
    "lambda_preference": 0.05,
}

UI_MODE = "no_gui_or_graphics"

# ========================================================================
# END CONFIGURATION
# ========================================================================


def main() -> None:
    assert len(SWEEP_2D) == 2, "SWEEP_2D must have exactly 2 coefficients."

    config_path = CONFIG_FILE.resolve()
    cfg = ExperimentConfig.model_validate_json(config_path.read_text())
    repo_root = config_path.parent.parent

    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)

    output_dir = repo_root / "results" / "experiments" / cfg.experiment_id / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gedcp_2d_sweep.csv"

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #
    print("Loading DNS and building loss function...")
    flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_coords, dns_fields = flow_case.load_dns(dns_path)

    for key, val in LAMBDAS.items():
        if key in cfg.objective.options:
            cfg.objective.options[key] = val

    loss_fn = build_loss_fn(cfg.objective, dns_coords, dns_fields)
    print(f"DNS loaded: {dns_coords.shape[0]} points\n")

    print("Ensuring mesh...")
    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, UI_MODE)
    time.sleep(5)

    # ------------------------------------------------------------------ #
    # Build grid                                                          #
    # ------------------------------------------------------------------ #
    (coef_x, (x_min, x_max, n_x)), (coef_y, (y_min, y_max, n_y)) = SWEEP_2D.items()

    x_vals = np.linspace(x_min, x_max, n_x)
    y_vals = np.linspace(y_min, y_max, n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)              # (n_y, n_x)
    pairs   = list(zip(xx.ravel(), yy.ravel()))        # (n_x*n_y, 2)

    total = len(pairs)
    print(f"Grid: {n_x} × {n_y} = {total} simulations")
    print(f"  {coef_x}: [{x_min}, {x_max}]")
    print(f"  {coef_y}: [{y_min}, {y_max}]\n")

    # ------------------------------------------------------------------ #
    # Run simulations                                                     #
    # ------------------------------------------------------------------ #
    history: list[dict] = []

    with PeriodicHillSolver(
        flow_case.case_config,
        mesh_path,
        fluent_work_dir,
        ui_mode=UI_MODE,
        flow_case=flow_case,
        residual_criteria=cfg.residual_criteria,
    ) as solver:

        for idx, (x_val, y_val) in enumerate(pairs, 1):
            print(f"[{idx}/{total}] {coef_x}={x_val:.4f}  {coef_y}={y_val:.4f}")

            try:
                t_start = time.time()

                params = dict(FIXED_PARAMS)
                params[coef_x] = float(x_val)
                params[coef_y] = float(y_val)

                trial_case = flow_case.make_trial_case(params)
                outputs    = solver.run_trial(trial_case)

                cost = time.time() - t_start

                run_result = flow_case.build_run_result(
                    run_id=trial_case.case_id,
                    parameters=params,
                    ascii_path=outputs["ascii"],
                    cost_seconds=cost,
                )

                objective = loss_fn(run_result)

                row = {
                    coef_x:      float(x_val),
                    coef_y:      float(y_val),
                    "objective": objective,
                    "cost_s":    cost,
                }
                history.append(row)
                print(f"  GEDCP={objective:.6f}  ({cost:.1f}s)")

                # Save incrementally so a crash doesn't lose data
                pd.DataFrame(history).to_csv(csv_path, index=False)

            except Exception:
                import traceback
                traceback.print_exc()

    if not history:
        print("No successful runs. Exiting.")
        return

    # ------------------------------------------------------------------ #
    # Plot                                                                #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(history)
    _plot(df, coef_x, coef_y, x_vals, y_vals, output_dir)
    print(f"\nCSV:  {csv_path}")


def _plot(
    df: pd.DataFrame,
    coef_x: str,
    coef_y: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    output_dir: Path,
) -> None:
    """Interpolate scattered results onto a fine grid and plot a heatmap."""

    pts = df[[coef_x, coef_y]].to_numpy()
    vals = df["objective"].to_numpy()

    # Fine grid for interpolation
    xi = np.linspace(x_vals.min(), x_vals.max(), 200)
    yi = np.linspace(y_vals.min(), y_vals.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(pts, vals, (Xi, Yi), method="cubic")

    fig, ax = plt.subplots(figsize=(9, 7))

    cf = ax.contourf(Xi, Yi, Zi, levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="GEDCP Objective")

    # Overlay actual sample points
    sc = ax.scatter(
        df[coef_x], df[coef_y],
        c=vals, cmap="viridis",
        edgecolors="white", linewidths=0.5,
        s=40, zorder=5,
    )

    # Mark default values
    if coef_x in DEFAULTS and coef_y in DEFAULTS:
        ax.plot(
            DEFAULTS[coef_x], DEFAULTS[coef_y],
            marker="+", markersize=14, markeredgewidth=2.5,
            color="red", zorder=10, label="Default",
        )
        ax.legend(fontsize=11)

    ax.set_xlabel(coef_x, fontsize=12, fontweight="bold")
    ax.set_ylabel(coef_y, fontsize=12, fontweight="bold")
    ax.set_title(f"GEDCP Objective — {coef_x} vs {coef_y}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, color="white")

    fig.tight_layout()
    fig_path = output_dir / f"gedcp_2d_{coef_x}_{coef_y}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {fig_path}")


if __name__ == "__main__":
    main()