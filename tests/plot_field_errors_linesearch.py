"""
Plot GEDCP objective vs parameter sweeps by running Fluent simulations.

Runs 30 simulations per coefficient, computes full GEDCP objective (field errors + preference),
and plots vs coefficient value.

Configure SWEEPS, DEFAULTS, and LAMBDAS at the top.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.objective import build_loss_fn
from geko_bayesopt.fluent.mesh_generator import MeshGenerator
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh


# ========================================================================
# CONFIGURATION
# ========================================================================

CONFIG_FILE = Path(__file__).parent.parent / "configs" / "periodic_hills_2800.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Sweeps: {coefficient_name: (min, max)} — will generate 30 points per sweep
SWEEPS = {
    # "geko_csep": (0.7, 2.5),
    "geko_cnw": (-2.0, 2.0),
}

# Default values (marked on plots with red line)
DEFAULTS = {
    # "geko_csep": 1.75,
    "geko_cnw": 0.5,
    #"geko_cmix": 0.0
    }

# GEDCP lambda weights (from objective config or override here)
LAMBDAS = {
    "lambda_field": 1.0,
    "lambda_integral": 0.0,
    "lambda_preference": 0.05,
}

# UI mode for Fluent
UI_MODE = "no_gui_or_graphics"

# ========================================================================
# END CONFIGURATION
# ========================================================================


def main():
    """Run simulations for each sweep and plot GEDCP objective."""
    config_path = CONFIG_FILE.resolve()
    cfg = ExperimentConfig.model_validate_json(config_path.read_text())
    repo_root = config_path.parent.parent

    print("Setting up...\n")

    # Resolve paths
    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)

    # Load DNS and build loss function
    print("Loading DNS and building loss function...")
    flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_coords, dns_fields = flow_case.load_dns(dns_path)
    
    # Update lambdas in config if provided
    for key, val in LAMBDAS.items():
        if key in cfg.objective.options:
            cfg.objective.options[key] = val
    
    loss_fn = build_loss_fn(cfg.objective, dns_coords, dns_fields)
    print(f"DNS loaded: {dns_coords.shape[0]} points\n")

    # Ensure mesh exists
    print("Ensuring mesh...")
    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, UI_MODE)
    time.sleep(5)

    # Output directory
    output_dir = repo_root / "results" / "experiments" / cfg.experiment_id / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulations for each sweep
    with PeriodicHillSolver(
        flow_case.case_config,
        mesh_path,
        fluent_work_dir,
        ui_mode=UI_MODE,
        flow_case=flow_case,
        residual_criteria=cfg.residual_criteria,
    ) as solver:

        for coef_name, (coef_min, coef_max) in SWEEPS.items():
            print(f"{'='*70}")
            print(f"Sweeping {coef_name} ({coef_min} to {coef_max})")
            print(f"{'='*70}\n")

            # Generate 30 points + default
            coef_values = np.linspace(coef_min, coef_max, 29)
            default_value = DEFAULTS.get(coef_name)
            if default_value is not None:
                coef_values = np.concatenate([coef_values, [default_value]])
            coef_values = np.unique(np.sort(coef_values))

            history = []

            for i, coef_val in enumerate(coef_values, 1):
                print(f"[{i}/{len(coef_values)}] {coef_name}={coef_val:.6f}")

                try:
                    t_start = time.time()

                    # Build parameters with this coefficient value
                    params = DEFAULTS.copy()
                    params[coef_name] = coef_val

                    # Run simulation
                    trial_case = flow_case.make_trial_case(params)
                    outputs = solver.run_trial(trial_case)

                    # Build run result
                    cost = time.time() - t_start
                    run_result = flow_case.build_run_result(
                        run_id=trial_case.case_id,
                        parameters=params,
                        ascii_path=outputs["ascii"],
                        cost_seconds=cost,
                    )

                    # Compute GEDCP objective
                    objective = loss_fn(run_result)

                    history.append({
                        coef_name: float(coef_val),
                        "objective": objective,
                        "cost_s": cost,
                    })

                    print(f" GEDCP={objective:.6f} ({cost:.1f}s)\n")

                except Exception as e:
                    import traceback
                    traceback.print_exc()

            if not history:
                print(f"No data for {coef_name}. Skipping.\n")
                continue

            df = pd.DataFrame(history).sort_values(coef_name)
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x = df[coef_name].to_numpy()
            y = df["objective"].to_numpy()
            
            x_smooth = np.linspace(x.min(), x.max(), 200)
            interp = PchipInterpolator(x, y)
            y_smooth = interp(x_smooth)

            ax.plot(x_smooth, y_smooth, linewidth=2.5, label="GEDCP Objective", color="steelblue")
            ax.scatter(x, y, s=50, alpha=0.7, color="steelblue")

            if default_value is not None:
                ax.axvline(default_value, linestyle="--", linewidth=2, color="red",
                          label=f"Default ({default_value:.4f})", alpha=0.8)

            ax.set_xlabel(coef_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("GEDCP Objective", fontsize=12, fontweight="bold")
            ax.set_title(f"GEDCP vs {coef_name}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            fig.tight_layout()

            fig_path = output_dir / f"gedcp_{coef_name}.png"
            fig.savefig(fig_path, dpi=150)
            print(f"Plot: {fig_path}\n")
            plt.close(fig)

    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
