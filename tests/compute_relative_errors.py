"""
Compute relative errors for given GEKO coefficients vs DNS reference.

Usage:
    python compute_relative_errors.py

Configure PARAMS list at the bottom (list of dicts with csep, cnw, cmix keys).
Can omit any key — defaults will be used for missing ones.

Example configs:
    [{"csep": 0.9759795067, "cnw": 0.3555436965, "cmix": 1.0}]  # Full spec
    [{"csep": 0.9759795067}]  # Only csep; cnw, cmix use defaults
    [{"csep": x} for x in [0.7, 0.9, 1.1]]  # Sweep csep only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path so we can import geko_bayesopt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import time

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.objective import build_loss_fn
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "periodic_hills_2800.json"

# Defaults for missing parameters (with geko_ prefix)
DEFAULTS = {
    "geko_csep": 1.0,
    "geko_cnw": 0.0,
    "geko_cmix": 0.5,
}

# UI mode for Fluent
UI_MODE = "no_gui_or_graphics"




def main():
    """Load config, DNS, run simulations, and compute errors for specified parameters."""
    print("Loading config...")
    config_path = CONFIG_PATH.resolve()
    cfg = ExperimentConfig.load(config_path)
    repo_root = config_path.parent.parent

    print(f"Config: {config_path}")
    print(f"Repo root: {repo_root}\n")

    # ---- Resolve paths ----
    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)
    print(f"Fluent work dir: {fluent_work_dir}")
    print(f"DNS data: {dns_path}\n")

    # ---- Build flow case and load DNS ----
    print("Building flow case and loading DNS...")
    flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_coords, dns_fields = flow_case.load_dns(dns_path)
    print(f"DNS loaded: {dns_coords.shape[0]} points, fields: {list(dns_fields.keys())}")

    # Build loss function
    loss_fn = build_loss_fn(cfg.objective, dns_coords, dns_fields)
    print(f"Loss function: {cfg.objective.kind}\n")

    # ---- Ensure mesh exists ----
    print("Ensuring mesh exists...")
    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, UI_MODE)
    print(f"Mesh: {mesh_path}\n")
    time.sleep(5)

    # ---- Define parameter sweep ----
    # Edit this list to change parameter combinations
    params_list = [
        # Best so far
        {"geko_csep": 0.95, "geko_cnw": 0.5424927097, "geko_cmix": 0.5641841089},
        {"geko_csep": 1.05, "geko_cnw": 0.5424927097, "geko_cmix": 0.5641841089},

        # Around current local best: vary Cnw
        {"geko_csep": 0.9967409948, "geko_cnw": 0.45, "geko_cmix": 0.5641841089},
        {"geko_csep": 0.9967409948, "geko_cnw": 0.65, "geko_cmix": 0.5641841089},

        # Around current local best: vary Cmix
        {"geko_csep": 0.9967409948, "geko_cnw": 0.5424927097, "geko_cmix": 0.45},
        {"geko_csep": 0.9967409948, "geko_cnw": 0.5424927097, "geko_cmix": 0.70},
    ]

    results = []

    # ---- Run trials ----
    print("Starting simulations...\n")
    with PeriodicHillSolver(
        flow_case.case_config,
        mesh_path,
        fluent_work_dir,
        ui_mode=UI_MODE,
        flow_case=flow_case,
        residual_criteria=cfg.residual_criteria,
    ) as solver:
        for i, partial_params in enumerate(params_list, 1):
            # Fill in defaults for missing keys
            full_params = {**DEFAULTS, **partial_params}

            print(f"[{i}/{len(params_list)}] csep={full_params['geko_csep']:.4f}, "
                  f"cnw={full_params['geko_cnw']:.4f}, cmix={full_params['geko_cmix']:.4f}")

            try:
                t_start = time.time()

                # Make trial case and run
                trial_case = flow_case.make_trial_case(full_params)
                outputs = solver.run_trial(trial_case)

                # Build run result
                cost = time.time() - t_start
                run_result = flow_case.build_run_result(
                    run_id=trial_case.case_id,
                    parameters=full_params,
                    ascii_path=outputs["ascii"],
                    cost_seconds=cost,
                )

                # Compute error
                error = loss_fn(run_result)

                results.append({
                    "csep": full_params["geko_csep"],
                    "cnw": full_params["geko_cnw"],
                    "cmix": full_params["geko_cmix"],
                    "error": error,
                    "cost_s": cost,
                })
                print(f"  ✓ MAE = {error:.6f} ({cost:.1f}s)\n")

            except Exception as e:
                print(f"  ❌ Error: {e}\n")

    # ---- Display results ----
    print()
    if results:
        df = pd.DataFrame(results)
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        best_idx = df["error"].idxmin()
        print(f"\n🏆 Best:")
        print(f"   csep={df.loc[best_idx, 'csep']:.6f}, "
              f"cnw={df.loc[best_idx, 'cnw']:.6f}, "
              f"cmix={df.loc[best_idx, 'cmix']:.6f}")
        print(f"   MAE={df.loc[best_idx, 'error']:.6f} (cost {df.loc[best_idx, 'cost_s']:.1f}s)")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
