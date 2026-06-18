"""
Mesh sensitivity study: DNS error across mesh levels.

1. Define GEKO params and reference BayOpt error at the top.
2. Phase 1: generate all meshes.
3. Phase 2: run Fluent + compute DNS error for each mesh.
4. Save CSV and plots.
"""

from pathlib import Path
import sys
import copy
import time
import traceback
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh
from geko_bayesopt.objective import build_loss_fn


# ======================================================================
# USER SETTINGS  —  edit this block only
# ======================================================================

CONFIG_FILE = Path(__file__).parent.parent / "configs" / "periodic_hills_2800.json"

# Best GEKO parameters found by BayOpt.
GEKO_PARAMS = {
    "geko_csep": 0.8675467412808036,
    "geko_cnw":  0.5036803172006827,
}

# DNS error reported by BayOpt for these parameters (insert manually).
# Set to None to skip the reference line.
BAYOPT_DNS_ERROR = 0.709356347   # <-- replace with correct value from BayOpt results

# Mesh levels to test. Level 0 = BASE_MESH_PARAMS exactly.
# Negative = coarser, positive = finer.
MESH_LEVELS = [-7,-6,-5,-4,-3,-2, -1, 0, 1, 2]

# Refinement ratio per level (sqrt(2) ≈ 1.41 is standard).
LENGTH_REFINEMENT_RATIO = np.sqrt(1.2)


# Base mesh parameters.
BASE_MESH_PARAMS = {
    "length_unit": "mm",
    "cad_route": "Workbench",
    "cad_extension": "pmdb",
    "min_size": 5,
    "max_size": 15,
    "growth_rate": 1.15,
    "curvature_normal_angle": 12,
    "bl_first_layer_height": 0.5,
    "bl_number_of_layers": 22,
    "bl_growth_rate": 1.2,
    "generate_quads": True,
}

# These three scale with the mesh level; the rest are dimensionless and stay fixed.
SCALE_FIELDS = ["min_size", "max_size", "bl_first_layer_height"]

UI_MODE = "no_gui_or_graphics"

SLEEP_AFTER_FAILURE   = 60
SLEEP_BETWEEN_MESHES  = 90
SLEEP_BETWEEN_SOLVERS = 90


# ======================================================================
# HELPERS
# ======================================================================

def kill_fluent_processes():
    for proc in ["fluent.exe", "cortex.exe", "meshing.exe",
                 "fluent_mpi.exe", "mpiexec.exe", "hydra_service.exe"]:
        subprocess.run(["taskkill", "/F", "/IM", proc],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def set_mesh_params(mesh, params):
    mesh = copy.deepcopy(mesh)
    for name, value in params.items():
        if hasattr(mesh, name):
            setattr(mesh, name, value)
    return mesh


def make_mesh_variant(base_mesh, level: int):
    length_scale = LENGTH_REFINEMENT_RATIO ** (-level)
    mesh = copy.deepcopy(base_mesh)
    for field in SCALE_FIELDS:
        setattr(mesh, field, getattr(mesh, field) * length_scale)
    return mesh, length_scale


def create_mesh(rec, repo_root):
    rec["work_dir"].mkdir(parents=True, exist_ok=True)
    print(f"[{rec['label']}] Creating mesh...")
    mesh_path = Path(_ensure_mesh(rec["flow_case"], rec["work_dir"], repo_root, UI_MODE))
    print(f"[{rec['label']}] Mesh ready: {mesh_path}")
    return mesh_path


def run_solver(rec, cfg):
    print(f"[{rec['label']}] Running Fluent...")
    t0 = time.time()
    with PeriodicHillSolver(
        rec["flow_case"].case_config,
        rec["mesh_path"],
        rec["work_dir"],
        ui_mode=UI_MODE,
        flow_case=rec["flow_case"],
        residual_criteria=cfg.residual_criteria,
    ) as solver:
        trial_case = rec["flow_case"].make_trial_case(GEKO_PARAMS)
        outputs = solver.run_trial(trial_case)
    return Path(outputs["ascii"]), time.time() - t0


def plot_errors(df: pd.DataFrame, output_dir: Path):
    """Absolute DNS error vs number of nodes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["nodes"], df["dns_error"], marker="o", linewidth=2,
            color="#378ADD", label="Mesh levels")

    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["nodes"], row["dns_error"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)

    if BAYOPT_DNS_ERROR is not None:
        ax.axhline(BAYOPT_DNS_ERROR, linestyle="--", linewidth=1.5,
                   color="#E24B4A", label=f"BayOpt reference ({BAYOPT_DNS_ERROR:.4f})")

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Error")
    ax.set_title("Mesh study")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mesh_sensitivity_error.png", dpi=150)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():
    config_path = CONFIG_FILE.resolve()
    cfg = ExperimentConfig.model_validate_json(config_path.read_text())
    repo_root = config_path.parent.parent

    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)
    output_dir = results_dir / "mesh_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MESH SENSITIVITY STUDY")
    print("=" * 70)
    print(f"Config     : {config_path}")
    print(f"Output     : {output_dir}")
    print(f"GEKO params: {GEKO_PARAMS}")
    print(f"BayOpt ref : {BAYOPT_DNS_ERROR}")
    print(f"Ratio      : sqrt(2) = {LENGTH_REFINEMENT_RATIO:.4f}")

    base_mesh = set_mesh_params(cfg.mesh, BASE_MESH_PARAMS)

    records = []
    for level in MESH_LEVELS:
        mesh_cfg, length_scale = make_mesh_variant(base_mesh, level)
        label    = f"level_{level:+d}"
        work_dir = fluent_work_dir / "mesh_sensitivity" / label
        flow_case = build_flow_case(cfg.case, mesh_cfg)
        records.append({
            "level": level, "label": label,
            "mesh_cfg": mesh_cfg, "length_scale": length_scale,
            "work_dir": work_dir, "flow_case": flow_case,
            "mesh_path": None,
        })
        print(f"  {label:12s}  h={length_scale:.4f}  "
              f"min={mesh_cfg.min_size:.4g}  max={mesh_cfg.max_size:.4g}  "
              f"bl_h={mesh_cfg.bl_first_layer_height:.4g}")

    # ------------------------------------------------------------------
    # Phase 1: mesh generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: MESH GENERATION")
    print("=" * 70)

    valid = []
    for rec in records:
        try:
            rec["mesh_path"] = create_mesh(rec, repo_root)
            valid.append(rec)
            kill_fluent_processes()
            time.sleep(SLEEP_BETWEEN_MESHES)
        except Exception:
            print(f"[{rec['label']}] Mesh failed — skipping.")
            traceback.print_exc()
            kill_fluent_processes()
            time.sleep(SLEEP_AFTER_FAILURE)

    if not valid:
        print("No meshes generated.")
        return

    # ------------------------------------------------------------------
    # Phase 2: Fluent + DNS error
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: FLUENT SIMULATIONS + DNS ERROR")
    print("=" * 70)

    # Build DNS loss once
    base_flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_coords, dns_fields = base_flow_case.load_dns(dns_path)
    loss_fn = build_loss_fn(cfg.objective, dns_coords, dns_fields)

    results = []
    for rec in valid:
        try:
            ascii_path, cost_s = run_solver(rec, cfg)

            run_result = rec["flow_case"].build_run_result(
                run_id=rec["flow_case"].make_trial_case(GEKO_PARAMS).case_id,
                parameters=GEKO_PARAMS,
                ascii_path=ascii_path,
                cost_seconds=cost_s,
            )

            dns_error = loss_fn(run_result)

            nodes = None
            try:
                cc = rec["flow_case"].case_config
                from geko_bayesopt.fluent.extract import parse_fluent_ascii
                coords, _ = parse_fluent_ascii(
                    ascii_path,
                    hill_height=cc.hill_height,
                    u_bulk=cc.u_bulk,
                    fluid_density=cc.fluid_density,
                )
                nodes = coords.shape[0]
            except Exception:
                nodes = None

            print(f"[{rec['label']}] nodes={nodes}  dns_error={dns_error:.6f}  cost={cost_s:.1f}s")
            results.append({
                "level":     rec["level"],
                "label":     rec["label"],
                "nodes":     nodes,
                "dns_error": dns_error,
                "cost_s":    cost_s,
            })
            time.sleep(SLEEP_BETWEEN_SOLVERS)

        except Exception:
            print(f"[{rec['label']}] Simulation failed — skipping.")
            traceback.print_exc()
            kill_fluent_processes()
            time.sleep(SLEEP_AFTER_FAILURE)

    if not results:
        print("No simulations completed.")
        return

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values("level").reset_index(drop=True)

    # Add BayOpt reference row for the CSV.
    if BAYOPT_DNS_ERROR is not None:
        ref_row = pd.DataFrame([{
            "level": "ref", "label": "bayopt_ref",
            "nodes": None, "dns_error": BAYOPT_DNS_ERROR, "cost_s": None,
        }])
        df_csv = pd.concat([ref_row, df], ignore_index=True)
    else:
        df_csv = df

    csv_path = output_dir / "mesh_sensitivity.csv"
    df_csv.to_csv(csv_path, index=False)

    plot_errors(df, output_dir)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df[["label", "nodes", "dns_error"]].to_string(index=False))
    if BAYOPT_DNS_ERROR is not None:
        print(f"\nBayOpt reference: {BAYOPT_DNS_ERROR:.6f}")
        df["rel_pct"] = 100.0 * df["dns_error"] / BAYOPT_DNS_ERROR
        print(df[["label", "rel_pct"]].rename(columns={"rel_pct": "% of ref"}).to_string(index=False))

    print(f"\nCSV  : {csv_path}")
    print(f"Plot1: {output_dir / 'mesh_sensitivity_error.png'}")
    print(f"Plot2: {output_dir / 'mesh_sensitivity_relative.png'}")


if __name__ == "__main__":
    main()
