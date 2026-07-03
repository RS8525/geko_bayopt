"""
Mesh sensitivity study: DNS error across mesh levels.
Warning: this test runs multiple Fluent simulations and mesh creation processes, which may crash sometimes.
Consider adjusting SLEEP_AFTER_FAILURE SLEEP_BETWEEN_MESHES SLEEP_BETWEEN_SOLVERS according to your system's performance

1. Study settings (GEKO params, mesh levels, output dir, ...) are read from a JSON
   file in scripts/mesh_validation/configs/. Pick which one with:
       python mesh_validation.py <config_name.json>
   (defaults to DEFAULT_STUDY_CONFIG below if no argument is given).
2. Phase 1: generate all meshes.
3. Phase 2: run Fluent + compute DNS error for each mesh.
4. Save CSV and plots.
"""

from pathlib import Path
import sys
import copy
import json
import time
import traceback
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh
from geko_bayesopt.objective import build_loss_fn


def _find_repo_root(start_path: Path) -> Path:
    """Locate the repository root by walking up from a path.

    Works whether `start_path` is the study config, the experiment config,
    or the script itself — it just needs to live somewhere under the repo,
    since it walks up to the first ancestor containing pyproject.toml or .git.
    """
    candidate = start_path.resolve().parent
    while candidate != candidate.parent:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    return start_path.resolve().parents[1]


# ======================================================================
# STUDY CONFIG LOADING
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"

# Used when no config name is passed on the command line.
DEFAULT_STUDY_CONFIG = "periodic_hills_2800_finner_mesh.json"


def _resolve_study_config_path(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        # Allow either just a filename ("foo.json") or a relative path.
        path = CONFIGS_DIR / path
    if not path.is_file():
        available = sorted(p.name for p in CONFIGS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"Study config not found: {path}\n"
            f"Available configs in {CONFIGS_DIR}: {available}"
        )
    return path


def load_study_config(name: str) -> dict:
    study_config_path = _resolve_study_config_path(name)
    with study_config_path.open("r", encoding="utf-8") as f:
        study_cfg = json.load(f)
    study_cfg["_study_config_path"] = study_config_path
    return study_cfg


# ======================================================================
# USER SETTINGS
# ======================================================================

UI_MODE = "no_gui_or_graphics"

SLEEP_AFTER_FAILURE   = 60
SLEEP_BETWEEN_MESHES  = 90
SLEEP_BETWEEN_SOLVERS = 10


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


def make_mesh_variant(base_mesh, level: int, length_refinement_ratio: float, scale_fields: list[str]):
    length_scale = length_refinement_ratio ** (-level)
    mesh = copy.deepcopy(base_mesh)
    for field in scale_fields:
        setattr(mesh, field, getattr(mesh, field) * length_scale)
    return mesh, length_scale


def create_mesh(rec, repo_root):
    rec["work_dir"].mkdir(parents=True, exist_ok=True)
    existing_meshes = {path.resolve() for path in rec["work_dir"].glob("*.msh.h5")}
    print(f"[{rec['label']}] Checking mesh...")
    mesh_path = Path(_ensure_mesh(rec["flow_case"], rec["work_dir"], repo_root, UI_MODE))
    created = mesh_path.resolve() not in existing_meshes
    status = "created" if created else "reused"
    print(f"[{rec['label']}] Mesh {status}: {mesh_path}")
    return mesh_path, created


def trial_case_for(rec, geko_params):
    return rec["flow_case"].make_trial_case(geko_params)


def expected_ascii_path(rec, geko_params):
    return rec["work_dir"] / f"{trial_case_for(rec, geko_params).case_id}.ascii"


def run_solver(rec, cfg, geko_params):
    existing_ascii = expected_ascii_path(rec, geko_params)
    if existing_ascii.is_file():
        print(f"[{rec['label']}] Reusing existing simulation: {existing_ascii}")
        return existing_ascii, None, True

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
        outputs = solver.run_trial(trial_case_for(rec, geko_params))
    return Path(outputs["ascii"]), time.time() - t0, False


def build_loss_fn_from_config(cfg, dns_coords, dns_fields):
    return build_loss_fn(cfg.objective, dns_coords, dns_fields)


def build_records(levels, base_mesh, output_dir, case_cfg, length_refinement_ratio, scale_fields):
    records = []
    for level in levels:
        mesh_cfg, length_scale = make_mesh_variant(base_mesh, level, length_refinement_ratio, scale_fields)
        label = f"level_{level:+d}"
        plot_label = str(level)
        work_dir = output_dir / label
        flow_case = build_flow_case(case_cfg, mesh_cfg)
        records.append(
            {
                "level": level,
                "label": label,
                "plot_label": plot_label,
                "completed": False,
                "mesh_cfg": mesh_cfg,
                "length_scale": length_scale,
                "work_dir": work_dir,
                "flow_case": flow_case,
                "mesh_path": None,
            }
        )
    return records


def build_result_row(rec, dns_errors, nodes, cost_s, reused_simulation, scale_fields):
    row = {
        "level": rec["level"],
        "label": rec["label"],
        "plot_label": rec["plot_label"],
        "nodes": nodes,
        "reused_simulation": reused_simulation,
    }
    for field in scale_fields:
        row[field] = getattr(rec["mesh_cfg"], field, None)
    row.update(dns_errors)
    return row


def plot_errors(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    write_legacy_name=False,
):
    """Percentual DNS error vs mesh level, including level 0."""
    error_col = "dns_error"
    pct_col = "percentual_error"
    df_plot = df.dropna(subset=["nodes", error_col, pct_col]).copy()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(df_plot))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df_plot)))
    bars = ax.bar(
        x_positions,
        df_plot[pct_col].astype(float),
        color=colors,
        width=0.8,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(level)) for level in df_plot["level"]], fontsize=9)
    ax.set_xlabel("Mesh level")
    ax.set_ylabel("Percentual error vs level 0")
    ax.set_title("Mesh study")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    legend_labels = [
        f"level {int(level)} ({int(nodes)} nodes)"
        for level, nodes in zip(df_plot["level"], df_plot["nodes"])
    ]
    ax.legend(
        bars,
        legend_labels,
        title="Level (nodes)",
        fontsize=7,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )

    for bar, value in zip(bars, df_plot[pct_col].astype(float)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "mesh_sensitivity_error.png", dpi=150)
    if write_legacy_name:
        fig.savefig(output_dir / "mesh_sensitivity_error_legacy.png", dpi=150)
    plt.close(fig)


def save_results(
    results,
    output_dir: Path,
    scale_fields: list[str],
    *,
    make_plots: bool = True,
):
    df = pd.DataFrame(results).sort_values("level").reset_index(drop=True)
    df = add_percent_errors_vs_level0(df)

    csv_path = output_dir / "mesh_sensitivity.csv"
    df[output_columns(df, scale_fields)].to_csv(csv_path, index=False)

    if make_plots:
        plot_errors(df, output_dir)
    return df, csv_path, df


def add_percent_errors_vs_level0(df: pd.DataFrame) -> pd.DataFrame:
    """Add percent error differences relative to level 0."""
    df = df.copy()
    reference_rows = df[df["level"] == 0]
    if reference_rows.empty:
        return df

    reference = reference_rows.iloc[0]
    error_col = "dns_error"
    pct_col = "percentual_error"
    ref_value = reference.get(error_col)
    if pd.isna(ref_value) or ref_value == 0:
        df[pct_col] = np.nan
    else:
        df[pct_col] = 100.0 * (df[error_col] - ref_value) / ref_value
    return df


def output_columns(df: pd.DataFrame, scale_fields: list[str]) -> list[str]:
    columns = ["level", "nodes", *scale_fields, "dns_error", "percentual_error"]
    return [column for column in columns if column in df.columns]


def load_existing_results(csv_path: Path, levels: list[int], scale_fields: list[str], records: list[dict] | None = None):
    """Load completed rows from a previous CSV so reruns can skip work."""
    if not csv_path.is_file():
        return []

    df = pd.read_csv(csv_path)
    if "level" not in df.columns:
        return []

    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df = df[df["level"].isin(levels)].copy()
    if df.empty:
        return []

    if "dns_error" not in df.columns and "optimized_dns_error" in df.columns:
        df["dns_error"] = df["optimized_dns_error"]

    if "reused_simulation" not in df.columns and "optimized_reused_simulation" in df.columns:
        df["reused_simulation"] = df["optimized_reused_simulation"]
    if "plot_label" not in df.columns:
        df["plot_label"] = df["level"].astype(int).astype(str)
    if "label" not in df.columns:
        df["label"] = df["level"].apply(lambda level: f"level_{int(level):+d}")

    record_map = {int(rec["level"]): rec for rec in (records or [])}

    required = ["nodes", "dns_error"]
    existing = []
    for _, row in df.iterrows():
        if any(col not in df.columns or pd.isna(row.get(col)) for col in required):
            continue
        level = int(row["level"])
        out = {
            "level": level,
            "label": row.get("label", f"level_{level:+d}"),
            "plot_label": str(level),
            "nodes": row["nodes"],
            "reused_simulation": row.get("reused_simulation", True),
        }
        if level in record_map:
            rec = record_map[level]
            for field in scale_fields:
                out[field] = getattr(rec["mesh_cfg"], field, None)
        out["dns_error"] = row["dns_error"]
        existing.append(out)
    return existing


# ======================================================================
# MAIN
# ======================================================================

def main():
    study_config_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STUDY_CONFIG
    study_cfg = load_study_config(study_config_name)
    study_config_path = study_cfg["_study_config_path"]

    # Repo root is found by walking up from the study config itself (it
    # lives under scripts/mesh_validation/configs/ inside the repo).
    repo_root = _find_repo_root(study_config_path)

    config_file = Path(study_cfg["config_file"])
    if not config_file.is_absolute():
        config_file = repo_root / config_file

    geko_params: dict = study_cfg["geko_params"]
    mesh_levels: list[int] = study_cfg["mesh_levels"]
    length_refinement_ratio = np.sqrt(study_cfg["length_refinement_ratio_base"])
    scale_fields: list[str] = study_cfg["scale_fields"]
    output_dir_setting = study_cfg.get("output_dir")

    config_path = config_file.resolve()
    cfg = ExperimentConfig.model_validate_json(config_path.read_text())

    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)
    output_dir = Path(output_dir_setting) if output_dir_setting is not None else results_dir / "mesh_sensitivity"
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_mesh = cfg.mesh
    records = build_records(mesh_levels, base_mesh, output_dir, cfg.case, length_refinement_ratio, scale_fields)

    csv_path = output_dir / "mesh_sensitivity.csv"
    results = load_existing_results(csv_path, mesh_levels, scale_fields, records)
    completed_levels = {int(row["level"]) for row in results}

    print("=" * 70)
    print("MESH SENSITIVITY STUDY")
    print("=" * 70)
    print(f"Study config: {study_config_path}")
    print(f"Config     : {config_path}")
    print(f"Output     : {output_dir}")
    print(f"GEKO params: {geko_params}")
    print("Error metric: objective value from JSON config")
    print(f"CSV cache  : {len(completed_levels)} completed level(s)")
    print(f"Ratio      : sqrt({study_cfg['length_refinement_ratio_base']}) = {length_refinement_ratio:.4f}")

    for rec in records:
        rec["completed"] = rec["level"] in completed_levels
        print(
            f"  {rec['label']:12s}  h={rec['length_scale']:.4f}  "
            f"min={rec['mesh_cfg'].min_size:.4g}  max={rec['mesh_cfg'].max_size:.4g}  "
            f"bl_h={rec['mesh_cfg'].bl_first_layer_height:.4g}"
        )

    # ------------------------------------------------------------------
    # Phase 1: mesh generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: MESH GENERATION")
    print("=" * 70)

    valid = []
    for rec in records:
        if rec["completed"]:
            print(f"[{rec['label']}] Already complete in CSV - skipping mesh and Fluent.")
            continue
        try:
            rec["mesh_path"], mesh_created = create_mesh(rec, repo_root)
            valid.append(rec)
            if mesh_created:
                kill_fluent_processes()
                time.sleep(SLEEP_BETWEEN_MESHES)
        except Exception:
            print(f"[{rec['label']}] Mesh failed — skipping.")
            traceback.print_exc()
            kill_fluent_processes()
            time.sleep(SLEEP_AFTER_FAILURE)

    if not valid and not results:
        print("No meshes generated and no completed CSV rows found.")
        return

    # ------------------------------------------------------------------
    # Phase 2: Fluent + DNS error
    # ------------------------------------------------------------------
    if valid:
        print("\n" + "=" * 70)
        print("PHASE 2: FLUENT SIMULATIONS + DNS ERROR")
        print("=" * 70)

        # Build the objective loss once from the JSON config.
        base_flow_case = build_flow_case(cfg.case, cfg.mesh)
        dns_coords, dns_fields = base_flow_case.load_dns(dns_path)
        loss_fn = build_loss_fn_from_config(cfg, dns_coords, dns_fields)

    for rec in valid:
        try:
            ascii_path, cost_s, reused_simulation = run_solver(rec, cfg, geko_params)

            run_result = rec["flow_case"].build_run_result(
                run_id=trial_case_for(rec, geko_params).case_id,
                parameters=geko_params,
                ascii_path=ascii_path,
                cost_seconds=cost_s or 0.0,
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

            cost_text = "reused" if reused_simulation else f"{cost_s:.1f}s"
            print(f"[{rec['label']}] nodes={nodes}  dns_error={dns_error:.6f}  cost={cost_text}")
            row = build_result_row(
                rec,
                {"dns_error": dns_error},
                nodes,
                cost_s,
                reused_simulation,
                scale_fields,
            )
            results.append(row)
            save_results(results, output_dir, scale_fields, make_plots=False)
            if not reused_simulation:
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
    df, csv_path, scale_df = save_results(results, output_dir, scale_fields)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    display_cols = output_columns(df, scale_fields)
    print(df[display_cols].to_string(index=False))

    print(f"\nCSV  : {csv_path}")
    print(f"Plot: {output_dir / 'mesh_sensitivity_error.png'}")

if __name__ == "__main__":
    main()
