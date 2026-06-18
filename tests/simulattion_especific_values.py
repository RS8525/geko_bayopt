"""
Run a single Fluent simulation with the given coefficients and save the ASCII output.

Usage:
    python run_single.py

Configure COEFS below.
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.fluent.solver import PeriodicHillSolver
from geko_bayesopt.experiment import _resolve_paths, _ensure_mesh


# ========================================================================
# CONFIGURATION
# ========================================================================

CONFIG_FILE = Path(__file__).parent.parent / "configs" / "periodic_hills_2800.json"

COEFS = {
    "geko_csep": 1.75,
}

UI_MODE = "no_gui_or_graphics"

# ========================================================================


if __name__ == "__main__":
    config_path = CONFIG_FILE.resolve()
    cfg = ExperimentConfig.model_validate_json(config_path.read_text())
    repo_root = config_path.parent.parent

    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)
    flow_case = build_flow_case(cfg.case, cfg.mesh)

    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, UI_MODE)
    time.sleep(5)

    with PeriodicHillSolver(
        flow_case.case_config,
        mesh_path,
        fluent_work_dir,
        ui_mode=UI_MODE,
        flow_case=flow_case,
        residual_criteria=cfg.residual_criteria,
    ) as solver:
        trial_case = flow_case.make_trial_case(COEFS)
        outputs = solver.run_trial(trial_case)
