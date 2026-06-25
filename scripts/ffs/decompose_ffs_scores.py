"""Print per-field objective contributions for FFS experiment results."""

from __future__ import annotations

import csv
from pathlib import Path

from geko_bayesopt.cases import build_flow_case
from geko_bayesopt.config import ExperimentConfig
from geko_bayesopt.fluent.extract import build_run_result
from geko_bayesopt.objective.GEDCP import GEKO_DEFAULTS, coefficient_preference, gedcp
from geko_bayesopt.objective.field_error import FieldErrorCalculator


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "ffs_retired"
    / "ffs_csep_v3_density_validation.json"
)


def main() -> None:
    cfg = ExperimentConfig.load(CONFIG_PATH)
    flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_path = ROOT / cfg.case.options["dns_path"]
    dns_coords, dns_fields = flow_case.load_dns(dns_path)
    options = cfg.objective.options
    field_names = options["field_names"]
    field_calc = FieldErrorCalculator(
        dns_coords,
        dns_fields,
        {"Ux": 1.0, "Uy": 1.0, "cp": 1.0},
        mask_hill=options.get("mask_hill", False),
        area_weight_mode=options.get("area_weight_mode", "auto"),
        evaluation_mode=options.get("evaluation_mode", "dns_points"),
        common_grid_nx=options.get("common_grid_nx", 360),
        common_grid_ny=options.get("common_grid_ny", 120),
        common_grid_floor=options.get("common_grid_floor"),
    )

    results_dir = ROOT / "results" / "experiments" / cfg.experiment_id
    fluent_dir = ROOT / "results" / "fluent" / cfg.experiment_id
    metadata_path = results_dir / "metadata.csv"

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"Config: {CONFIG_PATH}")
    print(f"Metadata: {metadata_path}")
    print("run_id, csep, metadata_score, field_sum, cp, Ux, Uy, preference, recomputed_score")
    for row in rows:
        run_id = row["run_id"]
        csep = float(row["geko_csep"])
        run = build_run_result(
            run_id=run_id,
            parameters={"geko_csep": csep},
            ascii_path=fluent_dir / f"{run_id}.ascii",
            hill_height=flow_case.case_config.hill_height,
            u_bulk=flow_case.case_config.u_bulk,
            fluid_density=flow_case.case_config.fluid_density,
            cost_seconds=float(row["cost_seconds"]),
            converged=row["converged"].lower() == "true",
        )
        contributions = {
            name: field_calc.calculate_error(run.grid_coords, run.fields, field_name=name)
            for name in field_names
        }
        field_sum = sum(contributions.values())
        pref = coefficient_preference(run.parameters, GEKO_DEFAULTS)
        score = gedcp(
            field_error=field_sum,
            integral_error=0.0,
            coefficient_preference=pref,
            lambda_field=options.get("lambda_field", 1.0),
            lambda_integral=options.get("lambda_integral", 1.0),
            lambda_preference=options.get("lambda_preference", 0.0),
        )
        print(
            f"{run_id}, {csep:.6g}, {float(row['score']):.6g}, {field_sum:.6g}, "
            f"{contributions.get('cp', 0.0):.6g}, {contributions.get('Ux', 0.0):.6g}, "
            f"{contributions.get('Uy', 0.0):.6g}, {pref:.6g}, {score:.6g}"
        )


if __name__ == "__main__":
    main()
