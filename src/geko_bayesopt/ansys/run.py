"""
Entry point: runs a single periodic-hill case using the periodic_hill package.

For a single run:        python run.py
For a BO loop, see ``BO USAGE`` at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path

from .periodic_hill import CaseConfig, MeshConfig, run_case

# ---- Paths -----------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "outputs"

# ---- Case ------------------------------------------------------------------
CASE = CaseConfig(
    alpha=1.0,
    hill_height=0.028,
    ly_over_h=3.036,
    fluid_density=1.0,
    fluid_viscosity=1.0e-5,
    re_h=5600,
    # GEKO coefficients -- None means "use Fluent default".
    # Defaults: Csep=1.75, Cnw=0.5, Cmix=0.0, Cjet=0.9, Ccorner=1.0
    geko_csep=2.0,
    geko_cnw=None,
    geko_cmix=None,
    geko_cjet=None,
    geko_ccorner=None,
    iter_count=2000,
    zone_inlet="inlet",
    zone_outlet="outlet",
    zone_top="wall",
    zone_bottom="wall_lower",
)

# ---- Mesh ------------------------------------------------------------------
MESH = MeshConfig(
    length_unit="mm",
    cad_route="DSCO",
    min_size=0.02,
    max_size=0.5,
    growth_rate=1.2,
    curvature_normal_angle=12,
    bl_first_layer_height=0.0009,
    bl_number_of_layers=22,
    bl_growth_rate=1.15,
    generate_quads=True,
)


def main() -> None:
    print(f"Running case: {CASE.case_id}")
    outputs = run_case(CASE, MESH, DATA_DIR)
    print("\nDone. Outputs:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()


# ============================================================================
# BO USAGE -- example pattern, not executed when this file runs as a script
# ============================================================================
#
# from periodic_hill import open_session, run_geko_trial
#
# def loss_against_dns(ascii_path) -> float:
#     # Load ascii, rescale by H and U_b, compare to DNS .vtr, return scalar
#     ...
#
# with open_session(CASE, MESH, DATA_DIR) as session:
#     # Fluent launches once here. Mesh load + base setup done once.
#
#     for trial_idx in range(n_trials):
#         # Whatever your BO library proposes:
#         params = bo.suggest()
#         # e.g. {"geko_csep": 1.83, "geko_cmix": 0.15, "geko_cnw": 0.6}
#
#         outputs = run_geko_trial(params, session, base_case=CASE)
#         loss = loss_against_dns(outputs["ascii"])
#         bo.observe(params, loss)
#
# # Fluent closes here automatically (context manager).
# ============================================================================
