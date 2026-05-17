from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from geko_bayesopt.ansys.run import CASE, MESH, DATA_DIR
from geko_bayesopt.ansys.periodic_hill.runner import open_session
from geko_bayesopt.objective.field_error import FieldErrorCalculator
from geko_bayesopt.utils.utilities import objective_geko
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData


def objective_function_plot(n_csep=30):
    # ---------------------------------------------------------------------
    # Reference Csep from fake DNS
    # ---------------------------------------------------------------------
    csep_ref = 0.8870889544486 #avpid overwriten

    csep_min = 0.7
    csep_max = 2.5

    lambdas = {
        "field": 1.0,
        "integral": 0.0,
        "preference": 0.5,
    }

    # Only these three fields are used for the GEDCP field score
    field_parameters = ["cp", "Ux", "Uy"]

    field_weights = {
        "Ux": 1.0,
        "Uy": 1.0,
        "cp": 1.0,
    }

    residual_criteria = {
    "continuity": 1e-12,
    "x-velocity": 1e-12,
    "y-velocity": 1e-12,
    "k": 1e-12,
    "omega": 1e-12,
}

    # ---------------------------------------------------------------------
    # Load fake DNS/reference data
    # ---------------------------------------------------------------------
    PROJECT_DIR = Path(__file__).resolve().parent.parent

    reference_path = (
        PROJECT_DIR
        / "src/geko_bayesopt/ansys/outputs/alpha1.0_Re5600_Csep0.8870889544487.ascii"
    )

    dns_coords, dns_fields = getSimulationData(reference_path)

    field_calc = FieldErrorCalculator(
        dns_coords=dns_coords,
        dns_fields=dns_fields,
        field_weights=field_weights,
    )

    # ---------------------------------------------------------------------
    # Choose Csep values for line search
    # ---------------------------------------------------------------------
    csep_values = np.linspace(csep_min, csep_max, n_csep - 1)

    # Ensure reference value is included exactly
    csep_values = np.concatenate([
        csep_values,
        np.array([csep_ref]),
    ])

    csep_values = np.unique(csep_values)
    csep_values = np.sort(csep_values)

    history = []

    # ---------------------------------------------------------------------
    # Run Fluent line search
    # ---------------------------------------------------------------------
    with open_session(CASE, MESH, DATA_DIR, residual_criteria=residual_criteria) as session:
        for csep in csep_values:
            params = {
                "geko_csep": float(csep),
            }

            # Your objective_geko returns only details when return_details=True
            details = objective_geko(
                geko_params=params,
                session=session,
                base_case=CASE,
                field_calc=field_calc,
                field_names=field_parameters,
                lambdas=lambdas,
                return_details=True,
            )

            history.append({
                "geko_csep": float(csep),
                "Ux_score": details.get("Ux_score", np.nan),
                "Uy_score": details.get("Uy_score", np.nan),
                "cp_score": details.get("cp_score", np.nan),
                "field_score": details.get("field_score", np.nan),
            })

            print(
                f"Csep = {csep:.12f} | "
                f"Ux_score = {details.get('Ux_score', np.nan):.8e} | "
                f"Uy_score = {details.get('Uy_score', np.nan):.8e} | "
                f"cp_score = {details.get('cp_score', np.nan):.8e} | "
                f"field_score = {details.get('field_score', np.nan):.8e}"
            )

    # ---------------------------------------------------------------------
    # Save raw data
    # ---------------------------------------------------------------------
    df = pd.DataFrame(history)
    df = df.sort_values("geko_csep")

    output_dir = Path(DATA_DIR)

    csv_path = output_dir / "objective_scores_linesearch_csep.csv"
    df.to_csv(csv_path, index=False)

    # ---------------------------------------------------------------------
    # Interpolated plot
    # ---------------------------------------------------------------------
    x = df["geko_csep"].to_numpy()
    x_smooth = np.linspace(x.min(), x.max(), 500)

    scores_to_plot = {
        r"$-E_{Ux}$": df["Ux_score"].to_numpy(),
        r"$-E_{Uy}$": df["Uy_score"].to_numpy(),
        r"$-E_{cp}$": df["cp_score"].to_numpy(),
        r"$f_{GEDCP}$": df["field_score"].to_numpy(),
    }

    plt.figure(figsize=(7, 4.5))

    for label, y in scores_to_plot.items():
        interpolator = PchipInterpolator(x, y)
        y_smooth = interpolator(x_smooth)

        plt.plot(
            x_smooth,
            y_smooth,
            linewidth=1.8,
            label=label,
        )

        plt.scatter(
            x,
            y,
            s=18,
        )

    plt.axvline(
        csep_ref,
        linestyle="--",
        linewidth=1,
        label=r"Reference $C_{sep}$",
    )

    plt.xlabel(r"$C_{sep}$")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = output_dir / "objective_scores_linesearch_csep.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"\nCSV saved to:    {csv_path}")
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    objective_function_plot()