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


def objective_function_plot(n_csep=50):
    # ---------------------------------------------------------------------
    # Reference Csep from fake DNS
    # ---------------------------------------------------------------------
    csep_ref = 0.8870889544486

    # GEKO default, if you want to show it
    csep_default = 1.75
    csep_min = 0.7
    csep_max = 2.5

    # ---------------------------------------------------------------------
    # IMPORTANT for fake DNS:
    # no preference term, otherwise you bias the optimum toward the default
    # ---------------------------------------------------------------------
    lambdas = {
        "field": 1.0,
        "integral": 1.0,
        "preference": 0.5,
    }

    field_parameters = ["cp", "Ux", "Uy"]

    field_weights = {
        "Ux": 1.0,
        "Uy": 1.0,
        "cp": 1.0,
    }

    # ---------------------------------------------------------------------
    # Load fake DNS/reference data
    # ---------------------------------------------------------------------
    PROJECT_DIR = Path(__file__).resolve().parent.parent

    dns_coords, dns_fields = getSimulationData(
        PROJECT_DIR / "src/geko_bayesopt/ansys/outputs/alpha1.0_Re5600_Csep0.8870889544486.ascii"
    )

    field_calc = FieldErrorCalculator(
        dns_coords=dns_coords,
        dns_fields=dns_fields,
        field_weights=field_weights)

    # ---------------------------------------------------------------------
    # Choose Csep values for line search
    # Use broad range + denser region near the known reference
    # ---------------------------------------------------------------------
    csep_values = np.linspace(csep_min, csep_max, n_csep-2)

    # Ensure reference and default values are included
    csep_values = np.concatenate([
        csep_values,
        np.array([csep_ref, csep_default])
    ])

    # Remove duplicates and sort
    csep_values = np.unique(csep_values)
    csep_values = np.sort(csep_values)

    history = []

    # ---------------------------------------------------------------------
    # Run Fluent line search
    # ---------------------------------------------------------------------
    with open_session(CASE, MESH, DATA_DIR) as session:
        for csep in csep_values:
            params = {
                "geko_csep": float(csep),
            }

            target = objective_geko(
                geko_params=params,
                session=session,
                base_case=CASE,
                field_calc=field_calc,
                field_names=field_parameters,
                lambdas=lambdas,
            )

            error = -target

            history.append({
                "geko_csep": float(csep),
                "target": float(target),
                "error": float(error),
            })

            print(
                f"Csep = {csep:.12f} | "
                f"error = {error:.8e} | "
                f"target = {target:.8e}"
            )

    df = pd.DataFrame(history)

    # ---------------------------------------------------------------------
    # Save raw data
    # ---------------------------------------------------------------------
    output_dir = Path(DATA_DIR)
    csv_path = output_dir / "objective_linesearch_csep.csv"
    df.to_csv(csv_path, index=False)

    # ---------------------------------------------------------------------
    # Interpolation for visualization only
    # ---------------------------------------------------------------------
    x = df["geko_csep"].to_numpy()
    y = df["target"].to_numpy()

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    interpolator = PchipInterpolator(x, y)

    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = interpolator(x_smooth)

    # Interpolated value at reference Csep
    ref_target_interp = interpolator(csep_ref)

    # Interpolated value at default Csep, if inside range
    default_target_interp = interpolator(csep_default)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    plt.figure(figsize=(6, 4))

    plt.plot(
        x_smooth,
        y_smooth,
        color="black",
        linewidth=1.8,
        label=r"Interpolated $f(C_{sep})$",
    )

    plt.scatter(
        x,
        y,
        color="black",
        s=25,
        label="RANS evaluations",
    )

    plt.scatter(
        csep_ref,
        ref_target_interp,
        marker="s",
        color="black",
        s=70,
        label=r"Reference $C_{sep}$",
    )


    plt.scatter(
        csep_default,
        default_target_interp,
        marker="^",
        color="black",
        s=70,
        label=r"Default $C_{sep}$",
    )

    plt.xlabel(r"$C_{sep}$")
    plt.ylabel(r"$f_{GEDCP}(C_{sep})$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = output_dir / "objective_linesearch_csep.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()



if __name__ == "__main__":
    objective_function_plot()