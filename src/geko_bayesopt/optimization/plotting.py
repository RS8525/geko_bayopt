import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline


class Plotting:
    """
    A class for visualizing optimization results and related data.

    Attributes:
        history (list): The optimization history, typically a list of dictionaries containing iteration data.
        number_of_sobol_sampling_points (int): The number of initial Sobol sampling points used in the optimization.
        output_dir (str): Directory where all plots will be saved.
        GEKO_default_dir (str, optional): Directory containing GEKO default results for comparison.
        DNS_dir (str, optional): Directory containing DNS results for comparison.
        GEKO_optimal_dir (str, optional): Directory containing GEKO optimal results for comparison.
        figsize (tuple): Default figure size for all plots.
    """

    def __init__(self,
                 history: list,
                 input_dir: str,
                 output_dir: str,
                 parameter_space: dict = {"geko_csep": (0.7, 2.5),
                                          "geko_cnw": (-2.0, 2.0)},
                 GEKO_default_dir: str = None,
                 DNS_dir: str = None,
                 GEKO_optimal_dir: str = None,
                 number_of_sobol_sampling_points: int = None,
                 figsize: tuple = (10, 6)
                 ):
        """
        Args:
            history (list): The optimization history.
            number_of_sobol_sampling_points (int): Number of initial Sobol sampling points.
            input_dir (str): Directory containing input data.
            output_dir (str): Directory where plots will be saved.
            parameter_space (dict): The parameter space definition.
            GEKO_default_dir (str, optional): Directory with GEKO default results.
            DNS_dir (str, optional): Directory with DNS results.    
            GEKO_optimal_dir (str, optional): Directory with GEKO optimal results.
            figsize (tuple): Default figure size. Defaults to (10, 6).
        """
        self.history = history
        self.number_of_sobol_sampling_points = number_of_sobol_sampling_points
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.parameter_space = parameter_space
        self.DNS_dir = DNS_dir
        self.GEKO_default_dir = GEKO_default_dir
        self.GEKO_optimal_dir = GEKO_optimal_dir
        self.figsize = figsize
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, filename: str):
        """Save the current figure to output_dir and close it."""
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    # ------------------------------------------------------------------
    # Bayesian Optimization
    # ------------------------------------------------------------------

    def plot_RunningMaximum(
        self,
        analytic_maximum: float = None,
    ):
        """
        Plots the optimization history for a 1D Bayesian Optimization case and saves the figure.

        Args:
            analytic_maximum (float, optional): The known maximum value of the function, if available.
        """
        print(self.history[0].keys())

        key1 = list(self.history[0].keys())[0]
        key2 = list(self.history[0].keys())[2]  # NOTE: index 2 skips the second key; adjust to [1] if that was intended

        x_values = [point[key1] for point in self.history]
        y_values = [point[key2] for point in self.history]
        y_values_running_max = [max(y_values[:i + 1]) for i in range(len(y_values))]

        plt.figure(figsize=self.figsize)
        plt.plot(
            x_values,
            y_values_running_max,
            marker="s",
            linestyle="--",
            color="black",
            label="Bayesian Optimization (Running Maximum)",
        )

        if analytic_maximum is not None:
            plt.axhline(y=analytic_maximum, color="blue", linestyle="-", label="Analytic Maximum")

        if self.number_of_sobol_sampling_points is not None:
            plt.axvline(
                x=self.number_of_sobol_sampling_points,
                color="red",
                linestyle="--",
                label="End of Sobol Sampling",
            )

        plt.xlabel("Iterations")
        plt.ylabel("Cost Function Value (Running Maximum)")
        plt.title("Bayesian Optimization History")
        plt.legend()
        plt.grid(True)

        self._save("bayopt_history_1D.png")

    def Interpolate_1d(self,
                       C_ref: float = None,
                       C_ref_label: str = "geko_csep",
                       ):
        """Interpolate the 1D optimization history to create a smooth curve."""

        # Check if C_ref_label is in the history keys before proceeding
        if C_ref_label not in self.parameter_space.keys():
            raise ValueError(f"C_ref_label '{C_ref_label}' not found in parameter_space keys: {list(self.parameter_space.keys())}")
        

        df = pd.DataFrame(self.history)


        df = df.sort_values(C_ref_label)

        csv_path = os.path.join(self.input_dir, "objective_scores_linesearch_csep.csv")
        df.to_csv(csv_path, index=False)

        x = df[C_ref_label].to_numpy()
        x_smooth = np.linspace(x.min(), x.max(), 500)

        scores_to_plot = {
            r"$-E_{Ux}$": df["Ux_score"].to_numpy(),
            r"$-E_{Uy}$": df["Uy_score"].to_numpy(),
            r"$-E_{cp}$": df["cp_score"].to_numpy(),
            r"$f_{GEDCP}$": df["field_score"].to_numpy(),
        }

        plt.figure(figsize=self.figsize)

        for label, y in scores_to_plot.items():
            interpolator = PchipInterpolator(x, y)
            y_smooth = interpolator(x_smooth)

            plt.plot(x_smooth, y_smooth, linewidth=1.8, label=label)
            plt.scatter(x, y, s=18)

        if C_ref is not None:
            plt.axvline(
                C_ref,
                linestyle="--",
                linewidth=1,
                label=r"Reference $C_{sep}$",
            )

        plt.xlabel(r"$C_{sep}$")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        self._save("Objective_Interpolated.png")

    def Interpolate_2d(self,
                       C_ref: float = None,
                       creal_ref: float = None,
                       score_key: str = "field_score",
                       n_smooth: int = 100,
                       ):
        """
        Interpolate and visualise a single score over a 2-D parameter grid
        (geko_csep × geko_creal) using a bivariate spline and plot it as a
        filled-contour surface with an optional scatter overlay of the raw
        sample points.

        The history must contain the keys ``geko_csep``, ``geko_creal``, and
        whatever ``score_key`` is passed (default ``"field_score"``).

        Args:
            csep_ref (float, optional):  Reference C_sep value drawn as a
                                         vertical dashed line.
            creal_ref (float, optional): Reference C_real value drawn as a
                                         horizontal dashed line.
            score_key (str):             Column name of the score to plot.
                                         Defaults to ``"field_score"``.
            n_smooth (int):              Resolution of the interpolation grid
                                         on each axis. Defaults to 100.
        """
        df = pd.DataFrame(self.history)
        df = df.sort_values(["geko_csep", "geko_creal"])

        csv_path = os.path.join(
            self.output_dir, f"objective_scores_2d_{score_key}.csv"
        )
        df.to_csv(csv_path, index=False)

        x = df["geko_csep"].to_numpy()   # shape (N,)
        y = df["geko_creal"].to_numpy()  # shape (N,)
        z = df[score_key].to_numpy()     # shape (N,)

        # Build a regular grid from the unique parameter values so that
        # RectBivariateSpline (which requires a regular grid) can be used.
        x_unique = np.unique(x)
        y_unique = np.unique(y)

        if len(x_unique) < 3 or len(y_unique) < 3:
            raise ValueError(
                "Interpolate_2d requires at least 3 unique values along each "
                "parameter axis; consider using Interpolate_1d instead."
            )

        # Reshape z into a (len(x_unique), len(y_unique)) grid.
        # This assumes the history covers all combinations of the unique values.
        z_grid = z.reshape(len(x_unique), len(y_unique))

        spline = RectBivariateSpline(x_unique, y_unique, z_grid)

        x_smooth = np.linspace(x.min(), x.max(), n_smooth)
        y_smooth = np.linspace(y.min(), y.max(), n_smooth)
        X, Y = np.meshgrid(x_smooth, y_smooth, indexing="ij")
        Z = spline(x_smooth, y_smooth)   # shape (n_smooth, n_smooth)

        fig, ax = plt.subplots(figsize=self.figsize)

        cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label=score_key)

        # Scatter the raw sample points on top for reference
        sc = ax.scatter(
            x, y, c=z,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.4,
            s=30,
            zorder=5,
            label="Sample points",
        )

        if csep_ref is not None:
            ax.axvline(
                csep_ref,
                linestyle="--",
                linewidth=1.2,
                color="red",
                label=r"Reference $C_{sep}$",
            )
        if creal_ref is not None:
            ax.axhline(
                creal_ref,
                linestyle="--",
                linewidth=1.2,
                color="orange",
                label=r"Reference $C_{real}$",
            )

        ax.set_xlabel(r"$C_{sep}$")
        ax.set_ylabel(r"$C_{real}$")
        ax.set_title(f"2-D Interpolated Score: {score_key}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        self._save(f"Objective_Interpolated_2D_{score_key}.png")





from pathlib import Path

from geko_bayesopt.ansys.run import CASE, MESH, DATA_DIR
from geko_bayesopt.ansys.periodic_hill.runner import open_session
from geko_bayesopt.objective.field_error import FieldErrorCalculator
from geko_bayesopt.utils.utilities import objective_geko
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData


def objective_function_plot(n_csep=3):
    # ---------------------------------------------------------------------
    # Reference Csep from fake DNS
    # ---------------------------------------------------------------------
    csep_ref = 0.8870889544486 

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
    "continuity": 1e-5,
    "x-velocity": 1e-5,
    "y-velocity": 1e-5,
    "k": 1e-5,
    "omega": 1e-5,
}

    # ---------------------------------------------------------------------
    # Load fake DNS/reference data
    # ---------------------------------------------------------------------
    PROJECT_DIR = Path(__file__).resolve().parent

    reference_path = (
        PROJECT_DIR
        / "reference_data/alpha1.0_Re5600_Csep0.8870889544486.ascii"
    )

    dns_coords, dns_fields = getSimulationData(reference_path)

    field_calc = FieldErrorCalculator(
        dns_coords=dns_coords,
        dns_fields=dns_fields,
        field_weights=field_weights,
    )

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

    Plotting(
        history=history,
        number_of_sobol_sampling_points=None,
        input_dir=Path(DATA_DIR),
        output_dir=os.path.join(PROJECT_DIR, "plots"),
        parameter_space={"geko_csep": (csep_min, csep_max)},
    ).Interpolate_1d(C_ref=csep_ref)


if __name__ == "__main__":
    objective_function_plot()