import numpy as np

from geko_bayesopt.ansys.periodic_hill.runner import run_geko_trial
from geko_bayesopt.objective.GEDCP import gedcp
from geko_bayesopt.objective.field_error import FieldErrorCalculator
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData

import os
import matplotlib.pyplot as plt 

GEKO_DEFAULTS = {
    "geko_csep": 1.75,
    "geko_cnw": 0.5,
    "geko_cmix": 0.0,
    "geko_cjet": 0.9,
    "geko_ccorner": 1.0,
}

FIELD_NAMES = ["Ux", "Uy", "cp"]

LAMBDAS = {
    "field": 1.0,
    "integral": 1.0,
    "preference": 0.5,
}


#from __future__ import annotations

import numpy as np


def coefficient_preference(
    coef_dict: dict[str, float],
    coef_default_dict: dict[str, float],
) -> float:
    """Default coefficient preference term.

    p = mean(|c_default - c_current| / |c_default|)
    """
    if not coef_dict:
        return 0.0

    penalties = []

    for name, value in coef_dict.items():

        default = coef_default_dict[name]

        if default == 0.0:
            penalties.append(abs(value - default))
        else:
            penalties.append(abs((default - value) / default))

    return float(np.mean(penalties))

def objective_geko(
    geko_params: dict[str, float],
    *,
    session,
    base_case,
    field_calc: FieldErrorCalculator,
    field_names: list[str] | None = None,
    lambdas: dict[str, float] | None = None,
) -> float:
    """Objective function for Bayesian optimization of GEKO coefficients."""

    if field_names is None:
        field_names = FIELD_NAMES

    if lambdas is None:
        lambdas = LAMBDAS

    outputs = run_geko_trial(
        geko_params=geko_params,
        session=session,
        base_case=base_case,
        reinitialize=False,
    )

    sim_coords, sim_fields = getSimulationData(outputs["ascii"])

    # Eq. (7): E_F = sum over fields
    field_error = 0.0

    for field_name in field_names:
        field_error += field_calc.calculate_error(
            sim_coords=sim_coords,
            sim_fields=sim_fields,
            field_name=field_name,
        )

    # No integral error yet
    integral_error = None

    # Eq. (9): default coefficient preference term
    preference = coefficient_preference(
        coef_dict=geko_params,
        coef_default_dict=GEKO_DEFAULTS,
    )

    score = gedcp(
        field_error=field_error,
        integral_error=integral_error,
        coefficient_preference=preference,
        lambda_field=lambdas["field"],
        lambda_integral=lambdas["integral"],
        lambda_preference=lambdas["preference"],
    )

    return float(score)

def get_sobol_sampling_points(sobol_sampling_points, pbounds):
    from scipy.stats import qmc

    # Create a Sobol sequence sampler
    sampler = qmc.Sobol(d=len(pbounds), scramble=True)

    # Generate Sobol sampling points
    sample = sampler.random_base2(m=int(sobol_sampling_points).bit_length())

    # Scale the sample to the bounds
    scaled_sample = qmc.scale(sample, [pbounds[key][0] for key in pbounds], [pbounds[key][1] for key in pbounds])

    # Convert to list of dictionaries
    sampling_points = []
    for point in scaled_sample:
        sampling_points.append({key: point[i] for i, key in enumerate(pbounds)})

    return sampling_points

# Analytic maximum of the function is at x = 0 with f(x) = 2
def quadratic_1D(x):
    return -x ** 2 + 2 

# Analytic maximum of the function is at (x, y) = (0, 1) with f(x, y) = 1
def quadratic_2D(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

def plot_and_save_BayOpt(history, output_dir, analytic_maximum=None, number_of_sobol_sampling_points=None):
    """
    Plots the optimization history for a 1D Bayesian Optimization case and saves the figure.

    Args:
        history (list of tuples): The optimization history, where each tuple contains (x, y) values.
        output_dir (str): The directory where the plot will be saved.
        analytic_maximum (float, optional): The known maximum value of the function, if available. Defaults to None.
        number_of_sobol_sampling_points (integer, optional): Number of initial Sobol sampling points, if used. Defaults to None.
    """
    print(history[0].keys())

    key1 = list(history[0].keys())[0]  
    key2 = list(history[0].keys())[2] 

    # Extract x and y values from history
    x_values = [point[key1] for point in history]
    y_values = [point[key2] for point in history]
    y_values_running_max = [max(y_values[:i+1]) for i in range(len(y_values))]



    # Plot the optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values_running_max, marker='s', linestyle='--', color='black', label='Bayesian Optimization (Running Maximum)')

    if analytic_maximum is not None:
        plt.axhline(y=analytic_maximum, color='blue', linestyle='-', label='Analytic Maximum')

    if number_of_sobol_sampling_points is not None:
        plt.axvline(x=number_of_sobol_sampling_points, color='red', linestyle='--', label='End of Sobol Sampling')

    plt.xlabel('Iterations')
    plt.ylabel('Cost Function Value (Running Maximum)')
    plt.title('Bayesian Optimization History')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'bayopt_history_1D.png'))
    plt.close()
