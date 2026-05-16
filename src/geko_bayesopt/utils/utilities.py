import numpy as np

from geko_bayesopt.ansys.periodic_hill.runner import run_geko_trial
from geko_bayesopt.objective.GEDCP import gedcp
from geko_bayesopt.objective.field_error import FieldErrorCalculator
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData


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
        reinitialize=True,
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