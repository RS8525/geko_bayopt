import numpy as np

from geko_bayesopt.ansys.periodic_hill.runner import run_geko_trial
from geko_bayesopt.objective.integral_and_field_error import FieldErrorCalculator
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

    # Run the GEKO trial with the given parameters and retrieve simulation outputs
    outputs = run_geko_trial(
        geko_params=geko_params,
        session=session,
        base_case=base_case,
        reinitialize=True,
    )
    sim_coords, sim_fields = getSimulationData(outputs["ascii"])

    # Compute field error by comparing simulation results to DNS reference data
    # TODO: Check what is the field_names thing
    field_error = 0.0
    for field_name in field_names:
        field_error += field_calc.calculate_error(
            sim_coords=sim_coords,
            sim_fields=sim_fields,
            field_name=field_name,
        )

    # Probably not used in the current implementation
    integral_error = None

    # Compute preference error based on deviation from default coefficients
    penalties = []
    for name, value in geko_params.items():

        default = GEKO_DEFAULTS[name]

        if default == 0.0:
            penalties.append(abs(value - default))
        else:
            penalties.append(abs((default - value) / default))
    
    preference_error = float(np.mean(penalties))


    # Combine errors into a single score for optimization.
    total_error = 0.0

    if field_error is not None:
        total_error += lambdas["field"] * field_error

    if integral_error is not None:
        total_error += lambdas["integral"] * integral_error

    score = -1.0 * total_error * (1.0 + lambdas["preference"] * preference_error)

    return float(score)