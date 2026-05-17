"""
General Error and Default Coefficient Preference (GEDCP) score.

This module contains pure math, no orchestration. Takes pre-computed
numerical inputs (field error, integral error, coefficient deviations)
and returns a single scalar. The orchestrator in ``objective/__init__.py``
is what wires it into the BO loop.

The GEDCP form is::

    f_GEDCP = (lambda_F * E_F + lambda_I * E_I) * (1 + lambda_p * p)

where:
    E_F = field error (>= 0; smaller is better)
    E_I = integral error (>= 0; smaller is better)
    p   = coefficient preference penalty (>= 0; smaller = closer to defaults)

The optimizer **minimizes** this score.
"""

from __future__ import annotations

import numpy as np


# Fluent's default GEKO coefficients. Used as the "preferred" anchor point
# for the coefficient-preference penalty. Pass a custom dict if your
# reference is different.
GEKO_DEFAULTS: dict[str, float] = {
    "geko_csep": 1.75,
    "geko_cnw": 0.5,
    "geko_cmix": 0.0,
    "geko_cjet": 0.9,
    "geko_ccorner": 1.0,
}


def coefficient_preference(
    coef_dict: dict[str, float],
    coef_default_dict: dict[str, float],
) -> float:
    """Default coefficient preference term.

        p = mean( |c_default - c_current| / |c_default| )

    Coefficients with a default value of zero use the absolute deviation
    instead (relative deviation is undefined when the reference is zero).
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


def gedcp(
    *,
    field_error: float = 0.0,
    integral_error: float = 0.0,
    coefficient_preference: float = 0.0,
    lambda_field: float = 1.0,
    lambda_integral: float = 1.0,
    lambda_preference: float = 0.0,
) -> float:
    """Combine field + integral errors and the preference penalty.

    Parameters
    ----------
    field_error
        Pre-computed E_F (e.g. summed MSE over selected fields).
    integral_error
        Pre-computed E_I (e.g. weighted integral-error total). Pass 0
        to skip.
    coefficient_preference
        Pre-computed p, e.g. from ``coefficient_preference(...)``.
    lambda_field, lambda_integral, lambda_preference
        Weights from GEDCP Eq. (7-9). Set lambda_preference to 0 to
        disable the preference term entirely.

    Returns
    -------
    float
        The minimization-form GEDCP score.
    """
    total_error = (
        lambda_field * field_error + lambda_integral * integral_error
    )
    return float(total_error * (1.0 + lambda_preference * coefficient_preference))
