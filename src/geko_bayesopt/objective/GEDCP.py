from __future__ import annotations


def gedcp(
    *,
    field_error: float | None = None,
    integral_error: float | None = None,
    coefficient_preference: float = 0.0,
    lambda_field: float = 1.0,
    lambda_integral: float = 1.0,
    lambda_preference: float = 0.5,
) -> float:
    """General Error and Default Coefficient Preference objective.

    Implements:

        f_GEDCP = -(lambda_field * E_F + lambda_integral * E_I)
                  * (1 + lambda_preference * p)

    where:
        E_F = field error
        E_I = integral parameter error
        p   = default coefficient preference term
    """

    total_error = 0.0

    if field_error is not None:
        total_error += lambda_field * field_error

    if integral_error is not None:
        total_error += lambda_integral * integral_error

    score = -total_error * (1.0 + lambda_preference * coefficient_preference)

    return float(score)