#first try with AI

class IntegralErrorCalculator:
    """
    Computes the error between reference integral quantities and RANS
    simulation integral quantities.
    """

    def __init__(self, ref_integrals: dict[str, float]):
        self.ref_integrals = ref_integrals

    def calculate_error(
        self,
        sim_integrals: dict[str, float],
        integral_name: str,
    ) -> float:
        """
        Calculates relative squared error for one integral quantity.

        error = ((sim - ref) / ref)^2
        """
        if integral_name not in self.ref_integrals or integral_name not in sim_integrals:
            raise KeyError(
                f"Integral quantity '{integral_name}' must be present in both "
                "reference and simulation dictionaries."
            )

        ref_val = float(self.ref_integrals[integral_name])
        sim_val = float(sim_integrals[integral_name])

        if ref_val == 0.0:
            raise ValueError(
                f"Cannot compute relative error for '{integral_name}' because "
                "the reference value is zero."
            )

        error = ((sim_val - ref_val) / ref_val) ** 2

        return float(error)