from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from geko_bayesopt.ansys.run import CASE, MESH, DATA_DIR
from geko_bayesopt.ansys.periodic_hill.runner import open_session, run_case
from geko_bayesopt.objective.field_error import FieldErrorCalculator
from geko_bayesopt.utils.load_dns_periodic_hill import getData
from geko_bayesopt.utils.utilities import objective_geko
from geko_bayesopt.utils.utilities import get_sobol_sampling_points
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData

# -------------------------------------------------------------------------
# Bayesian optimization settings
# -------------------------------------------------------------------------

acq = acquisition.UpperConfidenceBound(kappa=2.5)

# Parameter bounds
pbounds = {
    "geko_csep": (0.5, 2.5),
    "geko_cnw": (-2, 0),
}

number_of_sobol_sampling_points = 8
itmax = 5 * number_of_sobol_sampling_points

sobol_sampling_points = get_sobol_sampling_points(
    number_of_sobol_sampling_points,
    pbounds,
)

optimizer = BayesianOptimization(
    f=None,
    acquisition_function=acq,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

lambdas={"field": 1.0, "integral": 1.0,"preference": 0}

residual_criteria = {
    "continuity": 1e-5,
    "x-velocity": 1e-5,
    "y-velocity": 1e-5,
    "k": 1e-5,
    "omega": 1e-5,
}

field_parameters = ["cp", "Ux", "Uy"]

# -------------------------------------------------------------------------
# Load DNS
# -------------------------------------------------------------------------
#Specify if you want ro run it with correct DNS data (problem: is not scaled) or with a "fake" DNS data with a known CSEP(0.8870889544487). Notice that not setting lambda_p=0 will make Csep pushed to the default value


from geko_bayesopt.ansys.periodic_hill.runner import run_case

from pathlib import Path

from pathlib import Path
import geko_bayesopt

PACKAGE_DIR = Path(geko_bayesopt.__file__).resolve().parent

DNS_CSEP0887 = getSimulationData(
    PACKAGE_DIR / "ansys/outputs/alpha1.0_Re5600_Csep0.8719157334417105_Cnw-1.6299342457205057.ascii"
)

sim_coords,sim_fields=DNS_CSEP0887
                                        


field_calc = FieldErrorCalculator(
                     dns_coords=sim_coords,
                     dns_fields=sim_fields)



# -------------------------------------------------------------------------
# Direct Csep test loop
# -------------------------------------------------------------------------


test_history = []
with open_session(CASE, MESH, DATA_DIR, residual_criteria=residual_criteria) as session:

        geko_params = {
            "geko_csep": 0.870889544487}

        target = objective_geko(
            geko_params=geko_params,
            session=session,
            base_case=CASE,
            field_calc=field_calc,
            field_names=field_parameters,
            lambdas=lambdas,
            return_details=True
        )

        # test_history.append({
        #     "geko_csep": geko_params["geko_csep"],
        #     "geko_cnw": geko_params["geko_cnw"],
        #     "target": float(target),
        # })
#         print(f"Csep:   {geko_params['geko_csep']:.13f}")
#         print(f"Target: {target:.10f}")


# print("\n=== Direct Csep test history ===")
# print(f"{'Iteration':>10} | {'geko_csep':>18} | {'geko_cnw':>18} | {'Target':>14}")
# print("-" * 70)

# for row in test_history:
#     print(
#         f"{row['iteration']:>10d} | "
#         f"{row['geko_csep']:>18.13f} | "
#         f"{row['geko_cnw']:>18.13f} | "
#         f"{row['target']:>14.10f}"
#     )

print(target)