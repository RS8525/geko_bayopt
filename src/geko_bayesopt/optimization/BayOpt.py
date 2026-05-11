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

lambdas={"field": 1.0, "integral": 1.0,"preference": 0.5,}

field_parameters = ["cp", "Ux", "Uy"]

# -------------------------------------------------------------------------
# Load DNS
# -------------------------------------------------------------------------
#Specify if you want ro run it with correct DNS data (problem: is not scaled) or with a "fake" DNS data with a known CSEP(0.8870889544487). Notice that not setting lambda_p=0 will make Csep pushed to the default value
test_case = 0

if test_case == 1:
    dns_coords, dns_fields = getData(CaseName="alph10-9-3036")

    field_calc = FieldErrorCalculator(
                        dns_coords=dns_coords,
                        dns_fields=dns_fields)

if test_case == 0:
    from geko_bayesopt.ansys.periodic_hill.runner import run_case

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    DNS_CSEP0887 = getSimulationData(
        BASE_DIR / "ansys/outputs/alpha1.0_Re5600_Csep0.8870889544487.ascii"
    )
    sim_coords,sim_fields=DNS_CSEP0887
                                        


    field_calc = FieldErrorCalculator(
                         dns_coords=sim_coords,
                         dns_fields=sim_fields)



# -------------------------------------------------------------------------
# BO loop
# -------------------------------------------------------------------------
history = []

with open_session(CASE, MESH, DATA_DIR) as session:
    for i in range(itmax):

        if i < number_of_sobol_sampling_points:
            next_point_to_probe = sobol_sampling_points[i]
        else:
            next_point_to_probe = optimizer.suggest()

        target = objective_geko(
            geko_params=next_point_to_probe,
            session=session,
            base_case=CASE,
            field_calc=field_calc,
            field_names=field_parameters,
            lambdas=lambdas,
        )

        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )
        
        history.append({
            "iteration": i,
            "geko_csep": float(next_point_to_probe["geko_csep"]),
            "error": float(target),
        })


        print(f"\nIteration {i}")
        print(f"Params: {next_point_to_probe}")
        print(f"Target: {target}")


print("\n=== Optimization history ===")
print(f"{'Iteration':>10} | {'geko_csep':>12} | {'Error / target':>14}")
print("-" * 43)

for row in history:
    print(
        f"{row['iteration']:>10d} | "
        f"{row['geko_csep']:>12.6f} | "
        f"{row['error']:>14.8f}"
    )

best = optimizer.max
best_score = float(best["target"])
best_csep = float(best["params"]["geko_csep"])

print("\n=== Best result ===")
print(f"Best geko_csep: {best_csep:.6f}")
print(f"Best score:     {best_score:.8f}")











#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Imports the Bayesian Optimizer class
# from bayes_opt import BayesianOptimization

# # Imports the acquisition function, which will be specified as a parameter of the optimizer
# from bayes_opt import acquisition
# acq = acquisition.UpperConfidenceBound(kappa=2.5)
# # TODO: Check aquisition functions and their parameters


# from geko_bayesopt.optimization.utilities import objective_geko_csep, get_sobol_sampling_points


# # Parameter Space
# pbounds = {"geko_csep": (0.5, 2.5)} 

# number_of_sobol_sampling_points = 8 # Potency of 2 needed?
# itmax = 3*number_of_sobol_sampling_points
# sobol_sampling_points = get_sobol_sampling_points(number_of_sobol_sampling_points, pbounds)



# #NOTE: Initialization of optimizer with f = None is (probably) unncessaryly complicated, as the function can be specified directly in the constructor by an ANSYS call.
# #BUT: I was trying to tackle the general case.
# optimizer = BayesianOptimization(
#     f=None,
#     acquisition_function=acq,
#     pbounds=pbounds,
#     verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1, # state for reproducibility
# )

# for i in range(itmax):

#     # next_point_to_probe is delivered as a dictionary {'c_sep': 1.5}
    
#     if i < number_of_sobol_sampling_points:
#         next_point_to_probe = sobol_sampling_points[i]
#         target = objective_geko_csep(geko_params=next_point_to_probe)
#         optimizer.register(
#             params=next_point_to_probe,
#             target=target,
#         )
#     else:
#         next_point_to_probe = optimizer.suggest()
#         target = objective_geko_csep(geko_params=next_point_to_probe)
#         optimizer.register(
#             params=next_point_to_probe,
#             target=target,
#         )  


