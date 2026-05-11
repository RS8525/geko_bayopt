import os
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from geko_bayesopt.ansys.run import CASE, MESH, DATA_DIR
from geko_bayesopt.ansys.periodic_hill.runner import open_session, run_case
from geko_bayesopt.objective.integral_and_field_error import FieldErrorCalculator
from geko_bayesopt.utils.load_dns_periodic_hill import getData
from geko_bayesopt.objective.objective import objective_geko
from geko_bayesopt.utils.utilities import get_sobol_sampling_points
from geko_bayesopt.utils.periodic_hills_loader import getSimulationData
from geko_bayesopt.utils.utilities import plot_and_save_BayOpt

# Chose aquisition function
acq = acquisition.UpperConfidenceBound(kappa=2.5)

##################################################################################################################
######################### Construct Simulation with known optimum (CSEP=0.8870889544487) #########################
##################################################################################################################
from geko_bayesopt.ansys.periodic_hill.runner import run_case
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DNS_CSEP0887 = getSimulationData(BASE_DIR / "ansys/outputs/alpha1.0_Re5600_Csep0.8870889544487.ascii")

sim_coords,sim_fields = DNS_CSEP0887

field_calc = FieldErrorCalculator(
                         dns_coords=sim_coords,
                         dns_fields=sim_fields)

##################################################################################################################
##################################################################################################################
##################################################################################################################

# Sobol sampling bounds
pbounds = {
    "geko_csep": (0.5, 2.5),
}

number_of_sobol_sampling_points = 8
itmax = 5 * number_of_sobol_sampling_points

sobol_sampling_points = get_sobol_sampling_points(
    number_of_sobol_sampling_points,
    pbounds,
)

# Manually insert sampling point close to the optimum
sobol_sampling_points[0] = 0.87

optimizer = BayesianOptimization(
    f=None,
    acquisition_function=acq,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)


# Makes sense
field = 1.0

# Does not exist anyways i think
integral = 1.0

# Do not want to have default preference because we know the optimum is at 0.8870889544487, so we want to let the optimization explore freely
preference = 0.0

lambdas = {"field": field, "integral": integral,"preference": preference}

field_parameters = ["cp", "Ux", "Uy"]


# -------------------------------------------------------------------------
# BO loop
# -------------------------------------------------------------------------
history = []

# Very optimistic stopping criterion, just for testing purposes. In practice, it should be set to a more reasonable value or removed.
break_when =  None #1e-8

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

        # Change-in-Function-Value based stopping criterion
        if i > 0 and break_when is not None:
            if abs(target - history[i-1]["error"]) < break_when:
                print(f"Breaking optimization loop at iteration {i} due to target change below threshold.")
                break

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

# Should not produce errors, see successfull implementation in visualization_test.py
Do_you_have_balls_to_plot = True

if Do_you_have_balls_to_plot:
    plot_and_save_BayOpt(history,
                    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "experiments", "bayesian_optimization")),
                    number_of_sobol_sampling_points=number_of_sobol_sampling_points,
                    )


