import os
import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

from geko_bayesopt.utils.utilities import get_sobol_sampling_points
from geko_bayesopt.utils.utilities import quadratic_1D, quadratic_2D, plot_and_save_BayOpt

acq = acquisition.UpperConfidenceBound(kappa=2.5) # 2.576 is default

Dimension = 2

if Dimension == 1:   
    pbounds = {
                "x": (-2.5, 2.5),
                }

    number_of_sobol_sampling_points = 2
    itmax = 5 * number_of_sobol_sampling_points

if Dimension == 2:  
    pbounds = {
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                }

    number_of_sobol_sampling_points = 6
    itmax = 4 * number_of_sobol_sampling_points



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

history = []

# Very optimistic stopping criterion, just for testing purposes. In practice, it should be set to a more reasonable value or removed.
break_when =  1e-8

for i in range(itmax):

    if i < number_of_sobol_sampling_points:
        next_point_to_probe = sobol_sampling_points[i]
       
    else:
        next_point_to_probe = optimizer.suggest()

    if Dimension == 1:
        target = quadratic_1D(next_point_to_probe["x"])

    if Dimension == 2:
        target = quadratic_2D(next_point_to_probe["x"], next_point_to_probe["y"])

    optimizer.register(
        params=next_point_to_probe,
        target=target,
        )
        
    history.append({
        "iteration": i,
        "x": float(next_point_to_probe["x"]),
        "error": float(target),
        })


    print(f"\nIteration {i}")
    print(f"Params: {next_point_to_probe}")
    print(f"Target: {target}")

    # Change-in-Function-Value based stopping criterion
    if i > 0:
        if abs(target - history[i-1]["error"]) < break_when:
            print(f"Breaking optimization loop at iteration {i} due to target change below threshold.")
            break





plot_and_save_BayOpt(history,
                        output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "experiments", "bayesian_optimization")),
                        number_of_sobol_sampling_points=number_of_sobol_sampling_points,
                        analytic_maximum=2 if Dimension == 1 else 1 
                        )