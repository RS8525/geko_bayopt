import os
import numpy as np
import pytest
from geko_bayesopt.objective.integral_and_field_error import FieldErrorCalculator



def getData(FileName, **kwargs):
    """
    Reads the simulated RANS output data from the specified CSV file.
    
    Extracts x-coordinates, y-coordinates, and pressure from specific columns,
    and returns a normalized pressure coefficient.
    """
    delimiter = kwargs.get("delimiter", None)

    # Assuming CSV output from Fluent matching [node, x, y, u, v, p]
    data = np.genfromtxt(FileName, dtype=float, skip_header=kwargs.get("skip_header", 0), delimiter=delimiter)
    x_data = data[:, kwargs.get("x_col", 0)]
    y_data = data[:, kwargs.get("y_col", 1)]
    p_data = data[:, kwargs.get("p_col", 5)]

    # Translate to pressure coefficient
    cp_data = p_data - p_data[-1]
    return x_data, y_data, cp_data

def test_objective_on_real_data() -> float:
    """
    Integration test checking the error calculation using the specific real run data.
    
    This function reads concrete DNS outputs and parsed Fluent outputs from the disk,
    packages them into the data structures expected by `FieldErrorCalculator`, 
    and drives a full end-to-end evaluation of the actual project components.
    """
    # 1. Resolve relative paths to where the data should be stored
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dns_path = os.path.join(base_dir, "data", "dns", "periodic_hills", "pehill-29-cases-DNS", "alph10-9-3036", "mean_files.dat")
    sim_path = os.path.join(base_dir, "results", "experiments", "periodic_hills", "periodic_hill_2d_alpha_1.0.msh_solved_csep_1.75")
    
    # 2. Skip this test strictly if the required large datasets are not present locally
    if not os.path.exists(dns_path) or not os.path.exists(sim_path):
        pytest.skip("Test data not found.")

    # 3. Read specific domain definitions and dependent matrices using our utility loaders
    x_dns, y_dns, cp_dns = getData(dns_path, **{"p_col": 5})
    x_sim, y_sim, cp_sim = getData(sim_path, **{"x_col": 1, "y_col": 2, "p_col": 5, "skip_header": 1, "delimiter": ","})
    
    # 4. Form structure matrices to match interface of FieldErrorCalculator
    dns_coords = np.column_stack((x_dns, y_dns))
    dns_fields = {"cp": cp_dns}
    
    sim_coords = np.column_stack((x_sim, y_sim))
    sim_fields = {"cp": cp_sim}
    
    # 5. Execute computation
    calc = FieldErrorCalculator(dns_coords, dns_fields)
    mse = calc.calculate_error(sim_coords, sim_fields, field_name="cp")
    
    # 6. Basic sanity checks ensuring we return properly-calculated valid losses
    assert mse >= 0.0
    assert not np.isnan(mse)
    return mse

def test_field_error_identical_data() -> float:
    """
    Test that identical DNS and Sim fields compute an MSE of 0.0.
    
    This acts as a basic mathematical verification that there are no floating-point 
    accumulations or indexing faults when checking the exact same domain representation 
    against itself.
    """
    # Create simple 2x2 grid representation
    dns_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dns_fields = {"cp": np.array([1.0, 1.5, 2.0, 2.5])}
    
    calc = FieldErrorCalculator(dns_coords, dns_fields)
    
    # Prepare identical simulation definitions
    sim_coords = dns_coords.copy()
    sim_fields = {"cp": dns_fields["cp"].copy()}
    
    # Validation against analytic expectation (0)
    mse = calc.calculate_error(sim_coords, sim_fields, field_name="cp")
    assert np.isclose(mse, 0.0)
    return mse

def test_field_error_constant_offset() -> float:
    """
    Test that a uniform offset in values produces the mathematically expected MSE.
    
    By shifting the target cp output uniformly by an arbitrary amount, we test whether 
    MSE correctly performs (`sum((1+c) - 1)**2 / N`), which explicitly simplifies to `c**2`.
    """
    # Initialize basic domain definition
    dns_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dns_fields = {"cp": np.array([1.0, 1.5, 2.0, 2.5])}
    
    calc = FieldErrorCalculator(dns_coords, dns_fields)
    
    # Offset simulation metrics by explicit constant
    sim_coords = dns_coords.copy()
    offset = 2.0
    sim_fields = {"cp": dns_fields["cp"] + offset}
    
    # Check evaluated accuracy against arithmetic constraint
    mse = calc.calculate_error(sim_coords, sim_fields, field_name="cp")
    assert np.isclose(mse, offset**2)
    return mse

if __name__ == "__main__":
    print("Running integration test manually...")
    try:
        mse = test_objective_on_real_data()
        print(f"Success! 'test_objective_on_real_data' executed without errors. MSE: {mse}")
        mse = test_field_error_identical_data()
        print(f"Success! 'test_field_error_identical_data' passed. MSE: {mse}")
        mse = test_field_error_constant_offset()
        print(f"Success! 'test_field_error_constant_offset' passed. MSE: {mse}")
    except Exception as e:
        print(f"Test failed with error: {e}")

