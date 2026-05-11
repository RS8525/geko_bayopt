# Test script to visualize the DNS data and simulation data for the periodic hills case. 
# This script will generate scatter plots of the specified columns from the DNS and simulation datasets, allowing for a visual comparison of the fields.
# Expected directories for data:
# - DNS data: data/dns/periodic_hills/pehill-29-cases-DNS/{case_name}/mean_files.dat
# - Simulation data: results/experiments/periodic_hills/periodic_hill_2d_alpha_1.0.msh_solved_csep_1.75
# Change in main function if thats vary

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dns_data(case_name, output_dir):
    """
    Plots data from the specified DNS case and saves the figures.

    Args:
        case_name (str): The name of the DNS case directory.
        output_dir (str): The directory where the plots will be saved.
    """
    # Define the path to the data file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dns", "periodic_hills", "pehill-29-cases-DNS"))
    data_file = os.path.join(base_dir, case_name, "mean_files.dat")

    # Load the data
    data = np.genfromtxt(data_file, dtype=float, skip_header=0, delimiter=None)

    # Extract columns
    coords = data[:, :2]  # First two columns for coordinates
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    col_2 = data[:, 2]           # Column with index 2
    col_5 = data[:, 5]           # Column with index 5
    col_5 = col_5 - col_5[-1]    # Normalize pressure

    # Plot column 2 (Ux)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c=col_2, cmap='viridis', label='Column 2', alpha=0.7)
    plt.colorbar(label='Column 2 Values')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'DNS Data Plot (Column 2) for {case_name}')
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'dns_{case_name}_column_2.png'))
    plt.show(block=False)  # Don't block execution

    # Plot column 5 (pressure)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c=col_5, cmap='plasma', label='Column 5', alpha=0.7)
    plt.colorbar(label='Column 5 Values')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'DNS Data Plot (Column 5) for {case_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'dns_{case_name}_column_5.png'))
    plt.show(block=False)  # Don't block execution

def plot_and_save_simulation_data(sim_file_path, output_dir):
    """
    Plots simulation data and saves each column plot as an individual figure.

    Args:
        sim_file_path (str): The path to the simulation data file.
        output_dir (str): The directory where the plots will be saved.
    """
    # Load simulation data
    sim_data = np.genfromtxt(sim_file_path, dtype=float, skip_header=1, delimiter=None)
    sim_coords = sim_data[:, 1:3]           # Columns 1 and 2 for coordinates
    sim_col_2 = sim_data[:, 3]              # This might need to be 2 or 3 depending on CSV
    sim_col_5 = sim_data[:, 5]              # Column with index 5
    sim_col_5 = sim_col_5 - sim_col_5[-1]   # Normalize pressure 

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save column 2 (Ux)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sim_coords[:, 0], sim_coords[:, 1], c=sim_col_2, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Simulation Column 2 Values')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulation Data Plot (Column 2)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'simulation_column_2.png'))
    plt.close()

    # Plot and save column 5 (pressure)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sim_coords[:, 0], sim_coords[:, 1], c=sim_col_5, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter, label='Simulation Column 5 Values')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulation Data Plot (Column 5)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'simulation_column_5.png'))
    plt.close()

if __name__ == "__main__":
    # Example usage
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "experiments", "periodic_hills"))
    plot_dns_data("alph10-9-3036", output_directory)
    plt.close('all')  # Close DNS plots before plotting simulation data
    
    sim_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "geko_bayesopt", "ansys", "outputs", "alpha1.0_Re5600_Csep1.678009572655402.ascii"))
    plot_and_save_simulation_data(sim_file, output_directory)
    
    plt.show()  # Keep the DNS plots open at the end

