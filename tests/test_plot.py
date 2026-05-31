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
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "dns",
            "periodic_hills",
            "pehill-29-cases-DNS",
            )
        )
    
    data_file = os.path.join(
        base_dir,
        case_name,
        "dns_avg_Re2800_columnwise_organized.ascii"
    )

    # Load the data
    data = np.genfromtxt(data_file, dtype=float, skip_header=1, delimiter=None)

    # Extract columns
    x_coords = data[:, 1]
    y_coords = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]
    k= data[:, 10]

    os.makedirs(output_dir, exist_ok=True)

    # Plot Ux
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c=ux, cmap='viridis', label='Ux', alpha=1, s=0.03)
    plt.colorbar(label='Ux')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'DNS Ux for {case_name}')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'dns_{case_name}_Ux.png'))
    plt.show(block=False)

    # Plot Uy
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c=uy, cmap='viridis', label='Uy', alpha=1, s=0.03)
    plt.colorbar(label='Uy')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'DNS Uy for {case_name}')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'dns_{case_name}_Uy.png'))
    plt.show(block=False)

    # Plot Turbulent Kinetic Energy (k)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c=k, cmap='viridis', label='Turbulent Kinetic Energy (k)', alpha=1, s=0.03)
    plt.colorbar(label='Turbulent Kinetic Energy (k)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'DNS Turbulent Kinetic Energy (k) for {case_name}')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'dns_{case_name}_k.png'))
    plt.show(block=False)


def plot_and_save_simulation_data(sim_file_path, output_dir):
    """
    Plots simulation data and saves each column plot as an individual figure.

    Args:
        sim_file_path (str): The path to the simulation data file.
        output_dir (str): The directory where the plots will be saved.
    """
    # Load simulation data
    sim_data = np.genfromtxt(sim_file_path, dtype=float, skip_header=1, delimiter=None)
    x_coords = sim_data[:, 1]
    y_coords = sim_data[:, 2]
    ux = sim_data[:, 4]
    uy = sim_data[:, 5]
    k = sim_data[:, 3]

    os.makedirs(output_dir, exist_ok=True)

    # Plot Ux
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_coords, y_coords, c=ux, cmap='viridis', alpha=1, s=0.03)
    plt.colorbar(scatter, label='Ux')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulation Ux')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'simulation_Ux.png'))
    plt.close()

    # Plot Uy
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_coords, y_coords, c=uy, cmap='viridis', alpha=1, s=0.03)
    plt.colorbar(scatter, label='Uy')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulation Uy')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'simulation_Uy.png'))
    plt.close()

    # Plot Turbulent Kinetic Energy (k)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_coords, y_coords, c=k, cmap='viridis', alpha=1, s=0.03)
    plt.colorbar(scatter, label='Turbulent Kinetic Energy (k)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulation Turbulent Kinetic Energy (k)')
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'simulation_k.png'))
    plt.close()

if __name__ == "__main__":
    output_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "experiments",
            "periodic_hills_2800_v1",
            "plots",
        )
    )

    plot_dns_data("alph10-9-3036", output_directory)  # Close DNS plots before plotting simulation data
    
    sim_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "fluent",
            "periodic_hills_2800_v1",
            "alpha1.0_Re2800_Csep0.8557517256760887_Cnw0.500973460097605.ascii",
        )
    )

    plot_and_save_simulation_data(sim_file, output_directory)
    
    #plt.show()  # Keep the DNS plots open at the end

