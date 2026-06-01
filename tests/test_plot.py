# Test script to visualize the DNS data and simulation data for the periodic hills case. 
# This script will generate scatter plots of the specified columns from the DNS and simulation datasets, allowing for a visual comparison of the fields.
# Expected directories for data:
# - DNS data: data/dns/periodic_hills/pehill-29-cases-DNS/{case_name}/mean_files.dat
# - Simulation data: results/experiments/periodic_hills/periodic_hill_2d_alpha_1.0.msh_solved_csep_1.75
# Change in main function if thats vary

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_field(ax, x, y, values, cmap='viridis'):
    triang = tri.Triangulation(x, y)

    # Mask triangles with long edges — these are usually the ones crossing the hill geometry
    tris = triang.triangles

    edge_01 = np.sqrt(
        (x[tris[:, 0]] - x[tris[:, 1]])**2
        + (y[tris[:, 0]] - y[tris[:, 1]])**2
    )
    edge_12 = np.sqrt(
        (x[tris[:, 1]] - x[tris[:, 2]])**2
        + (y[tris[:, 1]] - y[tris[:, 2]])**2
    )
    edge_20 = np.sqrt(
        (x[tris[:, 2]] - x[tris[:, 0]])**2
        + (y[tris[:, 2]] - y[tris[:, 0]])**2
    )

    max_edge = np.maximum.reduce([edge_01, edge_12, edge_20])

    triang.set_mask(max_edge > 0.15)

    tcf = ax.tricontourf(triang, values, levels=50, cmap=cmap)
    return tcf

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
    prod_k = data[:, 7]

    os.makedirs(output_dir, exist_ok=True)

    # Plot Ux
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, ux, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Ux')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'DNS Ux for {case_name}')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, f'dns_{case_name}_Ux.png'))

    # Plot Uy
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, uy, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Uy')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'DNS Uy for {case_name}')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, f'dns_{case_name}_Uy.png'))

    # Plot Turbulent Kinetic Energy (k)
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, k, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Turbulent Kinetic Energy (k)')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'DNS Turbulent Kinetic Energy (k) for {case_name}')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, f'dns_{case_name}_k.png'))

    # Plot Production of k
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, prod_k, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Production of k')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'DNS Production of k for {case_name}')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, f'dns_{case_name}_prod_k.png'))


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
    ux = sim_data[:, 5]
    uy = sim_data[:, 6]
    k = sim_data[:, 4]
    prod_k = sim_data[:, 3]

    os.makedirs(output_dir, exist_ok=True)

    # Plot Ux
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, ux, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Ux')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Simulation Ux')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'simulation_Ux.png'))
    plt.close(fig)

    # Plot Uy
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, uy, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Uy')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Simulation Uy')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'simulation_Uy.png'))
    plt.close(fig)

    # Plot Turbulent Kinetic Energy (k)
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, k, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Turbulent Kinetic Energy (k)')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Simulation Turbulent Kinetic Energy (k)')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'simulation_k.png'))
    plt.close(fig)

    # Plot Production of k
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x_coords, y_coords, prod_k, cmap='viridis')
    fig.colorbar(tcf, ax=ax, label='Production of k')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Simulation Production of k')
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'simulation_prod_k.png'))
    plt.close(fig)

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
            "alpha1.0_Re2800_Csep1.0207955748001365_Cnw0.8437503761014433_Cmix0.49642554136187106.ascii",
        )
    )

    plot_and_save_simulation_data(sim_file, output_directory)
    
    #plt.show()  # Keep the DNS plots open at the end

