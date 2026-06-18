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

def plot_field(ax, x, y, values, cmap='viridis', vmin=None, vmax=None):
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

    levels = np.linspace(vmin, vmax, 51) if (vmin is not None and vmax is not None) else 50
    tcf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    return tcf


def load_dns_data(case_name):
    """Carrega e devolve os campos do ficheiro DNS."""
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
    data = np.genfromtxt(data_file, dtype=float, skip_header=1, delimiter=None)

    x       = data[:, 1]
    y       = data[:, 2]
    ux      = data[:, 3]
    uy      = data[:, 4]
    k       = data[:, 10]
    prod_k  = data[:, 7]
    diss    = data[:, 8]
    density = data[:, 6]
    #turb_visc = 0.09 * density * k**2 / diss

    return dict(x=x, y=y, ux=ux, uy=uy, k=k, prod_k=prod_k,) #turb_visc=turb_visc)


def load_sim_data(sim_file_path):
    """Carrega e devolve os campos do ficheiro de simulação."""
    sim_data = np.genfromtxt(sim_file_path, dtype=float, skip_header=1, delimiter=None)

    x         = sim_data[:, 1]
    y         = sim_data[:, 2]
    ux        = sim_data[:, 6]
    uy        = sim_data[:, 7]
    k         = sim_data[:, 5]
    prod_k    = sim_data[:, 4]
    #turb_visc = sim_data[:, 3]

    return dict(x=x, y=y, ux=ux, uy=uy, k=k, prod_k=prod_k)# turb_visc=turb_visc)


def save_field_plot(x, y, values, title, label, filepath, cmap='viridis', vmin=None, vmax=None):
    """Renderiza um campo 2D e guarda em ficheiro."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x, y, values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, label=label)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(filepath)
    plt.close(fig)


def plot_dns_and_sim(case_name, sim_file_path, output_dir):
    """
    Plota campos do DNS e da simulação com a mesma escala de cor para cada campo.

    Args:
        case_name (str): Nome do caso DNS.
        sim_file_path (str): Caminho para o ficheiro de simulação.
        output_dir (str): Directório onde os plots são guardados.
    """
    dns = load_dns_data(case_name)
    sim = load_sim_data(sim_file_path)

    os.makedirs(output_dir, exist_ok=True)

    # Campos a comparar: (chave, label, cmap)
    fields = [
        ("ux",        "Ux",                        "viridis"),
        ("uy",        "Uy",                        "viridis"),
        ("k",         "Turbulent Kinetic Energy (k)", "viridis"),
        ("prod_k",    "Production of k",           "viridis"),
        #("turb_visc", "Turbulent Viscosity",        "viridis"),
    ]

    for key, label, cmap in fields:
        # Escala global: min e max entre DNS e simulação
        vmin = min(dns[key].min(), sim[key].min())
        vmax = max(dns[key].max(), sim[key].max())

        # Plot DNS
        save_field_plot(
            dns["x"], dns["y"], dns[key],
            title=f"DNS {label} — {case_name}",
            label=label,
            filepath=os.path.join(output_dir, f"dns_{case_name}_{key}.png"),
            cmap=cmap, vmin=vmin, vmax=vmax,
        )

        # Plot Simulação
        save_field_plot(
            sim["x"], sim["y"], sim[key],
            title=f"Simulation {label}",
            label=label,
            filepath=os.path.join(output_dir, f"simulation_{key}.png"),
            cmap=cmap, vmin=vmin, vmax=vmax,
        )



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

    sim_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "fluent",
            "periodic_hills_2800_v1",
            "alpha1.0_Re2800_Csep1.0539614348602349_Cnw2.0_Cmix0.46039441259276553_Cjet0.884168342743064_Cturb1.6647.ascii",
        )
    )

    plot_dns_and_sim(
        case_name="alph10-9-3036",
        sim_file_path=sim_file,
        output_dir=output_directory,
    )