# Test script to visualize the DNS data and simulation data for the periodic hills case.
# This script will generate scatter plots of the specified columns from the DNS and simulation datasets,
# allowing for a visual comparison of the fields.
#
# JSON contains:
#     name
#     simulation: path, x, y, fields
#     dns:        path, x, y, fields
#     plots:      fields, x_locations, x_tol

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DEFAULT_CONFIG_PATH = os.path.join(
    REPO_ROOT, "scripts", "per_hill", "plots", "plot_config.json"
)


def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(config_path):
    with open(repo_path(config_path), "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path):
    return pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)


def get_column(df, col):
    if isinstance(col, int):
        return df.iloc[:, col].to_numpy(dtype=float)
    return df[col].to_numpy(dtype=float)


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


def load_dns_data(dns_cfg):
    """Carrega e devolve os campos do ficheiro DNS."""
    df = read_table(repo_path(dns_cfg["path"]))

    x = get_column(df, dns_cfg["x"])
    y = get_column(df, dns_cfg["y"])

    out = {"x": x, "y": y}

    for field_name, col in dns_cfg["fields"].items():
        out[field_name] = get_column(df, col)

    return out


def load_sim_data(sim_cfg):
    """Carrega e devolve os campos do ficheiro de simulação."""
    df = read_table(repo_path(sim_cfg["path"]))

    x = get_column(df, sim_cfg["x"])
    y = get_column(df, sim_cfg["y"])

    out = {"x": x, "y": y}

    for field_name, col in sim_cfg["fields"].items():
        out[field_name] = get_column(df, col)

    return out


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


def plot_dns_and_sim(case_name, sim_cfg, dns_cfg, output_dir, fields):
    """
    Plota campos do DNS e da simulação com a mesma escala de cor para cada campo.
    """
    dns = load_dns_data(dns_cfg)
    sim = load_sim_data(sim_cfg)

    os.makedirs(output_dir, exist_ok=True)

    for field in fields:
        vmin = min(dns[field].min(), sim[field].min())
        vmax = max(dns[field].max(), sim[field].max())

        save_field_plot(
            dns["x"], dns["y"], dns[field],
            title=f"DNS {field} — {case_name}",
            label=field,
            filepath=os.path.join(output_dir, f"dns_{case_name}_{field}.png"),
            cmap="viridis", vmin=vmin, vmax=vmax,
        )

        save_field_plot(
            sim["x"], sim["y"], sim[field],
            title=f"Simulation {field}",
            label=field,
            filepath=os.path.join(output_dir, f"simulation_{field}.png"),
            cmap="viridis", vmin=vmin, vmax=vmax,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config. Default: scripts/per_hill/plots/plot_config.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_directory = os.path.abspath(
        os.path.join(REPO_ROOT, "results", "experiments", cfg["name"], "plots")
    )

    plot_dns_and_sim(
        case_name=cfg["name"],
        sim_cfg=cfg["simulation"],
        dns_cfg=cfg["dns"],
        output_dir=output_directory,
        fields=cfg["plots"]["fields"],
    )
