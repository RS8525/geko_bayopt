"""Quick visualization script for a Fluent ASCII export from the FFS case.

The color plots use triangular interpolation between the unstructured node
locations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


ASCII_PATH = Path(__file__).resolve().parents[1] / "results" / "fluent" / "ffs_csep_v3" / "alpha1.0_Re6000_Csep0.9407.ascii"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "plots"


def load_ascii(path: Path) -> np.ndarray:
    return np.genfromtxt(path, skip_header=1)


def plot_field(x: np.ndarray, y: np.ndarray, values: np.ndarray, title: str, cbar_label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    tri = mtri.Triangulation(x, y)
    tri.set_mask(_ffs_mask(tri, x, y))
    contour = ax.tricontourf(tri, values, levels=100, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _ffs_floor(x: np.ndarray) -> np.ndarray:
    """Piecewise floor for the forward-facing step plot.

    Upstream of the step (x < 0) the floor is at y = -0.01.
    Downstream (x >= 0) the floor is at y = 0.0.
    """

    return np.where(x < 0.0, -0.01, 0.0)


def _ffs_mask(tri: mtri.Triangulation, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask triangles that fall into the solid step region."""

    triangles = tri.triangles
    tri_x = x[triangles]
    tri_y = y[triangles]
    floor_y = _ffs_floor(tri_x)

    # Mask any triangle with at least one vertex below the local floor.
    # This prevents tricontourf from filling the gap under the step.
    return np.any(tri_y < floor_y - 1e-12, axis=1)


def main() -> None:
    data = load_ascii(ASCII_PATH)

    node_id = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]
    pressure = data[:, 5] - data[:, 5].min()  # simple gauge pressure convention
    turbulent_kinetic_energy = data[:, 6]


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_field(
        x,
        y,
        ux,
        title="FFS simulation Ux",
        cbar_label="Ux",
        output_path=OUTPUT_DIR / "ffs_sim_ux.png",
    )
    plot_field(
        x,
        y,
        uy,
        title="FFS simulation Uy",
        cbar_label="Uy",
        output_path=OUTPUT_DIR / "ffs_sim_uy.png",
    )
    plot_field(
        x,
        y,
        pressure,
        title="FFS simulation pressure",
        cbar_label="Pressure",
        output_path=OUTPUT_DIR / "ffs_sim_pressure.png",
    )
    plot_field(
        x,
        y,
        turbulent_kinetic_energy,
        title="FFS simulation turbulent kinetic energy",
        cbar_label="Turbulent kinetic energy",
        output_path=OUTPUT_DIR / "ffs_sim_tke.png",
    )

    print(f"Loaded {len(node_id)} nodes from {ASCII_PATH}")
    print(f"Plots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
