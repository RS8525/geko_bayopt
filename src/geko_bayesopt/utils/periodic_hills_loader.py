import numpy as np
from pathlib import Path


def getSimulationData(sim_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Reads RANS/Fluent simulation output data.

    Expected format:
        nodenumber x-coordinate y-coordinate x-velocity y-velocity pressure

    Returns:
        sim_coords: array with x and y coordinates, shape (N, 2)
        sim_fields: dictionary with fields {"cp", "Ux", "Uy"}
    """

    sim_path = Path(sim_path)

    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")

    data = np.genfromtxt(
        sim_path,
        dtype=float,
        skip_header=1,
        delimiter=None,
    )

    # Columns:
    # 0: nodenumber
    # 1: x-coordinate
    # 2: y-coordinate
    # 3: x-velocity
    # 4: y-velocity
    # 5: pressure
    sim_coords = data[:, 1:3]

    u_data = data[:, 3]
    v_data = data[:, 4]
    p_data = data[:, 5]

    # Same logic as DNS: pressure shifted by final pressure value
    cp_data = p_data - p_data[-1]

    sim_fields = {
        "cp": cp_data,
        "Ux": u_data,
        "Uy": v_data,
    }

    return sim_coords, sim_fields