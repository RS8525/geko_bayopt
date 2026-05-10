import os
import numpy as np

def getData(CaseName=str, **kwargs) ->  tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Reads the simulated RANS output data from the specified .dat file.
    
    Extracts x-coordinates, y-coordinates, umean, vmean, and pressure from specific columns,
     returns a normalized pressure coefficient.
    """


    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    dns_path = os.path.join(
        base_dir,
        "data",
        "dns",
        "periodic_hills",
        "pehill-29-cases-DNS",
        CaseName,
        "mean_files.dat",
    )

    # Assuming .dat output from Fluent matching [node, x, y, u, v, p]
    data = np.genfromtxt(dns_path, dtype=float, skip_header= 0, delimiter = None)
    # Directly get dns_coords from data
    dns_coords = data[:, :2]
    u_data = data[:,  2]
    v_data = data[:,  3]
    p_data = data[:,  5]

    # Translate to pressure coefficient
    cp_data = p_data - p_data[-1]
    dns_fields = {"cp": cp_data, "Ux": u_data, "Uy": v_data}

    return dns_coords, dns_fields