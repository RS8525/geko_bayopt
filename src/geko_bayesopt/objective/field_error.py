###Frist prelim try created by AI


import numpy as np
from scipy.interpolate import griddata

class FieldErrorCalculator:
    """
    Computes the error between DNS reference data and RANS simulation output.
    Interpolates RANS data onto the DNS grid for comparison.
    """
    
    def __init__(self,
                 dns_coords: np.ndarray,
                 dns_fields: dict[str, np.ndarray],
                 field_weights: dict[str, float] | None = None):
        """
        Store the DNS reference data for repeated evaluations.
        
        Args:
            dns_coords: DNS coordinates, shape (N, 2)
            dns_fields: Dictionary of DNS fields (e.g., {'cp': array, 'Ux': array})
        """
        self.dns_coords = dns_coords
        self.dns_fields = dns_fields
        self.field_weights = field_weights or {}
     

    def calculate_error(self, sim_coords: np.ndarray, sim_fields: dict[str, np.ndarray], field_name: str = "cp") -> float:
        """
        Calculates the Mean Squared Error (MSE) between simulated and DNS fields.
        Interpolates the simulation data onto the DNS coordinates.
        
        Args:
            sim_coords: Simulation point coordinates, shape (M, 2)
            sim_fields: Dictionary of simulation fields (e.g., from RunResult)
            field_name: The field to calculate error for (default: "cp")
            
        Returns:
            The Mean Squared Error as a float
        """
        if field_name not in self.dns_fields or field_name not in sim_fields:
            raise KeyError(f"Field '{field_name}' must be present in both DNS and simulation fields.")
            
        dns_vals = self.dns_fields[field_name]
        sim_vals = sim_fields[field_name]
        
        # Interpolate simulation results onto the DNS points using griddata
        # This handles the mapping between the unstructured 2D RANS grid and DNS grid.
        sim_interp = griddata(
            sim_coords,
            sim_vals,
            self.dns_coords,
            method="linear"
        )
        
        # Mask out points where interpolation fails (e.g. outside the simulation convex hull)
        valid_mask = ~np.isnan(sim_interp)
        
        sim_valid = sim_interp[valid_mask]
        dns_valid = dns_vals[valid_mask]
        
        # Calculate and return Mean Squared Error
        if len(sim_valid) == 0:
            raise ValueError(
                "Interpolation resulted in entirely NaN values. "
                "Check coordinate systems and domains."
            )
        eps = 1e-8

        denominator = np.maximum(np.abs(dns_valid), eps)
        error = np.mean(
            np.abs((dns_valid - sim_valid) / denominator)
        )


        weight = self.field_weights.get(field_name, 1.0)

        error_wighted = weight * error

        return float(error_wighted )