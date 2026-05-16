# import numpy as np
# from scipy.interpolate import griddata


# class SurfaceFieldErrorCalculator:
#     """
#     Computes the error between reference and simulated surface-field data.

#     This is intended for quantities defined on boundaries/surfaces, such as:
#     - x-wall-shear
#     - skin-friction-coefficient
#     - wall pressure
#     - wall heat flux
#     """

#     def __init__(
#         self,
#         ref_coords: np.ndarray,
#         ref_fields: dict[str, np.ndarray],
#         field_weights: dict[str, float] | None = None,
#         error_type: str = "mse",
#         normalize: bool = True,
#     ):
#         """
#         Args:
#             ref_coords:
#                 Reference surface coordinates, shape (N, 2).

#             ref_fields:
#                 Dictionary of reference surface fields.
#                 Example:
#                     {
#                         "x_wall_shear": array,
#                         "skin_friction": array,
#                     }

#             field_weights:
#                 Optional weights for each surface field.
#                 Fields not specified get weight 1.0 by default.

#             error_type:
#                 "mse" or "mae".

#             normalize:
#                 If True, normalize each surface-field error by the range
#                 of the corresponding reference field.
#         """

#         self.ref_coords = ref_coords
#         self.ref_fields = ref_fields
#         self.field_weights = field_weights or {}
#         self.error_type = error_type.lower()
#         self.normalize = normalize

#         if self.error_type not in ["mse", "mae"]:
#             raise ValueError(
#                 f"Unsupported error type: {error_type}. Use 'mse' or 'mae'."
#             )

#     def calculate_error(
#         self,
#         sim_coords: np.ndarray,
#         sim_fields: dict[str, np.ndarray],
#         field_name: str,
#     ) -> float:
#         """
#         Calculates the weighted error between simulated and reference
#         surface fields.

#         Args:
#             sim_coords:
#                 Simulation surface coordinates, shape (M, 2).

#             sim_fields:
#                 Dictionary of simulated surface fields.

#             field_name:
#                 Name of the surface field to compare.

#         Returns:
#             Weighted surface-field error.
#         """

#         if field_name not in self.ref_fields or field_name not in sim_fields:
#             raise KeyError(
#                 f"Surface field '{field_name}' must be present in both "
#                 f"reference and simulation fields."
#             )

#         ref_vals = self.ref_fields[field_name]
#         sim_vals = sim_fields[field_name]

#         # If both datasets are already on the same surface points, compare directly.
#         if (
#             sim_coords.shape == self.ref_coords.shape
#             and np.allclose(sim_coords, self.ref_coords)
#         ):
#             sim_valid = sim_vals
#             ref_valid = ref_vals

#         else:
#             # Interpolate simulation surface field onto reference surface points.
#             sim_interp = griddata(
#                 sim_coords,
#                 sim_vals,
#                 self.ref_coords,
#                 method="linear",
#             )

#             valid_mask = ~np.isnan(sim_interp)

#             sim_valid = sim_interp[valid_mask]
#             ref_valid = ref_vals[valid_mask]

#             if len(sim_valid) == 0:
#                 raise ValueError(
#                     "Interpolation resulted in entirely NaN values. "
#                     "Check surface coordinate systems."
#                 )

#         diff = sim_valid - ref_valid

#         if self.error_type == "mse":
#             error = np.mean(diff**2)

#         elif self.error_type == "mae":
#             error = np.mean(np.abs(diff))

#         # Normalize each surface field by its own reference scale.
#         if self.normalize:
#             field_range = np.max(ref_valid) - np.min(ref_valid)

#             if field_range < 1e-12:
#                 field_range = 1.0

#             if self.error_type == "mse":
#                 error = error / field_range**2

#             elif self.error_type == "mae":
#                 error = error / field_range

#         weight = self.field_weights.get(field_name, 1.0)

#         weighted_error = weight * error

#         return float(weighted_error)