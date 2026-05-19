"""
Mesh configuration: sizing parameters and boundary-layer settings.

Kept separate from CaseConfig because mesh parameters are usually tuned
once and reused across many physics cases (Csep sweeps, Re sweeps, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MeshConfig:
    """Mesh sizing parameters for the 2D Meshing workflow.

    Default values give y+ ~ 1 at the bottom wall for Re_h = 5600 with
    hill height H = 28 mm. Adjust ``bl_first_layer_height`` (Re-dependent)
    and the bulk sizes (geometry-dependent) when changing scales.
    """

    # ---- CAD reading --------------------------------------------------------
    length_unit: str = "mm"   # "m" if your Discovery file is in metres
    cad_route: str = "Workbench" # Import route: "Workbench" (for .pmdb), "DSCO" (for .dsco)
    cad_extension: str = "pmdb"  # "pmdb" or "dsco"

    # ---- Bulk mesh sizing ---------------------------------------------------
    min_size: float = 0.02    # smallest edge length on curved walls
    max_size: float = 0.5     # largest edge length in the bulk
    growth_rate: float = 1.2  # cell-to-cell growth factor
    curvature_normal_angle: int = 12  # degrees per cell along curved edges

    # Either "Curvature & Proximity" (recommended), "Curvature", or "Proximity"
    size_functions: str = "Curvature & Proximity"

    # ---- Boundary-layer sizing ---------------------------------------------
    bl_first_layer_height: float = 0.0009  # for y+ ~ 1 at Re_h = 5600
    bl_number_of_layers: int = 22
    bl_growth_rate: float = 1.15

    # ---- Surface mesh options ----------------------------------------------
    generate_quads: bool = True   # quad-dominant mesh (recommended)
