"""
.. _ref_geometry-mesh-fluent_01-geometry:

Geometry generation
###################

Generate a Periodic Hill geometry using PyAnsys Geometry.

Geometry will be saved in the `{DATA_DIR}` directory as a .scdocx file, which can be opened in Ansys SpaceClaim/Discovery.
Using it in Ansys Fluent for meshing possible.
The geometry will consist of a 2D sketch of the periodic hill and the surrounding fluid domain. 
Named selections will be created for the inlet, outlet, ceiling, and hill edges.

Main workhorse:

`create_perhill_geom_ansys()`: This function creates the periodic hill geometry in Ansys. 
It generates the bottom curve points, creates a 2D sketch of the fluid domain, converts it to a face, defines named selections for the edges
and saves the design.

Inputs: 
- `alpha`: Parameter controlling the shape of the hill (default: 1.0)
- `L`: Streamwise length of the domain (default: 9.0)
- `ceiling_param`: Ceiling height of the domain (default: 3.036)

Requiremnts for running the scripts are given in requirements.txt file. 
Additional requirement: Ansys Spaceclaim using version -261
Can be editied in line the third line after the start of the create_perhill_geom_ansys function.

"""  # noqa: D400, D415

import os
import numpy as np
import cadquery as cq
from pathlib import Path
from typing import List, Union

from ansys.geometry.core import launch_modeler
from ansys.geometry.core.connection import GeometryContainers
import ansys.geometry.core.connection.defaults as pygeom_defaults
from ansys.geometry.core.math import Plane, Point2D, Point3D
#from ansys.geometry.core.plotting import GeometryPlotter
from ansys.geometry.core.sketch import Sketch
from geko_bayesopt.ansys.periodic_hill.geometry_generation.hillShape import para_profile
import math


###############################################################################
# Preparing the environment
# -------------------------
# This section is only necessary for workflow runs and docs generation. It checks
# the environment variables to determine which image to use for the geometry service.
# If you are running this script outside of a workflow, you can ignore this section.
#
image = None
transport_mode = None
if "ANSYS_GEOMETRY_RELEASE" in os.environ:
    image_tag = os.environ["ANSYS_GEOMETRY_RELEASE"]
    for geom_services in GeometryContainers:
        if image_tag == f"{pygeom_defaults.GEOMETRY_SERVICE_DOCKER_IMAGE}:{geom_services.value[2]}":
            print(f"Using {image_tag} image")
            image = geom_services
            transport_mode = "insecure"
            break

# sphinx_gallery_start_ignore
# Check if the __file__ variable is defined. If not, set it.
# This is a workaround to run the script in Sphinx-Gallery.
if "__file__" not in locals():
    __file__ = Path(os.getcwd(), "periodic_hill_geom.py")
# sphinx_gallery_end_ignore

###############################################################################
# Parameters for the script
# -------------------------
# The following parameters are used to control the script execution. You can
# modify these parameters to suit your needs.
#


###############################################################################
# Parameters for the script
# -------------------------
# Graphics boolean
GRAPHICS_BOOL = False  # Set to True to display the graphs

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..","..", "..", "..", "data", "geometry", "perhill")

# sphinx_gallery_start_ignore
if "DOC_BUILD" in os.environ:
    GRAPHICS_BOOL = True
# sphinx_gallery_end_ignore

def generate_periodic_hill_shape(alpha: float = 1.0, L: float = 9.0) -> list[Point2D]:
    """
    Generate points for the bottom curve of the periodic hill geometry.
    """
    # Generate the bottom curve points with a strict 0.01 spacing
    yy = np.arange(0, L, 0.01)
    if not np.isclose(yy[-1], L): # Ensure the final domain length point is present
        yy = np.append(yy, L)
        
    ya, h = para_profile(yy, alpha, L)
    H = 1.0  # Scale factor for the hill height

    x_pts = (ya * H).tolist()
    y_pts = (h * H).tolist()
    
    return [Point2D([x, y]) for x, y in zip(x_pts, y_pts)]

def create_perhill_geom_ansys(alpha: float = 1.0, L: float = 9.0, ceiling_param: float = 3.036):
    # Instantiate the modeler, usning "insecure" transport mode for local execution. 
    # Runs faster but might be instable. Adjust the version as needed.
    modeler = launch_modeler(mode="spaceclaim", version=-261, transport_mode=None)
    print(modeler)

    # Parameters
    H = 1.0
    L = L / H  # streamwise length
    ceiling = ceiling_param / H  # ceiling height

    # Create the design
    design = modeler.create_design(f"Periodic_Hill_{alpha}_{L}_{ceiling_param}".replace(".", ""))

    # Generate the bottom curve
    bottom_points = generate_periodic_hill_shape(alpha, L)
    x_end = bottom_points[-1].x.m

    # Create the fluid domain sketch in 2D
    fluid_sketch = Sketch()
    
    # 1. Bottom profile (hill)
    for i in range(len(bottom_points) - 1):
        fluid_sketch.segment(bottom_points[i], bottom_points[i + 1])
        
    # 2. Outlet wall (Right)
    p_br = bottom_points[-1]
    p_tr = Point2D([x_end, ceiling])
    fluid_sketch.segment(p_br, p_tr)
    
    # 3. Ceiling (Top)
    p_tl = Point2D([0, ceiling])
    fluid_sketch.segment(p_tr, p_tl)
    
    # 4. Inlet wall (Left)
    p_bl = bottom_points[0]
    fluid_sketch.segment(p_tl, p_bl)

    # Convert the fluid domain sketch to a face
    fluid = design.create_surface("Fluid", fluid_sketch)

    # Define Named Selections using bounding boxes of edges
    fluid_edges = fluid.edges
    inlet_edges = []
    outlet_edges = []
    ceiling_edges = []
    hill_edges = []

    for edge in fluid_edges:
        bounds = edge.bounding_box
        mid_x = (bounds.min_corner.x.m + bounds.max_corner.x.m) / 2.0
        mid_y = (bounds.min_corner.y.m + bounds.max_corner.y.m) / 2.0
        
        if math.isclose(mid_x, 0.0, abs_tol=1e-3):
            inlet_edges.append(edge)
        elif math.isclose(mid_x, x_end, abs_tol=1e-3):
            outlet_edges.append(edge)
        elif math.isclose(mid_y, ceiling, abs_tol=1e-3):
            ceiling_edges.append(edge)
        else:
            hill_edges.append(edge)

    design.create_named_selection("Inlet", edges=inlet_edges)
    design.create_named_selection("Outlet", edges=outlet_edges)
    design.create_named_selection("Ceiling", edges=ceiling_edges)
    design.create_named_selection("Hill", edges=hill_edges)

    # Plot the design
    if GRAPHICS_BOOL:
        design.plot()

    # Save the design
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = design.export_to_pmdb(DATA_DIR)
    print(f"Design saved to {file_path}")

    # Close the server session.
    modeler.close()

if __name__ == "__main__":
    create_perhill_geom_ansys()
