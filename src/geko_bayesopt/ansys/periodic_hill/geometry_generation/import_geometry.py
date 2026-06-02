import numpy as np
import cadquery as cq
from hillShape import para_profile

# Parameters
alpha = 1.0          # try 0.5, 0.8, 1.2, 1.5 for stretched variants
H = 28.0             # one hill height in raw units
L = 9.0 * H          # streamwise length = 252
ceiling = 3.036 * H  # = 85.008
span = 4.5 * H       # spanwise extrusion = 126 (typical LES domain)

# Generate the bottom curve points
yy = np.linspace(0, 9, 1000)
ya, h = para_profile(yy, alpha)

x_pts = (ya * H).tolist()
y_pts = (h * H).tolist()
bottom = list(zip(x_pts, y_pts))

x_end = x_pts[-1]  # may differ from L if alpha != 1; clamp to L if needed

# Build the closed 2D fluid domain
domain_2d = (
    cq.Workplane("XY")
    .moveTo(0, ceiling)         # top-left corner
    .lineTo(x_end, ceiling)     # ceiling
    .lineTo(x_end, y_pts[-1])   # outlet wall
    .polyline(bottom[::-1])     # hill bottom, right-to-left
    .close()
)

# Export
cq.exporters.export(domain_2d.val(), f"periodic_hill_2d_alpha_{alpha}.step")