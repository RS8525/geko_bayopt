"""
Mesh generation: wraps Fluent Meshing's 2D Meshing workflow in a reusable class.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import ansys.fluent.core as pyfluent

from .case_config import CaseConfig
from .mesh_config import MeshConfig


class MeshGenerator:
    """Generate a 2D mesh from a Discovery .dsco file.

    Parameters
    ----------
    case : CaseConfig
        Used only for the geometry basename.
    mesh : MeshConfig
        Sizing and boundary-layer parameters.
    data_dir : str or Path
        Folder containing the input .dsco and where the .msh.h5 is written.
    ui_mode : str, optional
        PyFluent launch mode. Defaults to ``"no_gui_or_graphics"``.

    Example
    -------
    >>> case = CaseConfig()
    >>> mesh = MeshConfig()
    >>> path = MeshGenerator(case, mesh, "outputs").generate()
    >>> print(path)
    .../outputs/periodic_hill_2d_alpha_1.0.msh.h5
    """

    def __init__(
        self,
        case: CaseConfig,
        mesh: MeshConfig,
        data_dir: str | Path,
        ui_mode: str = "no_gui_or_graphics",
        container_dict: dict | None = None,
        geometry_path: str | Path | None = None,
    ):
        """Create a mesh generator bound to a case and an output directory.

        Parameters
        ----------
        case, mesh
            Configuration dataclasses.
        data_dir
            Directory where the output .msh.h5 is written. Also the
            fallback location for the input geometry if ``geometry_path``
            is not given.
        ui_mode
            PyFluent launch mode. Default headless.
        container_dict
            Optional Fluent-container configuration.
        geometry_path
            Explicit absolute path to the input geometry file (.dsco on
            Windows, .pmdb on Linux). If None, the generator falls back
            to ``data_dir / <case.geometry_basename>.<ext>`` for backward
            compatibility with the standalone ``run.py`` script.
        """
        self.case = case
        self.mesh = mesh
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ui_mode = ui_mode
        self.container_dict = container_dict
        self._explicit_geometry_path = (
            Path(geometry_path).resolve() if geometry_path is not None else None
        )

        # Output path the mesh will be written to
        self.mesh_path = self.data_dir / f"{case.geometry_basename}.msh.h5"

    # ---- Public API ---------------------------------------------------------

    def generate(self) -> Path:
        """Run the full 2D meshing workflow. Returns the path of the .msh.h5."""
        self._validate_inputs()
        meshing = self._launch()
        try:
            two_dim = meshing.two_dimensional_meshing()
            self._load_cad(two_dim)
            self._update_regions(two_dim)
            self._update_boundaries(two_dim)
            self._define_global_sizing(two_dim)
            self._add_boundary_layers(two_dim)
            self._generate_surface_mesh(two_dim)
            self._write_mesh(two_dim)
        finally:
            meshing.exit()
        return self.mesh_path

    # ---- Internal steps -----------------------------------------------------

    def _geometry_path(self) -> Path:
        """Pick the extension specified in MeshConfig.

        If ``geometry_path`` was provided to the constructor, use that
        directly. Otherwise fall back to ``data_dir / basename.<ext>``.
        """
        if self._explicit_geometry_path is not None:
            return self._explicit_geometry_path
        ext = self.mesh.cad_extension
        return self.data_dir / f"{self.case.geometry_basename}.{ext}"

    def _validate_inputs(self) -> None:
        """Surface clear errors before launching Fluent."""
        if self.container_dict is not None:
            return  # checks don't apply inside a container
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"data_dir does not exist: {self.data_dir!r}")
        geom = self._geometry_path()
        if not geom.is_file():
            raise FileNotFoundError(
                f"Geometry file not found: {geom!r}\n"
                "Pass an explicit geometry_path to MeshGenerator, "
                "or place the file matching geometry_basename in data_dir."
            )

    def _launch(self):
        """Launch Fluent Meshing (NOT in 2D mode at the launcher level --
        the 2D Meshing workflow is selected after launch). Passing
        ``version='2d'`` together with ``mode='meshing'`` triggers a
        Fluent bug that switches the session to the solver instead.
        """
        kwargs = dict(
            precision="double",
            processor_count=4,
            mode="meshing",
            cwd=str(self.data_dir),
        )
        if self.container_dict is not None:
            return pyfluent.launch_fluent(
                container_dict=self.container_dict,
                start_container=True,
                ui_mode="no_gui_or_graphics",
                cleanup_on_exit=False,
                start_timeout=300,
                **kwargs,
            )
        return pyfluent.launch_fluent(ui_mode=self.ui_mode, **kwargs)

    def _load_cad(self, two_dim) -> None:
        """Step 1: Load CAD Geometry."""
        task = two_dim.load_cad_geometry
        task.file_name = str(self._geometry_path())
        task.route = self.mesh.cad_route
        task.length_unit = self.mesh.length_unit
        task.refaceting.refacet = False
        task()

    def _update_regions(self, two_dim) -> None:
        """Step 2: Update Regions. Detects fluid regions in the CAD."""
        two_dim.update_regions()

    def _update_boundaries(self, two_dim) -> None:
        """Step 3: Update Boundaries. ``label`` selection mode picks up
        named selections from the .dsco (inlet, outlet, walls)."""
        task = two_dim.update_boundaries
        task.selection_type = "label"
        task()

    def _define_global_sizing(self, two_dim) -> None:
        """Step 4: Define Global Sizing."""
        task = two_dim.define_global_sizing
        task.min_size = self.mesh.min_size
        task.max_size = self.mesh.max_size
        task.growth_rate = self.mesh.growth_rate
        task.curvature_normal_angle = self.mesh.curvature_normal_angle
        task.size_functions = self.mesh.size_functions
        task()

    def _add_boundary_layers(self, two_dim) -> None:
        """Step 5: Add 2D Boundary Layers (prism layers on the walls)."""
        task = two_dim.add_2d_boundary_layers
        task.add_child = "yes"
        task.bl_control_name = "uniform_1"
        task.offset_method_type = "uniform"
        task.number_of_layers = self.mesh.bl_number_of_layers
        task.first_layer_height = self.mesh.bl_first_layer_height
        task.growth_rate = self.mesh.bl_growth_rate
        task.add_child_and_update()

    def _generate_surface_mesh(self, two_dim) -> None:
        """Step 6: Generate the Surface Mesh (the final 2D mesh)."""
        task = two_dim.generate_initial_surface_mesh
        task.generate_quads = self.mesh.generate_quads
        task()

    def _write_mesh(self, two_dim) -> None:
        """Step 7: Write the mesh via the workflow task (NOT TUI -- the
        TUI version produces files that the solver rejects as 'surface mesh')."""
        two_dim.write_2d_mesh.file_name = str(self.mesh_path)
        two_dim.write_2d_mesh()

        # Sanity check -- a truly empty mesh write is silent in some Fluent
        # builds. A periodic-hill mesh at our resolutions is several MB.
        size_mb = os.path.getsize(self.mesh_path) / (1024 * 1024)
        if size_mb < 0.5:
            raise RuntimeError(
                f"Mesh file is suspiciously small ({size_mb:.2f} MB). "
                "Mesh generation likely produced 0 cells."
            )
        print(f"[mesh] Wrote {self.mesh_path}  ({size_mb:.1f} MB)")
