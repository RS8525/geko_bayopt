"""
Test script for the 2D Meshing workflow using PyFluent.
"""

import os
from pathlib import Path

from geko_bayesopt.fluent.case_config import CaseConfig
from geko_bayesopt.fluent.mesh_config import MeshConfig
from geko_bayesopt.fluent.mesh_generator import MeshGenerator

def test_meshing_workflow():
    repo_root = Path(__file__).resolve().parent.parent
    
    # Path to an existing .dsco file in your data folder
    geometry_path = repo_root / "data" / "geometry" / "periodic_hill_2d_alpha_1.0.dsco"
    
    # We write the testing outputs to a dedicated test folder in results
    output_dir = repo_root / "results" / "fluent" / "meshing_test"
    
    print(f"Using geometry: {geometry_path}")
    print(f"Output directory: {output_dir}")

    # Set up case and mesh configs
    # We specify cad_route="DSCO" and cad_extension="dsco" to match the input geometry type
    case = CaseConfig(alpha=1.0)
    mesh = MeshConfig(cad_route="DSCO", cad_extension="dsco")

    # Initialize the generator bypassing the default generic lookup by 
    # explicitly passing `geometry_path`
    generator = MeshGenerator(
        case=case,
        mesh=mesh,
        data_dir=output_dir,
        ui_mode="no_gui_or_graphics",
        geometry_path=geometry_path
    )

    # Generate the mesh
    print("Starting mesh generation...")
    mesh_path = generator.generate()

    # Verify the output
    assert mesh_path.exists(), f"Failed to generate mesh file: {mesh_path}"
    
    size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
    assert size_mb > 0.5, f"Mesh file is too small ({size_mb:.2f} MB), expected > 0.5 MB"
    
    print(f"Meshing test passed! Mesh created successfully at: {mesh_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    test_meshing_workflow()
