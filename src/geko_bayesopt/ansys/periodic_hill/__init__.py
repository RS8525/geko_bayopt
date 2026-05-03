"""
Periodic-hill 2D RANS simulation package.

Public API:
    CaseConfig         -- physics + GEKO + zone names + solver settings
    MeshConfig         -- mesh sizing + boundary layer parameters
    MeshGenerator      -- reads .dsco, runs 2D meshing workflow, writes .msh.h5
    PeriodicHillSolver -- loads mesh, runs RANS/GEKO solve, exports results

    run_case           -- one-shot helper (mesh + solve + export)
    open_session       -- context manager: long-lived Fluent session for BO
    run_geko_trial     -- helper to run one GEKO trial against an open session
"""

from .case_config import CaseConfig
from .mesh_config import MeshConfig
from .mesh_generator import MeshGenerator
from .solver import PeriodicHillSolver
from .runner import run_case, open_session, run_geko_trial

__all__ = [
    "CaseConfig",
    "MeshConfig",
    "MeshGenerator",
    "PeriodicHillSolver",
    "run_case",
    "open_session",
    "run_geko_trial",
]
