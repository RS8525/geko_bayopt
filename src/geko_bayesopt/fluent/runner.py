"""
High-level helpers: one-shot runs and BO-friendly session reuse.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Generator

from .case_config import CaseConfig
from .mesh_config import MeshConfig
from .mesh_generator import MeshGenerator
from .solver import PeriodicHillSolver

import time


# ---------------------------------------------------------------------------
# One-shot run (mesh + solve + export, then exit Fluent)
# ---------------------------------------------------------------------------

def run_case(
    case: CaseConfig,
    mesh: MeshConfig,
    data_dir: str | Path,
    *,
    skip_meshing_if_exists: bool = True,
    ui_mode: str = "no_gui_or_graphics",
) -> dict[str, Path]:
    """Generate the mesh (if needed) and run the solver for one case.

    Use this for single runs. For BO loops use ``open_session`` instead --
    it keeps Fluent alive across trials, saving 30-60 seconds of launch
    overhead per trial.

    Parameters
    ----------
    case : CaseConfig
    mesh : MeshConfig
    data_dir : str or Path
    skip_meshing_if_exists : bool, default True
        Skip remeshing if .msh.h5 already exists.
    ui_mode : str
        Forwarded to PyFluent launches.
    """
    data_dir = Path(data_dir).resolve()

    mesh_generator = MeshGenerator(case, mesh, data_dir, ui_mode=ui_mode)
    if skip_meshing_if_exists and mesh_generator.mesh_path.is_file():
        print(f"[run_case] Reusing existing mesh: {mesh_generator.mesh_path}")
        mesh_path = mesh_generator.mesh_path
    else:
        print(f"[run_case] Generating mesh for case {case.case_id}")
        mesh_path = mesh_generator.generate()
        time.sleep(10)

    print(f"[run_case] Running solver for case {case.case_id}")
    return PeriodicHillSolver(case, mesh_path, data_dir, ui_mode=ui_mode).run()


# ---------------------------------------------------------------------------
# Persistent session (Fluent stays alive across multiple GEKO trials)
# ---------------------------------------------------------------------------

@contextmanager
def open_session(
    base_case: CaseConfig,
    mesh: MeshConfig,
    data_dir: str | Path,
    *,
    skip_meshing_if_exists: bool = True,
    ui_mode: str = "no_gui_or_graphics",
) -> Generator[PeriodicHillSolver, None, None]:
    """Open a long-lived Fluent session for repeated GEKO trials.

    The mesh is generated (or reused) once, Fluent is launched once, and
    the base setup (turbulence model, periodic, material, methods) is
    applied once. Each trial then only re-applies GEKO coefficients,
    re-initializes, iterates, and saves.

    Use as a context manager; Fluent is closed cleanly even if your BO
    loop crashes.

    Example
    -------
    >>> with open_session(BASE_CASE, MESH, "outputs") as session:
    ...     for trial_params in bo_trials:
    ...         trial_case = replace(BASE_CASE, **trial_params)
    ...         outputs = session.run_trial(trial_case)
    ...         loss = compute_loss(outputs["ascii"])

    Yields
    ------
    PeriodicHillSolver
        Live session. Call ``run_trial(trial_case)`` for each BO trial.
    """
    data_dir = Path(data_dir).resolve()

    # Mesh once
    mesh_generator = MeshGenerator(base_case, mesh, data_dir, ui_mode=ui_mode)
    if skip_meshing_if_exists and mesh_generator.mesh_path.is_file():
        print(f"[open_session] Reusing existing mesh: {mesh_generator.mesh_path}")
        mesh_path = mesh_generator.mesh_path
    else:
        print(f"[open_session] Generating mesh for case {base_case.case_id}")
        mesh_path = mesh_generator.generate()

    # Launch + base setup once, yield session, close on exit (even on error)
    session = PeriodicHillSolver(base_case, mesh_path, data_dir, ui_mode=ui_mode)
    session.start()
    try:
        yield session
    finally:
        session.close()


def run_geko_trial(
    geko_params: dict[str, float],
    session: PeriodicHillSolver,
    base_case: CaseConfig,
    *,
    reinitialize: bool = True,
    iter_count: int | None = None,
) -> dict[str, Path]:
    """Run one GEKO parameter trial against a live session.

    Convenience wrapper for BO loops. Builds a trial CaseConfig from
    ``base_case`` + ``geko_params`` and dispatches to ``session.run_trial``.

    Parameters
    ----------
    geko_params : dict
        Subset of ``{"geko_csep", "geko_cnw", "geko_cmix", "geko_cjet",
        "geko_ccorner"}`` mapping to numeric values. ``None`` means
        "use Fluent default".
    session : PeriodicHillSolver
        A live session, e.g. from ``open_session``.
    base_case : CaseConfig
        Baseline. GEKO fields here are overridden by ``geko_params``.
    reinitialize : bool, default True
        Re-initialize the flow before iterating (recommended for BO).
    iter_count : int, optional
        Override base_case.iter_count for this trial.

    Returns
    -------
    dict[str, Path]
        Output file paths, named per trial_case.case_id.
    """
    valid_keys = {
        "geko_csep", "geko_cnw", "geko_cmix", "geko_cjet", "geko_ccorner",
    }
    unknown = set(geko_params) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown GEKO keys: {unknown}. "
            f"Valid keys are: {sorted(valid_keys)}"
        )

    overrides = dict(geko_params)
    if iter_count is not None:
        overrides["iter_count"] = iter_count

    trial_case = replace(base_case, **overrides)
    return session.run_trial(trial_case, reinitialize=reinitialize)
