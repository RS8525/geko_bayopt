"""
Bayesian optimization loop.

Composition only — every concrete behaviour lives in a dispatched module:
    - flow case (BCs, DNS loading) -> ``cases``
    - loss function -> ``objective``
    - optimizer -> ``optimizer``
    - Fluent automation -> ``fluent``
    - persistence -> ``store``

This file knows how to glue them together. Adding a new loss, optimizer,
or flow case requires zero changes here.
"""

from __future__ import annotations

import time
from pathlib import Path

from .cases import FlowCase, build_flow_case
from .config import ExperimentConfig
from .fluent.mesh_generator import MeshGenerator
from .fluent.solver import PeriodicHillSolver
from .objective import build_loss_fn
from .optimizer import build_optimizer, vector_to_params, params_to_vector
from .store import ResultStore


def _resolve_paths(cfg: ExperimentConfig, root: Path) -> tuple[Path, Path, Path]:
    """Resolve the three runtime directories from config + repo root.

    Returns (fluent_work_dir, results_dir, dns_path).
    """
    fluent_work_dir = (
        Path(cfg.fluent_work_dir).resolve() if cfg.fluent_work_dir
        else (root / "results" / "fluent" / cfg.experiment_id).resolve()
    )
    fluent_work_dir.mkdir(parents=True, exist_ok=True)

    results_dir = (
        Path(cfg.results_dir).resolve() if cfg.results_dir
        else (root / "results" / "experiments" / cfg.experiment_id).resolve()
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # DNS path: case must declare ``dns_path`` in its options.
    dns_path_str = cfg.case.options.get("dns_path")
    if dns_path_str is None:
        raise ValueError(
            "case.options.dns_path is required (path to DNS reference data)."
        )
    dns_path = Path(dns_path_str)
    if not dns_path.is_absolute():
        dns_path = (root / dns_path).resolve()

    return fluent_work_dir, results_dir, dns_path


def _ensure_mesh(
    flow_case: FlowCase,
    fluent_work_dir: Path,
    repo_root: Path,
    ui_mode: str,
) -> Path:
    """Generate the mesh if missing, otherwise reuse it.

    The geometry file is read from ``flow_case.geometry_path`` (if set),
    resolved relative to ``repo_root`` when not absolute. The output
    .msh.h5 is written into ``fluent_work_dir``.
    """
    # Resolve the optional geometry path against repo_root.
    geom_path: Path | None = flow_case.geometry_path
    if geom_path is not None and not geom_path.is_absolute():
        geom_path = (repo_root / geom_path).resolve()

    generator = MeshGenerator(
        flow_case.case_config, flow_case.mesh_config,
        data_dir=fluent_work_dir, ui_mode=ui_mode,
        geometry_path=geom_path,
    )
    if generator.mesh_path.is_file():
        print(f"[experiment] Reusing mesh: {generator.mesh_path}")
        return generator.mesh_path
    if geom_path is not None and not geom_path.is_file():
        raise FileNotFoundError(
            f"Geometry file not found at {geom_path}. "
            "Update case.options.geometry_path in the JSON config."
        )
    print(f"[experiment] Generating mesh from {geom_path}...")
    return generator.generate()


def _replay_into_optimizer(optimizer, store, parameters) -> int:
    """Tell the optimizer about every completed trial in the store.

    Returns the number of trials replayed.
    """
    completed = store.load_completed_trials()
    if not completed:
        return 0

    print(f"[experiment] Resuming: replaying {len(completed)} completed trials...")
    for trial in completed:
        x = params_to_vector(trial.parameters, parameters)
        optimizer.tell(x, trial.score)
    return len(completed)


def run_experiment(
    config_path: str | Path,
    *,
    repo_root: Path | None = None,
    ui_mode: str = "no_gui_or_graphics",
) -> None:
    """Run the BO sweep defined in a JSON config.

    Parameters
    ----------
    config_path : str or Path
        Path to the experiment JSON.
    repo_root : Path, optional
        Repository root, used to resolve default output directories
        and the DNS path if it's given as a relative path.
        Defaults to the JSON file's grandparent (assumes
        ``<root>/configs/<name>.json``).
    ui_mode : str
        Forwarded to PyFluent launches.
    """
    config_path = Path(config_path).resolve()
    cfg = ExperimentConfig.load(config_path)

    if repo_root is None:
        # configs/<name>.json -> repo root is two levels up
        repo_root = config_path.parent.parent

    print(f"[experiment] Starting: {cfg.experiment_id}")
    print(f"[experiment] Config: {config_path}")
    print(f"[experiment] Repo root: {repo_root}")

    # ---- Resolve paths ----
    fluent_work_dir, results_dir, dns_path = _resolve_paths(cfg, repo_root)
    print(f"[experiment] Fluent work dir: {fluent_work_dir}")
    print(f"[experiment] Results dir:     {results_dir}")
    print(f"[experiment] DNS data:        {dns_path}")

    # ---- Build the three swappable components ----
    flow_case = build_flow_case(cfg.case, cfg.mesh)
    dns_coords, dns_fields = flow_case.load_dns(dns_path)
    loss_fn = build_loss_fn(cfg.objective, dns_coords, dns_fields)
    optimizer = build_optimizer(cfg.optimizer, cfg.parameters)
    store = ResultStore(results_dir, cfg.parameters)

    # ---- Resume from prior runs, if any ----
    n_completed = _replay_into_optimizer(optimizer, store, cfg.parameters)
    n_remaining = cfg.optimizer.n_iterations - n_completed
    if n_remaining <= 0:
        print(
            f"[experiment] All {cfg.optimizer.n_iterations} trials already completed. "
            "Nothing to do."
        )
        return

    # ---- Ensure mesh exists before any solver launches ----
    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, ui_mode)
    time.sleep(15)

    # ---- Main loop ----
    if cfg.session_strategy == "live":
        _run_live_session(
            cfg, flow_case, mesh_path, fluent_work_dir,
            optimizer, loss_fn, store, n_completed, ui_mode,
        )
    elif cfg.session_strategy == "per_trial":
        _run_per_trial(
            cfg, flow_case, mesh_path, fluent_work_dir,
            optimizer, loss_fn, store, n_completed, ui_mode,
        )
    else:  # pragma: no cover -- pydantic enforces the literal
        raise ValueError(f"Unknown session_strategy: {cfg.session_strategy}")

    print(f"[experiment] Done: {cfg.experiment_id}")


# --------------------------------------------------------------------- #
# Session strategies                                                    #
# --------------------------------------------------------------------- #

def _run_live_session(
    cfg, flow_case, mesh_path, fluent_work_dir,
    optimizer, loss_fn, store, n_completed, ui_mode,
) -> None:
    """One Fluent process, reused for all trials. Faster, slightly riskier."""
    solver = PeriodicHillSolver(
        flow_case.case_config, mesh_path, fluent_work_dir,
        ui_mode=ui_mode, flow_case=flow_case,
    )
    with solver:
        for i in range(n_completed, cfg.optimizer.n_iterations):
            _do_one_trial(i, cfg, flow_case, optimizer, loss_fn, store, solver=solver)


def _run_per_trial(
    cfg, flow_case, mesh_path, fluent_work_dir,
    optimizer, loss_fn, store, n_completed, ui_mode,
) -> None:
    """Launch + exit Fluent per trial. Safer on Student licenses."""
    for i in range(n_completed, cfg.optimizer.n_iterations):
        solver = PeriodicHillSolver(
            flow_case.case_config, mesh_path, fluent_work_dir,
            ui_mode=ui_mode, flow_case=flow_case,
        )
        with solver:
            _do_one_trial(i, cfg, flow_case, optimizer, loss_fn, store, solver=solver)


def _do_one_trial(
    iteration: int,
    cfg,
    flow_case,
    optimizer,
    loss_fn,
    store,
    *,
    solver,
) -> None:
    """Ask -> run -> score -> save -> tell. Order matters: save BEFORE tell."""
    t_start = time.time()

    # 1. Ask
    x = optimizer.ask()
    params = vector_to_params(x, cfg.parameters)
    print(
        f"\n[experiment] Trial {iteration + 1}/{cfg.optimizer.n_iterations}: "
        f"{params}"
    )

    # 2. Run
    trial_case = flow_case.make_trial_case(params)
    outputs = solver.run_trial(trial_case)

    # 3. Score
    cost = time.time() - t_start
    run_result = flow_case.build_run_result(
        run_id=trial_case.case_id,
        parameters=params,
        ascii_path=outputs["ascii"],
        cost_seconds=cost,
    )
    score = loss_fn(run_result)
    print(f"[experiment] Trial {iteration + 1} score = {score:.6g} (cost {cost:.1f}s)")

    # 4. Save (BEFORE tell -- recovers cleanly on crash)
    store.save_trial(run_result, score)

    # 5. Tell
    optimizer.tell(x, score)
    store.save_optimizer(optimizer)
