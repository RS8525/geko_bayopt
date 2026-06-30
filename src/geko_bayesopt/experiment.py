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
from dataclasses import replace
from pathlib import Path

from .cases import FlowCase, build_flow_case
from .config import ExperimentConfig
from .fluent.mesh_generator import MeshGenerator
from .fluent.solver import PeriodicHillSolver
from .geko_defaults import GEKO_DEFAULTS, defaults_for_parameters
from .objective import build_loss_fn
from .optimizer import build_optimizer, vector_to_params, params_to_vector
from .store import ResultStore, cleanup_non_best_case_files


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
    """Tell the optimizer about every completed optimizer trial in the store.

    Returns the number of trials replayed.
    """
    completed = store.load_completed_trials(trial_role="optimizer")
    if not completed:
        return 0

    print(f"[experiment] Resuming: replaying {len(completed)} completed trials...")
    for trial in completed:
        x = params_to_vector(trial.parameters, parameters)
        optimizer.tell(x, trial.score)
    return len(completed)


def _should_run_default_baseline(
    cfg: ExperimentConfig,
    store: ResultStore,
) -> bool:
    """Return whether this invocation should execute or backfill the baseline."""
    return cfg.evaluate_default_first and not store.has_baseline()


def _baseline_run_id(flow_case: FlowCase) -> str:
    """Return a deterministic filename-safe ID for the default GEKO run."""
    return f"{flow_case.case_config.case_id}_geko_default"


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
        Defaults to the nearest ancestor directory containing
        ``pyproject.toml`` or ``.git``.
    ui_mode : str
        Forwarded to PyFluent launches.
    """
    config_path = Path(config_path).resolve()
    cfg = ExperimentConfig.load(config_path)

    if repo_root is None:
        # Walk up from the config file to find the repo root (contains pyproject.toml or .git).
        # This is robust to configs being nested at any depth under configs/.
        candidate = config_path.parent
        while candidate != candidate.parent:
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                repo_root = candidate
                break
            candidate = candidate.parent
        else:
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
    n_calls = cfg.optimizer.stopping_criteria.get("n_calls", 32)
    n_completed = _replay_into_optimizer(optimizer, store, cfg.parameters)
    run_default_baseline = _should_run_default_baseline(cfg, store)
    n_remaining = n_calls - n_completed
    if n_remaining <= 0 and not run_default_baseline:
        print(f"[experiment] All {n_calls} trials already completed. Nothing to do.")
        return

    # ---- Ensure mesh exists before any solver launches ----
    mesh_path = _ensure_mesh(flow_case, fluent_work_dir, repo_root, ui_mode)
    time.sleep(15)

    # Pull residual criteria from config (None if not set).
    residual_criteria = cfg.residual_criteria

    # ---- Main loop ----
    if cfg.session_strategy == "live":
        _run_live_session(
            cfg, flow_case, mesh_path, fluent_work_dir,
            optimizer, loss_fn, store, n_completed, ui_mode, residual_criteria,
            run_default_baseline,
        )
    elif cfg.session_strategy == "per_trial":
        _run_per_trial(
            cfg, flow_case, mesh_path, fluent_work_dir,
            optimizer, loss_fn, store, n_completed, ui_mode, residual_criteria,
            run_default_baseline,
        )
    else:  # pragma: no cover -- pydantic enforces the literal
        raise ValueError(f"Unknown session_strategy: {cfg.session_strategy}")

    print(f"[experiment] Done: {cfg.experiment_id}")


# --------------------------------------------------------------------- #
# Session strategies                                                    #
# --------------------------------------------------------------------- #

def _run_live_session(
    cfg, flow_case, mesh_path, fluent_work_dir,
    optimizer, loss_fn, store, n_completed, ui_mode, residual_criteria,
    run_default_baseline,
) -> None:
    """One Fluent process, reused for all trials. Faster, slightly riskier."""
    solver = PeriodicHillSolver(
        flow_case.case_config, mesh_path, fluent_work_dir,
        ui_mode=ui_mode, flow_case=flow_case, residual_criteria=residual_criteria
    )
    with solver:
        if run_default_baseline:
            _do_default_baseline(
                cfg, flow_case, loss_fn, store, fluent_work_dir, solver=solver
            )
        for i in range(n_completed, cfg.optimizer.stopping_criteria.get("n_calls", 32)):
            _do_one_trial(
                i, cfg, flow_case, optimizer, loss_fn, store,
                fluent_work_dir, solver=solver,
            )
            if hasattr(optimizer, "should_stop") and optimizer.should_stop():
                print(f"[experiment] Early stop: epsilon convergence after {i + 1} trials.")
                break


def _run_per_trial(
    cfg, flow_case, mesh_path, fluent_work_dir,
    optimizer, loss_fn, store, n_completed, ui_mode, residual_criteria,
    run_default_baseline,
) -> None:
    """Launch + exit Fluent per trial. Safer on Student licenses."""
    if run_default_baseline:
        solver = PeriodicHillSolver(
            flow_case.case_config, mesh_path, fluent_work_dir,
            ui_mode=ui_mode, flow_case=flow_case, residual_criteria=residual_criteria
        )
        with solver:
            _do_default_baseline(
                cfg, flow_case, loss_fn, store, fluent_work_dir, solver=solver
            )

    for i in range(n_completed, cfg.optimizer.stopping_criteria.get("n_calls", 32)):
        solver = PeriodicHillSolver(
            flow_case.case_config, mesh_path, fluent_work_dir,
            ui_mode=ui_mode, flow_case=flow_case, residual_criteria=residual_criteria
        )
        with solver:
            _do_one_trial(
                i, cfg, flow_case, optimizer, loss_fn, store,
                fluent_work_dir, solver=solver,
            )
        if hasattr(optimizer, "should_stop") and optimizer.should_stop():
            print(f"[experiment] Early stop: epsilon convergence after {i + 1} trials.")
            break


def _do_default_baseline(
    cfg,
    flow_case,
    loss_fn,
    store,
    fluent_work_dir: Path,
    *,
    solver,
) -> None:
    """Run, score, and persist Fluent's defaults without touching the optimizer."""
    t_start = time.time()
    parameters = defaults_for_parameters(cfg.parameters)
    run_id = _baseline_run_id(flow_case)

    # Keep coefficient overrides as None so Fluent uses its configured defaults.
    baseline_case = replace(
        flow_case.case_config,
        base_case_name=run_id,
        **{name: None for name in GEKO_DEFAULTS},
    )
    print(f"\n[experiment] Baseline: GEKO defaults {parameters}")
    outputs = solver.run_trial(baseline_case)

    cost = time.time() - t_start
    run_result = flow_case.build_run_result(
        run_id=run_id,
        parameters=parameters,
        ascii_path=outputs["ascii"],
        cost_seconds=cost,
    )
    score = loss_fn(run_result)
    print(f"[experiment] Baseline score = {score:.6g} (cost {cost:.1f}s)")

    store.save_trial(run_result, score, trial_role="baseline")
    _cleanup_trial_outputs(cfg, store, fluent_work_dir)


def _do_one_trial(
    iteration: int,
    cfg,
    flow_case,
    optimizer,
    loss_fn,
    store,
    fluent_work_dir: Path,
    *,
    solver,
) -> None:
    """Ask -> run -> score -> save -> tell -> cleanup.

    Order matters: save BEFORE tell so a crash is recoverable.
    """
    t_start = time.time()

    # 1. Ask
    n_calls = cfg.optimizer.stopping_criteria.get("n_calls", 32)
    x = optimizer.ask()
    params = vector_to_params(x, cfg.parameters)
    print(f"\n[experiment] Trial {iteration + 1}/{n_calls}: {params}")

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

    # 6. Apply case/data and ASCII retention policies.
    _cleanup_trial_outputs(cfg, store, fluent_work_dir)


def _cleanup_trial_outputs(
    cfg: ExperimentConfig,
    store: ResultStore,
    fluent_work_dir: Path,
) -> None:
    """Apply configured retention while always protecting baseline ASCII."""
    best = store.best_trial(trial_role="optimizer")
    if best is None:
        best = store.best_trial(trial_role="baseline")
    best_run_id = best.run_id if best is not None else None
    baseline_run_ids = {
        trial.run_id
        for trial in store.load_completed_trials(trial_role="baseline")
    }
    cleanup_non_best_case_files(
        fluent_work_dir,
        best_run_id,
        keep_only_best_case_files=cfg.keep_only_best_case_files,
        keep_all_ascii_files=cfg.keep_all_ascii_files,
        protected_ascii_run_ids=baseline_run_ids,
    )
