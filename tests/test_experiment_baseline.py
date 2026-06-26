from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geko_bayesopt.config import ParameterSpec
from geko_bayesopt.experiment import (
    _do_default_baseline,
    _replay_into_optimizer,
    _run_live_session,
    _run_per_trial,
    _should_run_default_baseline,
)
from geko_bayesopt.fluent.case_config import CaseConfig
from geko_bayesopt.types import RunResult


class _FlowCase:
    def __init__(self) -> None:
        self.case_config = CaseConfig(base_case_name="case")
        self.seen_parameters: dict[str, float] | None = None

    def build_run_result(
        self,
        *,
        run_id,
        parameters,
        ascii_path,
        cost_seconds,
    ) -> RunResult:
        self.seen_parameters = parameters
        return RunResult(
            run_id=run_id,
            parameters=parameters,
            grid_coords=np.empty((0, 2)),
            fields={},
            ascii_path=ascii_path,
            cost_seconds=cost_seconds,
        )


class _Solver:
    def __init__(self, ascii_path: Path) -> None:
        self.ascii_path = ascii_path
        self.trial_case = None

    def run_trial(self, trial_case):
        self.trial_case = trial_case
        return {"ascii": self.ascii_path}


class _Store:
    def __init__(self) -> None:
        self.saved = []

    def save_trial(self, result, score, *, trial_role="optimizer") -> None:
        self.saved.append((result, score, trial_role))


def test_default_baseline_does_not_use_an_optimizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = SimpleNamespace(
        parameters=[
            ParameterSpec(name="geko_csep", low=0.5, high=2.5),
            ParameterSpec(name="geko_cnw", low=0.1, high=1.0),
        ]
    )
    flow_case = _FlowCase()
    solver = _Solver(tmp_path / "case_geko_default.ascii")
    store = _Store()
    monkeypatch.setattr(
        "geko_bayesopt.experiment._cleanup_trial_outputs",
        lambda *args, **kwargs: None,
    )

    _do_default_baseline(
        cfg,
        flow_case,
        lambda result: 3.0,
        store,
        tmp_path,
        solver=solver,
    )

    assert solver.trial_case.geko_csep is None
    assert solver.trial_case.geko_cnw is None
    assert flow_case.seen_parameters == {
        "geko_csep": 1.75,
        "geko_cnw": 0.5,
    }
    assert store.saved[0][2] == "baseline"


def test_replay_only_requests_optimizer_trials() -> None:
    completed = SimpleNamespace(parameters={"geko_csep": 1.2}, score=0.5)

    class Store:
        def load_completed_trials(self, *, trial_role=None):
            assert trial_role == "optimizer"
            return [completed]

    class Optimizer:
        def __init__(self):
            self.observations = []

        def tell(self, x, y):
            self.observations.append((x, y))

    optimizer = Optimizer()
    parameters = [ParameterSpec(name="geko_csep", low=0.5, high=2.5)]

    replayed = _replay_into_optimizer(optimizer, Store(), parameters)

    assert replayed == 1
    assert optimizer.observations == [([1.2], 0.5)]


def test_missing_baseline_is_backfilled_for_existing_experiment() -> None:
    cfg = SimpleNamespace(evaluate_default_first=True)
    store = SimpleNamespace(has_baseline=lambda: False)

    assert _should_run_default_baseline(cfg, store) is True


def test_session_strategies_run_baseline_before_optimizer_trials(
    monkeypatch,
) -> None:
    events = []

    class SolverSession:
        created = 0

        def __init__(self, *args, **kwargs):
            type(self).created += 1
            self.session_id = type(self).created

        def __enter__(self):
            events.append(("enter", self.session_id))
            return self

        def __exit__(self, *args):
            events.append(("exit", self.session_id))

    monkeypatch.setattr(
        "geko_bayesopt.experiment.PeriodicHillSolver",
        SolverSession,
    )
    monkeypatch.setattr(
        "geko_bayesopt.experiment._do_default_baseline",
        lambda *args, solver, **kwargs: events.append(
            ("baseline", solver.session_id)
        ),
    )
    monkeypatch.setattr(
        "geko_bayesopt.experiment._do_one_trial",
        lambda iteration, *args, solver, **kwargs: events.append(
            ("trial", iteration, solver.session_id)
        ),
    )

    cfg = SimpleNamespace(
        optimizer=SimpleNamespace(stopping_criteria={"n_calls": 2})
    )
    flow_case = SimpleNamespace(case_config=object())

    _run_live_session(
        cfg,
        flow_case,
        None,
        None,
        object(),
        object(),
        object(),
        0,
        None,
        None,
        True,
    )
    assert events == [
        ("enter", 1),
        ("baseline", 1),
        ("trial", 0, 1),
        ("trial", 1, 1),
        ("exit", 1),
    ]

    events.clear()
    SolverSession.created = 0
    _run_per_trial(
        cfg,
        flow_case,
        None,
        None,
        object(),
        object(),
        object(),
        0,
        None,
        None,
        True,
    )
    assert events == [
        ("enter", 1),
        ("baseline", 1),
        ("exit", 1),
        ("enter", 2),
        ("trial", 0, 2),
        ("exit", 2),
        ("enter", 3),
        ("trial", 1, 3),
        ("exit", 3),
    ]
