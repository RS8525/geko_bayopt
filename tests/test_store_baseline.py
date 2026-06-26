from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from geko_bayesopt.config import ParameterSpec
from geko_bayesopt.store import ResultStore, cleanup_non_best_case_files
from geko_bayesopt.types import RunResult


def _result(run_id: str, value: float) -> RunResult:
    return RunResult(
        run_id=run_id,
        parameters={"geko_csep": value},
        grid_coords=np.empty((0, 2)),
        fields={},
    )


def test_store_separates_baseline_and_optimizer_trials(tmp_path: Path) -> None:
    store = ResultStore(
        tmp_path,
        [ParameterSpec(name="geko_csep", low=0.5, high=2.5)],
    )

    store.save_trial(_result("default", 1.75), 2.0, trial_role="baseline")
    store.save_trial(_result("candidate", 1.2), 1.0)

    assert store.has_baseline()
    assert [trial.run_id for trial in store.load_completed_trials()] == [
        "default",
        "candidate",
    ]
    assert [
        trial.run_id
        for trial in store.load_completed_trials(trial_role="optimizer")
    ] == ["candidate"]
    assert store.best_trial(trial_role="optimizer").run_id == "candidate"


def test_store_migrates_legacy_metadata_before_append(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "run_id,score,cost_seconds,converged,geko_csep\n"
        "old,1.5,2.0,true,1.2\n",
        encoding="utf-8",
    )
    store = ResultStore(
        tmp_path,
        [ParameterSpec(name="geko_csep", low=0.5, high=2.5)],
    )

    store.save_trial(_result("default", 1.75), 2.0, trial_role="baseline")

    with metadata.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["trial_role"] == "optimizer"
    assert rows[1]["trial_role"] == "baseline"


def test_cleanup_always_keeps_baseline_ascii(tmp_path: Path) -> None:
    for name in (
        "baseline.ascii",
        "best.ascii",
        "other.ascii",
        "best_solved.cas.h5",
        "other_solved.cas.h5",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    cleanup_non_best_case_files(
        tmp_path,
        "best",
        keep_only_best_case_files=True,
        keep_all_ascii_files=False,
        protected_ascii_run_ids={"baseline"},
    )

    assert (tmp_path / "baseline.ascii").exists()
    assert (tmp_path / "best.ascii").exists()
    assert not (tmp_path / "other.ascii").exists()
    assert (tmp_path / "best_solved.cas.h5").exists()
    assert not (tmp_path / "other_solved.cas.h5").exists()


def test_cleanup_can_keep_all_ascii_independently(tmp_path: Path) -> None:
    for name in ("first.ascii", "second.ascii", "second_solved.cas.h5"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    cleanup_non_best_case_files(
        tmp_path,
        "first",
        keep_only_best_case_files=False,
        keep_all_ascii_files=True,
    )

    assert all(path.exists() for path in tmp_path.iterdir())
