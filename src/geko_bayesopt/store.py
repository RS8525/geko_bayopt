"""
Append-only metadata.csv plus optimizer pickle for crash recovery.

Two files per experiment:

    results/experiments/<experiment_id>/metadata.csv
    results/experiments/<experiment_id>/optimizer.pkl

CSV columns: run_id, trial_role, score, cost_seconds, converged, then one
column per parameter. ``trial_role`` is either ``baseline`` or ``optimizer``.
Numbers are written with full precision. The file is created with a header on
the first call; subsequent calls append one row each.

Optimizer state is pickled after each completed optimizer trial. On restart,
only rows with ``trial_role=optimizer`` are replayed into a fresh optimizer via
``tell()``. Baseline rows remain independent.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ParameterSpec
from .types import RunResult


@dataclass
class CompletedTrial:
    """One row from metadata.csv, hydrated for replaying into an optimizer."""
    run_id: str
    trial_role: str
    parameters: dict[str, float]
    score: float
    cost_seconds: float
    converged: bool


class ResultStore:
    """Persists each completed trial to disk immediately.

    Designed for crash recovery: a crash after ``save_trial`` but before
    the optimizer is updated is fully recoverable on restart. The reverse
    is not — so the experiment loop must call save_trial BEFORE telling
    the optimizer.
    """

    METADATA_FILENAME = "metadata.csv"
    OPTIMIZER_FILENAME = "optimizer.pkl"

    def __init__(self, results_dir: str | Path, parameters: list[ParameterSpec]):
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.parameters = parameters

        self.metadata_path = self.results_dir / self.METADATA_FILENAME
        self.optimizer_path = self.results_dir / self.OPTIMIZER_FILENAME

        # Column order is locked once; new columns at the end would
        # be additive but breaking-change-free.
        self._columns = (
            ["run_id", "trial_role", "score", "cost_seconds", "converged"]
            + [p.name for p in parameters]
        )

    # ------------------------------------------------------------------ #
    # Writing                                                            #
    # ------------------------------------------------------------------ #

    def save_trial(
        self,
        run_result: RunResult,
        score: float,
        *,
        trial_role: str = "optimizer",
    ) -> None:
        """Append one row to metadata.csv. Header written on first call."""
        if trial_role not in {"baseline", "optimizer"}:
            raise ValueError(
                "trial_role must be either 'baseline' or 'optimizer', "
                f"got {trial_role!r}."
            )
        self._migrate_legacy_metadata()
        is_new = not self.metadata_path.exists()
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(self._columns)
            row = [
                run_result.run_id,
                trial_role,
                f"{score:.10g}",
                f"{run_result.cost_seconds:.4f}",
                "true" if run_result.converged else "false",
            ]
            for p in self.parameters:
                value = run_result.parameters.get(p.name, "")
                row.append(f"{value:.10g}" if value != "" else "")
            writer.writerow(row)

    def _migrate_legacy_metadata(self) -> None:
        """Add ``trial_role=optimizer`` to metadata created before roles existed."""
        if not self.metadata_path.exists():
            return

        with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "trial_role" in reader.fieldnames:
                return
            rows = list(reader)

        temp_path = self.metadata_path.with_suffix(".csv.tmp")
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            for row in rows:
                migrated = dict(row)
                migrated["trial_role"] = "optimizer"
                writer.writerow(
                    {
                        column: migrated.get(column, "")
                        for column in self._columns
                    }
                )
        temp_path.replace(self.metadata_path)

    def save_optimizer(self, optimizer: Any) -> None:
        """Pickle the optimizer state. Overwrites any previous file."""
        with open(self.optimizer_path, "wb") as f:
            pickle.dump(optimizer, f)

    # ------------------------------------------------------------------ #
    # Loading (for resume)                                               #
    # ------------------------------------------------------------------ #

    def load_completed_trials(
        self,
        *,
        trial_role: str | None = None,
    ) -> list[CompletedTrial]:
        """Read metadata.csv and return completed trials.

        Pass ``trial_role`` to return only baseline or optimizer rows. Metadata
        created before roles existed is interpreted as optimizer-only data.

        Returns an empty list if the file doesn't exist yet (i.e. this
        is a fresh experiment).
        """
        if not self.metadata_path.exists():
            return []

        trials: list[CompletedTrial] = []
        param_names = [p.name for p in self.parameters]
        with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                params = {
                    name: float(row[name])
                    for name in param_names
                    if row.get(name) not in (None, "")
                }
                loaded_role = row.get("trial_role") or "optimizer"
                if trial_role is not None and loaded_role != trial_role:
                    continue
                trials.append(
                    CompletedTrial(
                        run_id=row["run_id"],
                        trial_role=loaded_role,
                        parameters=params,
                        score=float(row["score"]),
                        cost_seconds=float(row["cost_seconds"]),
                        converged=row["converged"].lower() == "true",
                    )
                )
        return trials

    def has_baseline(self) -> bool:
        """Return whether a completed independent baseline is already stored."""
        return bool(self.load_completed_trials(trial_role="baseline"))

    def load_optimizer(self) -> Any | None:
        """Load the pickled optimizer state, or None if no checkpoint exists."""
        if not self.optimizer_path.exists():
            return None
        with open(self.optimizer_path, "rb") as f:
            return pickle.load(f)

    def best_trial(
        self,
        *,
        trial_role: str | None = None,
    ) -> CompletedTrial | None:
        """Return the lowest-scoring matching trial, or None if empty.

        Used to identify which trial's .cas/.dat files should be kept on
        disk when ``keep_only_best_case_files`` is enabled.
        """
        trials = self.load_completed_trials(trial_role=trial_role)
        if not trials:
            return None
        return min(trials, key=lambda t: t.score)


def cleanup_non_best_case_files(
    fluent_work_dir: Path,
    best_run_id: str | None,
    *,
    keep_only_best_case_files: bool = True,
    keep_all_ascii_files: bool = False,
    protected_ascii_run_ids: set[str] | None = None,
) -> None:
    """Apply case/data and ASCII retention policies.

    When ``keep_only_best_case_files`` is true, only the best solved case/data
    files are retained and all initialization case files are deleted. When
    ``keep_all_ascii_files`` is false, only the best ASCII and explicitly
    protected ASCII run IDs are retained. Baseline run IDs are passed as
    protected IDs by the experiment loop.

    Parameters
    ----------
    fluent_work_dir : Path
        Directory containing the per-trial Fluent outputs.
    best_run_id : str | None
        Run ID of the current-best trial. If case-file cleanup is enabled, its
        solved case/data files are retained. If ASCII cleanup is enabled, its
        ASCII is retained in addition to protected IDs.
    """
    if not fluent_work_dir.is_dir():
        return

    keep_solved_cas = (
        f"{best_run_id}_solved.cas.h5" if best_run_id else None
    )
    keep_solved_dat = (
        f"{best_run_id}_solved.dat.h5" if best_run_id else None
    )

    protected_ascii_run_ids = protected_ascii_run_ids or set()
    deleted = 0
    for path in fluent_work_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name

        if keep_only_best_case_files and name.endswith("_init.cas.h5"):
            path.unlink()
            deleted += 1
            continue

        if keep_only_best_case_files and (
            name == keep_solved_cas or name == keep_solved_dat
        ):
            continue

        if keep_only_best_case_files and (
            name.endswith("_solved.cas.h5") or name.endswith("_solved.dat.h5")
        ):
            path.unlink()
            deleted += 1
            continue

        if path.suffix == ".ascii" and not keep_all_ascii_files:
            keep_ascii_ids = set(protected_ascii_run_ids)
            if best_run_id is not None:
                keep_ascii_ids.add(best_run_id.strip())
            if path.stem not in keep_ascii_ids:
                path.unlink()
                deleted += 1

    if deleted > 0:
        kept = best_run_id if best_run_id else "(none)"
        print(
            f"[store] Cleaned {deleted} stale trial files "
            f"(kept best={kept}, keep_all_ascii={keep_all_ascii_files})"
        )
