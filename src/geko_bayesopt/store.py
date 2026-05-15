"""
Append-only metadata.csv plus optimizer pickle for crash recovery.

Two files per experiment:

    results/experiments/<experiment_id>/metadata.csv
    results/experiments/<experiment_id>/optimizer.pkl

CSV columns: run_id, score, cost_seconds, converged, then one column per
parameter. Numbers are written with full precision. The file is created
with a header on the first call; subsequent calls append one row each.

Optimizer state is pickled after each completed trial. On restart,
``load_completed_trials`` replays everything into a fresh optimizer
via ``tell()`` so the search continues exactly where it left off.
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
            ["run_id", "score", "cost_seconds", "converged"]
            + [p.name for p in parameters]
        )

    # ------------------------------------------------------------------ #
    # Writing                                                            #
    # ------------------------------------------------------------------ #

    def save_trial(self, run_result: RunResult, score: float) -> None:
        """Append one row to metadata.csv. Header written on first call."""
        is_new = not self.metadata_path.exists()
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(self._columns)
            row = [
                run_result.run_id,
                f"{score:.10g}",
                f"{run_result.cost_seconds:.4f}",
                "true" if run_result.converged else "false",
            ]
            for p in self.parameters:
                value = run_result.parameters.get(p.name, "")
                row.append(f"{value:.10g}" if value != "" else "")
            writer.writerow(row)

    def save_optimizer(self, optimizer: Any) -> None:
        """Pickle the optimizer state. Overwrites any previous file."""
        with open(self.optimizer_path, "wb") as f:
            pickle.dump(optimizer, f)

    # ------------------------------------------------------------------ #
    # Loading (for resume)                                               #
    # ------------------------------------------------------------------ #

    def load_completed_trials(self) -> list[CompletedTrial]:
        """Read metadata.csv and return one CompletedTrial per row.

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
                trials.append(
                    CompletedTrial(
                        run_id=row["run_id"],
                        parameters=params,
                        score=float(row["score"]),
                        cost_seconds=float(row["cost_seconds"]),
                        converged=row["converged"].lower() == "true",
                    )
                )
        return trials

    def load_optimizer(self) -> Any | None:
        """Load the pickled optimizer state, or None if no checkpoint exists."""
        if not self.optimizer_path.exists():
            return None
        with open(self.optimizer_path, "rb") as f:
            return pickle.load(f)
