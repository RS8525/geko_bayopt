"""Shared types for the objective subsystem."""

from __future__ import annotations

from typing import Callable

from ..types import RunResult


#: A loss function: takes a RunResult, returns a scalar to minimize.
LossFn = Callable[[RunResult], float]
