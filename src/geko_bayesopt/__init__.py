"""
geko_bayesopt — Bayesian optimization of GEKO turbulence-model coefficients.

Entry points:
    cli:run_experiment   -- CLI wrapper (``geko-opt run config.json``)
    experiment           -- main BO loop
    config               -- Pydantic schemas for experiment JSONs
    types                -- shared dataclasses (RunResult)
"""

from .types import RunResult

__all__ = ["RunResult"]
