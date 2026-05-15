"""
Command-line entry point.

Installed as ``geko-opt`` by pyproject.toml. Usage::

    geko-opt run configs/periodic_hills_csep.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    # Imported lazily so ``geko-opt --help`` doesn't pull in pyfluent.
    from .experiment import run_experiment

    run_experiment(args.config, ui_mode=args.ui_mode)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Quick syntax check of a config without running anything."""
    from .config import ExperimentConfig

    cfg = ExperimentConfig.load(args.config)
    print(f"Config OK: experiment_id={cfg.experiment_id}")
    print(f"  Case: {cfg.case.kind}")
    print(f"  Parameters: {[p.name for p in cfg.parameters]}")
    print(f"  Objective: {cfg.objective.kind}")
    print(f"  Optimizer: {cfg.optimizer.kind} "
          f"(n_initial={cfg.optimizer.n_initial}, "
          f"n_iterations={cfg.optimizer.n_iterations})")
    print(f"  Session: {cfg.session_strategy}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geko-opt",
        description="Bayesian optimization of GEKO turbulence-model coefficients",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a BO experiment from a JSON config")
    p_run.add_argument("config", type=Path, help="Path to experiment JSON")
    p_run.add_argument(
        "--ui-mode",
        default="no_gui_or_graphics",
        choices=["gui", "hidden_gui", "no_gui", "no_gui_or_graphics"],
        help="PyFluent UI mode for Fluent launches",
    )
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="Validate a config without running")
    p_val.add_argument("config", type=Path, help="Path to experiment JSON")
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
