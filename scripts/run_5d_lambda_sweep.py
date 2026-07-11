"""
Run the 5D BO lambda-sweep configs under configs/optimizer_comparison_configs/5D/.

For each config this runs exactly the command you'd type by hand:

    geko-opt run <config>

so it must be launched from a terminal where `geko-opt` is already on PATH
(e.g. after `conda activate BayOpt_clean`). It does not guess an interpreter
path itself -- whatever `geko-opt` resolves to in your shell is what runs.

Launch this script itself with the ``BayOpt_clean`` python (or an activated
``BayOpt_clean`` shell's plain ``python``) -- NOT ``.venv\\Scripts\\python.exe``.
The repo-local ``.venv`` is a ``uv``-provisioned interpreter, and spawning
subprocesses from it poisons the child's Python environment (the geko-opt
subprocess fails with an unrelated ``threading``/``_thread`` AttributeError
from a mismatched stdlib, even though the same geko-opt.exe runs fine when
launched directly from a shell).

Each config launches a REAL Fluent CFD sweep. This is slow (many trials x
GEKO solver iterations per config) and requires a working PyFluent/Fluent
license. Runs are resumable: ``run_experiment`` replays completed trials
from ``results_dir/metadata.csv`` on start, so re-running this script after
an interruption picks up where each experiment left off instead of
restarting from scratch.

This script runs the lambda 0.05 / 0.15 / 0.25 / 0.45 configs. The 0.35
config is intentionally excluded because it is being run separately.

Usage::

    geko-opt run configs/optimizer_comparison_configs/5D/bo_5d_ph2800_lambda0.05.json  # what one iteration does, by hand
    python scripts/run_5d_lambda_sweep.py
    python scripts/run_5d_lambda_sweep.py --dry-run
    python scripts/run_5d_lambda_sweep.py --validate-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "optimizer_comparison_configs" / "5D"

# The 0.35 config is intentionally omitted (run separately).
CONFIG_NAMES = [
    "bo_5d_ph2800_lambda0.05.json",
    "bo_5d_ph2800_lambda0.15.json",
    "bo_5d_ph2800_lambda0.25.json",
    "bo_5d_ph2800_lambda0.45.json",
]


def discover_configs() -> list[Path]:
    return [CONFIG_DIR / name for name in CONFIG_NAMES]


def run_one(geko_opt: str, config: Path, *, ui_mode: str, validate_only: bool) -> bool:
    cmd = [geko_opt] + (["validate", str(config)] if validate_only
                        else ["run", str(config), "--ui-mode", ui_mode])
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ui-mode", default="no_gui_or_graphics",
                         choices=["gui", "hidden_gui", "no_gui", "no_gui_or_graphics"],
                         help="Forwarded to each geko-opt run (PyFluent UI mode)")
    parser.add_argument("--dry-run", action="store_true",
                         help="List the configs that would run, without running them")
    parser.add_argument("--validate-only", action="store_true",
                         help="Run `geko-opt validate` on each config instead of `run` (no Fluent launched)")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="Abort the batch on the first failed config (default: continue to the next one)")
    args = parser.parse_args()

    geko_opt = shutil.which("geko-opt")
    if geko_opt is None:
        print("[run-5d] `geko-opt` is not on PATH in this shell.")
        print("[run-5d] Activate the environment it's installed in first, e.g.:")
        print("[run-5d]   conda activate BayOpt_clean")
        return 1

    configs = discover_configs()
    missing = [c for c in configs if not c.is_file()]
    if missing:
        print("[run-5d] The following expected config(s) are missing:")
        for c in missing:
            print(f"  - {c.relative_to(REPO_ROOT)}")
        return 1

    print(f"[run-5d] Using: {geko_opt}")
    print(f"[run-5d] {len(configs)} config(s) queued:")
    for c in configs:
        print(f"  - {c.relative_to(REPO_ROOT)}")

    if args.dry_run:
        return 0

    failures: list[Path] = []
    t_batch = time.time()
    for i, config in enumerate(configs, 1):
        rel = config.relative_to(REPO_ROOT)
        print(f"\n{'=' * 70}\n[run-5d] ({i}/{len(configs)}) {rel}\n{'=' * 70}")
        t0 = time.time()
        ok = run_one(geko_opt, config, ui_mode=args.ui_mode, validate_only=args.validate_only)
        elapsed = time.time() - t0
        if ok:
            print(f"[run-5d] OK ({elapsed / 60:.1f} min): {rel}")
        else:
            failures.append(config)
            print(f"[run-5d] FAILED ({elapsed / 60:.1f} min): {rel}")
            if args.stop_on_error:
                break

    total_elapsed = time.time() - t_batch
    print(f"\n[run-5d] Finished in {total_elapsed / 60:.1f} min.")
    if failures:
        print(f"[run-5d] {len(failures)} failure(s):")
        for c in failures:
            print(f"  - {c.relative_to(REPO_ROOT)}")
        return 1

    print("[run-5d] All configs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
