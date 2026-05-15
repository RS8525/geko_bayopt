"""
Generic Fluent automation: meshing, solving, ASCII extraction.

Submodules import ``ansys.fluent.core`` directly, so this package's
``__init__`` is kept empty to avoid forcing that import on consumers
who only need ``extract`` (which has no pyfluent dependency).

Import what you need directly::

    from geko_bayesopt.fluent.extract import build_run_result   # no pyfluent
    from geko_bayesopt.fluent.solver import PeriodicHillSolver  # requires pyfluent
"""

