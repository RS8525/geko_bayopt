"""
Flow-case registry.

Add a new flow case by:
    1. Creating a subdirectory with a ``case.py`` containing your
       ``FlowCase`` subclass.
    2. Importing it here.
    3. Adding it to ``_REGISTRY`` keyed by its ``case_id``.
    4. Adding its ``case_id`` to the ``Literal`` in ``config.CaseSection``.
"""

from __future__ import annotations

from typing import Any

from ..config import CaseSection, MeshSection
from ..fluent.mesh_config import MeshConfig
from .base import FlowCase
from .periodic_hills import PeriodicHillsCase


# Registry: case_id -> FlowCase subclass.
_REGISTRY: dict[str, type[FlowCase]] = {
    PeriodicHillsCase.case_id: PeriodicHillsCase,
    # Add new cases here:
    # ForwardFacingStepCase.case_id: ForwardFacingStepCase,
}


def build_flow_case(case_section: CaseSection, mesh_section: MeshSection) -> FlowCase:
    """Dispatch on ``case_section.kind`` and instantiate the right class.

    Parameters
    ----------
    case_section : CaseSection
        The ``case`` block from the experiment JSON.
    mesh_section : MeshSection
        The ``mesh`` block from the experiment JSON (or defaults).

    Returns
    -------
    FlowCase
        An instantiated case object, ready to drive the solver.
    """
    cls = _REGISTRY.get(case_section.kind)
    if cls is None:
        valid = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown case kind: {case_section.kind!r}. "
            f"Valid kinds: {valid}"
        )

    # Build a MeshConfig from the mesh section (Pydantic model -> dataclass).
    mesh_config = MeshConfig(**mesh_section.model_dump())

    return cls(options=case_section.options, mesh_config=mesh_config)


__all__ = ["FlowCase", "PeriodicHillsCase", "build_flow_case"]
