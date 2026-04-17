"""Loads and essential (Dirichlet) data attached to the FE model before assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EssentialBoundaryCondition:
    field_name: str
    node_ids: list[int] | tuple[int, ...]
    value: float = 0.0


@dataclass(frozen=True)
class ElementEdgeLineLoad:
    """Line traction on one edge of a quadrilateral (see element local edge numbering)."""

    element_id: int
    edge_id: int
    magnitude: float | Callable[[float, float], float]


@dataclass(frozen=True)
class ElementSurfaceLoad:
    """Uniform or pointwise pressure-like load over the element mid-surface."""

    element_id: int
    magnitude: float | Callable[[float, float], float]
