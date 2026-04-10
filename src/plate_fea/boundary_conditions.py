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
    element_id: int
    edge_id: int
    magnitude: float | Callable[[float, float], float]


@dataclass(frozen=True)
class ElementSurfaceLoad:
    element_id: int
    magnitude: float | Callable[[float, float], float]
