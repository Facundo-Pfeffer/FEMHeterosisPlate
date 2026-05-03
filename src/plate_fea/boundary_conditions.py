"""Boundary condition and load dataclasses attached to a PlateModel before assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EssentialBoundaryCondition:
    """
    Prescribe a fixed value for one displacement or rotation field at a set of nodes.

    Args:
        field_name: Which DOF to constrain — one of "w", "theta_x", or "theta_y".
        node_ids:   Global w-node indices (from HeterosisMesh.node_coordinates) to constrain.
                    For theta_x / theta_y, these are theta-node indices.
        value:      Prescribed value; defaults to 0.0 (fixed / clamped).
    """

    field_name: str
    node_ids: list[int] | tuple[int, ...]
    value: float = 0.0


@dataclass(frozen=True)
class ElementEdgeLineLoad:
    """
    Distributed transverse traction along one edge of a Q8 element.

    The load is integrated over the edge arc length and applied to the three w-DOFs on that
    edge (two corner nodes and one midside node).

    Args:
        element_id: Global element index.
        edge_id:    Local edge number, 1–4 (bottom, right, top, left — see
                    HeterosisPlateElement.local_edge_nodes).
        magnitude:  Transverse load per unit edge length. Positive acts in the +w direction.
                    May be a callable f(x, y) → float for spatially varying tractions.
    """

    element_id: int
    edge_id: int
    magnitude: float | Callable[[float, float], float]


@dataclass(frozen=True)
class ElementSurfaceLoad:
    """
    Distributed transverse pressure over the mid-surface of one Q8 element.

    The load is integrated over the element area using a 3×3 Gauss rule and applied to
    the eight w-DOFs of the element.

    Args:
        element_id: Global element index.
        magnitude:  Transverse pressure (force per unit area). Positive acts in the +w direction.
                    May be a callable f(x, y) → float for spatially varying pressures.
    """

    element_id: int
    magnitude: float | Callable[[float, float], float]


@dataclass(frozen=True)
class NodalPointLoad:
    """
    Concentrated generalized nodal load.

    Args:
        field_name: One of "w", "theta_x", or "theta_y".
        node_id: Global node id for the corresponding field.
        value: Concentrated load value. Positive acts in the positive
            direction of the selected generalized displacement.
    """
    field_name: str
    node_id: int
    value: float