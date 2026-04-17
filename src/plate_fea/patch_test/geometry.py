"""Patch geometries: single-element, multi-element, regular and mildly distorted."""

from __future__ import annotations

import numpy as np

from plate_fea.mesh import HeterosisMesh
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh


def unit_square_single_element() -> HeterosisMesh:
    """One Q8 plate element on [0, 1]² (one internal element boundary absent)."""
    return generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=1, ny=1)


def unit_square_multi(nx: int = 2, ny: int = 2) -> HeterosisMesh:
    """Multi-element patch on [0, 1]² with interior element boundaries."""
    if nx < 2 or ny < 2:
        raise ValueError("multi-element patch expects nx >= 2 and ny >= 2 for an interior boundary.")
    return generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=nx, ny=ny)


def rectangle_patch(width: float, height: float, nx: int, ny: int) -> HeterosisMesh:
    """Axis-aligned rectangle [0, width] × [0, height]."""
    return generate_rectangular_heterosis_mesh(width=width, height=height, nx=nx, ny=ny)


def mildly_distorted_unit_square(nx: int = 2, ny: int = 2, amplitude: float = 0.04) -> HeterosisMesh:
    """
    Start from a regular ``nx×ny`` patch on the unit square, then perturb interior ``w``-nodes.

    Boundary nodes are fixed so the outer rectangle remains [0,1]² for BC logic.
    """
    mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=nx, ny=ny)
    xy = mesh.node_coordinates.copy()
    x, y = xy[:, 0], xy[:, 1]
    tol = 1e-9
    interior = (x > tol) & (x < 1.0 - tol) & (y > tol) & (y < 1.0 - tol)
    rng = np.random.default_rng(0)
    xy[interior, 0] += amplitude * (rng.random(np.sum(interior)) - 0.5)
    xy[interior, 1] += amplitude * (rng.random(np.sum(interior)) - 0.5)
    return HeterosisMesh.from_arrays(
        node_coordinates=xy,
        w_location_matrix=mesh.w_location_matrix,
        theta_location_matrix=None,
    )


def outer_rectangle_boundary_w_node_mask(mesh: HeterosisMesh, width: float, height: float) -> np.ndarray:
    """``True`` for w-nodes whose coordinates lie on the edges of ``[0,width] × [0,height]``."""
    tol = 1.0e-9
    x = mesh.node_coordinates[:, 0]
    y = mesh.node_coordinates[:, 1]
    return (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, width, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, height, atol=tol)
    )


def outer_rectangle_boundary_theta_node_mask(mesh: HeterosisMesh, width: float, height: float) -> np.ndarray:
    """``True`` for θ-nodes on the boundary of ``[0,width] × [0,height]`` (uses ``theta_node_coordinates``)."""
    tol = 1.0e-9
    x = mesh.theta_node_coordinates[:, 0]
    y = mesh.theta_node_coordinates[:, 1]
    return (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, width, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, height, atol=tol)
    )
