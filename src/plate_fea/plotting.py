"""Matplotlib helpers for visualising heterosis mesh connectivity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plate_fea.mesh import HeterosisMesh


def _q8_boundary_loop_local_ids() -> np.ndarray:
    """Local node order around the Q8 perimeter (matches HeterosisPlateElement edges)."""
    return np.array([0, 4, 1, 5, 2, 6, 3, 7, 0], dtype=int)


def plot_heterosis_mesh(
    mesh: HeterosisMesh,
    *,
    ax: Axes | None = None,
    show_w_nodes: bool = True,
    show_theta_nodes: bool = False,
    show_q9_center_nodes: bool = True,
    w_node_color: str = "C0",
    theta_node_color: str = "C3",
    q9_center_node_color: str | None = None,
    element_edge_color: str = "0.35",
    title: str | None = "Heterosis mesh (Q8 geometry, Q9 rotation interpolation)",
) -> tuple[Figure, Axes]:
    """
    Plot the mesh geometry: Q8 elements from ``w_location_matrix`` / ``node_coordinates``.

    When ``theta_location_matrix`` uses generated centers (default ``from_arrays``),
    set ``show_theta_nodes=True`` to overlay Q9 nodes, including element center rotations.
    ``show_q9_center_nodes=True`` overlays only the local Q9 center rotation node per element.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    loop = _q8_boundary_loop_local_ids()

    for e in range(mesh.total_element_number):
        geom_ids = mesh.w_location_matrix[loop, e]
        xy = mesh.node_coordinates[geom_ids, :]
        ax.plot(xy[:, 0], xy[:, 1], color=element_edge_color, linewidth=1.0)

    if show_w_nodes:
        ax.scatter(
            mesh.node_coordinates[:, 0],
            mesh.node_coordinates[:, 1],
            s=18,
            c=w_node_color,
            zorder=4,
            label="w nodes",
        )

    if show_theta_nodes:
        ax.scatter(
            mesh.theta_node_coordinates[:, 0],
            mesh.theta_node_coordinates[:, 1],
            s=10,
            c=theta_node_color,
            zorder=3,
            marker="x",
            label="θ nodes",
        )

    if show_q9_center_nodes:
        center_color = q9_center_node_color or element_edge_color
        center_theta_ids = np.unique(mesh.theta_location_matrix[8, :])
        center_xy = mesh.theta_node_coordinates[center_theta_ids, :]
        ax.scatter(
            center_xy[:, 0],
            center_xy[:, 1],
            s=8,
            c=center_color,
            zorder=5,
            marker="o",
            edgecolors="none",
            alpha=0.9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    return fig, ax


def show_mesh_plot(
    mesh: HeterosisMesh,
    *,
    show_w_nodes: bool = True,
    show_theta_nodes: bool = False,
    show_q9_center_nodes: bool = True,
    block: bool = True,
) -> None:
    """
    Convenience wrapper: build a figure and call ``plt.show()``.
    """
    plot_heterosis_mesh(
        mesh,
        show_w_nodes=show_w_nodes,
        show_theta_nodes=show_theta_nodes,
        show_q9_center_nodes=show_q9_center_nodes,
    )
    plt.show(block=block)
