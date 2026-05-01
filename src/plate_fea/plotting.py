"""
Matplotlib helpers for visualising mesh geometry and result fields.

Mesh:
    plot_heterosis_mesh(mesh)         — Q8 element edges and nodes
    show_mesh_plot(mesh)              — same, displayed interactively

Result fields (after solving):
    plot_w_field(mesh, displacement)  — transverse displacement w at Q8 nodes
    plot_field_at_quadrature_points(x, y, values) — any scalar field at scattered points
    plot_all_result_fields(mesh, model, displacement) — 2×4-panel summary of all result fields

Style:
    apply_report_style()              — apply clean, report-quality rcParams globally
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel
from plate_fea.postprocessing import SampledFields, sample_fields_at_quadrature_points


def apply_report_style() -> None:
    """Apply clean, report-quality rcParams (white background, inward ticks, sans-serif, grid)."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.linewidth":    1.0,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "font.size":         11,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "legend.fontsize":   10,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "mathtext.fontset":  "dejavusans",
    })


def _q8_boundary_loop_local_ids() -> np.ndarray:
    """Local node order around the Q8 perimeter (matches HeterosisPlateElement edges)."""
    return np.array([0, 4, 1, 5, 2, 6, 3, 7, 0], dtype=int)


def _symmetric_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    """Return n_levels evenly spaced from −|max| to +|max| for a symmetric colormap."""
    abs_max = max(float(np.max(np.abs(values))), 1e-14)
    return np.linspace(-abs_max, abs_max, n_levels)


def _masked_triangulation(
    x: np.ndarray,
    y: np.ndarray,
    hole_rect: tuple[float, float, float, float] | None = None,
) -> mtri.Triangulation:
    """
    Build a Delaunay triangulation of (x, y) and optionally mask triangles inside a rectangle.

    hole_rect, if given, is (x_min, x_max, y_min, y_max). Any triangle whose centroid
    falls strictly inside the rectangle is masked so ``tricontourf`` will not interpolate
    across the void.
    """
    triang = mtri.Triangulation(x, y)
    if hole_rect is not None:
        x_min, x_max, y_min, y_max = hole_rect
        cx = triang.x[triang.triangles].mean(axis=1)
        cy = triang.y[triang.triangles].mean(axis=1)
        mask = (cx > x_min) & (cx < x_max) & (cy > y_min) & (cy < y_max)
        triang.set_mask(mask)
    return triang


# ── Mesh geometry ──────────────────────────────────────────────────────────────

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
    Plot mesh element edges and optionally node markers.

    show_w_nodes:       overlay Q8 displacement nodes.
    show_theta_nodes:   overlay all Q9 rotation nodes (includes center nodes).
    show_q9_center_nodes: overlay only the Q9 center-of-element rotation nodes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    q8_perimeter_ids = _q8_boundary_loop_local_ids()

    for element_id in range(mesh.total_element_number):
        perimeter_node_ids = mesh.w_location_matrix[q8_perimeter_ids, element_id]
        perimeter_coordinates = mesh.node_coordinates[perimeter_node_ids, :]
        ax.plot(perimeter_coordinates[:, 0], perimeter_coordinates[:, 1], color=element_edge_color, linewidth=1.0)

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
            label="Q9 center nodes",
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
    """Convenience wrapper: build a mesh figure and call plt.show()."""
    plot_heterosis_mesh(
        mesh,
        show_w_nodes=show_w_nodes,
        show_theta_nodes=show_theta_nodes,
        show_q9_center_nodes=show_q9_center_nodes,
    )
    plt.show(block=block)


# ── Result fields ──────────────────────────────────────────────────────────────

def plot_w_field(
    mesh: HeterosisMesh,
    displacement: np.ndarray,
    *,
    ax: Axes | None = None,
    colormap: str = "RdBu_r",
    n_levels: int = 20,
    title: str = r"Transverse displacement $w$",
    hole_rect: tuple[float, float, float, float] | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot transverse displacement w at Q8 nodes as a filled contour map.

    w is the field directly available in the displacement vector; no interpolation is needed.
    The colormap is symmetric about zero (positive w = +z direction).

    hole_rect=(x_min, x_max, y_min, y_max): mask triangles inside the hole so the void is
    not interpolated across.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = mesh.node_coordinates[:, 0]
    y = mesh.node_coordinates[:, 1]
    w = displacement[:mesh.total_w_node_number]

    triang = _masked_triangulation(x, y, hole_rect)
    contour = ax.tricontourf(triang, w, levels=_symmetric_levels(w, n_levels), cmap=colormap, extend="both")
    fig.colorbar(contour, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return fig, ax


def plot_field_at_quadrature_points(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    ax: Axes | None = None,
    colormap: str = "RdBu_r",
    scatter_size: int = 15,
    title: str = "",
) -> tuple[Figure, Axes]:
    """
    Plot a scalar field at scattered quadrature-point positions as a scatter plot.

    Each marker is one quadrature point; no interpolation between points is performed.
    The colormap is symmetric about zero. Points inside any hole simply don't exist in the
    sampled data, so no masking is needed.

    Example::

        fields = sample_fields_at_quadrature_points(mesh, model, displacement)
        fig, ax = plot_field_at_quadrature_points(
            fields.x, fields.y, fields.gamma_xz, title=r"$\\gamma_{xz}$"
        )
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    vmax = max(float(np.abs(values).max()), 1e-14)
    sc = ax.scatter(x, y, c=values, cmap=colormap, vmin=-vmax, vmax=vmax, s=scatter_size, linewidths=0)
    fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_all_result_fields(
    mesh: HeterosisMesh,
    model: PlateModel,
    displacement: np.ndarray,
    *,
    quadrature_order: tuple[int, int] = (3, 3),
    colormap: str = "RdBu_r",
    n_levels: int = 20,
    scatter_size: int = 15,
    hole_rect: tuple[float, float, float, float] | None = None,
) -> Figure:
    """
    Multi-panel figure showing displacement, shear strains, bending moments, and shear forces.

    Displacement w is shown as a filled contour at Q8 nodes.
    All other fields are raw scatter plots at quadrature points — one dot per integration point,
    no interpolation between them.

    Panel layout (2 rows × 4 columns):

        w (nodes) | γ_xz | γ_yz | M_xx
        M_yy      | M_xy | Q_x  | Q_y

    hole_rect=(x_min, x_max, y_min, y_max): masks the hole in the w contour panel. Quadrature
    panels need no masking — points inside the void are never sampled.
    """
    fields = sample_fields_at_quadrature_points(mesh, model, displacement, quadrature_order)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # w at Q8 nodes — tricontourf with hole masking.
    node_triang = _masked_triangulation(mesh.node_coordinates[:, 0], mesh.node_coordinates[:, 1], hole_rect)
    w = displacement[:mesh.total_w_node_number]
    cf = axs[0, 0].tricontourf(node_triang, w, levels=_symmetric_levels(w, n_levels), cmap=colormap, extend="both")
    fig.colorbar(cf, ax=axs[0, 0])
    axs[0, 0].set_title(r"Displacement $w$")

    # Quadrature-point panels — scatter, one dot per integration point.
    quad_panels = [
        (axs[0, 1], fields.gamma_xz, r"Shear strain $\gamma_{xz}$"),
        (axs[0, 2], fields.gamma_yz, r"Shear strain $\gamma_{yz}$"),
        (axs[0, 3], fields.M_xx,     r"Bending moment $M_{xx}$"),
        (axs[1, 0], fields.M_yy,     r"Bending moment $M_{yy}$"),
        (axs[1, 1], fields.M_xy,     r"Bending moment $M_{xy}$"),
        (axs[1, 2], fields.Q_x,      r"Shear force $Q_x$"),
        (axs[1, 3], fields.Q_y,      r"Shear force $Q_y$"),
    ]
    for ax, values, label in quad_panels:
        vmax = max(float(np.abs(values).max()), 1e-14)
        sc = ax.scatter(fields.x, fields.y, c=values, cmap=colormap, vmin=-vmax, vmax=vmax,
                        s=scatter_size, linewidths=0)
        fig.colorbar(sc, ax=ax)
        ax.set_title(label)

    for ax in axs.flat:
        if ax.has_data():
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    fig.suptitle("Result fields — w at nodes (contour), stress resultants at quadrature points (scatter)", fontsize=12)
    fig.tight_layout()
    return fig
