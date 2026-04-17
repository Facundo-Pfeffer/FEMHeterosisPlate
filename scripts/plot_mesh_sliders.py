"""Interactive slider viewer for the plate-with-hole mesh."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from plate_fea.mesh_generation import PlateWithHoleGeometry, UniformBufferRingQ8Generator

PLOT_BOUNDS = (0.06, 0.14, 0.62, 0.80)
SLIDER_X = 0.74
SLIDER_W = 0.22
SLIDER_H = 0.03


def _q8_boundary_loop_local_ids() -> np.ndarray:
    return np.array([0, 4, 1, 5, 2, 6, 3, 7, 0], dtype=int)


def _rect_polyline(x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]], dtype=float)


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def _unique_q8_edge_segments(node_xy: np.ndarray, w_lm: np.ndarray) -> np.ndarray:
    """Build unique line segments for Q8 boundaries."""
    loop = _q8_boundary_loop_local_ids()
    seg_pairs: set[tuple[int, int]] = set()
    for e in range(w_lm.shape[1]):
        ordered = w_lm[:, e][loop]
        for a, b in zip(ordered[:-1], ordered[1:], strict=False):
            i, j = int(a), int(b)
            if i == j:
                continue
            seg_pairs.add((i, j) if i < j else (j, i))

    segments = np.zeros((len(seg_pairs), 2, 2), dtype=float)
    for k, (i, j) in enumerate(seg_pairs):
        segments[k, 0, :] = node_xy[i, :]
        segments[k, 1, :] = node_xy[j, :]
    return segments


def _configure_main_axes(ax: Axes, geometry: PlateWithHoleGeometry) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\;[\mathrm{mm}]$")
    ax.set_ylabel(r"$y\;[\mathrm{mm}]$")
    ax.set_title(r"Mesh exploration: plate with centered rectangular hole")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, color="0.85")
    for spine in ax.spines.values():
        spine.set_color("0.25")
    ax.set_xlim(0.0, geometry.outer_width)
    ax.set_ylim(0.0, geometry.outer_height)


@dataclass
class PlotArtists:
    edge_collection: LineCollection
    q9_center_scatter: any
    buffer_line_x: any
    info_text: any


@dataclass
class PlotControls:
    resolution: Slider
    hole_refine: Slider
    buffer: Slider


def _create_static_overlays(ax: Axes, geometry: PlateWithHoleGeometry) -> PlotArtists:
    # Mesh edge collection
    edge_collection = LineCollection([], colors=["0.35"], linewidths=0.6, alpha=0.55, zorder=2)
    ax.add_collection(edge_collection)
    q9_center_scatter = ax.scatter(
        [],
        [],
        c="0.35",
        s=8,
        zorder=5,
        alpha=0.9,
    )

    outer = _rect_polyline(0.0, geometry.outer_width, 0.0, geometry.outer_height)
    hole = _rect_polyline(geometry.hole_x_min, geometry.hole_x_max, geometry.hole_y_min, geometry.hole_y_max)
    ax.plot(outer[:, 0], outer[:, 1], color="0.1", linewidth=1.4, alpha=0.9, zorder=3, label="Outer boundary")
    ax.plot(hole[:, 0], hole[:, 1], color="0.0", linewidth=1.6, alpha=0.95, zorder=4, label="Hole boundary")

    # Buffer rectangle guide
    buffer_poly = _rect_polyline(geometry.hole_x_min, geometry.hole_x_max, geometry.hole_y_min, geometry.hole_y_max)
    (buffer_line,) = ax.plot(
        buffer_poly[:, 0],
        buffer_poly[:, 1],
        color="0.35",
        linewidth=1.0,
        alpha=0.8,
        linestyle=(0, (4, 3)),
        zorder=3,
        label="Buffer rectangle",
    )

    # Load and point A annotations
    ax.plot(
        [geometry.hole_x_min, geometry.hole_x_max],
        [geometry.hole_y_max, geometry.hole_y_max],
        color="#1f77b4",
        linewidth=2.6,
        alpha=0.95,
        zorder=5,
        label=r"Applied shear $q$ on hole top edge",
    )
    ax.text(
        0.5 * (geometry.hole_x_min + geometry.hole_x_max),
        geometry.hole_y_max + 9.0,
        r"$q=1\,\mathrm{kN/mm}$",
        color="#1f77b4",
        ha="center",
        va="bottom",
        fontsize=9,
        zorder=6,
    )
    ax.scatter([geometry.hole_x_max], [geometry.hole_y_min], c="#d62728", s=36, zorder=6, label="Point A")
    ax.text(geometry.hole_x_max + 6.0, geometry.hole_y_min - 8.0, r"$A$", color="#d62728")

    info_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    return PlotArtists(
        edge_collection=edge_collection,
        q9_center_scatter=q9_center_scatter,
        buffer_line_x=buffer_line,
        info_text=info_text,
    )


def _create_controls(fig: Figure, geometry: PlateWithHoleGeometry) -> PlotControls:
    ax_res = fig.add_axes([SLIDER_X, 0.76, SLIDER_W, SLIDER_H])
    ax_refine = fig.add_axes([SLIDER_X, 0.72, SLIDER_W, SLIDER_H])
    ax_buffer = fig.add_axes([SLIDER_X, 0.66, SLIDER_W, SLIDER_H])

    max_buffer = min(
        min(geometry.hole_x_min, geometry.outer_width - geometry.hole_x_max),
        min(geometry.hole_y_min, geometry.outer_height - geometry.hole_y_max),
    ) - 1e-6

    return PlotControls(
        resolution=Slider(ax_res, "resolution", 1, 6, valinit=2, valstep=1),
        hole_refine=Slider(ax_refine, "hole_refine", 0, 6, valinit=2, valstep=1),
        buffer=Slider(ax_buffer, "buffer", 1.0, max_buffer, valinit=min(30.0, max_buffer), valstep=1.0),
    )


def _update_plot(fig: Figure, geometry: PlateWithHoleGeometry, controls: PlotControls, artists: PlotArtists) -> None:
    resolution = _clamp_int(controls.resolution.val, 1, 6)
    hole_refine = _clamp_int(controls.hole_refine.val, 0, 6)
    buffer = float(controls.buffer.val)

    # Update guide rectangle
    b = _rect_polyline(
        geometry.hole_x_min - buffer,
        geometry.hole_x_max + buffer,
        geometry.hole_y_min - buffer,
        geometry.hole_y_max + buffer,
    )
    artists.buffer_line_x.set_data(b[:, 0], b[:, 1])

    # Sliders may hit invalid mesh settings; catch any generator failure and show it in the figure.
    try:
        mesh = UniformBufferRingQ8Generator(
            geometry=geometry,
            resolution=resolution,
            hole_refine=hole_refine,
            buffer=buffer,
        ).generate()
    except Exception as exc:  # noqa: BLE001
        artists.edge_collection.set_segments([])
        artists.q9_center_scatter.set_offsets(np.empty((0, 2), dtype=float))
        artists.info_text.set_text(f"Error:\n  {exc}")
        fig.canvas.draw_idle()
        return

    segments = _unique_q8_edge_segments(mesh.node_coordinates, mesh.w_location_matrix)
    artists.edge_collection.set_segments(segments)
    q9_center_theta_ids = np.unique(mesh.theta_location_matrix[8, :])
    q9_center_xy = mesh.theta_node_coordinates[q9_center_theta_ids, :]
    artists.q9_center_scatter.set_offsets(q9_center_xy)
    artists.info_text.set_text(
        f"\nParams:\n"
        f"  resolution   = {resolution}\n"
        f"  hole_refine  = {hole_refine}\n"
        f"  buffer       = {buffer:.1f}\n"
        f"\nCounts:\n"
        f"  elements = {mesh.total_element_number}\n"
        f"  w_nodes  = {mesh.total_w_node_number}\n"
        f"  q9 ctrs  = {q9_center_xy.shape[0]}\n"
        f"  dofs     = {mesh.total_dof_number}"
    )
    fig.canvas.draw_idle()


def main() -> None:
    geometry = PlateWithHoleGeometry()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes(PLOT_BOUNDS)
    _configure_main_axes(ax, geometry)
    artists = _create_static_overlays(ax, geometry)
    controls = _create_controls(fig, geometry)

    def on_change(_value: float) -> None:
        _update_plot(fig, geometry, controls, artists)

    controls.resolution.on_changed(on_change)
    controls.hole_refine.on_changed(on_change)
    controls.buffer.on_changed(on_change)

    _update_plot(fig, geometry, controls, artists)
    plt.show()


if __name__ == "__main__":
    main()

