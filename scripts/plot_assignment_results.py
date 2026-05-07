"""
Solve and visualise the plate-with-hole assignment problem.

Problem specification (STATEMENT.md):
  Plate     : 500 mm × 300 mm (width × height)
  Cut-out   : 250 mm × 180 mm, centred → corners at (125,60)–(375,240) mm
  Thickness : 20 mm
  Material  : E = 200 000 N/mm², ν = 0.25
  BCs       : left edge (x = 0) and top edge (y = 300 mm) clamped;
              right edge (x = 500 mm) and bottom edge (y = 0) free
  Load      : 1 kN/mm downward on the inner top edge of the cut-out (y = 240 mm)
  Point A   : bottom-right corner of the cut-out (375, 60) mm

Outputs (saved to --save-dir, default output/assignment):
  01_mesh.png          – mesh with loaded edge and Point A marked.
  02_w.png             – transverse displacement w
  03.1_gamma_xz.png    – shear strain γ_xz
  03.2_gamma_yz.png    – shear strain γ_yz
  04.1_M_xx.png        – bending moment M_xx
  04.2_M_yy.png        – bending moment M_yy
  04.3_M_xy.png        – bending moment M_xy
  05.1_Q_x.png         – shear force Q_x
  05.2_Q_y.png         – shear force Q_y

Usage:
  python scripts/plot_assignment_results.py
  python scripts/plot_assignment_results.py --resolution 4 --hole-refine 3
  python scripts/plot_assignment_results.py --save-dir output/assignment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle

from plate_fea.plotting import (
    apply_report_style,
    plot_heterosis_mesh,
    plot_w_field,
)
from plate_fea.postprocessing import sample_fields_at_quadrature_points
from plate_fea.problem_orchestrator import ProblemConfig, solve_plate_problem

_DPI = 220

_UNIT_NOTE = (
    r"Units: $x$, $y$ in mm.  "
    r"Loaded inner edge: $q = 1\ \mathrm{kN/mm}$ downward.  "
    r"Left and top outer edges clamped; right and bottom free."
)

# Figure 01 (mesh): loaded boundary and Point A markers
_LOADED_BOUNDARY_COLOR = "#1f77b4"
_POINT_A_COLOR = "crimson"
_CLAMP_BAND_MM = 10.0
_CLAMP_EDGE_COLOR = "0.32"
_CLAMP_FILL = "#dcdcdc"
# Matplotlib ``/`` hatch draws parallel lines at 45° to the axes (AutoCAD ANSI31-style);
# repeat characters to increase line density in the pattern.
_CLAMP_HATCH = "////////"
_CLAMP_HATCH_LINEWIDTH = 0.55
# Extra margin around the mesh axes (mm) so the exterior hatch and annotations breathe.
_MESH_VIEW_PAD_MM = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and plot the plate-with-hole assignment problem."
    )
    parser.add_argument("--resolution",  type=int, default=3,                   help="Global mesh density. Default: 3")
    parser.add_argument("--hole-refine", type=int, default=2,                   help="Extra refinement near hole. Default: 2")
    parser.add_argument("--save-dir",    type=str, default="output/assignment", help="Directory to save figures. Default: output/assignment")
    return parser.parse_args()


def print_summary(config: ProblemConfig, result) -> None:
    g = config.geometry
    print()
    print("┌── Assignment: plate with centred cut-out ─────────────────────────────┐")
    print(f"│  Plate         : {g.outer_width:.0f} × {g.outer_height:.0f} mm")
    print(f"│  Cut-out       : {g.hole_width:.0f} × {g.hole_height:.0f} mm  (centred)")
    print(f"│  Thickness     : {config.thickness:.0f} mm")
    print(f"│  E / ν         : {config.young_modulus:.0f} N/mm²  /  {config.poisson_ratio}")
    print(f"│  Clamped edges : {', '.join(config.clamped_outer_edges)}")
    print(f"│  Load          : {abs(config.hole_top_shear_load):.0f} N/mm downward on hole top edge")
    print("├───────────────────────────────────────────────────────────────────────┤")
    print(f"│  Elements      : {result.model.mesh.total_element_number}")
    print(f"│  w-nodes       : {result.model.mesh.total_w_node_number}")
    print(f"│  Total DOFs    : {result.model.mesh.total_dof_number}")
    print("├───────────────────────────────────────────────────────────────────────┤")
    print(f"│  Point A node  : {result.point_a_node_id}  (nearest to hole corner (375, 60) mm)")
    print(f"│  w at Point A  : {result.point_a_deflection:.6e} mm")
    print("└───────────────────────────────────────────────────────────────────────┘")
    print()


def _add_caption(fig: plt.Figure, caption: str, *, figure_bottom: float = 0.18) -> None:
    """Add a small grey unit-note below the axes — identical to the convergence script."""
    fig.subplots_adjust(bottom=figure_bottom)
    fig.text(
        0.5, 0.02, caption,
        transform=fig.transFigure,
        ha="center", va="bottom",
        fontsize=8.5, color="0.32", linespacing=1.35,
    )


def _save(fig: plt.Figure, save_dir: Path, filename: str) -> None:
    path = save_dir / filename
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def _elementwise_average_from_quadrature(
    values: np.ndarray,
    n_elements: int,
) -> np.ndarray:
    """Average quadrature-point samples inside each element."""
    if values.size % n_elements != 0:
        raise ValueError(
            "Quadrature samples are not divisible by element count: "
            f"n_samples={values.size}, n_elements={n_elements}"
        )
    n_points_per_element = values.size // n_elements
    return values.reshape(n_elements, n_points_per_element).mean(axis=1)


def _plot_elementwise_constant_field(
    mesh,
    element_values: np.ndarray,
    *,
    title: str,
    colormap: str = "RdBu_r",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a per-element scalar field as constant-filled Q8 polygons."""
    fig, ax = plt.subplots()

    q8_loop = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=int)
    polygons: list[np.ndarray] = []
    for element_id in range(mesh.total_element_number):
        perimeter_node_ids = mesh.w_location_matrix[q8_loop, element_id]
        polygons.append(mesh.node_coordinates[perimeter_node_ids, :])

    vmax = max(float(np.max(np.abs(element_values))), 1.0e-14)
    collection = PolyCollection(
        polygons,
        array=np.asarray(element_values, dtype=float),
        cmap=colormap,
        edgecolors="0.35",
        linewidths=0.5,
    )
    collection.set_clim(-vmax, vmax)
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax)

    x = mesh.node_coordinates[:, 0]
    y = mesh.node_coordinates[:, 1]
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    return fig, ax


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    apply_report_style()

    config = ProblemConfig(
        mesh_strategy="gmsh_boundary_sensitive",
        resolution=args.resolution,
        hole_refine=args.hole_refine,
        clamped_outer_edges=("left", "top"),
    )

    result = solve_plate_problem(config)
    print_summary(config, result)

    g          = config.geometry
    point_a_xy = np.array([g.hole_x_max, g.hole_y_min])
    hole_rect  = (g.hole_x_min, g.hole_x_max, g.hole_y_min, g.hole_y_max)
    n_el       = result.model.mesh.total_element_number
    subtitle   = rf"$N_{{\mathrm{{el}}}} = {n_el}$, resolution $= {args.resolution}$"

    # ── 01: mesh ───────────────────────────────────────────────────────────────
    apply_report_style()
    fig_mesh, ax_mesh = plot_heterosis_mesh(
        result.model.mesh,
        show_w_nodes=False,
        show_theta_nodes=False,
        title=subtitle,
    )
    ax_mesh.set_xlabel(r"$x$ $\mathrm{(mm)}$")
    ax_mesh.set_ylabel(r"$y$ $\mathrm{(mm)}$")

    # Clamped outer edges (left + top): hatch sits OUTSIDE the plate (support region), like drafting.
    # Left strip: x in [-b, 0]; top strip: y in [H, H+b]; corner square outside (0,H).
    w_plate = g.outer_width
    h_plate = g.outer_height
    b = _CLAMP_BAND_MM
    _pad = _MESH_VIEW_PAD_MM
    with plt.rc_context(
        {
            "hatch.linewidth": _CLAMP_HATCH_LINEWIDTH,
            "hatch.color": _CLAMP_EDGE_COLOR,
        }
    ):
        left_band = Rectangle(
            (-b, 0.0),
            b,
            h_plate,
            facecolor=_CLAMP_FILL,
            edgecolor=_CLAMP_EDGE_COLOR,
            linewidth=0.7,
            hatch=_CLAMP_HATCH,
            alpha=0.72,
            zorder=1,
        )
        top_band = Rectangle(
            (0.0, h_plate),
            w_plate,
            b,
            facecolor=_CLAMP_FILL,
            edgecolor=_CLAMP_EDGE_COLOR,
            linewidth=0.7,
            hatch=_CLAMP_HATCH,
            alpha=0.72,
            zorder=1,
        )
        corner_band = Rectangle(
            (-b, h_plate),
            b,
            b,
            facecolor=_CLAMP_FILL,
            edgecolor=_CLAMP_EDGE_COLOR,
            linewidth=0.7,
            hatch=_CLAMP_HATCH,
            alpha=0.72,
            zorder=1,
        )
        ax_mesh.add_patch(left_band)
        ax_mesh.add_patch(top_band)
        ax_mesh.add_patch(corner_band)
    # Extend limits so exterior hatch and frame are not clipped; small symmetric air on all sides.
    ax_mesh.set_xlim(-b - _pad, w_plate + _pad)
    ax_mesh.set_ylim(-0.03 * h_plate - 0.35 * _pad, h_plate + b + _pad)
    ax_mesh.plot(
        [0.0, 0.0],
        [0.0, h_plate],
        color=_CLAMP_EDGE_COLOR,
        linewidth=2.6,
        solid_capstyle="projecting",
        zorder=5,
    )
    ax_mesh.plot(
        [0.0, w_plate],
        [h_plate, h_plate],
        color=_CLAMP_EDGE_COLOR,
        linewidth=2.6,
        solid_capstyle="projecting",
        zorder=5,
    )

    # Loaded inner edge (blue) + labelled callout.
    x_load = 0.5 * (g.hole_x_min + g.hole_x_max)
    ax_mesh.plot(
        [g.hole_x_min, g.hole_x_max], [g.hole_y_max, g.hole_y_max],
        color=_LOADED_BOUNDARY_COLOR,
        linewidth=2.8,
        solid_capstyle="round",
        zorder=7,
        label="Loaded boundary",
    )
    ax_mesh.annotate(
        "Loaded boundary",
        xy=(x_load, g.hole_y_max),
        xytext=(x_load, 0.5 * (g.hole_y_min + g.hole_y_max)),
        ha="center",
        va="center",
        fontsize=8.5,
        color=_LOADED_BOUNDARY_COLOR,
        arrowprops=dict(arrowstyle="->", color=_LOADED_BOUNDARY_COLOR, lw=1.0),
        bbox=dict(
            facecolor="white",
            edgecolor=_LOADED_BOUNDARY_COLOR,
            boxstyle="round,pad=0.2",
            alpha=0.95,
        ),
        annotation_clip=False,
        zorder=8,
    )

    # Point A (crimson).
    ax_mesh.scatter(
        *point_a_xy,
        s=40,
        color=_POINT_A_COLOR,
        zorder=10,
        edgecolors="white",
        linewidths=0.6,
        label="Point A",
    )
    ax_mesh.annotate(
        "Point A",
        xy=point_a_xy,
        xytext=(point_a_xy[0] + 55, point_a_xy[1] - 35),
        ha="center",
        va="center",
        fontsize=8.5,
        color=_POINT_A_COLOR,
        arrowprops=dict(arrowstyle="->", color=_POINT_A_COLOR, lw=1.0),
        bbox=dict(
            facecolor="white",
            edgecolor=_POINT_A_COLOR,
            boxstyle="round,pad=0.2",
            alpha=0.95,
        ),
        annotation_clip=False,
        zorder=8,
    )

    # Legend below the x-axis label (clamped support hatch, Q9 centers, load, point A).
    with plt.rc_context(
        {
            "hatch.linewidth": _CLAMP_HATCH_LINEWIDTH,
            "hatch.color": _CLAMP_EDGE_COLOR,
        }
    ):
        clamp_legend_patch = Patch(
            facecolor=_CLAMP_FILL,
            edgecolor=_CLAMP_EDGE_COLOR,
            linewidth=0.7,
            hatch=_CLAMP_HATCH,
            alpha=0.72,
        )
    handles, labels = ax_mesh.get_legend_handles_labels()
    handles = [clamp_legend_patch, *handles]
    labels = ["Clamped edges", *labels]
    ax_mesh.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        columnspacing=1.15,
        handletextpad=0.6,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    # Bottom margin for x-label + legend only (no footer caption on mesh figure).
    fig_mesh.subplots_adjust(bottom=0.28)
    _save(fig_mesh, save_dir, "01_mesh.png")

    # ── Result fields ──────────────────────────────────────────────────────────
    fields = sample_fields_at_quadrature_points(result.model.mesh, result.model, result.solution)

    # 02 – w at nodes (tricontourf with hole masking).
    apply_report_style()
    fig_w, ax_w = plot_w_field(
        result.model.mesh, result.solution,
        hole_rect=hole_rect,
        title=r"Transverse displacement $w\ \mathrm{(mm)}$",
    )
    ax_w.set_xlabel(r"$x$ $\mathrm{(mm)}$")
    ax_w.set_ylabel(r"$y$ $\mathrm{(mm)}$")
    ax_w.scatter(*point_a_xy, s=50, color="k", zorder=10)
    ax_w.text(point_a_xy[0] + 8, point_a_xy[1], "A", fontsize=9, fontweight="bold", va="center")
    _add_caption(fig_w, _UNIT_NOTE)
    _save(fig_w, save_dir, "02_w.png")

    # 03–05 – per-element fields using average over each element's quadrature points.
    n_elements = result.model.mesh.total_element_number
    quad_fields = [
        (fields.gamma_xz, r"Shear strain $\gamma_{xz}$",                  "03.1_gamma_xz.png"),
        (fields.gamma_yz, r"Shear strain $\gamma_{yz}$",                  "03.2_gamma_yz.png"),
        (fields.M_xx,     r"Bending moment $M_{xx}\ \mathrm{(N)}$",       "04.1_M_xx.png"),
        (fields.M_yy,     r"Bending moment $M_{yy}\ \mathrm{(N)}$",       "04.2_M_yy.png"),
        (fields.M_xy,     r"Bending moment $M_{xy}\ \mathrm{(N)}$",       "04.3_M_xy.png"),
        (fields.Q_x,      r"Shear force $Q_x\ \mathrm{(N/mm)}$",          "05.1_Q_x.png"),
        (fields.Q_y,      r"Shear force $Q_y\ \mathrm{(N/mm)}$",          "05.2_Q_y.png"),
    ]
    for values, title, filename in quad_fields:
        element_avg_values = _elementwise_average_from_quadrature(values, n_elements)
        apply_report_style()
        fig, ax = _plot_elementwise_constant_field(
            result.model.mesh,
            element_avg_values,
            title=f"{title} (element-avg from quadrature)",
        )
        ax.set_xlabel(r"$x$ $\mathrm{(mm)}$")
        ax.set_ylabel(r"$y$ $\mathrm{(mm)}$")
        _add_caption(fig, _UNIT_NOTE)
        _save(fig, save_dir, filename)

    print(f"\n  All figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
