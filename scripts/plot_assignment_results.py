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

from plate_fea.plotting import (
    apply_report_style,
    plot_field_at_quadrature_points,
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


def _add_caption(fig: plt.Figure, caption: str) -> None:
    """Add a small grey unit-note below the axes — identical to the convergence script."""
    fig.subplots_adjust(bottom=0.18)
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

    # Loaded inner edge — black line + labelled callout (style from plot_mesh_strategies_comparison.py).
    x_load = 0.5 * (g.hole_x_min + g.hole_x_max)
    ax_mesh.plot(
        [g.hole_x_min, g.hole_x_max], [g.hole_y_max, g.hole_y_max],
        color="black", linewidth=2.8, solid_capstyle="round", zorder=7,
    )
    ax_mesh.annotate(
        "Loaded boundary",
        xy=(x_load, g.hole_y_max),
        xytext=(x_load, 0.5 * (g.hole_y_min + g.hole_y_max)),
        ha="center", va="center", fontsize=8.5, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.95),
        annotation_clip=False, zorder=8,
    )

    # Point A — same callout style.
    ax_mesh.scatter(*point_a_xy, s=40, color="black", zorder=10)
    ax_mesh.annotate(
        "Point A",
        xy=point_a_xy,
        xytext=(point_a_xy[0] + 55, point_a_xy[1] - 35),
        ha="center", va="center", fontsize=8.5, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.95),
        annotation_clip=False, zorder=8,
    )

    # Legend for node types (Q9 center nodes label added inside plot_heterosis_mesh).
    ax_mesh.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)

    _add_caption(fig_mesh, _UNIT_NOTE)
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

    # 03–05 – quadrature-point fields (scatter, no interpolation).
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
        apply_report_style()
        fig, ax = plot_field_at_quadrature_points(fields.x, fields.y, values, title=title)
        ax.set_xlabel(r"$x$ $\mathrm{(mm)}$")
        ax.set_ylabel(r"$y$ $\mathrm{(mm)}$")
        _add_caption(fig, _UNIT_NOTE)
        _save(fig, save_dir, filename)

    print(f"\n  All figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
