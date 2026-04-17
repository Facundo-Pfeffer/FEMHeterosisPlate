"""
Plot a professional comparison of the two supported mesh strategies.

Includes:
  1) baseline (uniform buffer ring),
  2) gmsh boundary-sensitive quads (skipped with a note if gmsh/libGLU unavailable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plate_fea.mesh import HeterosisMesh
from plate_fea.mesh_generation import (
    GmshBoundarySensitiveQ8Generator,
    PlateWithHoleGeometry,
    UniformBufferRingQ8Generator,
)
from plate_fea.plotting import plot_heterosis_mesh


def apply_professional_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )


def _caption(fig: Figure, text: str) -> None:
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.18, hspace=0.38, wspace=0.18)
    fig.text(0.5, 0.03, text, ha="center", va="bottom", fontsize=8.7, color="0.30", linespacing=1.35)


def _draw_focus_annotations(ax: Axes, g: PlateWithHoleGeometry, *, gutter: float) -> None:
    # Loaded boundary: draw and label the full top hole segment (not a point marker).
    x_load = 0.5 * (g.hole_x_min + g.hole_x_max)
    y_load = g.hole_y_max
    ax.plot(
        [g.hole_x_min, g.hole_x_max],
        [y_load, y_load],
        color="black",
        linewidth=2.8,
        solid_capstyle="round",
        zorder=7,
    )
    ax.annotate(
        "Loaded boundary",
        xy=(x_load, y_load),
        xytext=(x_load, 0.5 * (g.hole_y_min + g.hole_y_max)),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.95),
        annotation_clip=False,
        zorder=8,
    )

    # BC transition: label text sits in the margin (inside expanded limits), not on top of the mesh.
    bl_txt_x = -0.72 * gutter
    bl_txt_y = -0.72 * gutter
    tr_txt_x = g.outer_width + 0.55 * gutter
    tr_txt_y = g.outer_height + 0.55 * gutter
    common = dict(
        fontsize=8.4,
        color="black",
        ha="center",
        va="center",
        annotation_clip=False,
        zorder=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.05, shrinkA=2, shrinkB=2),
        bbox=dict(facecolor="white", edgecolor="black", linewidth=1.0, boxstyle="round,pad=0.28", alpha=0.98),
    )
    ann_bl = ax.annotate(
        "BC transition\n(clamped to free)",
        xy=(0.0, 0.0),
        xytext=(bl_txt_x, bl_txt_y),
        textcoords="data",
        **common,
    )
    ann_tr = ax.annotate(
        "BC transition\n(clamped to free)",
        xy=(g.outer_width, g.outer_height),
        xytext=(tr_txt_x, tr_txt_y),
        textcoords="data",
        **common,
    )
    ann_bl.set_clip_on(False)
    ann_tr.set_clip_on(False)


def _try_gmsh_mesh(g: PlateWithHoleGeometry, resolution: int, hole_refine: int) -> tuple[HeterosisMesh | None, str | None]:
    try:
        mesh = GmshBoundarySensitiveQ8Generator(
            geometry=g,
            resolution=resolution,
            hole_refine=hole_refine,
        ).generate()
        return mesh, None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and gmsh mesh strategies.")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Directory to save the figure.")
    parser.add_argument("--out-name", type=str, default="mesh_strategies_comparison.png", help="Output file name.")
    parser.add_argument("--resolution", type=int, default=2, help="Base resolution control.")
    parser.add_argument("--hole-refine", type=int, default=2, help="Extra near-hole refinement.")
    parser.add_argument("--buffer", type=float, default=30.0, help="Buffer-ring thickness [mm].")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window.")
    args = parser.parse_args()

    g = PlateWithHoleGeometry()
    res = int(args.resolution)
    hr = max(2, int(args.hole_refine) + 1)
    buf = float(args.buffer)

    base = UniformBufferRingQ8Generator(
        geometry=g,
        resolution=res,
        hole_refine=int(args.hole_refine),
        buffer=buf,
    ).generate()
    gmsh_c, gmsh_err = _try_gmsh_mesh(g, resolution=res, hole_refine=hr)

    apply_professional_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax_flat = np.atleast_1d(axes).ravel()

    entries: list[tuple[Axes, HeterosisMesh | None, str, str, str | None]] = [
        (ax_flat[0], base, "Current: uniform buffer ring", "#355C7D", None),
        (ax_flat[1], gmsh_c, "gmsh boundary-sensitive quads", "#d62728", gmsh_err),
    ]

    meshes_for_bounds = [m for _, m, _, _, err in entries if m is not None]
    all_xy = np.vstack([m.node_coordinates for m in meshes_for_bounds])
    span = float(np.max(all_xy.max(axis=0) - all_xy.min(axis=0)))
    pad = 0.02 * span if span > 0.0 else 1.0
    # Extra margin so BC callout labels (placed outside the plate) stay inside the axes view.
    gutter = max(52.0, 0.09 * span) if span > 0.0 else 52.0
    xlim = (float(all_xy[:, 0].min() - pad - gutter), float(all_xy[:, 0].max() + pad + gutter))
    ylim = (float(all_xy[:, 1].min() - pad - gutter), float(all_xy[:, 1].max() + pad + gutter))

    for ax, mesh, title, edge_color, err in entries:
        if mesh is not None:
            plot_heterosis_mesh(
                mesh,
                ax=ax,
                show_w_nodes=False,
                show_theta_nodes=False,
                element_edge_color=edge_color,
                title=f"{title}\n$N_{{el}}={mesh.total_element_number}$, DOF={mesh.total_dof_number}",
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            _draw_focus_annotations(ax, g, gutter=gutter)
        else:
            ax.set_title(f"{title}\n(not available)", fontsize=11)
            ax.text(
                0.5,
                0.5,
                "Option C requires gmsh + libGLU.\n"
                "Install: pip install --upgrade gmsh\n"
                "and system package libglu1-mesa (or equivalent).\n\n"
                f"Error: {err or 'unknown'}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="0.25",
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        if mesh is None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_xlabel(r"$x$ (mm)")
        ax.set_ylabel(r"$y$ (mm)")

    fig.suptitle("Plate-with-hole mesh strategies", fontsize=13, y=0.98)
    _caption(
        fig,
        (
            "Baseline: uniform buffer ring. "
            "gmsh strategy: distance-field-driven sizing near hole, load, and BC-transition regions "
            "(requires gmsh runtime)."
        ),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    fig.savefig(out_path, dpi=230)

    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")
    if gmsh_err:
        print(f"Note: Option C omitted mesh plot ({gmsh_err[:120]}...)" if len(gmsh_err) > 120 else f"Note: Option C omitted mesh plot ({gmsh_err})")


if __name__ == "__main__":
    main()
