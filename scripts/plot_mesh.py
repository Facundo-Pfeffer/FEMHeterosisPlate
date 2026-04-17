"""
Generate and plot the structured Q8 mesh for the plate-with-hole problem.

Usage:
    python scripts/plot_mesh.py
    python scripts/plot_mesh.py --out mesh.png --no-show
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from plate_fea.mesh_generation import (
    PlateWithHoleGeometry,
    UniformBufferRingQ8Generator,
)
from plate_fea.plotting import plot_heterosis_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structured Q8 mesh for the plate-with-hole problem.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plotting window.")
    parser.add_argument("--out", type=str, default="", help="Optional path to save the figure.")
    parser.add_argument("--resolution", type=int, default=2, help="Global mesh density (>=1).")
    parser.add_argument("--hole-refine", type=int, default=2, help="Extra refinement near the hole (>=0).")
    parser.add_argument("--buffer", type=float, default=30.0, help="Symmetric buffer ring thickness away from hole [mm].")
    args = parser.parse_args()

    geometry = PlateWithHoleGeometry()
    generator = UniformBufferRingQ8Generator(
        geometry=geometry,
        resolution=int(args.resolution),
        hole_refine=int(args.hole_refine),
        buffer=float(args.buffer),
    )

    mesh = generator.generate()

    fig, ax = plot_heterosis_mesh(
        mesh,
        show_w_nodes=False,
        show_theta_nodes=False,
        title="Mesh: 500x300 plate with centered 250x180 hole",
    )

    # Point A: bottom-right corner of the inner rectangle.
    ax.scatter([geometry.hole_x_max], [geometry.hole_y_min], c="C3", s=40, zorder=5)
    ax.text(geometry.hole_x_max + 6.0, geometry.hole_y_min - 8.0, "A", color="C3")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

