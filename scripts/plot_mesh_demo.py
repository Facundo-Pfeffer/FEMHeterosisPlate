"""
Plot the same single Q8 element mesh as ``run_smoke_test.py`` (square plate).

Usage (from repo root, after ``pip install -e .``):

    python scripts/plot_mesh_demo.py

Add ``--no-show`` to only build the figure (e.g. in headless environments).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plate_fea.mesh import HeterosisMesh
from plate_fea.plotting import plot_heterosis_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a demo heterosis plate mesh.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (still creates the figure).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="If set, save figure to this path (e.g. mesh.png).",
    )
    args = parser.parse_args()

    node_coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    w_location_matrix = np.array(
        [[0], [1], [2], [3], [4], [5], [6], [7]],
        dtype=int,
    )
    mesh = HeterosisMesh.from_arrays(
        node_coordinates=node_coordinates,
        w_location_matrix=w_location_matrix,
    )

    _, ax = plot_heterosis_mesh(mesh, show_theta_nodes=True)
    ax.grid(True, alpha=0.3)

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
