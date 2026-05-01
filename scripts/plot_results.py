"""
Solve a rectangular plate and display displacement, shear-strain, and stress-resultant plots.

Default case: SSSS square plate, uniform downward pressure (same parameters as the Navier
validation test). Switch to CCCC with --bc clamped.

Usage examples:

    python scripts/plot_results.py
    python scripts/plot_results.py --nx 16 --ny 16
    python scripts/plot_results.py --bc clamped --nx 12 --ny 12
    python scripts/plot_results.py --nx 8 --ny 8 --save-fig output/results.png
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.plotting import plot_all_result_fields
from plate_fea.reference_solutions import (
    kirchhoff_cccc_uniform_load_center_deflection_square,
    kirchhoff_ssss_uniform_load_center_deflection_square,
)
from plate_fea.solver import solve_displacement_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a plate and plot displacement, shear strains, and stress resultants.",
    )
    parser.add_argument("--side",          type=float, default=1.0,     help="Edge length [m]. Default: 1.0")
    parser.add_argument("--nx",            type=int,   default=10,      help="Elements along x. Default: 10")
    parser.add_argument("--ny",            type=int,   default=10,      help="Elements along y. Default: 10")
    parser.add_argument("--pressure",      type=float, default=-10.0e3, help="Uniform pressure [Pa]; negative = downward. Default: -10 kPa")
    parser.add_argument("--young-modulus", type=float, default=200.0e9, help="E [Pa]. Default: 200 GPa")
    parser.add_argument("--poisson-ratio", type=float, default=0.3,     help="ν [-]. Default: 0.3")
    parser.add_argument("--thickness",     type=float, default=5.0e-3,  help="Plate thickness t [m]. Default: 5 mm")
    parser.add_argument("--bc",            choices=["ssss", "clamped"], default="ssss",
                        help="Boundary conditions: ssss (simply supported) or clamped. Default: ssss")
    parser.add_argument("--save-fig",      type=str,   default=None,    help="Save figure to this path instead of showing interactively.")
    return parser.parse_args()


def apply_ssss_boundary_conditions(model: PlateModel, side: float, tolerance: float = 1.0e-9) -> None:
    """Pin w = 0 on all four edges (simply supported; rotations are free)."""
    xy = model.mesh.node_coordinates
    boundary_nodes = np.flatnonzero(
        np.isclose(xy[:, 0], 0.0,  atol=tolerance)
        | np.isclose(xy[:, 0], side, atol=tolerance)
        | np.isclose(xy[:, 1], 0.0,  atol=tolerance)
        | np.isclose(xy[:, 1], side, atol=tolerance)
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="w", node_ids=boundary_nodes.tolist(), value=0.0)
    )


def apply_clamped_boundary_conditions(model: PlateModel, side: float, tolerance: float = 1.0e-9) -> None:
    """Clamp w = θ_x = θ_y = 0 on all four edges."""
    xy = model.mesh.node_coordinates
    boundary_nodes = np.flatnonzero(
        np.isclose(xy[:, 0], 0.0,  atol=tolerance)
        | np.isclose(xy[:, 0], side, atol=tolerance)
        | np.isclose(xy[:, 1], 0.0,  atol=tolerance)
        | np.isclose(xy[:, 1], side, atol=tolerance)
    )
    for field_name in ("w", "theta_x", "theta_y"):
        model.add_essential_condition(
            EssentialBoundaryCondition(field_name=field_name, node_ids=boundary_nodes.tolist(), value=0.0)
        )


def print_summary(
    args: argparse.Namespace,
    mesh_node_coordinates: np.ndarray,
    displacement: np.ndarray,
    n_elements: int,
) -> None:
    side = args.side
    centre = np.array([0.5 * side, 0.5 * side])
    centre_node = int(np.argmin(np.linalg.norm(mesh_node_coordinates - centre, axis=1)))
    w_centre = float(displacement[centre_node])

    if args.bc == "ssss":
        w_ref = kirchhoff_ssss_uniform_load_center_deflection_square(
            side=side,
            pressure=abs(args.pressure),
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            thickness=args.thickness,
        )
        bc_label = "SSSS (simply supported)"
        ref_label = "Navier series (Kirchhoff)"
    else:
        w_ref = kirchhoff_cccc_uniform_load_center_deflection_square(
            side=side,
            pressure=abs(args.pressure),
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            thickness=args.thickness,
        )
        bc_label = "CCCC (clamped)"
        ref_label = "Kirchhoff β factor (Timoshenko)"

    rel_error = abs((w_centre + w_ref) / w_ref) if w_ref != 0.0 else float("nan")

    print()
    print(f"  Boundary conditions : {bc_label}")
    print(f"  Mesh                : {args.nx} × {args.ny} elements  ({n_elements} total)")
    print(f"  Side / thickness    : {side} m  /  {args.thickness * 1e3:.1f} mm  (t/a = {args.thickness / side:.4f})")
    print(f"  Pressure            : {args.pressure:.3e} Pa")
    print()
    print(f"  Centre w  (FE)      : {w_centre:.6e} m")
    print(f"  Centre w  ({ref_label}) : {-w_ref:.6e} m  (sign matched to FE)")
    print(f"  Relative error      : {rel_error:.4%}")
    print()


def main() -> None:
    args = parse_args()

    mesh = generate_rectangular_heterosis_mesh(
        width=args.side, height=args.side, nx=args.nx, ny=args.ny
    )
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            thickness=args.thickness,
        ),
        element_formulation=HeterosisPlateElement(),
    )

    if args.bc == "ssss":
        apply_ssss_boundary_conditions(model, args.side)
    else:
        apply_clamped_boundary_conditions(model, args.side)

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=args.pressure))

    _, _, displacement = solve_displacement_system(model)

    print_summary(args, mesh.node_coordinates, displacement, mesh.total_element_number)

    fig = plot_all_result_fields(mesh, model, displacement)
    bc_label = "SSSS" if args.bc == "ssss" else "CCCC"
    fig.suptitle(
        f"{bc_label} square plate — {args.nx}×{args.ny} elements, "
        f"q = {args.pressure:.3g} Pa, t = {args.thickness * 1e3:.1f} mm",
        fontsize=11,
    )

    if args.save_fig:
        save_path = Path(args.save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Figure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
