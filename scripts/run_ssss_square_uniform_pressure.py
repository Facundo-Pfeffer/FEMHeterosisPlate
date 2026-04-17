"""
SSSS square plate benchmark (same case as ``tests/validation/test_ssss_uniform_pressure_vs_navier.py``).

SI throughout: ``a`` [m], ``t`` [m], ``E`` [Pa], ``q`` [Pa]; centre deflection [m] vs Navier.
"""

from __future__ import annotations

import argparse

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.reference_solutions import kirchhoff_ssss_uniform_load_center_deflection_square
from plate_fea.solver import solve_linear_system


def main() -> None:
    parser = argparse.ArgumentParser(description="SSSS square plate, uniform pressure vs Navier (SI).")
    parser.add_argument("--side", type=float, default=1.0, help="Edge length a [m].")
    parser.add_argument("--nx", type=int, default=20, help="Q8 elements along x.")
    parser.add_argument("--ny", type=int, default=20, help="Q8 elements along y.")
    parser.add_argument(
        "--pressure",
        type=float,
        default=-10.0e3,
        help="Uniform pressure [Pa]; negative = downward if +z is upward.",
    )
    parser.add_argument("--young-modulus", type=float, default=200.0e9, help="E [Pa].")
    parser.add_argument("--poisson-ratio", type=float, default=0.3, help="ν [-].")
    parser.add_argument("--thickness", type=float, default=5.0e-3, help="Thickness t [m].")
    args = parser.parse_args()

    a_m = args.side
    mesh = generate_rectangular_heterosis_mesh(width=a_m, height=a_m, nx=args.nx, ny=args.ny)
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            thickness=args.thickness,
        ),
        element_formulation=HeterosisPlateElement(),
    )

    xy = mesh.node_coordinates
    tol_m = 1.0e-9
    boundary = np.flatnonzero(
        np.isclose(xy[:, 0], 0.0, atol=tol_m)
        | np.isclose(xy[:, 0], a_m, atol=tol_m)
        | np.isclose(xy[:, 1], 0.0, atol=tol_m)
        | np.isclose(xy[:, 1], a_m, atol=tol_m)
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="w", node_ids=boundary.tolist(), value=0.0)
    )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=float(args.pressure)))

    k = assemble_stiffness_matrix(model)
    f = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(k, f, bc_ess, bc_val)

    centre = np.array([0.5 * a_m, 0.5 * a_m])
    centre_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre, axis=1)))
    w_centre_m = float(u[centre_node])

    w_navier_m = kirchhoff_ssss_uniform_load_center_deflection_square(
        side=a_m,
        pressure=abs(float(args.pressure)),
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        thickness=args.thickness,
    )
    rel = abs((w_centre_m + w_navier_m) / w_navier_m) if w_navier_m != 0.0 else float("nan")

    print("=== SSSS square plate, uniform pressure (SI) ===")
    print(f"mesh: {args.nx} x {args.ny} Q8 elements")
    print(f"centre w (FE)     [m]: {w_centre_m:.8e}")
    print(f"centre w (Navier) [m]: {-w_navier_m:.8e}  (sign matched to FE)")
    print(f"relative |FE + w_Navier| / w_Navier: {rel:.4%}")


if __name__ == "__main__":
    main()
