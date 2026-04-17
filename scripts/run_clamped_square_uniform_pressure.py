"""
CCCC square plate under uniform pressure: FE vs Kirchhoff tabulated centre factor β.

SI defaults match ``tests/validation/test_clamped_square_uniform_pressure.py``.
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
from plate_fea.reference_solutions import kirchhoff_cccc_uniform_load_center_deflection_square
from plate_fea.solver import solve_linear_system


def main() -> None:
    parser = argparse.ArgumentParser(description="CCCC square plate, uniform pressure vs Kirchhoff β (SI).")
    parser.add_argument("--side", type=float, default=1.0, help="Edge length a [m].")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--pressure", type=float, default=-10.0e3, help="Uniform pressure [Pa].")
    parser.add_argument("--young-modulus", type=float, default=200.0e9, help="E [Pa].")
    parser.add_argument("--poisson-ratio", type=float, default=0.3, help="ν [-].")
    parser.add_argument("--thickness", type=float, default=5.0e-3, help="t [m].")
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
    for field_name in ("w", "theta_x", "theta_y"):
        model.add_essential_condition(
            EssentialBoundaryCondition(field_name=field_name, node_ids=boundary.tolist(), value=0.0)
        )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=float(args.pressure)))

    k = assemble_stiffness_matrix(model)
    f = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(k, f, bc_ess, bc_val)

    centre = np.array([0.5 * a_m, 0.5 * a_m])
    centre_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre, axis=1)))
    w_fe_m = float(u[centre_node])

    w_k_m = kirchhoff_cccc_uniform_load_center_deflection_square(
        side=a_m,
        pressure=abs(float(args.pressure)),
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        thickness=args.thickness,
    )
    rel = abs((w_fe_m + w_k_m) / w_k_m) if w_k_m != 0.0 else float("nan")

    print("=== CCCC square plate, uniform pressure (SI) ===")
    print(f"Kirchhoff β (centre) ≈ 0.00126532  (ν ≈ 0.3)")
    print(f"centre w (FE)        [m]: {w_fe_m:.8e}")
    print(f"centre w (Kirchhoff) [m]: {-w_k_m:.8e}  (sign matched to FE)")
    print(f"relative |FE + w_K| / w_K: {rel:.4%}")


if __name__ == "__main__":
    main()
