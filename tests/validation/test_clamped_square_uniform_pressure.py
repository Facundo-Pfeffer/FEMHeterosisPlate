"""
CCCC square plate, uniform pressure: FE vs Kirchhoff centre deflection (tabulated β).

Classical thin plate: all edges clamped (w = ∂w/∂n = 0 in Kirchhoff); uniform q.
FE: w = θ_x = θ_y = 0 on boundary w-nodes (clamped heterosis plate).

Reference: w = β q a⁴ / D with β ≈ 0.00126532 (ν ≈ 0.3), e.g. Timoshenko & Woinowsky-Krieger.

Sign: model pressure < 0 ⇒ w < 0; reference uses +|q| ⇒ compare to −w_ref.

rtol=0.002: heterosis plate shear + Q8 mesh vs thin-plate β (same SI case as the SSSS test).
"""

from __future__ import annotations

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.reference_solutions import kirchhoff_cccc_uniform_load_center_deflection_square
from plate_fea.solver import solve_linear_system


def test_clamped_square_uniform_pressure_center_matches_kirchhoff_factor() -> None:
    a_m = 1.0
    nx = ny = 20

    young_pa = 200.0e9
    nu = 0.3
    thickness_m = 5.0e-3
    pressure_pa = -10.0e3

    mesh = generate_rectangular_heterosis_mesh(width=a_m, height=a_m, nx=nx, ny=ny)
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=young_pa, poisson_ratio=nu, thickness=thickness_m
        ),
        element_formulation=HeterosisPlateElement(),
    )

    xy = mesh.node_coordinates
    x_m = xy[:, 0]
    y_m = xy[:, 1]
    geom_tol_m = 1.0e-9
    boundary_w = np.flatnonzero(
        np.isclose(x_m, 0.0, atol=geom_tol_m)
        | np.isclose(x_m, a_m, atol=geom_tol_m)
        | np.isclose(y_m, 0.0, atol=geom_tol_m)
        | np.isclose(y_m, a_m, atol=geom_tol_m)
    )
    for field_name in ("w", "theta_x", "theta_y"):
        model.add_essential_condition(
            EssentialBoundaryCondition(
                field_name=field_name, node_ids=boundary_w.tolist(), value=0.0
            )
        )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=pressure_pa))

    k = assemble_stiffness_matrix(model)
    f = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(k, f, bc_ess, bc_val)

    centre_m = np.array([0.5 * a_m, 0.5 * a_m])
    centre_w_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre_m, axis=1)))
    w_centre_m = float(u[centre_w_node])

    w_kirchhoff_m = kirchhoff_cccc_uniform_load_center_deflection_square(
        side=a_m,
        pressure=abs(pressure_pa),
        young_modulus=young_pa,
        poisson_ratio=nu,
        thickness=thickness_m,
    )

    assert w_centre_m < 0.0
    np.testing.assert_allclose(
        w_centre_m,
        -w_kirchhoff_m,
        rtol=0.002,
        atol=0.0,
        err_msg="Centre w vs Kirchhoff CCCC β factor (thin plate + shear/mesh margin)",
    )
