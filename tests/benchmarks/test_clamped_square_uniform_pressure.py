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

from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_displacement_system


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

    _, _, displacement = solve_displacement_system(model)

    centre_m = np.array([0.5 * a_m, 0.5 * a_m])
    centre_w_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre_m, axis=1)))
    w_centre_m = float(displacement[centre_w_node])

    # Kirchhoff CCCC square centre deflection:
    # w = beta * q * a^4 / D, D = E t^3 / (12 (1 - nu^2)), beta ~= 0.00126532 for nu ~= 0.3.
    d = young_pa * thickness_m**3 / (12.0 * (1.0 - nu**2))
    beta = 0.00126532
    w_kirchhoff_m = beta * abs(pressure_pa) * a_m**4 / d

    assert w_centre_m < 0.0
    np.testing.assert_allclose(
        w_centre_m,
        -w_kirchhoff_m,
        rtol=0.002,
        atol=0.0,
        err_msg="Centre w vs Kirchhoff CCCC β factor (thin plate + shear/mesh margin)",
    )
