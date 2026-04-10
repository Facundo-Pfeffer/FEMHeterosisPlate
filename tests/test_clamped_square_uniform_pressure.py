from __future__ import annotations

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_q8_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_linear_system


def test_clamped_square_uniform_pressure_center_is_downward() -> None:
    mesh = generate_rectangular_q8_mesh(width=1.0, height=1.0, nx=6, ny=6)
    model = PlateModel(
        mesh=mesh,
        material=PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2),
        element=HeterosisPlateElement(),
    )

    xy = mesh.node_coordinates
    x = xy[:, 0]
    y = xy[:, 1]
    tol = 1.0e-10
    boundary = np.flatnonzero(
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, 1.0, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, 1.0, atol=tol)
    )
    for field_name in ("w", "theta_x", "theta_y"):
        model.add_essential_condition(EssentialBoundaryCondition(field_name=field_name, node_ids=boundary.tolist(), value=0.0))

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=-1.0))

    K = assemble_stiffness_matrix(model)
    F = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(K, F, bc_ess, bc_val)

    center = np.array([0.5, 0.5], dtype=float)
    center_id = int(np.argmin(np.linalg.norm(mesh.node_coordinates - center[None, :], axis=1)))
    w_center = float(u[center_id])

    assert np.isfinite(w_center)
    assert w_center < 0.0

