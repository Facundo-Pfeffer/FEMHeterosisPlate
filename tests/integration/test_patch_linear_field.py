from __future__ import annotations

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_linear_system


def test_linear_patch_closed_form_solution() -> None:
    """Exact linear-field patch test for the implemented heterosis element."""
    mesh = generate_rectangular_heterosis_mesh(width=2.0, height=1.0, nx=4, ny=3)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    model = PlateModel(mesh=mesh, constitutive_material=material, element_formulation=HeterosisPlateElement())

    a, b, c = 0.7, -0.35, 0.15
    xy = mesh.node_coordinates
    w_exact = a * xy[:, 0] + b * xy[:, 1] + c

    x = xy[:, 0]
    y = xy[:, 1]
    on_boundary = np.isclose(x, 0.0) | np.isclose(x, 2.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
    boundary_ids = np.flatnonzero(on_boundary)

    for node_id in boundary_ids.tolist():
        model.add_essential_condition(EssentialBoundaryCondition(field_name="w", node_ids=[node_id], value=float(w_exact[node_id])))
        model.add_essential_condition(EssentialBoundaryCondition(field_name="theta_x", node_ids=[node_id], value=a))
        model.add_essential_condition(EssentialBoundaryCondition(field_name="theta_y", node_ids=[node_id], value=b))

    K = assemble_stiffness_matrix(model)
    F = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(K, F, bc_ess, bc_val)

    n_w = mesh.total_w_node_number
    w_num = u[:n_w]
    theta_x_num = u[n_w::2]
    theta_y_num = u[n_w + 1 :: 2]

    assert np.max(np.abs(w_num - w_exact)) < 1.0e-8
    assert np.max(np.abs(theta_x_num - a)) < 1.0e-8
    assert np.max(np.abs(theta_y_num - b)) < 1.0e-8

