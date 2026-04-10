"""
Closed-form feasibility check: linear-field patch test (Heterosis plate model).

Exact solution used:
    w(x,y) = a*x + b*y + c
    theta_x = a
    theta_y = b
with zero external load. This gives:
    kappa = 0, gamma = grad(w)-theta = 0
so it is an exact equilibrium solution.
"""

from __future__ import annotations

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_linear_system


def main() -> None:
    # Mesh and material
    mesh = generate_rectangular_heterosis_mesh(width=2.0, height=1.0, nx=4, ny=3)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    model = PlateModel(mesh=mesh, material=material, element=HeterosisPlateElement())

    # Exact closed-form linear field
    a, b, c = 0.7, -0.35, 0.15
    xy = mesh.node_coordinates
    w_exact = a * xy[:, 0] + b * xy[:, 1] + c

    # Apply exact values only on boundary nodes.
    x = xy[:, 0]
    y = xy[:, 1]
    on_boundary = np.isclose(x, 0.0) | np.isclose(x, 2.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
    boundary_ids = np.flatnonzero(on_boundary)

    for node_id in boundary_ids.tolist():
        model.add_essential_condition(EssentialBoundaryCondition(field_name="w", node_ids=[node_id], value=float(w_exact[node_id])))
        model.add_essential_condition(EssentialBoundaryCondition(field_name="theta_x", node_ids=[node_id], value=a))
        model.add_essential_condition(EssentialBoundaryCondition(field_name="theta_y", node_ids=[node_id], value=b))

    # No external loads
    K = assemble_stiffness_matrix(model)
    F = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(K, F, bc_ess, bc_val)

    # Extract solved fields
    n_w = mesh.total_w_node_number
    w_num = u[:n_w]
    theta_x_num = u[n_w::2]
    theta_y_num = u[n_w + 1 :: 2]

    # Exact theta field at theta nodes (same values at every point for this linear patch).
    theta_x_exact = np.full_like(theta_x_num, a, dtype=float)
    theta_y_exact = np.full_like(theta_y_num, b, dtype=float)

    err_w = float(np.max(np.abs(w_num - w_exact)))
    err_tx = float(np.max(np.abs(theta_x_num - theta_x_exact)))
    err_ty = float(np.max(np.abs(theta_y_num - theta_y_exact)))

    print("=== Linear Patch Test (closed-form) ===")
    print(f"max|w_num - w_exact|        = {err_w:.6e}")
    print(f"max|theta_x_num - exact|    = {err_tx:.6e}")
    print(f"max|theta_y_num - exact|    = {err_ty:.6e}")


if __name__ == "__main__":
    main()

