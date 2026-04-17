"""Tiny single-element sanity check: assemble K/F, apply BCs, solve, report timings."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_linear_system


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
    [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ],
    dtype=int,
)

mesh = HeterosisMesh.from_arrays(
    node_coordinates=node_coordinates,
    w_location_matrix=w_location_matrix,
)

material = PlateMaterial(
    young_modulus=200.0,
    poisson_ratio=0.25,
    thickness=20.0,
)

element = HeterosisPlateElement()
model = PlateModel(mesh=mesh, constitutive_material=material, element_formulation=element)

left_nodes = mesh.find_w_nodes_on_line(axis="x", value=0.0)
for field_name in ("w", "theta_x", "theta_y"):
    model.add_essential_condition(EssentialBoundaryCondition(field_name=field_name, node_ids=left_nodes, value=0.0))


def time_call(function_handle, repeats: int = 20):
    t0 = perf_counter()
    out = None
    for _ in range(repeats):
        out = function_handle()
    t1 = perf_counter()
    return (t1 - t0) / repeats, out


bc_ess, bc_val = model.build_essential_boundary_arrays()

tK, K = time_call(lambda: assemble_stiffness_matrix(model), repeats=20)
tF, F = time_call(lambda: assemble_force_vector(model), repeats=20)
tS, u = time_call(lambda: solve_linear_system(K, F, bc_ess, bc_val), repeats=20)

print(f"Total degrees of freedom: {mesh.total_dof_number}")
print(f"K assembly time (avg): {tK*1e3:.6f} ms")
print(f"F assembly time (avg): {tF*1e3:.6f} ms")
print(f"Solve time (avg):      {tS*1e3:.6f} ms")
print(f"Total (avg):           {(tK+tF+tS)*1e3:.6f} ms")
print(f"Symmetry check:        {np.linalg.norm((K - K.T).toarray()):.6e}")
print(f"Solution norm:         {np.linalg.norm(u):.6e}")
