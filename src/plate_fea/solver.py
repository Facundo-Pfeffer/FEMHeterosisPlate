"""
Constrained linear-system solve for the assembled plate system.

Partitions K and F into free/prescribed DOF blocks, forms the reduced system
K_ff u_f = F_f − K_fc u_c, and solves via sparse direct factorisation. Returns the full
displacement vector with prescribed values already inserted.  solve_displacement_system
is the convenience wrapper that assembles K and F from a PlateModel before solving.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.model import PlateModel


def solve_linear_system(
    K: csr_matrix, F: np.ndarray, prescribed_dof_indices: np.ndarray, prescribed_values: np.ndarray
) -> np.ndarray:
    total_dof_number = F.size
    displacement = np.zeros(total_dof_number, dtype=float)
    displacement[prescribed_dof_indices] = prescribed_values

    free_mask = np.ones(total_dof_number, dtype=bool)
    free_mask[prescribed_dof_indices] = False
    free_dof_ids = np.flatnonzero(free_mask)

    # K_ff u_f = F_f - K_fc u_c  (fixed DOFs c, free DOFs f)
    K_ff = K[free_dof_ids][:, free_dof_ids]
    K_fc = K[free_dof_ids][:, prescribed_dof_indices]
    F_f = F[free_dof_ids] - K_fc @ prescribed_values

    displacement[free_dof_ids] = spsolve(K_ff, F_f)
    return displacement


def solve_displacement_system(model: PlateModel) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """End-to-end equilibrium solve: K and F from ``model``, then u with essential BCs applied."""
    stiffness = assemble_stiffness_matrix(model)
    force = assemble_force_vector(model)
    prescribed_dof_indices, prescribed_values = model.build_essential_boundary_arrays()
    displacement = solve_linear_system(stiffness, force, prescribed_dof_indices, prescribed_values)
    return stiffness, force, displacement
