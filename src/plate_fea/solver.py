"""Constrained sparse solve: partition K and F by prescribed (essential) DOFs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.model import PlateModel


def solve_linear_system(
    K: csr_matrix, F: np.ndarray, bc_ess: np.ndarray, bc_val: np.ndarray
) -> np.ndarray:
    total_dof_number = F.size
    u = np.zeros(total_dof_number, dtype=float)
    u[bc_ess] = bc_val

    free_mask = np.ones(total_dof_number, dtype=bool)
    free_mask[bc_ess] = False
    free_dof_ids = np.flatnonzero(free_mask)

    # K_ff u_f = F_f - K_fc u_c  (fixed DOFs c, free DOFs f)
    K_ff = K[free_dof_ids][:, free_dof_ids]
    K_fc = K[free_dof_ids][:, bc_ess]
    F_f = F[free_dof_ids] - K_fc @ bc_val

    u[free_dof_ids] = spsolve(K_ff, F_f)
    return u


def solve_displacement_system(model: PlateModel) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """End-to-end equilibrium solve: K and F from ``model``, then u with essential BCs applied."""
    stiffness = assemble_stiffness_matrix(model)
    force = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    displacement = solve_linear_system(stiffness, force, bc_ess, bc_val)
    return stiffness, force, displacement
