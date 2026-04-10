from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve


def solve_linear_system(K, F: np.ndarray, bc_ess: np.ndarray, bc_val: np.ndarray) -> np.ndarray:
    total_dof_number = F.size
    u = np.zeros(total_dof_number, dtype=float)
    u[bc_ess] = bc_val

    free_mask = np.ones(total_dof_number, dtype=bool)
    free_mask[bc_ess] = False
    free_dof_ids = np.flatnonzero(free_mask)

    K_ff = K[free_dof_ids][:, free_dof_ids]
    K_fc = K[free_dof_ids][:, bc_ess]
    F_f = F[free_dof_ids] - K_fc @ bc_val

    u[free_dof_ids] = spsolve(K_ff, F_f)
    return u
