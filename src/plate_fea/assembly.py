from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from plate_fea.model import PlateModel


def assemble_stiffness_matrix(model: PlateModel):
    total_dof_number = model.mesh.total_dof_number
    K = lil_matrix((total_dof_number, total_dof_number), dtype=float)

    for element_id in range(model.mesh.total_element_number):
        k_local = model.element.compute_stiffness_matrix(model.mesh, model.material, element_id)
        element_dof_ids = model.element.local_to_global_dof_indices(model.mesh, element_id)
        K[np.ix_(element_dof_ids, element_dof_ids)] += k_local

    return K.tocsr()


def assemble_force_vector(model: PlateModel) -> np.ndarray:
    F = np.zeros(model.mesh.total_dof_number, dtype=float)

    for load in model.line_loads:
        f_local = model.element.compute_edge_force_vector(
            model.mesh,
            load.element_id,
            load.edge_id,
            load.magnitude,
        )
        element_dof_ids = model.element.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += f_local

    for load in model.surface_loads:
        f_local = model.element.compute_surface_force_vector(
            model.mesh,
            load.element_id,
            load.magnitude,
        )
        element_dof_ids = model.element.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += f_local

    return F
