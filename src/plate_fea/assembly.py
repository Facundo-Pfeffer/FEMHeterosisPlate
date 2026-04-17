"""Global assembly: scatter element stiffness and consistent nodal loads into K and F."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from plate_fea.model import PlateModel


def assemble_stiffness_matrix(model: PlateModel) -> csr_matrix:
    total_dof_number = model.mesh.total_dof_number
    # lil_matrix: efficient incremental fill; convert to CSR once for the linear solve.
    K = lil_matrix((total_dof_number, total_dof_number), dtype=float)

    kw = getattr(model, "element_stiffness_kwargs", None) or {}
    for element_id in range(model.mesh.total_element_number):
        k_local = model.element_formulation.compute_stiffness_matrix(
            model.mesh, model.constitutive_material, element_id, **kw
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, element_id)
        K[np.ix_(element_dof_ids, element_dof_ids)] += k_local

    return K.tocsr()


def assemble_force_vector(model: PlateModel) -> np.ndarray:
    """Sum edge tractions and surface pressures into the global load vector F."""
    F = np.zeros(model.mesh.total_dof_number, dtype=float)

    for load in model.line_loads:
        f_local = model.element_formulation.compute_edge_force_vector(
            model.mesh,
            load.element_id,
            load.edge_id,
            load.magnitude,
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += f_local

    for load in model.surface_loads:
        f_local = model.element_formulation.compute_surface_force_vector(
            model.mesh,
            load.element_id,
            load.magnitude,
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += f_local

    return F
