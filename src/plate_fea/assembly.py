"""
Global stiffness and load assembly.

Iterates over elements, calls compute_stiffness_matrix / compute_*_force_vector on each,
and scatters the local (26×26 or 26-vector) results into the global sparse K and dense F
using local_to_global_dof_indices. The two public functions are the entry point between
the element layer and the solver layer.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from plate_fea.model import PlateModel


def assemble_stiffness_matrix(model: PlateModel) -> csr_matrix:
    """Assemble the global sparse stiffness matrix from all element contributions."""
    total_dof_number = model.mesh.total_dof_number
    # lil_matrix for efficient incremental scatter; converted to CSR once assembly is done.
    K = lil_matrix((total_dof_number, total_dof_number), dtype=float)

    for element_id in range(model.mesh.total_element_number):
        element_stiffness_matrix = model.element_formulation.compute_stiffness_matrix(
            model.mesh, model.constitutive_material, element_id, **model.element_stiffness_kwargs
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, element_id)
        K[np.ix_(element_dof_ids, element_dof_ids)] += element_stiffness_matrix

    return K.tocsr()


def assemble_force_vector(model: PlateModel) -> np.ndarray:
    """Sum edge tractions and surface pressures into the global load vector F."""
    F = np.zeros(model.mesh.total_dof_number, dtype=float)

    for load in model.line_loads:
        element_force_vector = model.element_formulation.compute_edge_force_vector(
            model.mesh,
            load.element_id,
            load.edge_id,
            load.magnitude,
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += element_force_vector

    for load in model.surface_loads:
        element_force_vector = model.element_formulation.compute_surface_force_vector(
            model.mesh,
            load.element_id,
            load.magnitude,
        )
        element_dof_ids = model.element_formulation.local_to_global_dof_indices(model.mesh, load.element_id)
        F[element_dof_ids] += element_force_vector

    return F
