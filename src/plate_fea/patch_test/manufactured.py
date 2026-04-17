"""
Manufactured nodal loads for higher-order displacement patterns.

For any global vector ``u``, defining ``f_mfg := K @ u`` makes ``K u - f_mfg`` vanish exactly
(Test A with that load vector). No physical body force or traction need be derived: this checks
that assembly of ``K`` is consistent with arbitrary ``u`` (e.g. quadratic ``w`` with ``θ = ∇w``).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from plate_fea.assembly import assemble_stiffness_matrix
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel

from plate_fea.patch_test.diagnostics import residual_norm


def manufactured_force_from_displacement(
    mesh: HeterosisMesh,
    material: PlateMaterial,
    u: np.ndarray,
    *,
    element_stiffness_kwargs: dict[str, object] | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Build ``K`` on ``mesh`` with ``material`` and return ``f_mfg = K @ u``.

    The model has no essential BCs and no separate load assembly: ``f_mfg`` is purely ``K @ u``.
    """
    model = PlateModel(
        mesh=mesh,
        constitutive_material=material,
        element_formulation=HeterosisPlateElement(),
        element_stiffness_kwargs=dict(element_stiffness_kwargs or {}),
    )
    K = assemble_stiffness_matrix(model)
    f_mfg = K @ u
    return K, f_mfg


def patch_test_a_residual_norm(
    K: csr_matrix,
    u: np.ndarray,
    f: np.ndarray,
) -> float:
    """Euclidean norm ``||K u - f||`` (Test A residual for a chosen ``u`` and load ``f``)."""
    return residual_norm(K, u, f)
