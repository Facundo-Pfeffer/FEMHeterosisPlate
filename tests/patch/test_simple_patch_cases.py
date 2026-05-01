"""
Simple patch tests on the distorted 5-element patch.

Two tests, each targeting one strain group independently:

  1. Constant shear  — w linear, θ constant.
     Expected: γ_xz and γ_yz constant; κ_xx = κ_yy = κ_xy = 0.
     Sampled at the 2×2 shear integration points.

  2. Constant kappa  — θ linear, w = 0.
     Expected: κ_xx, κ_yy, κ_xy constant.
     Sampled at the 3×3 bending integration points.

Both fields are exactly representable by Q8/Q9 on any element geometry, so all
strain components must agree with the analytical field to machine precision (1e-10).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.model import PlateModel

from ._helpers import (
    ConstantKappaKinematicField,
    ConstantShearKinematicField,
    assemble_constant_kappa_patch_displacement_vector,
    assemble_constant_shear_patch_displacement_vector,
    build_distorted_five_element_patch,
    compute_generalized_strains_at_quadrature_points,
    evaluate_expected_constant_kappa_strains,
    evaluate_expected_constant_shear_strains,
)

pytestmark = pytest.mark.patch


def _build_model() -> tuple:
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2),
        element_formulation=HeterosisPlateElement(),
    )
    return mesh, model


def test_constant_shear_exact_recovery() -> None:
    """
    w linear, θ constant → γ constant and κ = 0 at the 2×2 shear quadrature points.

    The 2×2 sampling matches the element's selective shear integration rule.
    Since θ is constant its gradients are identically zero, so κ = 0 everywhere.
    Since w is linear ∂w/∂x and ∂w/∂y are constant, giving constant γ.
    """
    mesh, model = _build_model()
    field = ConstantShearKinematicField()

    displacement, _ = assemble_constant_shear_patch_displacement_vector(mesh, model, field)
    sampled = compute_generalized_strains_at_quadrature_points(mesh, model, displacement, quadrature_order=(2, 2))
    expected = evaluate_expected_constant_shear_strains(sampled, field)

    for name in ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz"):
        np.testing.assert_allclose(
            sampled[name], expected[name], atol=1e-10,
            err_msg=f"{name}: FEM vs analytical exceeds tolerance",
        )


def test_constant_kappa_exact_recovery() -> None:
    """
    θ linear, w = 0 → κ_xx, κ_yy, κ_xy constant at the 3×3 bending quadrature points.

    The 3×3 sampling matches the element's full bending integration rule.
    Linear rotations have constant gradients, so all three curvatures are exactly
    constant and must be recovered to machine precision.
    """
    mesh, model = _build_model()
    field = ConstantKappaKinematicField()

    displacement, _ = assemble_constant_kappa_patch_displacement_vector(mesh, model, field)
    sampled = compute_generalized_strains_at_quadrature_points(mesh, model, displacement, quadrature_order=(3, 3))
    expected = evaluate_expected_constant_kappa_strains(sampled, field)

    for name in ("kappa_xx", "kappa_yy", "kappa_xy"):
        np.testing.assert_allclose(
            sampled[name], expected[name], atol=1e-10,
            err_msg=f"{name}: FEM vs analytical exceeds tolerance",
        )
