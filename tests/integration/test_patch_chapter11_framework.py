"""Fast pytest coverage for the patch-test suite (Ch. 11, Zienkiewicz et al., *FEM: Basis and Fundamentals*, 8th ed.)."""

from __future__ import annotations

import numpy as np

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.patch_test.engine import (
    PatchTestSuiteConfig,
    acceptance_all_pass,
    run_patch_test_suite,
)
from plate_fea.patch_test.manufactured import manufactured_force_from_displacement, patch_test_a_residual_norm
from plate_fea.patch_test.nodal_vector import assemble_quadratic_patch_displacement, assemble_linear_patch_displacement
from plate_fea.patch_test.plate_exact_states import LinearHeterosisPatchState, quadratic_gamma_free_catalog
from plate_fea.model import PlateModel


def test_chapter11_suite_smoke_fast() -> None:
    cfg = PatchTestSuiteConfig(
        run_single_element=True,
        run_multi_element=True,
        run_distorted=False,
        test_all_linear_modes=False,
        full_quadrature_sweep=False,
        nodal_tol=1.0e-6,
        residual_tol=1.0e-5,
    )
    reports = run_patch_test_suite(cfg)
    assert acceptance_all_pass(reports)


def test_manufactured_quadratic_force_balance() -> None:
    """f_mfg := K ũ ⇒ Test A residual vanishes (assembly consistency for higher-order displacement pattern)."""
    mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=2, ny=2)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    model = PlateModel(mesh=mesh, constitutive_material=material, element_formulation=HeterosisPlateElement())
    for st in quadratic_gamma_free_catalog():
        u = assemble_quadratic_patch_displacement(mesh, model, st)
        K, f = manufactured_force_from_displacement(mesh, material, u)
        r = patch_test_a_residual_norm(K, u, f)
        assert r < 1.0e-8


def test_linear_nodal_vector_three_modes() -> None:
    mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=1, ny=1)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    model = PlateModel(mesh=mesh, constitutive_material=material, element_formulation=HeterosisPlateElement())
    for st in (
        LinearHeterosisPatchState(1.0, 0.0, 0.0),
        LinearHeterosisPatchState(0.0, 1.0, 0.0),
        LinearHeterosisPatchState(0.0, 0.0, 1.0),
    ):
        u = assemble_linear_patch_displacement(mesh, model, st)
        xy = mesh.node_coordinates
        w_exp = st.w(xy[:, 0], xy[:, 1])
        assert np.allclose(u[: mesh.total_w_node_number], w_exp)
