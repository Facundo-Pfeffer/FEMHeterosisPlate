"""
Element-level Jacobian and stiffness checks used in normal assembly.

- Area Jacobian det(∂(x,y)/∂(ξ,η)) must be strictly positive at all stiffness (and surface)
  quadrature points for valid Q8 geometry (enforced in ``HeterosisPlateElement`` via
  ``positive_area_jacobian_det``).
- Local stiffness is symmetric; ``eigvalsh`` returns ``n_dof`` real eigenvalues; for an
  unconstrained free element, ``K`` is symmetric positive semi-definite. With default
  (3×3)/(2×2) bending/shear rules, exactly three eigenvalues are near-null (rigid modes);
  the deliberately under-integrated (2×2)/(1×1) case has extra mechanisms (12 near-null).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_fea.assembly import assemble_stiffness_matrix
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.quadrature import tensor_product_rule

LOCAL_DOF = 26
SPD_TOL = -1.0e-8
# Bending + shear blocks accumulate in different orders; expect symmetry only within float tolerance.
SYM_RTOL = 1.0e-12
SYM_ATOL = 1.0e-10

# Unconstrained heterosis plate (w, θx, θy): three rigid null modes. Count uses λ_max as scale
# so the check is stable with material scaling.
EXPECTED_RIGID_MODES = 3
NULL_EIGENVALUE_REL_TOL = 1.0e-8


def _near_null_eigenvalue_count(lam: np.ndarray, *, rel: float) -> int:
    lam_max = float(np.max(lam))
    scale = max(lam_max, 1.0)
    tol = scale * rel
    return int(np.sum(lam < tol))


def _parent_points_for_stiffness_orders(bx: int, by: int, sx: int, sy: int) -> np.ndarray:
    b = tensor_product_rule(order_x=bx, order_y=by)
    s = tensor_product_rule(order_x=sx, order_y=sy)
    return np.vstack([b.points, s.points])


@pytest.mark.parametrize("nx,ny", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize(
    "orders",
    [
        (3, 3, 2, 2),
        (2, 2, 2, 2),
        (3, 3, 1, 1),
    ],
)
def test_area_jacobian_det_positive_at_all_stiffness_quadrature_points(
    nx: int,
    ny: int,
    orders: tuple[int, int, int, int],
) -> None:
    mesh = generate_rectangular_heterosis_mesh(width=2.0, height=1.25, nx=nx, ny=ny)
    el = HeterosisPlateElement()
    bx, by, sx, sy = orders
    pts = _parent_points_for_stiffness_orders(bx, by, sx, sy)
    for element_id in range(mesh.total_element_number):
        geom = mesh.get_geometry_coordinates(element_id)
        for row in pts:
            xi, eta = float(row[0]), float(row[1])
            J = el.geometry_jacobian(xi, eta, geom)
            det = el.positive_area_jacobian_det(J, element_id, context="unit test sweep")
            assert det > 0.0


@pytest.mark.parametrize("nx,ny", [(1, 1), (2, 3)])
def test_surface_pressure_quadrature_jacobian_det_positive(nx: int, ny: int) -> None:
    mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=nx, ny=ny)
    el = HeterosisPlateElement()
    area_rule = tensor_product_rule(order_x=3, order_y=3)
    for element_id in range(mesh.total_element_number):
        geom = mesh.get_geometry_coordinates(element_id)
        for point, _w in zip(area_rule.points, area_rule.weights):
            xi, eta = float(point[0]), float(point[1])
            J = el.geometry_jacobian(xi, eta, geom)
            det = el.positive_area_jacobian_det(
                J, element_id, context="surface pressure unit test",
            )
            assert det > 0.0


@pytest.mark.parametrize("nx,ny", [(1, 1), (2, 2)])
def test_element_stiffness_symmetric_positive_semidefinite(nx: int, ny: int) -> None:
    mesh = generate_rectangular_heterosis_mesh(width=1.5, height=1.0, nx=nx, ny=ny)
    material = PlateMaterial(young_modulus=210.0, poisson_ratio=0.3, thickness=0.2)
    el = HeterosisPlateElement()
    for element_id in range(mesh.total_element_number):
        K = el.compute_stiffness_matrix(mesh, material, element_id)
        assert K.shape == (LOCAL_DOF, LOCAL_DOF)
        assert np.allclose(K, K.T, rtol=SYM_RTOL, atol=SYM_ATOL)
        lam = np.linalg.eigvalsh(0.5 * (K + K.T))
        assert lam.size == LOCAL_DOF
        assert np.all(lam >= SPD_TOL)
        n_null = _near_null_eigenvalue_count(lam, rel=NULL_EIGENVALUE_REL_TOL)
        assert n_null == EXPECTED_RIGID_MODES


@pytest.mark.parametrize(
    "bend,shear",
    [
        ((3, 3), (2, 2)),
        ((2, 2), (1, 1)),
    ],
)
def test_element_stiffness_symmetric_with_quadrature_kwargs(
    bend: tuple[int, int],
    shear: tuple[int, int],
) -> None:
    mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=2, ny=2)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.15)
    el = HeterosisPlateElement()
    kw = {"bending_quadrature_order": bend, "shear_quadrature_order": shear}
    K = el.compute_stiffness_matrix(mesh, material, 0, **kw)
    assert K.shape == (LOCAL_DOF, LOCAL_DOF)
    assert np.allclose(K, K.T, rtol=SYM_RTOL, atol=SYM_ATOL)
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert lam.size == LOCAL_DOF
    assert np.all(lam >= SPD_TOL)
    n_null = _near_null_eigenvalue_count(lam, rel=NULL_EIGENVALUE_REL_TOL)
    if bend == (3, 3) and shear == (2, 2):
        assert n_null == EXPECTED_RIGID_MODES
    else:
        # (2, 2) bending + (1, 1) shear: under-integrated shear — extra zero-energy modes.
        assert bend == (2, 2) and shear == (1, 1)
        assert n_null == 12


def test_global_assembled_K_is_symmetric() -> None:
    mesh = generate_rectangular_heterosis_mesh(width=2.0, height=1.0, nx=3, ny=2)
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    model = PlateModel(mesh=mesh, constitutive_material=material, element_formulation=HeterosisPlateElement())
    K = assemble_stiffness_matrix(model)
    diff = K - K.T
    assert diff.nnz == 0 or float(np.max(np.abs(diff.data))) < 1.0e-12
