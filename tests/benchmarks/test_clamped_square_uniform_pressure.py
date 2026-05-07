"""
CCCC square plate, uniform pressure: FE vs Kirchhoff centre deflection.

Classical thin plate: all edges clamped (w = ∂w/∂n = 0 in Kirchhoff); uniform q.
FE: w = θ_x = θ_y = 0 on boundary w-nodes (clamped heterosis plate).

Reference:
1) Taylor-Govindjee double-cosine Ritz center value (200 terms).
2) Classical tabulated Kirchhoff coefficient β ≈ 0.00126532 (ν ≈ 0.3).

Sign: model pressure < 0 ⇒ w < 0; reference uses +|q| ⇒ compare to −w_ref.

rtol=0.002: heterosis plate shear + Q8 mesh vs thin-plate reference (same SI case as SSSS test).
"""

from __future__ import annotations

import numpy as np

from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_displacement_system


def _taylor_govindjee_clamped_square_center_deflection(
    *,
    side_m: float,
    pressure_pa: float,
    bending_stiffness: float,
    terms: int = 200,
) -> float:
    if side_m <= 0.0:
        raise ValueError("side_m must be positive")
    if bending_stiffness <= 0.0:
        raise ValueError("bending_stiffness must be positive")
    if terms < 1:
        raise ValueError("terms must be >= 1")

    m = np.arange(1, terms + 1, dtype=np.float64)
    n = np.arange(1, terms + 1, dtype=np.float64)
    m_grid = m[None, :]
    n_grid = n[:, None]

    # Square plate: a=b => aspect=1.
    a_inv = 1.0 / (m_grid**2 + n_grid**2) ** 2
    load_val = pressure_pa * side_m**4 / (4.0 * np.pi**4 * bending_stiffness)
    a_inv_b = load_val * a_inv

    d_diag = np.concatenate((2.0 * m**4, 2.0 * n**4))
    size = 2 * terms
    reduced = np.zeros((size, size), dtype=np.float64)
    reduced[:terms, :terms] = np.diag(a_inv.sum(axis=0))
    reduced[:terms, terms:] = a_inv.T
    reduced[terms:, :terms] = a_inv
    reduced[terms:, terms:] = np.diag(a_inv.sum(axis=1))
    reduced += np.diag(1.0 / d_diag)

    rhs = np.concatenate((a_inv_b.sum(axis=0), a_inv_b.sum(axis=1)))
    scale = 1.0 / np.sqrt(np.diag(reduced))
    z = np.linalg.solve(scale[:, None] * reduced * scale[None, :], scale * rhs)
    sol = scale * z

    correction = (sol[:terms][None, :] + sol[terms:][:, None]) * a_inv
    coeff = a_inv_b - correction

    # At center x=y=a/2: basis terms are 1-cos(m*pi).
    basis = 1.0 - np.cos(np.pi * m)
    return float(basis @ coeff @ basis)


def test_clamped_square_uniform_pressure_center_matches_kirchhoff_factor() -> None:
    a_m = 1.0
    nx = ny = 20

    young_pa = 200.0e9
    nu = 0.3
    thickness_m = 5.0e-3
    pressure_pa = -10.0e3

    mesh = generate_rectangular_heterosis_mesh(width=a_m, height=a_m, nx=nx, ny=ny)
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=young_pa, poisson_ratio=nu, thickness=thickness_m
        ),
        element_formulation=HeterosisPlateElement(),
    )

    xy = mesh.node_coordinates
    x_m = xy[:, 0]
    y_m = xy[:, 1]
    geom_tol_m = 1.0e-9
    boundary_w = np.flatnonzero(
        np.isclose(x_m, 0.0, atol=geom_tol_m)
        | np.isclose(x_m, a_m, atol=geom_tol_m)
        | np.isclose(y_m, 0.0, atol=geom_tol_m)
        | np.isclose(y_m, a_m, atol=geom_tol_m)
    )
    for field_name in ("w", "theta_x", "theta_y"):
        model.add_essential_condition(
            EssentialBoundaryCondition(
                field_name=field_name, node_ids=boundary_w.tolist(), value=0.0
            )
        )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=pressure_pa))

    _, _, displacement = solve_displacement_system(model)

    centre_m = np.array([0.5 * a_m, 0.5 * a_m])
    centre_w_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre_m, axis=1)))
    w_centre_m = float(displacement[centre_w_node])

    # Kirchhoff CCCC square center deflection references.
    d = young_pa * thickness_m**3 / (12.0 * (1.0 - nu**2))
    beta_tabulated = 0.00126532
    w_kirchhoff_tabulated_m = -beta_tabulated * abs(pressure_pa) * a_m**4 / d

    w_taylor_govindjee_m = _taylor_govindjee_clamped_square_center_deflection(
        side_m=a_m,
        pressure_pa=pressure_pa,
        bending_stiffness=d,
        terms=200,
    )

    assert w_centre_m < 0.0

    np.testing.assert_allclose(
        w_taylor_govindjee_m,
        w_kirchhoff_tabulated_m,
        rtol=1.0e-6,
        atol=0.0,
        err_msg="Taylor-Govindjee 200-term reference vs tabulated Kirchhoff beta",
    )

    np.testing.assert_allclose(
        w_centre_m,
        w_taylor_govindjee_m,
        rtol=0.002,
        atol=0.0,
        err_msg="Centre w vs Taylor-Govindjee/Kirchhoff clamped plate reference",
    )
