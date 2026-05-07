"""
CCCC square plate, uniform pressure: FE vs Kirchhoff centre deflection.

Classical thin plate: all edges clamped (w = ∂w/∂n = 0 in Kirchhoff); uniform q.
FE: w = theta_x = theta_y = 0 on boundary w-nodes (clamped heterosis plate).

Reference:
1. Kirchhoff tabulated coefficient: w = beta q a^4 / D, beta ~= 0.00126532.
2. Taylor-Govindjee double-cosine Ritz solution for the same clamped
   Kirchhoff plate problem, solved through the Sherman-Morrison-Woodbury
   reduction.

Sign: model pressure < 0 implies w < 0; reference uses +|q|, then the sign is restored.

rtol=0.002: heterosis plate shear + Q8 mesh vs thin-plate reference.
"""

from __future__ import annotations

import numpy as np

from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_displacement_system


def _taylor_govindjee_clamped_rectangular_plate_deflection(
    *,
    x_m: float,
    y_m: float,
    width_m: float,
    height_m: float,
    pressure_pa: float,
    bending_stiffness: float,
    terms_x: int = 200,
    terms_y: int = 200,
) -> float:
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Plate dimensions must be positive.")
    if bending_stiffness <= 0.0:
        raise ValueError("Plate bending stiffness must be positive.")
    if terms_x < 1 or terms_y < 1:
        raise ValueError("The number of series terms must be positive.")

    xi = x_m / width_m
    eta = y_m / height_m
    if not (0.0 <= xi <= 1.0 and 0.0 <= eta <= 1.0):
        raise ValueError("The evaluation point must lie within the plate domain.")

    m = np.arange(1, terms_x + 1, dtype=np.float64)
    n = np.arange(1, terms_y + 1, dtype=np.float64)

    m_grid = m[None, :]
    n_grid = n[:, None]

    aspect = width_m / height_m

    a_diag = (m_grid**2 + (aspect * n_grid) ** 2) ** 2
    a_inv = 1.0 / a_diag

    load_vector_value = (
        pressure_pa * width_m**4 / (4.0 * np.pi**4 * bending_stiffness)
    )
    a_inv_b = load_vector_value * a_inv

    d_diag = np.concatenate(
        (
            2.0 * m**4,
            2.0 * (aspect * n) ** 4,
        )
    )

    reduced_size = terms_x + terms_y
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=np.float64)

    reduced_matrix[:terms_x, :terms_x] = np.diag(a_inv.sum(axis=0))
    reduced_matrix[:terms_x, terms_x:] = a_inv.T
    reduced_matrix[terms_x:, :terms_x] = a_inv
    reduced_matrix[terms_x:, terms_x:] = np.diag(a_inv.sum(axis=1))

    reduced_matrix += np.diag(1.0 / d_diag)

    reduced_rhs = np.concatenate(
        (
            a_inv_b.sum(axis=0),
            a_inv_b.sum(axis=1),
        )
    )

    scale = 1.0 / np.sqrt(np.diag(reduced_matrix))
    scaled_matrix = scale[:, None] * reduced_matrix * scale[None, :]
    scaled_rhs = scale * reduced_rhs

    scaled_solution = np.linalg.solve(scaled_matrix, scaled_rhs)
    reduced_solution = scale * scaled_solution

    correction = (
        reduced_solution[:terms_x][None, :]
        + reduced_solution[terms_x:][:, None]
    ) * a_inv
    coefficients = a_inv_b - correction

    basis_x = 1.0 - np.cos(2.0 * np.pi * m * xi)
    basis_y = 1.0 - np.cos(2.0 * np.pi * n * eta)

    return float(basis_y @ coefficients @ basis_x)


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
            young_modulus=young_pa,
            poisson_ratio=nu,
            thickness=thickness_m,
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
                field_name=field_name,
                node_ids=boundary_w.tolist(),
                value=0.0,
            )
        )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(
            ElementSurfaceLoad(element_id=element_id, magnitude=pressure_pa)
        )

    _, _, displacement = solve_displacement_system(model)

    centre_m = np.array([0.5 * a_m, 0.5 * a_m])
    centre_w_node = int(
        np.argmin(np.linalg.norm(mesh.node_coordinates - centre_m, axis=1))
    )
    w_centre_m = float(displacement[centre_w_node])

    bending_stiffness = (
        young_pa * thickness_m**3 / (12.0 * (1.0 - nu**2))
    )

    beta_kirchhoff_tabulated = 0.00126532
    w_kirchhoff_tabulated_m = (
        -beta_kirchhoff_tabulated
        * abs(pressure_pa)
        * a_m**4
        / bending_stiffness
    )

    w_taylor_govindjee_m = _taylor_govindjee_clamped_rectangular_plate_deflection(
        x_m=0.5 * a_m,
        y_m=0.5 * a_m,
        width_m=a_m,
        height_m=a_m,
        pressure_pa=pressure_pa,
        bending_stiffness=bending_stiffness,
        terms_x=200,
        terms_y=200,
    )

    beta_taylor_govindjee_200_terms = 1.265319036e-3
    w_taylor_govindjee_table_m = (
        -beta_taylor_govindjee_200_terms
        * abs(pressure_pa)
        * a_m**4
        / bending_stiffness
    )

    np.testing.assert_allclose(
        w_taylor_govindjee_m,
        w_taylor_govindjee_table_m,
        rtol=1.0e-9,
        atol=0.0,
        err_msg="Taylor-Govindjee 200-term coefficient check",
    )

    np.testing.assert_allclose(
        w_taylor_govindjee_m,
        w_kirchhoff_tabulated_m,
        rtol=1.0e-6,
        atol=0.0,
        err_msg="Taylor-Govindjee reference vs Kirchhoff tabulated beta",
    )

    assert w_centre_m < 0.0

    np.testing.assert_allclose(
        w_centre_m,
        w_taylor_govindjee_m,
        rtol=0.002,
        atol=0.0,
        err_msg="Centre w vs Taylor-Govindjee clamped Kirchhoff plate reference",
    )