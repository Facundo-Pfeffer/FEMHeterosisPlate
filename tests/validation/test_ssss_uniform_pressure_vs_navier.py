"""
SSSS square plate, uniform pressure: FE vs Kirchhoff–Navier centre deflection.

Classical problem: thin Kirchhoff plate on [0, a]², w = 0 and M_n = 0 on all edges, uniform q.
FE: essential w = 0 on the boundary w-nodes only; edge moments natural (heterosis plate weak form).

Sign convention: downward pressure is negative in the model ⇒ centre w < 0; Navier uses +|q| ⇒ compare to −w_Navier.

Tolerance rtol=0.006: shear in the heterosis element plus floating-point slack (thin: t/a = 1/200).
"""

from __future__ import annotations

import numpy as np

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.reference_solutions import kirchhoff_ssss_uniform_load_center_deflection_square
from plate_fea.solver import solve_linear_system


def test_ssss_uniform_pressure_center_matches_navier_series() -> None:
    # SI: midsurface coordinates x,y [m]; transverse displacement w [m].
    a_m = 1.0  # Span of simply supported square [m].
    nx = ny = 20  # element count per direction (structured mesh).

    young_pa = 200.0e9  # Young's modulus E [Pa] (~steel).
    nu = 0.3  # Poisson's ratio [-].
    thickness_m = 5.0e-3  # Thickness t = 5 mm [m]; t/a = 1/200.

    pressure_pa = -10.0e3  # Uniform pressure [Pa]; negative ⇒ downward (+z up).

    mesh = generate_rectangular_heterosis_mesh(width=a_m, height=a_m, nx=nx, ny=ny)
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=young_pa, poisson_ratio=nu, thickness=thickness_m
        ),
        element_formulation=HeterosisPlateElement(),
    )

    xy_m = mesh.node_coordinates
    x_m = xy_m[:, 0]
    y_m = xy_m[:, 1]
    geom_tol_m = 1.0e-9  # Node-on-edge tolerance [m].
    boundary_w = np.flatnonzero(
        np.isclose(x_m, 0.0, atol=geom_tol_m)
        | np.isclose(x_m, a_m, atol=geom_tol_m)
        | np.isclose(y_m, 0.0, atol=geom_tol_m)
        | np.isclose(y_m, a_m, atol=geom_tol_m)
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="w", node_ids=boundary_w.tolist(), value=0.0)
    )

    for element_id in range(mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=pressure_pa))

    k = assemble_stiffness_matrix(model)
    f = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()
    u = solve_linear_system(k, f, bc_ess, bc_val)  # u: w DOFs then θx, θy [rad]

    centre_m = np.array([0.5 * a_m, 0.5 * a_m])
    centre_w_node = int(np.argmin(np.linalg.norm(mesh.node_coordinates - centre_m, axis=1)))
    w_centre_m = float(u[centre_w_node])

    w_navier_m = kirchhoff_ssss_uniform_load_center_deflection_square(
        side=a_m,
        pressure=abs(pressure_pa),
        young_modulus=young_pa,
        poisson_ratio=nu,
        thickness=thickness_m,
    )

    np.testing.assert_allclose(
        w_centre_m,
        -w_navier_m,
        rtol=0.006,
        atol=0.0,
        err_msg="Centre w vs Navier (Kirchhoff) within shear + numerical margin",
    )
