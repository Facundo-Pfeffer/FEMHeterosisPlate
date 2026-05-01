"""
Circular plate convergence test — Hughes & Cohen (1978), Figures 5.3.18–5.3.19.

Only one quadrant is discretised (Fig. 5.3.18); symmetry conditions are applied on the
two straight edges (x = 0 and y = 0).

Parameters (as stated by Hughes):
    E = 10.92 × 10⁵,  ν = 0.3,  R = 5,  t = 2

Two load/BC combinations:
    SS₁-U : simply-supported outer edge, uniform pressure
    CL-U  : clamped outer edge, uniform pressure

Reference solution — Reissner-Mindlin plate (first-order shear correction):
    w_M(r) = w_K(r) + q(R² − r²) / (4κGt)

where w_K is the Kirchhoff thin-plate solution and κ = 5/6 is the shear factor. The
same correction applies to both SS₁ and CL; only w_K differs between the two cases.

The mesh has a small inner hole (r_min = R / (4 n_el)) to avoid degenerate elements
at the origin. The FEM centre deflection is compared to the analytical solution
evaluated at r = r_min; the remaining quadrature error is O((r_min/R)²).

Convergence check: n_el = 6 must be within 5 % of the reference, and it must be
closer to the reference than n_el = 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_fea.boundary_conditions import ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_quarter_circle_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_displacement_system

pytestmark = pytest.mark.benchmark

# ── Problem parameters (Hughes) ────────────────────────────────────────────────
E   = 10.92e5
NU  = 0.3
R   = 5.0
T   = 2.0
Q   = -1.0          # uniform transverse pressure, downward
KAPPA = 5.0 / 6.0  # Reissner shear-correction factor

TOL = 1.0e-9        # node-on-boundary tolerance


# ── Analytical reference (Reissner-Mindlin first-order correction) ─────────────

def _flexural_rigidity() -> float:
    return E * T**3 / (12.0 * (1.0 - NU**2))


def _shear_stiffness() -> float:
    G = E / (2.0 * (1.0 + NU))
    return KAPPA * G * T


def w_mindlin_ss1(r: float) -> float:
    """Reissner-Mindlin centre deflection for SS₁, uniform load, evaluated at radius r."""
    D  = _flexural_rigidity()
    kGt = _shear_stiffness()
    r2, R2 = r**2, R**2
    w_kirchhoff = Q * (R2 - r2) * ((5 + NU) * R2 - (1 + NU) * r2) / (64.0 * D * (1.0 + NU))
    w_shear     = Q * (R2 - r2) / (4.0 * kGt)
    return w_kirchhoff + w_shear


def w_mindlin_cl(r: float) -> float:
    """Reissner-Mindlin centre deflection for CL, uniform load, evaluated at radius r."""
    D  = _flexural_rigidity()
    kGt = _shear_stiffness()
    r2, R2 = r**2, R**2
    w_kirchhoff = Q * (R2 - r2)**2 / (64.0 * D)
    w_shear     = Q * (R2 - r2) / (4.0 * kGt)
    return w_kirchhoff + w_shear


# ── FEM helpers ────────────────────────────────────────────────────────────────

def _build_model(n_el: int) -> tuple[PlateModel, float]:
    """Return (model, r_min) for a quarter-circle mesh with n_el elements per direction."""
    mesh = generate_quarter_circle_heterosis_mesh(R, n_el)
    material = PlateMaterial(young_modulus=E, poisson_ratio=NU, thickness=T)
    model = PlateModel(
        mesh=mesh,
        constitutive_material=material,
        element_formulation=HeterosisPlateElement(),
    )
    r_min = R / (4.0 * n_el)
    return model, r_min


def _apply_symmetry_bcs(model: PlateModel) -> None:
    """
    Symmetry on straight edges of the quarter disc:
        y = 0  (θ = 0):    θ_y = 0  (slope ∂w/∂y vanishes by symmetry)
        x = 0  (θ = π/2):  θ_x = 0  (slope ∂w/∂x vanishes by symmetry)
    """
    xy = model.mesh.node_coordinates
    on_x_axis  = np.flatnonzero(np.abs(xy[:, 1]) < TOL)   # y ≈ 0
    on_y_axis  = np.flatnonzero(np.abs(xy[:, 0]) < TOL)   # x ≈ 0

    model.add_essential_condition(
        EssentialBoundaryCondition("theta_y", on_x_axis.tolist(), 0.0)
    )
    model.add_essential_condition(
        EssentialBoundaryCondition("theta_x", on_y_axis.tolist(), 0.0)
    )


def _apply_outer_bc(model: PlateModel, bc: str) -> None:
    """
    Outer circular boundary (r ≈ R):
        SS₁ : w = 0  (hard simply-supported; M_n = 0 is a natural condition)
        CL  : w = θ_x = θ_y = 0
    """
    xy = model.mesh.node_coordinates
    r  = np.linalg.norm(xy, axis=1)
    on_outer = np.flatnonzero(np.abs(r - R) < TOL)

    model.add_essential_condition(
        EssentialBoundaryCondition("w", on_outer.tolist(), 0.0)
    )
    if bc == "clamped":
        for field in ("theta_x", "theta_y"):
            model.add_essential_condition(
                EssentialBoundaryCondition(field, on_outer.tolist(), 0.0)
            )


def _apply_uniform_load(model: PlateModel) -> None:
    for element_id in range(model.mesh.total_element_number):
        model.add_surface_load(ElementSurfaceLoad(element_id=element_id, magnitude=Q))


def _solve_and_get_centre_w(n_el: int, bc: str) -> float:
    """Full solve for the quarter-circle plate; return w at the node closest to r = 0."""
    model, _ = _build_model(n_el)
    _apply_symmetry_bcs(model)
    _apply_outer_bc(model, bc)
    _apply_uniform_load(model)

    _, _, displacement = solve_displacement_system(model)

    xy = model.mesh.node_coordinates
    r  = np.linalg.norm(xy, axis=1)
    centre_node = int(np.argmin(r))
    return float(displacement[centre_node])


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_el", [3, 6])
def test_ss1_uniform_load(n_el: int) -> None:
    """
    SS₁-U: simply-supported outer edge, uniform pressure.

    At n_el = 6 the FEM result must be within 5 % of the Reissner-Mindlin reference
    (evaluated at the innermost mesh node r = r_min). At n_el = 3 a wider 15 %
    window is allowed, reflecting the coarser mesh.
    """
    r_min = R / (4.0 * n_el)
    w_ref = w_mindlin_ss1(r_min)
    w_fem = _solve_and_get_centre_w(n_el, bc="simply_supported")

    rtol = 0.05 if n_el == 6 else 0.15
    np.testing.assert_allclose(w_fem, w_ref, rtol=rtol,
        err_msg=f"SS₁-U n_el={n_el}: w_fem={w_fem:.4e}, w_ref={w_ref:.4e}")


@pytest.mark.parametrize("n_el", [3, 6])
def test_cl_uniform_load(n_el: int) -> None:
    """
    CL-U: clamped outer edge, uniform pressure.

    Shear deformation accounts for ~70 % of the total deflection at t/R = 0.4,
    so the Reissner-Mindlin reference is essential here (Kirchhoff alone would
    be ~2.5× too small). Tolerance: 5 % at n_el = 6, 15 % at n_el = 3.
    """
    r_min = R / (4.0 * n_el)
    w_ref = w_mindlin_cl(r_min)
    w_fem = _solve_and_get_centre_w(n_el, bc="clamped")

    rtol = 0.05 if n_el == 6 else 0.15
    np.testing.assert_allclose(w_fem, w_ref, rtol=rtol,
        err_msg=f"CL-U n_el={n_el}: w_fem={w_fem:.4e}, w_ref={w_ref:.4e}")


def test_convergence_ss1_uniform_load() -> None:
    """Finer mesh (n_el = 6) must be closer to the reference than the coarse mesh (n_el = 3)."""
    errors = {}
    for n_el in (3, 6):
        r_min = R / (4.0 * n_el)
        w_ref = w_mindlin_ss1(r_min)
        w_fem = _solve_and_get_centre_w(n_el, bc="simply_supported")
        errors[n_el] = abs(w_fem - w_ref) / abs(w_ref)

    assert errors[6] < errors[3], (
        f"Expected convergence: error at n_el=6 ({errors[6]:.4f}) "
        f"should be less than at n_el=3 ({errors[3]:.4f})"
    )


def test_convergence_cl_uniform_load() -> None:
    """Finer mesh (n_el = 6) must be closer to the reference than the coarse mesh (n_el = 3)."""
    errors = {}
    for n_el in (3, 6):
        r_min = R / (4.0 * n_el)
        w_ref = w_mindlin_cl(r_min)
        w_fem = _solve_and_get_centre_w(n_el, bc="clamped")
        errors[n_el] = abs(w_fem - w_ref) / abs(w_ref)

    assert errors[6] < errors[3], (
        f"Expected convergence: error at n_el=6 ({errors[6]:.4f}) "
        f"should be less than at n_el=3 ({errors[3]:.4f})"
    )
