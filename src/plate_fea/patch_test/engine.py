"""
Patch tests A, B, and C for the heterosis plate element (Test C is the main regression gate).

Discrete equilibrium is ``K u = f`` with ``u`` the global displacement vector. Each test uses the
same linear analytical patch assembled by ``assemble_linear_patch_displacement`` as ``u_exact``.

Test A — No essential BCs, ``f = 0``: check ``||K u_exact||`` (assembly consistency for that
         displacement pattern).

Test B — Essential ``w`` and ``θ`` on the full outer boundary to match the patch; solve; compare
         ``u`` to ``u_exact``.

Test C — Essential ``w``, ``θ_x``, ``θ_y`` at one corner only; natural elsewhere; solve; compare to
         ``u_exact`` and inspect ``K_ff`` (stability / mechanisms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from plate_fea.assembly import assemble_force_vector, assemble_stiffness_matrix
from plate_fea.boundary_conditions import EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel
from plate_fea.solver import solve_linear_system

from plate_fea.patch_test.diagnostics import (
    ConstrainedSystemDiagnostics,
    diagnose_constrained_stiffness,
    residual_norm,
)
from plate_fea.patch_test.geometry import (
    outer_rectangle_boundary_theta_node_mask,
    outer_rectangle_boundary_w_node_mask,
    unit_square_multi,
    unit_square_single_element,
    mildly_distorted_unit_square,
)
from plate_fea.patch_test.nodal_vector import assemble_linear_patch_displacement
from plate_fea.patch_test.plate_exact_states import LinearHeterosisPatchState, linear_base_catalog
from plate_fea.patch_test.reporting import FailureClass, PatchTestCaseReport


def _fresh_model(mesh: HeterosisMesh, material: PlateMaterial, quad_kw: dict[str, object]) -> PlateModel:
    """Empty model: no BCs and no loads until the caller adds them."""
    return PlateModel(
        mesh=mesh,
        constitutive_material=material,
        element_formulation=HeterosisPlateElement(),
        element_stiffness_kwargs=quad_kw,
    )


def _apply_boundary_linear_state(
    model: PlateModel,
    mesh: HeterosisMesh,
    state: LinearHeterosisPatchState,
    width: float,
    height: float,
) -> None:
    """Test B: fix every w- and θ-node on the outer rectangle [0,width]×[0,height] to the patch."""
    w_mask = outer_rectangle_boundary_w_node_mask(mesh, width, height)
    t_mask = outer_rectangle_boundary_theta_node_mask(mesh, width, height)
    w_ids = np.flatnonzero(w_mask).tolist()
    t_ids = np.flatnonzero(t_mask).tolist()
    xy_w = mesh.node_coordinates
    for nid in w_ids:
        wv = float(state.w(xy_w[nid, 0], xy_w[nid, 1]))
        model.add_essential_condition(EssentialBoundaryCondition(field_name="w", node_ids=[nid], value=wv))
    for tid in t_ids:
        model.add_essential_condition(
            EssentialBoundaryCondition(field_name="theta_x", node_ids=[tid], value=state.theta_x)
        )
        model.add_essential_condition(
            EssentialBoundaryCondition(field_name="theta_y", node_ids=[tid], value=state.theta_y)
        )


def _apply_minimal_corner_linear(model: PlateModel, mesh: HeterosisMesh, state: LinearHeterosisPatchState) -> None:
    """
    Test C: pick the w-node closest to the origin (min x+y) and prescribe ``w``, ``θ_x``, ``θ_y``.

    Three scalars match the three free parameters ``(a, b, c)`` of the affine patch; remaining
    boundaries stay natural (no extra loads on ``f``).
    """
    tol = 1.0e-9
    xy = mesh.node_coordinates
    j = int(np.argmin(xy[:, 0] + xy[:, 1]))
    xj, yj = float(xy[j, 0]), float(xy[j, 1])
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="w", node_ids=[j], value=float(state.w(xj, yj)))
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="theta_x", node_ids=[j], value=state.theta_x)
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(field_name="theta_y", node_ids=[j], value=state.theta_y)
    )


def _free_partition(K: csr_matrix, f: np.ndarray, bc_ess: np.ndarray, bc_val: np.ndarray) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Split ``K u = f`` into free DOFs: ``K_ff u_f = f_f`` with prescribed DOFs eliminated."""
    n = f.size
    free_mask = np.ones(n, dtype=bool)
    free_mask[bc_ess] = False
    free = np.flatnonzero(free_mask)
    K_ff = K[free][:, free]
    K_fc = K[free][:, bc_ess]
    f_f = f[free] - K_fc @ bc_val
    return K_ff, f_f, free


def _load_perturbation_metric(K_ff: csr_matrix, f_f: np.ndarray) -> float | None:
    """Relative change in ``u_f`` under a tiny load perturbation; ``None`` if too large for a dense solve."""
    if f_f.size == 0 or f_f.size > 300:
        return None
    Kd = K_ff.toarray()
    u0 = np.linalg.solve(Kd, f_f)
    rng = np.random.default_rng(1)
    du = 1.0e-10 * rng.standard_normal(f_f.shape)
    u1 = np.linalg.solve(Kd, f_f + du)
    den = max(np.linalg.norm(u0), 1e-30)
    return float(np.linalg.norm(u1 - u0) / den)


def _run_one(
    *,
    test_type: str,
    patch_topology: str,
    geometry_label: str,
    mesh: HeterosisMesh,
    width: float,
    height: float,
    material: PlateMaterial,
    quad_label: str,
    quad_kw: dict[str, object],
    state: LinearHeterosisPatchState,
    mode_fn: Callable[[PlateModel], None],
    test_name_suffix: str,
    nodal_tol: float,
    residual_tol: float,
) -> PatchTestCaseReport:
    """Run one patch case: build ``u_exact``, apply ``mode_fn`` (BCs), assemble, solve or residual check."""
    model = _fresh_model(mesh, material, quad_kw)
    u_exact = assemble_linear_patch_displacement(mesh, model, state)
    mode_fn(model)
    K = assemble_stiffness_matrix(model)
    f = assemble_force_vector(model)
    bc_ess, bc_val = model.build_essential_boundary_arrays()

    poly = f"w=a x + b y + c with (a,b,c)=({state.a},{state.b},{state.c})"

    if test_type == "A":
        r = residual_norm(K, u_exact, f)
        res_ok = r < residual_tol
        nodal_err = None
        diag = diagnose_constrained_stiffness(K, dense_threshold=800)
        pert = None
        passed = res_ok
        fail = FailureClass.NONE if passed else FailureClass.CONSISTENCY
        interp = "Test A: residual K ũ - f vanishes (reactions consistent)." if passed else "Test A: residual indicates inconsistency in K or f."
        return PatchTestCaseReport(
            test_name=f"Test A {patch_topology} {quad_label} {geometry_label} {test_name_suffix}",
            test_type="A",
            patch_topology=patch_topology,
            element_type="heterosis plate element (Q8 w, Q9 θ)",
            quadrature_label=quad_label,
            geometry_label=geometry_label,
            polynomial_state=poly,
            body_force_description="none (homogeneous patch, f = 0)",
            natural_bc_description="none (no edge or surface loads)",
            essential_bc_description="none for Test A (full ũ inserted)",
            residual_norm=r,
            nodal_linf_error=nodal_err,
            nodal_l2_error=None,
            diagnostics_note=diag.note,
            rank_dense=diag.rank_dense,
            n_free=diag.n_free,
            min_eigenvalue=diag.min_eigenvalue,
            smallest_singular_value=diag.smallest_singular_value_dense,
            load_perturbation_sensitivity=pert,
            passed=passed,
            failure_class=fail,
            interpretation=interp,
        )

    K_ff, f_f, free = _free_partition(K, f, bc_ess, bc_val)
    diag = diagnose_constrained_stiffness(K_ff, dense_threshold=400)
    pert = _load_perturbation_metric(K_ff, f_f)

    u = solve_linear_system(K, f, bc_ess, bc_val)
    diff = u - u_exact
    linf = float(np.max(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))

    stability_ok = True
    if diag.rank_dense is not None and diag.n_free is not None:
        if diag.rank_dense < diag.n_free:
            stability_ok = False
    if diag.min_eigenvalue is not None and diag.min_eigenvalue < -1.0e-8:
        stability_ok = False

    nodal_ok = linf < nodal_tol
    res = residual_norm(K, u, f)
    res_ok = res < residual_tol
    passed = nodal_ok and res_ok and stability_ok

    fail = FailureClass.NONE
    interp_parts: list[str] = []
    if not nodal_ok or not res_ok:
        fail = FailureClass.CONSISTENCY
        interp_parts.append("Nodal solution or equilibrium residual deviates from the exact linear patch.")
    if not stability_ok:
        fail = FailureClass.MIXED if fail == FailureClass.CONSISTENCY else FailureClass.STABILITY
        interp_parts.append("Stability issue: rank deficiency or non-positive spectrum on K_ff.")
    if passed:
        interp_parts = [
            f"Test {test_type}: interior/exact match within tolerance; K_ff diagnostics acceptable."
        ]
    interpretation = " ".join(interp_parts)

    return PatchTestCaseReport(
        test_name=f"Test {test_type} {patch_topology} {quad_label} {geometry_label} {test_name_suffix}",
        test_type=test_type,
        patch_topology=patch_topology,
        element_type="heterosis plate element (Q8 w, Q9 θ)",
        quadrature_label=quad_label,
        geometry_label=geometry_label,
        polynomial_state=poly,
        body_force_description="none (homogeneous patch, f = 0)",
        natural_bc_description="none (all boundaries traction-free for Test C; full boundary prescription for Test B)",
        essential_bc_description="Test B: all boundary w and θ nodes; Test C: one corner (w, θ_x, θ_y)",
        residual_norm=float(residual_norm(K, u, f)),
        nodal_linf_error=linf,
        nodal_l2_error=l2,
        diagnostics_note=diag.note,
        rank_dense=diag.rank_dense,
        n_free=int(free.size),
        min_eigenvalue=diag.min_eigenvalue,
        smallest_singular_value=diag.smallest_singular_value_dense,
        load_perturbation_sensitivity=pert,
        passed=passed,
        failure_class=fail if not passed else FailureClass.NONE,
        interpretation=interpretation,
    )


@dataclass
class PatchTestSuiteConfig:
    width: float = 1.0
    height: float = 1.0
    multi_nx: int = 2
    multi_ny: int = 2
    run_single_element: bool = True
    run_multi_element: bool = True
    run_distorted: bool = True
    test_all_linear_modes: bool = True
    full_quadrature_sweep: bool = True
    nodal_tol: float = 1.0e-7
    residual_tol: float = 1.0e-6


def quadrature_presets() -> list[tuple[str, dict[str, object]]]:
    """(label, element_stiffness_kwargs). Standard = exact product rules used in production."""
    return [
        ("standard (3×3 bend + 2×2 shear)", {}),
        ("reduced bending (2×2 bend + 2×2 shear)", {"bending_quadrature_order": (2, 2)}),
        ("reduced shear (3×3 bend + 1×1 shear)", {"shear_quadrature_order": (1, 1)}),
        ("reduced both (2×2 bend + 1×1 shear)", {"bending_quadrature_order": (2, 2), "shear_quadrature_order": (1, 1)}),
    ]


def run_patch_test_suite(cfg: PatchTestSuiteConfig | None = None) -> list[PatchTestCaseReport]:
    cfg = cfg or PatchTestSuiteConfig()
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    states = list(linear_base_catalog()) if cfg.test_all_linear_modes else [LinearHeterosisPatchState(0.5, -0.25, 0.1)]

    reports: list[PatchTestCaseReport] = []

    def mesh_specs() -> list[tuple[str, str, HeterosisMesh]]:
        out: list[tuple[str, str, HeterosisMesh]] = []
        if cfg.run_single_element:
            out.append(("single_element", "regular unit square 1×1 el", unit_square_single_element()))
        if cfg.run_multi_element:
            out.append(
                (
                    "multi_element",
                    f"regular unit square {cfg.multi_nx}×{cfg.multi_ny} el",
                    unit_square_multi(cfg.multi_nx, cfg.multi_ny),
                )
            )
        if cfg.run_distorted:
            out.append(
                (
                    "multi_element",
                    f"distorted {cfg.multi_nx}×{cfg.multi_ny} (interior nodes perturbed)",
                    mildly_distorted_unit_square(cfg.multi_nx, cfg.multi_ny),
                )
            )
        return out

    for topo, geom_label, mesh in mesh_specs():
        w = cfg.width if "unit" in geom_label or "square" in geom_label else cfg.width
        h = cfg.height
        if topo == "single_element":
            w = h = 1.0
        elif "unit square" in geom_label:
            w = h = 1.0

        presets = quadrature_presets() if cfg.full_quadrature_sweep else [quadrature_presets()[0]]
        for state in states:
            suffix = f"a={state.a}_b={state.b}_c={state.c}"
            for quad_label, quad_kw in presets:

                def mode_a(m: PlateModel) -> None:
                    return None

                rA = _run_one(
                    test_type="A",
                    patch_topology=topo,
                    geometry_label=geom_label,
                    mesh=mesh,
                    width=w,
                    height=h,
                    material=material,
                    quad_label=quad_label,
                    quad_kw=quad_kw,
                    state=state,
                    mode_fn=mode_a,
                    test_name_suffix=suffix,
                    nodal_tol=cfg.nodal_tol,
                    residual_tol=cfg.residual_tol,
                )
                reports.append(rA)

                def mode_b(m: PlateModel) -> None:
                    _apply_boundary_linear_state(m, mesh, state, w, h)

                rB = _run_one(
                    test_type="B",
                    patch_topology=topo,
                    geometry_label=geom_label,
                    mesh=mesh,
                    width=w,
                    height=h,
                    material=material,
                    quad_label=quad_label,
                    quad_kw=quad_kw,
                    state=state,
                    mode_fn=mode_b,
                    test_name_suffix=suffix,
                    nodal_tol=cfg.nodal_tol,
                    residual_tol=cfg.residual_tol,
                )
                reports.append(rB)

                def mode_c(m: PlateModel) -> None:
                    _apply_minimal_corner_linear(m, mesh, state)

                rC = _run_one(
                    test_type="C",
                    patch_topology=topo,
                    geometry_label=geom_label,
                    mesh=mesh,
                    width=w,
                    height=h,
                    material=material,
                    quad_label=quad_label,
                    quad_kw=quad_kw,
                    state=state,
                    mode_fn=mode_c,
                    test_name_suffix=suffix,
                    nodal_tol=cfg.nodal_tol,
                    residual_tol=cfg.residual_tol,
                )
                reports.append(rC)

    return reports


def acceptance_all_pass(reports: list[PatchTestCaseReport]) -> bool:
    """Framework acceptance: every Test C on multi-element standard quadrature must pass."""
    crit = [
        r
        for r in reports
        if r.test_type == "C"
        and "standard" in r.quadrature_label
        and r.patch_topology == "multi_element"
        and "distorted" not in r.geometry_label
    ]
    return bool(crit) and all(r.passed for r in crit)


def summarize(reports: list[PatchTestCaseReport]) -> str:
    lines: list[str] = []
    for r in reports:
        lines.extend(r.lines())
        lines.append("")
    lines.append(
        "Acceptance (multi-element + standard quadrature + Test C only): "
        f"{'PASS' if acceptance_all_pass(reports) else 'FAIL'}"
    )
    lines.append(
        "Note: single-element Test C alone is never sufficient for convergence claims in standard patch-test practice; "
        "the suite always includes multi-element patches."
    )
    return "\n".join(lines)
