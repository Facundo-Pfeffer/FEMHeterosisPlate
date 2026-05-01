"""
Equilibrium validation for the plate-with-hole assignment problem.

Three mechanical checks on the assembled and solved system:

  1. Load assembly  — the w-DOF entries of the force vector sum to the expected total
     applied force: q × hole_width = 1 kN/mm × 250 mm = -250 000 N (downward).

  2. Force equilibrium — transverse reactions at clamped w-DOFs balance the applied load.

         R_i = (K @ u)_i   at each clamped w-DOF i  (F_i = 0 there)
         sum(R_w) = -sum(F_applied_w) = +250 000 N

  3. Moment equilibrium — reactions also balance the moment of the applied load.

     Clamped supports exert two types of reaction:
       • Transverse force reactions R_w at clamped w-DOFs (need position × force).
       • Moment (couple) reactions R_θ at clamped θ-DOFs (already in N·mm, no lever arm).

     Two independent moment equations about the origin:

       About x-axis:  sum(y·R_w) + sum(R_θy) = −sum(y·F_applied_w)
       About y-axis:  sum(x·R_w) + sum(R_θx) = −sum(x·F_applied_w)

     The sign rule follows from virtual work on rigid-body rotation modes:
       • Rotation δφ about x: δw = y·δφ, δθ_y = δφ  → θ_y couples enter M_x.
       • Rotation δφ about y: δw = x·δφ, δθ_x = δφ  → θ_x couples enter M_y.

The assembled K is never modified by the solver, so K @ u recovers both applied
loads (at free DOFs) and reactions (at constrained DOFs).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_fea.problem_orchestrator import ProblemConfig, solve_plate_problem

pytestmark = pytest.mark.final_project_specific

_CONFIG = ProblemConfig(mesh_strategy="uniform_buffer_ring", resolution=2)


@pytest.fixture(scope="module")
def solved_result():
    return solve_plate_problem(_CONFIG)


def _boundary_reactions(solved_result):
    """
    Compute all reaction components at clamped DOFs from K @ u.

    Returns:
        xy_w    : coordinates of clamped w-nodes,     shape (n_clamped_w, 2)  [mm]
        R_w     : transverse reactions at clamped w-DOFs                       [N]
        R_theta_x: moment reactions at clamped θ_x-DOFs                       [N·mm]
        R_theta_y: moment reactions at clamped θ_y-DOFs                       [N·mm]
    """
    mesh = solved_result.model.mesh
    n_w = mesh.total_w_node_number

    all_constrained, _ = solved_result.model.build_essential_boundary_arrays()
    clamped_w_dofs     = all_constrained[all_constrained < n_w]
    clamped_theta_dofs = all_constrained[all_constrained >= n_w]

    # θ_x DOF for node j: n_w + 2j  (even offset); θ_y DOF: n_w + 2j + 1 (odd offset).
    offsets = clamped_theta_dofs - n_w
    clamped_theta_x_dofs = clamped_theta_dofs[offsets % 2 == 0]
    clamped_theta_y_dofs = clamped_theta_dofs[offsets % 2 == 1]

    internal = solved_result.stiffness_matrix @ solved_result.solution

    return (
        mesh.node_coordinates[clamped_w_dofs],
        internal[clamped_w_dofs],
        internal[clamped_theta_x_dofs],
        internal[clamped_theta_y_dofs],
    )


def test_total_applied_force(solved_result) -> None:
    """
    The w-DOF entries of the assembled force vector must sum to q × hole_width.

    For 1 kN/mm downward over 250 mm: expected = -1000 × 250 = -250 000 N.
    """
    mesh = solved_result.model.mesh
    total_applied = float(solved_result.force_vector[:mesh.total_w_node_number].sum())

    expected = _CONFIG.hole_top_shear_load * _CONFIG.geometry.hole_width
    np.testing.assert_allclose(total_applied, expected, rtol=1e-6)


def test_reaction_forces_balance_applied_load(solved_result) -> None:
    """
    Transverse reactions at clamped w-DOFs must sum to +250 000 N (opposing the applied load).

    Reaction at each clamped w-DOF:  R_i = (K @ u)_i   (F_i = 0 at constrained nodes)
    Equilibrium requirement:          sum(R_w) = −sum(F_applied_w) = +250 000 N
    """
    _, R_w, _, _ = _boundary_reactions(solved_result)
    total_applied = float(solved_result.force_vector[:solved_result.model.mesh.total_w_node_number].sum())

    np.testing.assert_allclose(float(R_w.sum()), -total_applied, rtol=1e-4)


def test_moment_equilibrium(solved_result) -> None:
    """
    External moment must equal and oppose the total reaction moment about each in-plane axis.

    About x-axis:  M_ext_x  =  sum(y · F_applied_w)  ≈ −60 × 10⁶ N·mm
                   M_react_x =  sum(y · R_w) + sum(R_θy)  (force lever arm + θ_y couples)

    About y-axis:  M_ext_y  =  sum(x · F_applied_w)  ≈ −62.5 × 10⁶ N·mm
                   M_react_y =  sum(x · R_w) + sum(R_θx)  (force lever arm + θ_x couples)
    """
    mesh = solved_result.model.mesh
    xy   = mesh.node_coordinates
    F_w  = solved_result.force_vector[:mesh.total_w_node_number]

    xy_w, R_w, R_theta_x, R_theta_y = _boundary_reactions(solved_result)

    M_ext_x   = float((xy[:, 1] * F_w).sum())
    M_react_x = float((xy_w[:, 1] * R_w).sum()) + float(R_theta_y.sum())

    M_ext_y   = float((xy[:, 0] * F_w).sum())
    M_react_y = float((xy_w[:, 0] * R_w).sum()) + float(R_theta_x.sum())

    np.testing.assert_allclose(M_react_x, -M_ext_x, rtol=1e-4,
                               err_msg="Moment about x-axis not balanced")
    np.testing.assert_allclose(M_react_y, -M_ext_y, rtol=1e-4,
                               err_msg="Moment about y-axis not balanced")
