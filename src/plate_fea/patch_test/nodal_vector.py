"""
Build global displacement vectors ``u`` for analytical patch fields.

``PlateModel`` (and the solver) use a fixed global DOF order: all ``w`` components first
(one index per w-node, same order as ``mesh.node_coordinates``), then ``theta_x`` and
``theta_y`` for each θ-node in order (see ``PlateModel.get_theta_x_dof`` /
``get_theta_y_dof``). These functions fill ``u`` to match that layout so patch tests compare
apples-to-apples with ``assemble_stiffness_matrix`` / ``solve_linear_system``.
"""

from __future__ import annotations

import numpy as np

from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel

from plate_fea.patch_test.plate_exact_states import LinearHeterosisPatchState, QuadraticHeterosisPatchState


def assemble_linear_patch_displacement(
    mesh: HeterosisMesh,
    model: PlateModel,
    state: LinearHeterosisPatchState,
) -> np.ndarray:
    """
    Nodal values for the affine Mindlin patch ``w = a x + b y + c``, ``θ_x = a``, ``θ_y = b``.

    For this state, rotations are **spatially constant**, so every θ-node receives the same
    ``theta_x`` and ``theta_y``. ``w`` is evaluated at each **w-node** position; θ does not
    use ``mesh.theta_node_coordinates`` here because the field is constant.

    Returns
    -------
    u
        Length ``mesh.total_dof_number``. Leading slice ``u[0 : n_w]`` holds ``w`` at w-nodes;
        remaining entries follow ``model.get_theta_*_dof(theta_node_id)``.
    """
    n_dof = mesh.total_dof_number
    n_w = mesh.total_w_node_number
    u = np.zeros(n_dof, dtype=float)

    xy_w = mesh.node_coordinates
    u[:n_w] = state.w(xy_w[:, 0], xy_w[:, 1])

    tx = float(state.theta_x)
    ty = float(state.theta_y)
    for theta_node_id in range(mesh.total_theta_node_number):
        u[model.get_theta_x_dof(theta_node_id)] = tx
        u[model.get_theta_y_dof(theta_node_id)] = ty

    return u


def assemble_quadratic_patch_displacement(
    mesh: HeterosisMesh,
    model: PlateModel,
    state: QuadraticHeterosisPatchState,
) -> np.ndarray:
    """
    Nodal values for a quadratic ``w`` with ``θ = ∇w`` (transverse shear ``γ = 0``).

    Unlike :func:`assemble_linear_patch_displacement`, rotations **vary in space**: ``θ_x`` and
    ``θ_y`` are evaluated at **θ-node** coordinates (``mesh.theta_node_coordinates``), which
    need not coincide with w-node positions for heterosis meshes.

    Returns
    -------
    u
        Same global layout as :func:`assemble_linear_patch_displacement`.
    """
    n_dof = mesh.total_dof_number
    n_w = mesh.total_w_node_number
    u = np.zeros(n_dof, dtype=float)

    xy_w = mesh.node_coordinates
    u[:n_w] = state.w(xy_w[:, 0], xy_w[:, 1])

    xy_t = mesh.theta_node_coordinates
    tx = state.theta_x_field(xy_t[:, 0], xy_t[:, 1])
    ty = state.theta_y_field(xy_t[:, 0], xy_t[:, 1])
    for theta_node_id in range(mesh.total_theta_node_number):
        u[model.get_theta_x_dof(theta_node_id)] = float(tx[theta_node_id])
        u[model.get_theta_y_dof(theta_node_id)] = float(ty[theta_node_id])

    return u
