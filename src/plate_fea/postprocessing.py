"""
Post-processing: evaluate generalised strains and stress resultants at element integration points.

Quantities are computed directly at the Gauss points of each element using the element
B-matrices — no nodal averaging or stress projection is performed.

Constitutive law used to convert strains to stress resultants:

    [M_xx, M_yy, M_xy] = D_b @ [κ_xx, κ_yy, κ_xy]   (bending moment resultants)
    [Q_x,  Q_y]        = D_s @ [γ_xz, γ_yz]           (transverse shear force resultants)

Entry point: sample_fields_at_quadrature_points(mesh, model, displacement) → SampledFields
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plate_fea.elements import HeterosisPlateElement
from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel
from plate_fea.quadrature import tensor_product_rule


@dataclass
class SampledFields:
    """
    Generalised strains and stress resultants at element quadrature points.

    All arrays have shape (n_samples,) where n_samples = n_elements × n_points_per_element.
    x and y hold the physical coordinates of each sampling point.
    """

    x: np.ndarray
    y: np.ndarray

    # Bending curvatures
    kappa_xx: np.ndarray    # ∂θ_x/∂x
    kappa_yy: np.ndarray    # ∂θ_y/∂y
    kappa_xy: np.ndarray    # ∂θ_x/∂y + ∂θ_y/∂x

    # Transverse shear strains
    gamma_xz: np.ndarray    # ∂w/∂x − θ_x
    gamma_yz: np.ndarray    # ∂w/∂y − θ_y

    # Bending moment resultants:  M = D_b @ [κ_xx, κ_yy, κ_xy]
    M_xx: np.ndarray
    M_yy: np.ndarray
    M_xy: np.ndarray

    # Transverse shear force resultants:  Q = D_s @ [γ_xz, γ_yz]
    Q_x: np.ndarray
    Q_y: np.ndarray


def sample_fields_at_quadrature_points(
    mesh: HeterosisMesh,
    model: PlateModel,
    displacement: np.ndarray,
    quadrature_order: tuple[int, int] = (3, 3),
) -> SampledFields:
    """
    Evaluate generalised strains and stress resultants at Gauss points of every element.

    Strains are computed from the element B-matrices applied to the local displacement vector.
    Stress resultants are obtained by multiplying by the constitutive matrices (D_b, D_s).
    No projection to nodes is performed.

    Args:
        quadrature_order: (n_xi, n_eta) tensor-product Gauss rule.  The default (3, 3) yields
                          9 samples per element and resolves the bending field accurately.
    """
    element = HeterosisPlateElement()
    D_b = model.constitutive_material.bending_constitutive_matrix
    D_s = model.constitutive_material.shear_constitutive_matrix
    quadrature_points = tensor_product_rule(*quadrature_order).points

    x_list: list[float] = []
    y_list: list[float] = []
    kappa_xx_list: list[float] = []
    kappa_yy_list: list[float] = []
    kappa_xy_list: list[float] = []
    gamma_xz_list: list[float] = []
    gamma_yz_list: list[float] = []
    M_xx_list: list[float] = []
    M_yy_list: list[float] = []
    M_xy_list: list[float] = []
    Q_x_list: list[float] = []
    Q_y_list: list[float] = []

    for element_id in range(mesh.total_element_number):
        geom = mesh.get_geometry_coordinates(element_id)
        dof_indices = element.local_to_global_dof_indices(mesh, element_id)
        u_local = displacement[dof_indices]

        for pt in quadrature_points:
            xi, eta = float(pt[0]), float(pt[1])
            J = element.geometry_jacobian(xi, eta, geom)

            # Physical position of this quadrature point.
            N_w = element.q8_shape_functions(xi, eta)
            x_list.append(float(N_w @ geom[:, 0]))
            y_list.append(float(N_w @ geom[:, 1]))

            # Bending curvatures: B_b @ u_local, using Q9 rotation gradients.
            dN9_dxi, dN9_deta = element.q9_shape_function_gradients_parent(xi, eta)
            dN9_dx, dN9_dy = element.parent_to_physical_gradients(dN9_dxi, dN9_deta, J)
            bending_strain = element.bending_B_matrix(dN9_dx, dN9_dy) @ u_local
            kappa_xx_list.append(float(bending_strain[0]))
            kappa_yy_list.append(float(bending_strain[1]))
            kappa_xy_list.append(float(bending_strain[2]))

            # Transverse shear strains: B_s @ u_local (Q8 w-gradients minus Q9 rotations).
            dN8_dxi, dN8_deta = element.q8_shape_function_gradients_parent(xi, eta)
            dN8_dx, dN8_dy = element.parent_to_physical_gradients(dN8_dxi, dN8_deta, J)
            N9 = element.q9_shape_functions(xi, eta)
            shear_strain = element.shear_B_matrix(dN8_dx, dN8_dy, N9) @ u_local
            gamma_xz_list.append(float(shear_strain[0]))
            gamma_yz_list.append(float(shear_strain[1]))

            # Stress resultants from constitutive law.
            M = D_b @ bending_strain
            M_xx_list.append(float(M[0]))
            M_yy_list.append(float(M[1]))
            M_xy_list.append(float(M[2]))

            Q = D_s @ shear_strain
            Q_x_list.append(float(Q[0]))
            Q_y_list.append(float(Q[1]))

    return SampledFields(
        x=np.array(x_list, dtype=float),
        y=np.array(y_list, dtype=float),
        kappa_xx=np.array(kappa_xx_list, dtype=float),
        kappa_yy=np.array(kappa_yy_list, dtype=float),
        kappa_xy=np.array(kappa_xy_list, dtype=float),
        gamma_xz=np.array(gamma_xz_list, dtype=float),
        gamma_yz=np.array(gamma_yz_list, dtype=float),
        M_xx=np.array(M_xx_list, dtype=float),
        M_yy=np.array(M_yy_list, dtype=float),
        M_xy=np.array(M_xy_list, dtype=float),
        Q_x=np.array(Q_x_list, dtype=float),
        Q_y=np.array(Q_y_list, dtype=float),
    )
