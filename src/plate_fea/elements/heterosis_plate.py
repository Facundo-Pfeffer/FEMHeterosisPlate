from __future__ import annotations

from typing import Callable

import numpy as np

from plate_fea.elements.base import PlateElementBase
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.quadrature import gauss_legendre_1d, tensor_product_rule


class HeterosisPlateElement(PlateElementBase):
    """
    Heterosis Mindlin-Reissner plate element.

    Local ordering:
        [w_1..w_8, theta_x1, theta_y1, ..., theta_x9, theta_y9]
    """

    local_edge_nodes = {
        1: np.array([0, 4, 1], dtype=int),
        2: np.array([1, 5, 2], dtype=int),
        3: np.array([2, 6, 3], dtype=int),
        4: np.array([3, 7, 0], dtype=int),
    }

    @staticmethod
    def q8_shape_functions(xi: float, eta: float) -> np.ndarray:
        return np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
                0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
                0.50 * (1.0 - xi**2) * (1.0 - eta),
                0.50 * (1.0 + xi) * (1.0 - eta**2),
                0.50 * (1.0 - xi**2) * (1.0 + eta),
                0.50 * (1.0 - xi) * (1.0 - eta**2),
            ],
            dtype=float,
        )

    @staticmethod
    def q8_shape_function_gradients_parent(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
        dN_dxi = np.array(
            [
                -0.25 * (eta - 1.0) * (eta + 2.0 * xi),
                0.25 * (eta - 1.0) * (eta - 2.0 * xi),
                0.25 * (eta + 1.0) * (eta + 2.0 * xi),
                -0.25 * (eta + 1.0) * (eta - 2.0 * xi),
                xi * (eta - 1.0),
                -0.50 * (eta - 1.0) * (eta + 1.0),
                -xi * (eta + 1.0),
                0.50 * (eta - 1.0) * (eta + 1.0),
            ],
            dtype=float,
        )
        dN_deta = np.array(
            [
                -0.25 * (2.0 * eta + xi) * (xi - 1.0),
                0.25 * (2.0 * eta - xi) * (xi + 1.0),
                0.25 * (2.0 * eta + xi) * (xi + 1.0),
                -0.25 * (2.0 * eta - xi) * (xi - 1.0),
                0.50 * (xi - 1.0) * (xi + 1.0),
                -eta * (xi + 1.0),
                -0.50 * (xi - 1.0) * (xi + 1.0),
                eta * (xi - 1.0),
            ],
            dtype=float,
        )
        return dN_dxi, dN_deta

    @staticmethod
    def q9_shape_functions(xi: float, eta: float) -> np.ndarray:
        L_xi = np.array(
            [0.50 * xi * (xi - 1.0), 1.0 - xi**2, 0.50 * xi * (xi + 1.0)],
            dtype=float,
        )
        L_eta = np.array(
            [0.50 * eta * (eta - 1.0), 1.0 - eta**2, 0.50 * eta * (eta + 1.0)],
            dtype=float,
        )

        return np.array(
            [
                L_xi[0] * L_eta[0],
                L_xi[2] * L_eta[0],
                L_xi[2] * L_eta[2],
                L_xi[0] * L_eta[2],
                L_xi[1] * L_eta[0],
                L_xi[2] * L_eta[1],
                L_xi[1] * L_eta[2],
                L_xi[0] * L_eta[1],
                L_xi[1] * L_eta[1],
            ],
            dtype=float,
        )

    @staticmethod
    def q9_shape_function_gradients_parent(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (dN/dxi, dN/deta) for the Q9 interpolation in parent space."""
        L_xi = np.array(
            [0.50 * xi * (xi - 1.0), 1.0 - xi**2, 0.50 * xi * (xi + 1.0)],
            dtype=float,
        )
        dL_xi = np.array([xi - 0.50, -2.0 * xi, xi + 0.50], dtype=float)

        L_eta = np.array(
            [0.50 * eta * (eta - 1.0), 1.0 - eta**2, 0.50 * eta * (eta + 1.0)],
            dtype=float,
        )
        dL_eta = np.array([eta - 0.50, -2.0 * eta, eta + 0.50], dtype=float)

        dN_dxi = np.array(
            [
                dL_xi[0] * L_eta[0],
                dL_xi[2] * L_eta[0],
                dL_xi[2] * L_eta[2],
                dL_xi[0] * L_eta[2],
                dL_xi[1] * L_eta[0],
                dL_xi[2] * L_eta[1],
                dL_xi[1] * L_eta[2],
                dL_xi[0] * L_eta[1],
                dL_xi[1] * L_eta[1],
            ],
            dtype=float,
        )

        dN_deta = np.array(
            [
                L_xi[0] * dL_eta[0],
                L_xi[2] * dL_eta[0],
                L_xi[2] * dL_eta[2],
                L_xi[0] * dL_eta[2],
                L_xi[1] * dL_eta[0],
                L_xi[2] * dL_eta[1],
                L_xi[1] * dL_eta[2],
                L_xi[0] * dL_eta[1],
                L_xi[1] * dL_eta[1],
            ],
            dtype=float,
        )
        return dN_dxi, dN_deta

    @staticmethod
    def edge_quadratic_shape_functions(s: float) -> np.ndarray:
        return np.array(
            [0.50 * s * (s - 1.0), 1.0 - s**2, 0.50 * s * (s + 1.0)],
            dtype=float,
        )

    @staticmethod
    def edge_quadratic_shape_function_derivatives(s: float) -> np.ndarray:
        return np.array([s - 0.50, -2.0 * s, s + 0.50], dtype=float)

    @staticmethod
    def geometry_jacobian(xi: float, eta: float, geometry_coordinates: np.ndarray) -> np.ndarray:
        """Compute the 2x2 geometric Jacobian d(x,y)/d(xi,eta) from Q8 geometry mapping."""
        dN_dxi, dN_deta = HeterosisPlateElement.q8_shape_function_gradients_parent(xi, eta)
        jacobian = np.zeros((2, 2), dtype=float)
        jacobian[0, 0] = dN_dxi @ geometry_coordinates[:, 0]
        jacobian[0, 1] = dN_deta @ geometry_coordinates[:, 0]
        jacobian[1, 0] = dN_dxi @ geometry_coordinates[:, 1]
        jacobian[1, 1] = dN_deta @ geometry_coordinates[:, 1]
        return jacobian

    @staticmethod
    def parent_to_physical_gradients(
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map parent-space gradients to physical-space gradients via J^{-T}."""
        inv_jacobian = np.linalg.inv(jacobian)
        gradients_parent = np.vstack([dN_dxi, dN_deta])
        gradients_physical = inv_jacobian.T @ gradients_parent
        return gradients_physical[0, :], gradients_physical[1, :]

    @staticmethod
    def bending_B_matrix(dN_theta_dx: np.ndarray, dN_theta_dy: np.ndarray) -> np.ndarray:
        """Assemble bending strain-displacement matrix B_b (curvatures from theta gradients)."""
        B_b = np.zeros((3, 26), dtype=float)
        for local_node_id in range(9):
            local_dof_x = 8 + 2 * local_node_id
            local_dof_y = local_dof_x + 1
            B_b[0, local_dof_x] = dN_theta_dx[local_node_id]
            B_b[1, local_dof_y] = dN_theta_dy[local_node_id]
            B_b[2, local_dof_x] = dN_theta_dy[local_node_id]
            B_b[2, local_dof_y] = dN_theta_dx[local_node_id]
        return B_b

    @staticmethod
    def shear_B_matrix(dN_w_dx: np.ndarray, dN_w_dy: np.ndarray, N_theta: np.ndarray) -> np.ndarray:
        """Assemble shear strain-displacement matrix B_s for gamma_xz, gamma_yz."""
        B_s = np.zeros((2, 26), dtype=float)
        for local_node_id in range(8):
            B_s[0, local_node_id] = dN_w_dx[local_node_id]
            B_s[1, local_node_id] = dN_w_dy[local_node_id]

        for local_node_id in range(9):
            local_dof_x = 8 + 2 * local_node_id
            local_dof_y = local_dof_x + 1
            B_s[0, local_dof_x] = -N_theta[local_node_id]
            B_s[1, local_dof_y] = -N_theta[local_node_id]

        return B_s

    def local_to_global_dof_indices(self, mesh: HeterosisMesh, element_id: int) -> np.ndarray:
        """
        Map this element's 26 local DOFs to global vector indices.

        Local order is:
        [w1..w8, theta_x1, theta_y1, ..., theta_x9, theta_y9]

        Global layout is:
        - w(node_id) -> node_id
        - theta_x(theta_node_id) -> n_w_total + 2*theta_node_id
        - theta_y(theta_node_id) -> n_w_total + 2*theta_node_id + 1
        Rotational DOFs are stacked in pairs after all translational have been inserted.
        """
        w_node_ids = mesh.w_location_matrix[:, element_id]
        theta_node_ids = mesh.theta_location_matrix[:, element_id]

        final_w_gdof_id = mesh.total_w_node_number
        global_dof_indices = list(w_node_ids.tolist())
        for theta_node_id in theta_node_ids:
            global_dof_indices.append(final_w_gdof_id + 2 * theta_node_id)
            global_dof_indices.append(final_w_gdof_id + 2 * theta_node_id + 1)

        return np.asarray(global_dof_indices, dtype=int)

    def compute_stiffness_matrix(
        self,
        mesh: HeterosisMesh,
        material: PlateMaterial,
        element_id: int,
    ) -> np.ndarray:
        """
        Compute local 26x26 stiffness matrix for one heterosis plate element.

        Uses split integration:
        - bending part with 3x3 Gauss rule
        - shear part with 2x2 Gauss rule
        """
        geometry_coordinates = mesh.get_geometry_coordinates(element_id)

        D_b = material.bending_constitutive_matrix
        D_s = material.shear_constitutive_matrix

        K_b = np.zeros((26, 26), dtype=float)
        K_s = np.zeros((26, 26), dtype=float)

        # Bending: curvatures depend on theta gradients.
        bending_rule = tensor_product_rule(order_x=3, order_y=3)
        for point, weight in zip(bending_rule.points, bending_rule.weights):
            xi, eta = point
            jacobian = self.geometry_jacobian(xi, eta, geometry_coordinates)
            det_jacobian = np.linalg.det(jacobian)
            if det_jacobian <= 0.0:
                raise ValueError(f"Non-positive Jacobian detected in element {element_id}.")

            dN_theta_dxi, dN_theta_deta = self.q9_shape_function_gradients_parent(xi, eta)
            dN_theta_dx, dN_theta_dy = self.parent_to_physical_gradients(
                dN_theta_dxi,
                dN_theta_deta,
                jacobian,
            )
            B_b = self.bending_B_matrix(dN_theta_dx, dN_theta_dy)
            K_b += weight * (B_b.T @ D_b @ B_b) * det_jacobian

        # Shear: gamma = grad(w) - theta.
        shear_rule = tensor_product_rule(order_x=2, order_y=2)
        for point, weight in zip(shear_rule.points, shear_rule.weights):
            xi, eta = point
            jacobian = self.geometry_jacobian(xi, eta, geometry_coordinates)
            det_jacobian = np.linalg.det(jacobian)
            if det_jacobian <= 0.0:
                raise ValueError(f"Non-positive Jacobian detected in element {element_id}.")

            dN_w_dxi, dN_w_deta = self.q8_shape_function_gradients_parent(xi, eta)
            dN_w_dx, dN_w_dy = self.parent_to_physical_gradients(dN_w_dxi, dN_w_deta, jacobian)
            N_theta = self.q9_shape_functions(xi, eta)
            B_s = self.shear_B_matrix(dN_w_dx, dN_w_dy, N_theta)
            K_s += weight * (B_s.T @ D_s @ B_s) * det_jacobian

        return K_b + K_s

    def compute_edge_force_vector(
        self,
        mesh: HeterosisMesh,
        element_id: int,
        edge_id: int,
        traction: float | Callable[[float, float], float],
    ) -> np.ndarray:
        """
        Integrate distributed transverse traction on one element edge into local DOFs.

        The load contributes only to w DOFs located on the selected quadratic edge
        (3 edge nodes in local Q8 numbering).
        """
        if edge_id not in self.local_edge_nodes:
            raise ValueError("edge_id must be one of 1, 2, 3, 4.")

        local_force = np.zeros(26, dtype=float)
        edge_node_ids_local = self.local_edge_nodes[edge_id]
        geometry_coordinates = mesh.get_geometry_coordinates(element_id)
        edge_coordinates = geometry_coordinates[edge_node_ids_local, :]

        edge_rule = gauss_legendre_1d(order=3)
        for s, weight in zip(edge_rule.points, edge_rule.weights):
            N_edge = self.edge_quadratic_shape_functions(float(s))
            dN_edge_ds = self.edge_quadratic_shape_function_derivatives(float(s))

            # Physical quadrature point and 1D edge Jacobian (line metric).
            x_q = float(N_edge @ edge_coordinates[:, 0])
            y_q = float(N_edge @ edge_coordinates[:, 1])
            jacobian_edge = float(
                np.sqrt(
                    (dN_edge_ds @ edge_coordinates[:, 0]) ** 2
                    + (dN_edge_ds @ edge_coordinates[:, 1]) ** 2
                )
            )

            q_q = traction(x_q, y_q) if callable(traction) else float(traction)
            local_force[edge_node_ids_local] += weight * N_edge * q_q * jacobian_edge

        return local_force

    def compute_surface_force_vector(
        self,
        mesh: HeterosisMesh,
        element_id: int,
        traction: float | Callable[[float, float], float],
    ) -> np.ndarray:
        """
        Integrate distributed transverse surface traction on one element.

        The contribution acts on the transverse displacement DOFs (w1..w8).
        """
        local_force = np.zeros(26, dtype=float)
        geometry_coordinates = mesh.get_geometry_coordinates(element_id)
        area_rule = tensor_product_rule(order_x=3, order_y=3)

        for point, weight in zip(area_rule.points, area_rule.weights):
            xi, eta = point
            N_w = self.q8_shape_functions(float(xi), float(eta))
            jacobian = self.geometry_jacobian(float(xi), float(eta), geometry_coordinates)
            det_jacobian = float(np.linalg.det(jacobian))
            if det_jacobian <= 0.0:
                raise ValueError(f"Non-positive Jacobian detected in element {element_id}.")

            # Physical quadrature point for callable tractions q(x,y).
            x_q = float(N_w @ geometry_coordinates[:, 0])
            y_q = float(N_w @ geometry_coordinates[:, 1])
            q_q = traction(x_q, y_q) if callable(traction) else float(traction)

            # Only w DOFs are loaded by transverse pressure.
            local_force[:8] += weight * N_w * q_q * det_jacobian

        return local_force
