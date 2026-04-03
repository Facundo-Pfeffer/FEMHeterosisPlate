from __future__ import annotations

from typing import Callable

import numpy as np


class HeterosisPlateElement(PlateElementBase):
    """
    Heterosis Mindlin-Reissner plate element.

    Local ordering:
        [theta_x1, theta_y1, ..., theta_x9, theta_y9,  w_1..w_8]
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