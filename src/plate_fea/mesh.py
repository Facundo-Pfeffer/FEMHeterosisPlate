"""Heterosis mesh layout: Q8 nodes for ``w``, Q9 nodes for rotations (extra center node per element)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeterosisMesh:
    node_coordinates: np.ndarray
    w_location_matrix: np.ndarray
    theta_node_coordinates: np.ndarray
    theta_location_matrix: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        node_coordinates: np.ndarray,
        w_location_matrix: np.ndarray,
        theta_location_matrix: np.ndarray | None = None,
    ) -> "HeterosisMesh":
        node_coordinates = np.asarray(node_coordinates, dtype=float)
        w_location_matrix = np.asarray(w_location_matrix, dtype=int)

        if node_coordinates.ndim != 2 or node_coordinates.shape[1] != 2:
            raise ValueError("node_coordinates must have shape (n_node, 2).")
        if w_location_matrix.ndim != 2 or w_location_matrix.shape[0] != 8:
            raise ValueError("w_location_matrix must have shape (8, n_element).")

        if theta_location_matrix is None:
            n_w_node = node_coordinates.shape[0]
            n_element = w_location_matrix.shape[1]

            center_coordinates = np.zeros((n_element, 2), dtype=float)
            generated_theta_location_matrix = np.zeros((9, n_element), dtype=int)
            generated_theta_location_matrix[:8, :] = w_location_matrix

            # Q9 center node: geometric mean of Q8 corners, new global theta-only node per element.
            for element_id in range(n_element):
                geom_node_ids = w_location_matrix[:, element_id]
                center_coordinates[element_id, :] = node_coordinates[geom_node_ids, :].mean(axis=0)
                generated_theta_location_matrix[8, element_id] = n_w_node + element_id

            theta_node_coordinates = np.vstack([node_coordinates, center_coordinates])
            theta_location_matrix = generated_theta_location_matrix
        else:
            theta_location_matrix = np.asarray(theta_location_matrix, dtype=int)
            if theta_location_matrix.ndim != 2 or theta_location_matrix.shape[0] != 9:
                raise ValueError("theta_location_matrix must have shape (9, n_element).")
            theta_node_coordinates = node_coordinates.copy()

        return cls(
            node_coordinates=node_coordinates,
            w_location_matrix=w_location_matrix,
            theta_node_coordinates=theta_node_coordinates,
            theta_location_matrix=theta_location_matrix,
        )

    @property
    def total_w_node_number(self) -> int:
        return int(self.node_coordinates.shape[0])

    @property
    def total_theta_node_number(self) -> int:
        return int(self.theta_node_coordinates.shape[0])

    @property
    def total_element_number(self) -> int:
        return int(self.w_location_matrix.shape[1])

    @property
    def total_dof_number(self) -> int:
        return self.total_w_node_number + 2 * self.total_theta_node_number

    def get_geometry_coordinates(self, element_id: int) -> np.ndarray:
        node_ids = self.w_location_matrix[:, element_id]
        return self.node_coordinates[node_ids, :]

    def get_theta_coordinates(self, element_id: int) -> np.ndarray:
        node_ids = self.theta_location_matrix[:, element_id]
        return self.theta_node_coordinates[node_ids, :]

    def find_w_nodes_on_line(
        self,
        axis: str,
        value: float,
        interval: tuple[float, float] | None = None,
        tolerance: float = 1.0e-10,
    ) -> np.ndarray:
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'.")

        axis_index = 0 if axis == "x" else 1
        other_axis_index = 1 - axis_index

        mask = np.isclose(self.node_coordinates[:, axis_index], value, atol=tolerance)
        if interval is not None:
            lower, upper = interval
            coordinate = self.node_coordinates[:, other_axis_index]
            mask &= coordinate >= lower - tolerance
            mask &= coordinate <= upper + tolerance

        return np.flatnonzero(mask)
