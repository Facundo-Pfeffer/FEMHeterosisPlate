"""
PlateModel: collects mesh, material, element formulation, boundary conditions, and loads.

Build a model, attach boundary conditions and loads, then pass it to assemble_stiffness_matrix
and assemble_force_vector (or solve_displacement_system to do both in one call).

Global DOF layout:
    indices 0 .. n_w_node-1            :  w at each Q8 node (one DOF per node)
    indices n_w_node + 2*k             :  θ_x at theta-node k
    indices n_w_node + 2*k + 1         :  θ_y at theta-node k
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plate_fea.boundary_conditions import ElementEdgeLineLoad, ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements.base import PlateElementBase
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh


@dataclass
class PlateModel:
    """
    All inputs needed to assemble and solve one plate problem.

    Attributes:
        mesh:                    Mesh connectivity and node coordinates.
        constitutive_material:   Bending (D_b) and shear (D_s) constitutive matrices.
        element_formulation:     Element that computes local K and load vectors.
        essential_conditions:    Prescribed DOF values (Dirichlet BCs); append via add_essential_condition.
        line_loads:              Edge tractions; append via add_line_load.
        surface_loads:           Surface pressures; append via add_surface_load.
        element_stiffness_kwargs: Forwarded to element_formulation.compute_stiffness_matrix as
                                  keyword arguments — used for non-default quadrature orders in
                                  patch tests (e.g. bending_quadrature_order=(3,3)).
    """

    mesh: HeterosisMesh
    constitutive_material: PlateMaterial
    element_formulation: PlateElementBase
    essential_conditions: list[EssentialBoundaryCondition] = field(default_factory=list)
    line_loads: list[ElementEdgeLineLoad] = field(default_factory=list)
    surface_loads: list[ElementSurfaceLoad] = field(default_factory=list)
    element_stiffness_kwargs: dict[str, object] = field(default_factory=dict)

    def add_essential_condition(self, condition: EssentialBoundaryCondition) -> None:
        """Attach an essential (Dirichlet) boundary condition to the model."""
        self.essential_conditions.append(condition)

    def add_line_load(self, load: ElementEdgeLineLoad) -> None:
        """Attach a distributed edge traction to the model."""
        self.line_loads.append(load)

    def add_surface_load(self, load: ElementSurfaceLoad) -> None:
        """Attach a distributed surface pressure to the model."""
        self.surface_loads.append(load)

    def get_theta_x_dof(self, theta_node_id: int) -> int:
        """Global DOF index for θ_x at theta-node theta_node_id."""
        return self.mesh.total_w_node_number + 2 * int(theta_node_id)

    def get_theta_y_dof(self, theta_node_id: int) -> int:
        """Global DOF index for θ_y at theta-node theta_node_id."""
        return self.mesh.total_w_node_number + 2 * int(theta_node_id) + 1

    def build_essential_boundary_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten all essential conditions into sorted index and value arrays for the solver.

        Returns:
            prescribed_dof_indices: Sorted global DOF indices with prescribed values.
            prescribed_values:      Corresponding prescribed values.

        Raises:
            ValueError: If the same DOF is constrained to two different values.
        """
        dof_value_pairs: dict[int, float] = {}

        for condition in self.essential_conditions:
            for node_id in condition.node_ids:
                if condition.field_name == "w":
                    dof_id = int(node_id)
                elif condition.field_name == "theta_x":
                    dof_id = self.get_theta_x_dof(node_id)
                elif condition.field_name == "theta_y":
                    dof_id = self.get_theta_y_dof(node_id)
                else:
                    raise ValueError("field_name must be one of: 'w', 'theta_x', 'theta_y'.")

                if dof_id in dof_value_pairs and not np.isclose(dof_value_pairs[dof_id], condition.value):
                    raise ValueError(f"Conflicting essential boundary values found for dof {dof_id}.")
                dof_value_pairs[dof_id] = condition.value

        prescribed_dof_indices = np.array(sorted(dof_value_pairs.keys()), dtype=int)
        prescribed_values = np.array([dof_value_pairs[dof_id] for dof_id in prescribed_dof_indices], dtype=float)
        return prescribed_dof_indices, prescribed_values
