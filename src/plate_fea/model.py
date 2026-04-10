from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plate_fea.boundary_conditions import ElementEdgeLineLoad, ElementSurfaceLoad, EssentialBoundaryCondition
from plate_fea.elements.base import PlateElementBase
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh


@dataclass
class PlateModel:
    mesh: HeterosisMesh
    material: PlateMaterial
    element: PlateElementBase
    essential_conditions: list[EssentialBoundaryCondition] = field(default_factory=list)
    line_loads: list[ElementEdgeLineLoad] = field(default_factory=list)
    surface_loads: list[ElementSurfaceLoad] = field(default_factory=list)

    def add_essential_condition(self, condition: EssentialBoundaryCondition) -> None:
        self.essential_conditions.append(condition)

    def add_line_load(self, load: ElementEdgeLineLoad) -> None:
        self.line_loads.append(load)

    def add_surface_load(self, load: ElementSurfaceLoad) -> None:
        self.surface_loads.append(load)

    def get_w_dof(self, node_id: int) -> int:
        return int(node_id)

    def get_theta_x_dof(self, theta_node_id: int) -> int:
        return self.mesh.total_w_node_number + 2 * int(theta_node_id)

    def get_theta_y_dof(self, theta_node_id: int) -> int:
        return self.mesh.total_w_node_number + 2 * int(theta_node_id) + 1

    def build_essential_boundary_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        dof_value_pairs: dict[int, float] = {}

        for condition in self.essential_conditions:
            for node_id in condition.node_ids:
                if condition.field_name == "w":
                    dof_id = self.get_w_dof(node_id)
                elif condition.field_name == "theta_x":
                    dof_id = self.get_theta_x_dof(node_id)
                elif condition.field_name == "theta_y":
                    dof_id = self.get_theta_y_dof(node_id)
                else:
                    raise ValueError("field_name must be one of: 'w', 'theta_x', 'theta_y'.")

                if dof_id in dof_value_pairs and not np.isclose(dof_value_pairs[dof_id], condition.value):
                    raise ValueError(f"Conflicting essential boundary values found for dof {dof_id}.")
                dof_value_pairs[dof_id] = condition.value

        bc_ess = np.array(sorted(dof_value_pairs.keys()), dtype=int)
        bc_val = np.array([dof_value_pairs[dof_id] for dof_id in bc_ess], dtype=float)
        return bc_ess, bc_val
