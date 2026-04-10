from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh


class PlateElementBase(ABC):
    @abstractmethod
    def compute_stiffness_matrix(
        self,
        mesh: HeterosisMesh,
        material: PlateMaterial,
        element_id: int,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def local_to_global_dof_indices(
        self,
        mesh: HeterosisMesh,
        element_id: int,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_edge_force_vector(
        self,
        mesh: HeterosisMesh,
        element_id: int,
        edge_id: int,
        traction: float | Callable[[float, float], float],
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_surface_force_vector(
        self,
        mesh: HeterosisMesh,
        element_id: int,
        traction: float | Callable[[float, float], float],
    ) -> np.ndarray:
        raise NotImplementedError
