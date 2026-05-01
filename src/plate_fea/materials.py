"""
Isotropic Reissner-Mindlin plate material.

PlateMaterial precomputes and caches the bending constitutive matrix D_b (3×3) and shear
constitutive matrix D_s (2×2) from Young's modulus, Poisson's ratio, plate thickness, and
shear correction factor. The cached matrices are read-only and reused by every element
integration point without reallocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlateMaterial:
    young_modulus: float
    poisson_ratio: float
    thickness: float
    shear_correction_factor: float = 5.0 / 6.0

    def __post_init__(self) -> None:
        # Precompute constitutive matrices once. Assembly touches these properties per element;
        # recomputing and reallocating on every access was O(n_element) redundant work.
        nu = self.poisson_ratio
        young_modulus = self.young_modulus
        thickness = self.thickness
        shear_correction = self.shear_correction_factor
        shear_modulus = young_modulus / (2.0 * (1.0 + nu))
        bending_stiffness_factor = young_modulus * thickness**3 / (12.0 * (1.0 - nu**2))
        d_b = bending_stiffness_factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=float,
        )
        shear_stiffness_factor = shear_correction * shear_modulus * thickness
        d_s = shear_stiffness_factor * np.eye(2, dtype=float)
        d_b.setflags(write=False)
        d_s.setflags(write=False)
        object.__setattr__(self, "_bending_constitutive_matrix", d_b)
        object.__setattr__(self, "_shear_constitutive_matrix", d_s)

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bending_constitutive_matrix(self) -> np.ndarray:
        return self._bending_constitutive_matrix

    @property
    def shear_constitutive_matrix(self) -> np.ndarray:
        return self._shear_constitutive_matrix
