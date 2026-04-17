"""Isotropic heterosis plate stiffness: bending (D_b) and shear (D_s) matrices."""

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
        e = self.young_modulus
        t = self.thickness
        k = self.shear_correction_factor
        g = e / (2.0 * (1.0 + nu))
        factor_b = e * t**3 / (12.0 * (1.0 - nu**2))
        d_b = factor_b * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=float,
        )
        factor_s = k * g * t
        d_s = factor_s * np.eye(2, dtype=float)
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
