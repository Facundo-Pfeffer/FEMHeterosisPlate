"""Constitutive matrices are precomputed once per PlateMaterial (not rebuilt on each property access)."""

from __future__ import annotations

import numpy as np

from plate_fea.materials import PlateMaterial


def test_constitutive_matrices_identical_on_repeated_access() -> None:
    m = PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=20.0)
    db1 = m.bending_constitutive_matrix
    db2 = m.bending_constitutive_matrix
    ds1 = m.shear_constitutive_matrix
    ds2 = m.shear_constitutive_matrix
    assert db1 is db2
    assert ds1 is ds2


def test_constitutive_mathematics_unchanged() -> None:
    """Same formulas as the original @property implementations."""
    m = PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=20.0, shear_correction_factor=5.0 / 6.0)
    nu = m.poisson_ratio
    e = m.young_modulus
    t = m.thickness
    k = m.shear_correction_factor
    g = e / (2.0 * (1.0 + nu))
    factor_b = e * t**3 / (12.0 * (1.0 - nu**2))
    expected_db = factor_b * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ],
        dtype=float,
    )
    expected_ds = (k * g * t) * np.eye(2, dtype=float)
    assert np.allclose(m.bending_constitutive_matrix, expected_db)
    assert np.allclose(m.shear_constitutive_matrix, expected_ds)


def test_cached_arrays_are_read_only() -> None:
    m = PlateMaterial(young_modulus=1.0, poisson_ratio=0.3, thickness=1.0)
    assert not m.bending_constitutive_matrix.flags.writeable
    assert not m.shear_constitutive_matrix.flags.writeable
