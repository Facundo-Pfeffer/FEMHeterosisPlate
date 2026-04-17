from __future__ import annotations

import numpy as np

from plate_fea.elements import HeterosisPlateElement


def test_q8_partition_of_unity() -> None:
    xi = 0.23
    eta = -0.41
    N = HeterosisPlateElement.q8_shape_functions(xi, eta)
    assert np.isclose(N.sum(), 1.0)


def test_q9_partition_of_unity() -> None:
    xi = -0.17
    eta = 0.31
    N = HeterosisPlateElement.q9_shape_functions(xi, eta)
    assert np.isclose(N.sum(), 1.0)


def test_q8_kronecker_property_at_node_1() -> None:
    N = HeterosisPlateElement.q8_shape_functions(-1.0, -1.0)
    expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(N, expected)


def test_q9_kronecker_property_at_center() -> None:
    N = HeterosisPlateElement.q9_shape_functions(0.0, 0.0)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(N, expected)
