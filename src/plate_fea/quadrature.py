"""
Gauss-Legendre quadrature rules for element integration.

gauss_legendre_1d(order) returns 1D point/weight pairs; tensor_product_rule(order_x, order_y)
returns the 2D tensor product used inside compute_stiffness_matrix. Both functions are
memoised via lru_cache so repeated assembly calls (one per element) allocate no new arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class GaussRule:
    """
    Gauss quadrature points and weights over [-1, 1]^n.

    For 1D rules (gauss_legendre_1d): points has shape (n,), weights has shape (n,).
    For 2D tensor-product rules (tensor_product_rule): points has shape (n, 2) where each
    row is a (xi, eta) pair, and weights has shape (n,) with the combined weight w_xi * w_eta.
    Both arrays are read-only.
    """

    points: np.ndarray
    weights: np.ndarray


def _as_readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@lru_cache(maxsize=None)
def gauss_legendre_1d(order: int) -> GaussRule:
    points, weights = np.polynomial.legendre.leggauss(order)
    return GaussRule(points=_as_readonly(np.asarray(points, dtype=float)), weights=_as_readonly(np.asarray(weights, dtype=float)))


@lru_cache(maxsize=None)
def tensor_product_rule(order_x: int, order_y: int) -> GaussRule:
    rule_x = gauss_legendre_1d(order_x)
    rule_y = gauss_legendre_1d(order_y)

    accumulated_points: list[list[float]] = []
    accumulated_weights: list[float] = []
    for xi, weight_x in zip(rule_x.points, rule_x.weights):
        for eta, weight_y in zip(rule_y.points, rule_y.weights):
            accumulated_points.append([xi, eta])
            accumulated_weights.append(float(weight_x) * float(weight_y))

    return GaussRule(
        points=_as_readonly(np.asarray(accumulated_points, dtype=float)),
        weights=_as_readonly(np.asarray(accumulated_weights, dtype=float)),
    )
