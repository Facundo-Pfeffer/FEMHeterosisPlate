from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class GaussRule:
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

    point_list: list[list[float]] = []
    weight_list: list[float] = []
    for i, xi in enumerate(rule_x.points):
        for j, eta in enumerate(rule_y.points):
            point_list.append([xi, eta])
            weight_list.append(rule_x.weights[i] * rule_y.weights[j])

    return GaussRule(
        points=_as_readonly(np.asarray(point_list, dtype=float)),
        weights=_as_readonly(np.asarray(weight_list, dtype=float)),
    )
