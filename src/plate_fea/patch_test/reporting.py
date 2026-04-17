"""Structured patch-test reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    NONE = "none"
    CONSISTENCY = "consistency"
    STABILITY = "stability"
    MIXED = "mixed"


@dataclass
class PatchTestCaseReport:
    test_name: str
    test_type: str
    patch_topology: str
    element_type: str
    quadrature_label: str
    geometry_label: str
    polynomial_state: str
    body_force_description: str
    natural_bc_description: str
    essential_bc_description: str
    residual_norm: float | None
    nodal_linf_error: float | None
    nodal_l2_error: float | None
    diagnostics_note: str
    rank_dense: int | None
    n_free: int | None
    min_eigenvalue: float | None
    smallest_singular_value: float | None
    load_perturbation_sensitivity: float | None
    passed: bool
    failure_class: FailureClass
    interpretation: str
    extra: dict[str, Any] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [
            f"--- {self.test_name} ---",
            f"  test_type: {self.test_type}",
            f"  patch_topology: {self.patch_topology}",
            f"  element_type: {self.element_type}",
            f"  quadrature: {self.quadrature_label}",
            f"  geometry: {self.geometry_label}",
            f"  polynomial_state: {self.polynomial_state}",
            f"  body_force / source: {self.body_force_description}",
            f"  natural BC / loads: {self.natural_bc_description}",
            f"  essential BC: {self.essential_bc_description}",
            f"  residual ||K ũ - f||: {self._fmt(self.residual_norm)}",
            f"  nodal L∞ error: {self._fmt(self.nodal_linf_error)}",
            f"  nodal L2 error: {self._fmt(self.nodal_l2_error)}",
            f"  rank(K_ff) (dense): {self.rank_dense}",
            f"  n_free: {self.n_free}",
            f"  min eigenvalue (diag): {self._fmt(self.min_eigenvalue)}",
            f"  σ_min(K_ff) dense: {self._fmt(self.smallest_singular_value)}",
            f"  load perturbation sensitivity: {self._fmt(self.load_perturbation_sensitivity)}",
            f"  diagnostics: {self.diagnostics_note}",
            f"  PASS: {self.passed}  failure_class: {self.failure_class.value}",
            f"  interpretation: {self.interpretation}",
        ]
        return out

    @staticmethod
    def _fmt(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{x:.6e}"

    def __str__(self) -> str:
        return "\n".join(self.lines())
