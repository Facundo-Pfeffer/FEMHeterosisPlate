"""
Exact polynomial states for the heterosis plate element.

Strains used in the code:
    κ_xx = ∂θ_x/∂x,  κ_yy = ∂θ_y/∂y,  κ_xy = ∂θ_x/∂y + ∂θ_y/∂x
    γ_xz = ∂w/∂x − θ_x,  γ_yz = ∂w/∂y − θ_y

A stress-free homogeneous (no body-force) patch state with vanishing bending and shear
requires κ = 0 and γ = 0, hence θ_x and θ_y constant and θ = ∇w, so w is affine in (x, y).

The three-parameter family (independent linear patch modes) is:
    w(x, y) = a x + b y + c,   θ_x = a,   θ_y = b.

Each (a, b, c) direction can be tested separately; the catalog below uses an orthonormal
choice in coefficient space for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearHeterosisPatchState:
    """
    Affine deflection with rotations matching the slope of ``w`` (``θ = ∇w``).

    Coefficients ``(a, b, c)`` give ``w = a x + b y + c``, ``θ_x = a``, ``θ_y = b``, hence
    vanishing curvature and transverse shear in the Mindlin sense used in ``plate_fea``.
    """

    a: float
    b: float
    c: float

    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.a * x + self.b * y + self.c

    @property
    def theta_x(self) -> float:
        return float(self.a)

    @property
    def theta_y(self) -> float:
        return float(self.b)


def linear_base_catalog() -> tuple[LinearHeterosisPatchState, ...]:
    """
    Independent linear patch modes (three degrees of freedom in (a, b, c)).

    Modes are chosen as elementary directions in coefficient space.
    """
    return (
        LinearHeterosisPatchState(a=1.0, b=0.0, c=0.0),
        LinearHeterosisPatchState(a=0.0, b=1.0, c=0.0),
        LinearHeterosisPatchState(a=0.0, b=0.0, c=1.0),
    )


@dataclass(frozen=True)
class QuadraticHeterosisPatchState:
    """
    Quadratic transverse deflection with θ = ∇w (so γ = 0).

        w = p xx x² + p_xy x y + p_yy y² + (lower, optional)
        θ_x = ∂w/∂x,  θ_y = ∂w/∂y

    Bending curvature is generally non-zero; an equilibrium patch requires distributed
    ``q`` (pressure) consistent with the heterosis plate equilibrium equations. For framework
    diagnostics without deriving ``q`` analytically, use ``manufactured_force_vector``.
    """

    p_xx: float
    p_xy: float
    p_yy: float
    p_x: float = 0.0
    p_y: float = 0.0
    p_0: float = 0.0

    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (
            self.p_xx * x * x
            + self.p_xy * x * y
            + self.p_yy * y * y
            + self.p_x * x
            + self.p_y * y
            + self.p_0
        )

    def theta_x_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return 2.0 * self.p_xx * x + self.p_xy * y + self.p_x

    def theta_y_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.p_xy * x + 2.0 * self.p_yy * y + self.p_y


def quadratic_gamma_free_catalog() -> tuple[QuadraticHeterosisPatchState, ...]:
    """Independent quadratic terms in w with θ = ∇w (γ = 0, κ ≠ 0 in general)."""
    return (
        QuadraticHeterosisPatchState(p_xx=1.0, p_xy=0.0, p_yy=0.0),
        QuadraticHeterosisPatchState(p_xx=0.0, p_xy=1.0, p_yy=0.0),
        QuadraticHeterosisPatchState(p_xx=0.0, p_xy=0.0, p_yy=1.0),
    )
