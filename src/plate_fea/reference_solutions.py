"""
Closed-form and series references for plate benchmarks.

Use a **single consistent system** end-to-end (e.g. SI: lengths in metres, pressure in
pascals, Young's modulus in pascals) so FE assembly and analytical formulas stay comparable.
"""

from __future__ import annotations

import numpy as np


def kirchhoff_ssss_uniform_load_center_deflection_square(
    side: float,
    pressure: float,
    young_modulus: float,
    poisson_ratio: float,
    thickness: float,
    *,
    n_series_terms: int = 120,
) -> float:
    """
    Kirchhoff square plate SSSS, uniform pressure: Navier series at the centre.

    Domain ``[0, a]^2``, load ``q``, stiffness ``D = E t^3 / (12(1 - ν^2))``:

        w(a/2,a/2) = (16 q a^4)/(π^6 D) Σ_{m,n odd} sin(mπ/2)sin(nπ/2)/(mn(m²+n²)²).

    Positive ``pressure`` is positive ``q``; deflection is positive in the same sense as ``q``.
    """
    d = young_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))  # D [N·m]
    a = side  # a [m]
    q = pressure  # q [Pa] = [N/m²]

    pi = np.pi
    series_sum = 0.0
    for m in range(1, n_series_terms + 1, 2):
        for n in range(1, n_series_terms + 1, 2):
            series_sum += (np.sin(0.5 * m * pi) * np.sin(0.5 * n * pi)) / (m * n * (m * m + n * n) ** 2)

    return (16.0 * q * a**4) / (pi**6 * d) * series_sum  # w [m]


# Centre deflection factor w D / (q a^4) for a Kirchhoff square plate, all edges clamped (CCCC),
# uniform pressure; ν ≈ 0.3 (factor varies only slightly with ν). Tabulated e.g. in
# Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*.
_KIRCHHOFF_CCCC_SQUARE_UNIFORM_CENTER_FACTOR = 0.00126532


def kirchhoff_cccc_uniform_load_center_deflection_square(
    side: float,
    pressure: float,
    young_modulus: float,
    poisson_ratio: float,
    thickness: float,
) -> float:
    """
    Kirchhoff square plate CCCC, uniform pressure: centre deflection from the classical factor.

        w(a/2, a/2) = β * q * a^4 / D,   D = E t^3 / (12(1 - ν^2)),

    with ``β ≈ 0.00126532`` for a square and ``ν ≈ 0.3``. Positive ``pressure`` is positive ``q``.
    """
    d = young_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))
    return _KIRCHHOFF_CCCC_SQUARE_UNIFORM_CENTER_FACTOR * pressure * side**4 / d  # w [m]
