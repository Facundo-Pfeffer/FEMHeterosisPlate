from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.model import PlateModel

from ._helpers import (
    OUTPUT_DIR,
    assemble_polynomial_state,
    build_distorted_five_element_patch,
    compute_generalized_strains_at_quadrature_points,
    plot_patch_geometry,
)

matplotlib.use("Agg")
pytestmark = pytest.mark.integration


def test_patch_strain_fields_checked_componentwise() -> None:
    """
    Validate sampled generalized strains against expected constants on the full patch.

    Checked fields:
    - bending: kappa_xx, kappa_yy, kappa_xy
    - shear:   gamma_xz, gamma_yz

    Where values are checked:
    - all 5 elements in the enclosing patch
    - all 3x3 Gauss points per element (9 points/element, 45 total samples per field)

    How values are obtained:
    - samples are computed by `compute_generalized_strains_at_quadrature_points(...)`, which evaluates element B-matrix
      strains at each quadrature point in each element.
    """
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2),
        element_formulation=HeterosisPlateElement(),
    )
    u, expected = assemble_polynomial_state(mesh, model)
    samples = compute_generalized_strains_at_quadrature_points(mesh, model, u)

    max_abs_error_tolerance = {
        "kappa_xx": 1.0e-12,
        "kappa_yy": 1.0e-12,
        "kappa_xy": 1.0e-12,
        # Distorted enclosing patch is not an exact pointwise shear patch test.
        "gamma_xz": 7.0e-2,
        "gamma_yz": 7.0e-2,
    }
    spread_tolerance = {
        "kappa_xx": 1.0e-12,
        "kappa_yy": 1.0e-12,
        "kappa_xy": 1.0e-12,
        "gamma_xz": 1.5e-1,
        "gamma_yz": 1.5e-1,
    }

    # The comparison is performed for every sampled value (all elements, all 3x3 points).
    for name in ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz"):
        values = samples[name]
        assert values.size > 0
        assert np.all(np.isfinite(values))

        spread = float(values.max() - values.min())
        max_abs_error = float(abs(values - expected[name]).max())
        assert spread <= spread_tolerance[name]
        assert max_abs_error <= max_abs_error_tolerance[name]


def test_patch_strain_plots_and_report_saved_to_output() -> None:
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2),
        element_formulation=HeterosisPlateElement(),
    )
    u, expected = assemble_polynomial_state(mesh, model)
    samples = compute_generalized_strains_at_quadrature_points(mesh, model, u)

    outdir = OUTPUT_DIR / "patch_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = plot_patch_geometry(mesh, outdir)

    report_path = outdir / "strain_field_report.txt"
    lines = [
        "Patch diagnostic report (distorted 5-element enclosing patch)",
        "",
        "Element naming in figure: E_top, E_left, E_center, E_right, E_bottom",
        "Checked fields: kappa_xx, kappa_yy, kappa_xy, gamma_xz, gamma_yz",
        "Sampling: all elements, 3x3 quadrature points per element",
        "",
    ]
    for name in ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz"):
        v = samples[name]
        spread = float(v.max() - v.min())
        err = float(abs(v - expected[name]).max())
        lines.append(
            f"{name}: expected={expected[name]:+.8f}, min={v.min():+.8f}, max={v.max():+.8f}, spread={spread:.3e}, max|err|={err:.3e}"
        )
    lines.extend(
        [
            "",
            "Access chain used in test:",
            "- local_dofs = HeterosisPlateElement.local_to_global_dof_indices(mesh, element_id)",
            "- u_local = u[local_dofs]",
            "- kappa = HeterosisPlateElement.bending_B_matrix(...) @ u_local",
            "- gamma = HeterosisPlateElement.shear_B_matrix(...) @ u_local",
            "",
            f"Generated figure: {p1.name}",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    assert p1.exists()
    assert report_path.exists()
