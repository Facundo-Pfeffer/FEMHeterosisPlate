"""
Distorted 5-element patch tests for the Heterosis plate element.

The patch has a genuinely distorted centre element. Two tests run on it:
  1. Strain accuracy — all five generalized strains must match the analytical field to machine precision.
  2. Output files — geometry plot, strain-field maps, and a text report are written for manual inspection.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.model import PlateModel

from ._helpers import (
    OUTPUT_DIR,
    assemble_representable_distorted_patch_displacement_vector,
    build_distorted_five_element_patch,
    build_distorted_patch_strain_report_lines,
    compute_generalized_strains_at_quadrature_points,
    evaluate_expected_distorted_patch_strains_at_sample_points,
    plot_field_maps,
    plot_patch_geometry,
)

pytestmark = pytest.mark.patch


def test_distorted_patch_exact_strain_recovery() -> None:
    """
    Generalized strains must match analytical values to machine precision on the distorted patch.

    The prescribed field uses linear w and linear theta, which Q8/Q9 represent exactly on
    any element geometry — including distorted ones. All five strain components (kappa_xx,
    kappa_yy, kappa_xy, gamma_xz, gamma_yz) must agree with the analytical field to 1e-10.
    """
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2),
        element_formulation=HeterosisPlateElement(),
    )

    displacement, kinematic_field = assemble_representable_distorted_patch_displacement_vector(mesh, model)
    sampled = compute_generalized_strains_at_quadrature_points(mesh, model, displacement)
    expected = evaluate_expected_distorted_patch_strains_at_sample_points(sampled, kinematic_field)

    for name in ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz"):
        np.testing.assert_allclose(
            sampled[name],
            expected[name],
            atol=1e-10,
            err_msg=f"{name}: FEM vs analytical exceeds tolerance",
        )


def test_distorted_patch_output_files_saved() -> None:
    """Save geometry plot, strain-field maps, and a text report for manual inspection."""
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2),
        element_formulation=HeterosisPlateElement(),
    )

    displacement, kinematic_field = assemble_representable_distorted_patch_displacement_vector(mesh, model)
    sampled = compute_generalized_strains_at_quadrature_points(mesh, model, displacement)
    expected = evaluate_expected_distorted_patch_strains_at_sample_points(sampled, kinematic_field)

    output_directory = OUTPUT_DIR / "patch_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)

    geometry_plot_path = plot_patch_geometry(mesh, output_directory)
    strain_field_plot_path = plot_field_maps(sampled, expected, output_directory)

    report_path = output_directory / "strain_field_report.txt"
    report_path.write_text(
        "\n".join(
            build_distorted_patch_strain_report_lines(
                prescribed_kinematic_field=kinematic_field,
                sampled_strain_fields=sampled,
                expected_strain_fields=expected,
                geometry_plot_path=geometry_plot_path,
                strain_field_plot_path=strain_field_plot_path,
            )
        ),
        encoding="utf-8",
    )

    assert geometry_plot_path.exists()
    assert strain_field_plot_path.exists()
    assert report_path.exists()
