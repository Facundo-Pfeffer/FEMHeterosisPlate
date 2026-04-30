from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.model import PlateModel

from ._helpers import (
    OUTPUT_DIR,
    RepresentableDistortedPatchKinematicField,
    assemble_representable_distorted_patch_displacement_vector,
    build_distorted_five_element_patch,
    compute_generalized_strains_at_quadrature_points,
    evaluate_expected_distorted_patch_strains_at_sample_points,
    plot_field_maps,
    plot_patch_geometry,
)

pytestmark = pytest.mark.integration

STRAIN_FIELD_NAMES = ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz")


def build_distorted_patch_model() -> tuple[HeterosisMesh, PlateModel]:
    """Create the mesh and plate model used by the distorted-patch strain tests."""
    mesh = build_distorted_five_element_patch()
    model = PlateModel(
        mesh=mesh,
        constitutive_material=PlateMaterial(
            young_modulus=200000.0,
            poisson_ratio=0.25,
            thickness=0.2,
        ),
        element_formulation=HeterosisPlateElement(),
    )
    return mesh, model


def assert_sampled_strains_match_expected_strains(
    sampled_strain_fields: dict[str, np.ndarray],
    expected_strain_fields: dict[str, np.ndarray],
    max_abs_error_tolerance_by_field: dict[str, float],
) -> None:
    """Compare each sampled FEM strain value against the analytical value at the same point."""
    for strain_field_name, max_abs_error_tolerance in max_abs_error_tolerance_by_field.items():
        sampled_values = sampled_strain_fields[strain_field_name]
        expected_values = expected_strain_fields[strain_field_name]

        assert sampled_values.size > 0
        assert sampled_values.shape == expected_values.shape
        assert np.all(np.isfinite(sampled_values))
        assert np.all(np.isfinite(expected_values))

        max_abs_error = float(np.max(np.abs(sampled_values - expected_values)))
        assert max_abs_error <= max_abs_error_tolerance, (
            f"{strain_field_name} exceeded tolerance: "
            f"max_abs_error={max_abs_error:.3e}, "
            f"tolerance={max_abs_error_tolerance:.3e}"
        )


def test_distorted_patch_representable_strain_fields_checked_componentwise() -> None:
    """
    Check generalized strain fields on a distorted patch using a representable field.

    The global displacement vector is assembled from:
    - a linear transverse displacement field w(x, y)
    - linear rotation fields theta_x(x, y) and theta_y(x, y)

    The analytical bending strains are constant. The analytical shear strains are
    evaluated at the same quadrature-point coordinates used to sample the FEM strains.
    """
    mesh, model = build_distorted_patch_model()

    global_displacement_vector, prescribed_kinematic_field = (
        assemble_representable_distorted_patch_displacement_vector(mesh, model)
    )

    sampled_strain_fields = compute_generalized_strains_at_quadrature_points(
        mesh,
        model,
        global_displacement_vector,
    )
    expected_strain_fields = evaluate_expected_distorted_patch_strains_at_sample_points(
        sampled_strain_fields,
        prescribed_kinematic_field,
    )

    max_abs_error_tolerance_by_field = {
        "kappa_xx": 1.0e-10,
        "kappa_yy": 1.0e-10,
        "kappa_xy": 1.0e-10,
        "gamma_xz": 1.0e-10,
        "gamma_yz": 1.0e-10,
    }

    assert_sampled_strains_match_expected_strains(
        sampled_strain_fields,
        expected_strain_fields,
        max_abs_error_tolerance_by_field,
    )


def test_distorted_patch_representable_strain_plots_and_report_saved_to_output() -> None:
    """
    Save geometry, strain-field plots, and a text report for the distorted-patch test.
    """
    mesh, model = build_distorted_patch_model()

    global_displacement_vector, prescribed_kinematic_field = (
        assemble_representable_distorted_patch_displacement_vector(mesh, model)
    )

    sampled_strain_fields = compute_generalized_strains_at_quadrature_points(
        mesh,
        model,
        global_displacement_vector,
    )
    expected_strain_fields = evaluate_expected_distorted_patch_strains_at_sample_points(
        sampled_strain_fields,
        prescribed_kinematic_field,
    )

    output_directory = OUTPUT_DIR / "patch_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)

    geometry_plot_path = plot_patch_geometry(mesh, output_directory)
    strain_field_plot_path = plot_field_maps(
        sampled_strain_fields,
        expected_strain_fields,
        output_directory,
    )

    report_path = output_directory / "strain_field_report.txt"
    report_lines = build_distorted_patch_strain_report_lines(
        prescribed_kinematic_field=prescribed_kinematic_field,
        sampled_strain_fields=sampled_strain_fields,
        expected_strain_fields=expected_strain_fields,
        geometry_plot_path=geometry_plot_path,
        strain_field_plot_path=strain_field_plot_path,
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    assert geometry_plot_path.exists()
    assert strain_field_plot_path.exists()
    assert report_path.exists()


def build_distorted_patch_strain_report_lines(
    *,
    prescribed_kinematic_field: RepresentableDistortedPatchKinematicField,
    sampled_strain_fields: dict[str, np.ndarray],
    expected_strain_fields: dict[str, np.ndarray],
    geometry_plot_path: Path,
    strain_field_plot_path: Path,
) -> list[str]:
    """Build the diagnostic text report for the distorted-patch strain test."""
    report_lines = [
        "Patch diagnostic report (distorted 5-element enclosing patch)",
        "",
        "Element naming in figure: E_top, E_left, E_center, E_right, E_bottom",
        "Checked fields: kappa_xx, kappa_yy, kappa_xy, gamma_xz, gamma_yz",
        "Sampling: all elements, 3x3 quadrature points per element",
        "",
        "Prescribed representable field:",
        (
            f"- w(x, y) = {prescribed_kinematic_field.w_x_slope:+.8f} x "
            f"{prescribed_kinematic_field.w_y_slope:+.8f} y "
            f"{prescribed_kinematic_field.w_offset:+.8f}"
        ),
        (
            f"- theta_x(x, y) = {prescribed_kinematic_field.curvature_xx:+.8f} x "
            f"{0.5 * prescribed_kinematic_field.curvature_xy:+.8f} y "
            f"{prescribed_kinematic_field.theta_x_intercept:+.8f}"
        ),
        (
            f"- theta_y(x, y) = {0.5 * prescribed_kinematic_field.curvature_xy:+.8f} x "
            f"{prescribed_kinematic_field.curvature_yy:+.8f} y "
            f"{prescribed_kinematic_field.theta_y_intercept:+.8f}"
        ),
        "",
        "Expected strain fields:",
        "- kappa_xx is constant.",
        "- kappa_yy is constant.",
        "- kappa_xy is constant.",
        "- gamma_xz is evaluated as w_,x - theta_x at each sampled point.",
        "- gamma_yz is evaluated as w_,y - theta_y at each sampled point.",
        "",
    ]

    for strain_field_name in STRAIN_FIELD_NAMES:
        sampled_values = sampled_strain_fields[strain_field_name]
        expected_values = expected_strain_fields[strain_field_name]
        error_values = sampled_values - expected_values

        report_lines.append(
            f"{strain_field_name}: "
            f"expected_min={expected_values.min():+.8f}, "
            f"expected_max={expected_values.max():+.8f}, "
            f"sampled_min={sampled_values.min():+.8f}, "
            f"sampled_max={sampled_values.max():+.8f}, "
            f"max|err|={np.max(np.abs(error_values)):.3e}"
        )

    report_lines.extend(
        [
            "",
            "Access chain used in test:",
            "- local_dofs = HeterosisPlateElement.local_to_global_dof_indices(mesh, element_id)",
            "- u_local = u[local_dofs]",
            "- kappa = HeterosisPlateElement.bending_B_matrix(...) @ u_local",
            "- gamma = HeterosisPlateElement.shear_B_matrix(...) @ u_local",
            "",
            f"Generated geometry figure: {geometry_plot_path.name}",
            f"Generated strain-field figure: {strain_field_plot_path.name}",
        ]
    )

    return report_lines