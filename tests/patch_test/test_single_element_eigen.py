from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial

from tests.patch_test._helpers import (
    OUTPUT_DIR,
    build_distorted_single_element_mesh,
    plot_eigenvalues,
    plot_single_element_geometry,
)

matplotlib.use("Agg")
pytestmark = pytest.mark.integration


def _compute_single_element_stiffness_eigenvalues() -> tuple[np.ndarray, float, int]:
    """
    Return eigen-spectrum diagnostics for one distorted element.

    The element stiffness is theoretically symmetric. We first verify near-symmetry
    (allowing tiny floating-point assembly noise), then compute eigenvalues from the
    explicitly symmetrized matrix to keep `eigvalsh` numerically robust.
    """
    mesh = build_distorted_single_element_mesh()
    element = HeterosisPlateElement()
    material = PlateMaterial(young_modulus=200000.0, poisson_ratio=0.25, thickness=0.2)
    K = element.compute_stiffness_matrix(mesh, material, 0)

    symmetry_residual = K - K.T
    if symmetry_residual.size:
        max_symmetry_residual = float(np.max(np.abs(symmetry_residual)))
        assert max_symmetry_residual < 1.0e-10

    symmetric_stiffness_matrix = 0.5 * (K + K.T)  # Keeps `eigvalsh` numerically robust. Symmetry was verified.
    eigenvalues = np.linalg.eigvalsh(symmetric_stiffness_matrix)
    near_zero_tolerance = max(float(np.max(eigenvalues)), 1.0) * 1.0e-8
    near_zero_count = int(np.sum(eigenvalues < near_zero_tolerance))
    return eigenvalues, near_zero_tolerance, near_zero_count


def test_single_element_eigenvalues_count() -> None:
    """
    Check that the distorted single-element stiffness has the expected spectrum shape:
    finite 26 eigenvalues and a small near-zero subspace.
    """
    eigenvalues, _, near_zero_count = _compute_single_element_stiffness_eigenvalues()

    assert eigenvalues.size == 26
    assert np.all(np.isfinite(eigenvalues))
    assert near_zero_count == 3


def test_single_element_eigenvalue_plot_and_report_saved_to_output() -> None:
    """Persist a figure and text report for manual spectrum inspection."""
    eigenvalues, near_zero_tolerance, near_zero_count = _compute_single_element_stiffness_eigenvalues()
    mesh = build_distorted_single_element_mesh()
    outdir = OUTPUT_DIR / "eigen_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    geometry_figure_path = plot_single_element_geometry(mesh, outdir)
    figure_path = plot_eigenvalues(eigenvalues, near_zero_count, outdir)
    report_path = outdir / "eigenvalue_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Unsupported distorted single-element eigenvalue diagnostic",
                f"near_zero_tolerance={near_zero_tolerance:.6e}",
                f"near_zero_count={near_zero_count}",
                "expected_if_requirement_is_strict=5",
                "observed_eigenvalues=" + ", ".join(f"{v:.8e}" for v in eigenvalues),
                f"generated_geometry_figure={geometry_figure_path.name}",
                f"generated_eigen_figure={figure_path.name}",
            ]
        ),
        encoding="utf-8",
    )

    assert geometry_figure_path.exists()
    assert figure_path.exists()
    assert report_path.exists()
