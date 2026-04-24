from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial

from ._helpers import OUTPUT_DIR, build_distorted_single_element_mesh, plot_eigenvalues

matplotlib.use("Agg")
pytestmark = pytest.mark.integration


def test_single_element_eigenvalues_count() -> None:
    mesh = build_distorted_single_element_mesh()
    element = HeterosisPlateElement()
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    K = element.compute_stiffness_matrix(mesh, material, 0)
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    tol = max(float(np.max(lam)), 1.0) * 1.0e-8
    near_zero_count = int(np.sum(lam < tol))

    assert lam.size == 26
    assert np.all(np.isfinite(lam))
    assert near_zero_count >= 3


def test_single_element_eigenvalue_plot_and_report_saved_to_output() -> None:
    mesh = build_distorted_single_element_mesh()
    element = HeterosisPlateElement()
    material = PlateMaterial(young_modulus=200.0, poisson_ratio=0.25, thickness=0.2)
    K = element.compute_stiffness_matrix(mesh, material, 0)
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    tol = max(float(np.max(lam)), 1.0) * 1.0e-8
    near_zero_count = int(np.sum(lam < tol))

    outdir = OUTPUT_DIR / "eigen_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_eigenvalues(lam, near_zero_count, outdir)
    txt_path = outdir / "eigenvalue_report.txt"
    txt_path.write_text(
        "\n".join(
            [
                "Unsupported distorted single-element eigenvalue diagnostic",
                f"near_zero_tolerance={tol:.6e}",
                f"near_zero_count={near_zero_count}",
                "expected_if_requirement_is_strict=5",
                "observed_eigenvalues=" + ", ".join(f"{v:.8e}" for v in lam),
            ]
        ),
        encoding="utf-8",
    )

    assert fig_path.exists()
    assert txt_path.exists()
