from __future__ import annotations

import importlib
import numpy as np
import pytest

from plate_fea.problem_orchestrator import ProblemConfig, generate_mesh


def _gmsh_runtime_available() -> bool:
    try:
        importlib.import_module("gmsh")
        return True
    except Exception:
        return False


def _triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs(float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))


def _corner_min_cell_area(mesh, x_target: float, y_target: float) -> float:
    xy = mesh.node_coordinates
    wlm = mesh.w_location_matrix
    best_area = float("inf")
    best_dist = float("inf")
    for eid in range(wlm.shape[1]):
        n0, n1, n2, n3 = (int(wlm[i, eid]) for i in range(4))
        corners = xy[[n0, n1, n2, n3]]
        center = corners.mean(axis=0)
        dist = float(np.hypot(center[0] - x_target, center[1] - y_target))
        # Bilinear quad area via two triangles.
        tri1 = _triangle_area_2d(corners[0], corners[1], corners[2])
        tri2 = _triangle_area_2d(corners[0], corners[2], corners[3])
        area = float(tri1 + tri2)
        if dist < best_dist:
            best_dist = dist
            best_area = area
    return best_area


def test_baseline_mesh_strategy_is_available() -> None:
    base = generate_mesh(ProblemConfig(mesh_strategy="uniform_buffer_ring", resolution=2, hole_refine=2))
    assert base.total_element_number > 0


@pytest.mark.skipif(not _gmsh_runtime_available(), reason="gmsh runtime unavailable (missing package or libGLU)")
def test_gmsh_strategy_is_available_and_feature_sensitive() -> None:
    base = generate_mesh(ProblemConfig(mesh_strategy="uniform_buffer_ring", resolution=2, hole_refine=2))
    gmsh_b = generate_mesh(ProblemConfig(mesh_strategy="gmsh_boundary_sensitive", resolution=2, hole_refine=2))

    # Optional gmsh strategy must at least be a valid Q8 mesh for this case.
    assert gmsh_b.total_element_number > 0

    # Finer near BC transitions and load area than baseline.
    assert _corner_min_cell_area(gmsh_b, 0.0, 0.0) < _corner_min_cell_area(base, 0.0, 0.0)
    assert _corner_min_cell_area(gmsh_b, 500.0, 300.0) < _corner_min_cell_area(base, 500.0, 300.0)
    assert _corner_min_cell_area(gmsh_b, 250.0, 240.0) < _corner_min_cell_area(base, 250.0, 240.0)
