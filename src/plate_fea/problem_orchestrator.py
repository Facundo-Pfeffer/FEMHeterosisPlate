"""
High-level workflows for specific plate problems.

Convention for functions that take both ``ProblemConfig`` and another argument:
``config`` is always the first parameter (inputs / problem definition before state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from plate_fea.boundary_conditions import ElementEdgeLineLoad, EssentialBoundaryCondition
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh import HeterosisMesh
from plate_fea.mesh_generation import (
    GmshBoundarySensitiveQ8Generator,
    PlateWithHoleGeometry,
    UniformBufferRingQ8Generator,
    generate_rectangular_heterosis_mesh,
)
from plate_fea.model import PlateModel
from plate_fea.solver import solve_displacement_system

OuterEdgeName = Literal["left", "right", "bottom", "top"]
MeshStrategyName = Literal[
    "uniform_buffer_ring",
    "gmsh_boundary_sensitive",
]


@dataclass(frozen=True)
class ProblemConfig:
    """
    Plate-with-hole benchmark (assignment-style).

    Mesh strategies: ``uniform_buffer_ring`` (baseline) and
    ``gmsh_boundary_sensitive`` (requires gmsh + libGLU).

    **Consistent mm–N–MPa system:** coordinates and thickness in ``mm``; ``young_modulus`` in
    ``N/mm²`` (= MPa); line load ``hole_top_shear_load`` in ``N/mm`` (force per unit edge length,
    transverse ``w``). Default load matches **1 kN/mm downward** on the hole top edge → ``-1000``.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()

    # Mesh controls
    mesh_strategy: MeshStrategyName = "uniform_buffer_ring"
    resolution: int = 2
    hole_refine: int = 2
    buffer: float = 30.0

    # Material (figure: E = 200000 N/mm² after typo correction; ν = 0.25; t = 20 mm)
    young_modulus: float = 200000.0
    poisson_ratio: float = 0.25
    thickness: float = 20.0

    # BC and load (figure: clamp left + top; hole-top shear 1 kN/mm downward → −1000 N/mm)
    clamped_outer_edges: tuple[OuterEdgeName, ...] = ("left", "top")
    # Transverse line traction on hole top edge [N/mm]. Positive = +w; assignment uses downward → negative.
    hole_top_shear_load: float = -1000.0
    tolerance: float = 1.0e-9


@dataclass(frozen=True)
class ProblemResult:
    """Outputs of ``solve_plate_problem``: discrete system and sampled deflection."""

    model: PlateModel
    stiffness_matrix: csr_matrix
    force_vector: np.ndarray
    solution: np.ndarray
    point_a_node_id: int
    point_a_deflection: float


@dataclass(frozen=True)
class SquarePlateCaseConfig:
    """Fully clamped square plate with uniform line load on the top edge (separate from ``ProblemConfig``)."""

    side_length: float = 1.0
    nx: int = 8
    ny: int = 8

    young_modulus: float = 200.0
    poisson_ratio: float = 0.25
    thickness: float = 0.2

    # Uniform line traction on top boundary (negative = downward in w convention).
    top_edge_line_load: float = -1.0
    tolerance: float = 1.0e-9


@dataclass(frozen=True)
class SquarePlateCaseResult:
    """Outputs of ``solve_clamped_square_plate_line_load_case``."""

    model: PlateModel
    stiffness_matrix: csr_matrix
    force_vector: np.ndarray
    solution: np.ndarray
    center_node_id: int
    center_deflection: float


def generate_mesh(config: ProblemConfig) -> HeterosisMesh:
    """Build the assignment-style plate-with-hole Q8 mesh from ``config``."""
    if config.mesh_strategy == "uniform_buffer_ring":
        generator = UniformBufferRingQ8Generator(
            geometry=config.geometry,
            resolution=config.resolution,
            hole_refine=config.hole_refine,
            buffer=config.buffer,
        )
        return generator.generate()
    if config.mesh_strategy == "gmsh_boundary_sensitive":
        generator = GmshBoundarySensitiveQ8Generator(
            geometry=config.geometry,
            resolution=config.resolution,
            hole_refine=config.hole_refine,
            clamped_outer_edges=config.clamped_outer_edges,
        )
        return generator.generate()
    raise ValueError(f"Unsupported mesh_strategy: {config.mesh_strategy}")


def build_plate_model(config: ProblemConfig, mesh: HeterosisMesh) -> PlateModel:
    """Instantiate material and heterosis element, wrap with an empty ``PlateModel``."""
    constitutive_material = PlateMaterial(
        young_modulus=config.young_modulus,
        poisson_ratio=config.poisson_ratio,
        thickness=config.thickness,
    )
    element_formulation = HeterosisPlateElement()
    return PlateModel(
        mesh=mesh,
        constitutive_material=constitutive_material,
        element_formulation=element_formulation,
    )


def apply_essential_boundary_conditions(config: ProblemConfig, model: PlateModel) -> None:
    """Clamp selected outer edges: ``w`` and both rotations zero on boundary w-nodes."""
    edge_to_axis_value: dict[OuterEdgeName, tuple[str, float]] = {
        "left": ("x", 0.0),
        "right": ("x", config.geometry.outer_width),
        "bottom": ("y", 0.0),
        "top": ("y", config.geometry.outer_height),
    }
    for edge_name in config.clamped_outer_edges:
        axis_label, value = edge_to_axis_value[edge_name]
        w_nodes = model.mesh.find_w_nodes_on_line(axis=axis_label, value=value, tolerance=config.tolerance)
        for field_name in ("w", "theta_x", "theta_y"):
            model.add_essential_condition(
                EssentialBoundaryCondition(field_name=field_name, node_ids=w_nodes.tolist(), value=0.0)
            )


def apply_hole_top_line_loads(config: ProblemConfig, model: PlateModel) -> None:
    """Distribute ``hole_top_shear_load`` as line loads on element edges along the hole top."""
    y_target = config.geometry.hole_y_max
    x_lower = config.geometry.hole_x_min
    x_upper = config.geometry.hole_x_max
    node_xy = model.mesh.node_coordinates
    w_lm = model.mesh.w_location_matrix  # shape: (8, n_element)
    tol = config.tolerance

    # Vectorized edge detection over all elements/edges using connectivity + node coordinates.
    # This avoids per-element geometry extraction and nested Python loops over coordinates.
    for edge_id, local_edge_nodes in HeterosisPlateElement.local_edge_nodes.items():
        edge_node_ids = w_lm[local_edge_nodes, :]  # shape: (3, n_element)
        edge_x = node_xy[edge_node_ids, 0]  # shape: (3, n_element)
        edge_y = node_xy[edge_node_ids, 1]  # shape: (3, n_element)

        on_hole_top_y = np.all(np.abs(edge_y - y_target) <= tol, axis=0)
        within_hole_top_x = np.all((edge_x >= x_lower - tol) & (edge_x <= x_upper + tol), axis=0)
        hit_elements = np.flatnonzero(on_hole_top_y & within_hole_top_x)

        for element_id in hit_elements.tolist():
            model.add_line_load(
                ElementEdgeLineLoad(
                    element_id=int(element_id),
                    edge_id=edge_id,
                    magnitude=config.hole_top_shear_load,
                )
            )


def extract_point_a_deflection(config: ProblemConfig, mesh: HeterosisMesh, solution: np.ndarray) -> tuple[int, float]:
    """Nearest mesh node to the hole corner (x_max, y_min); return its index and ``w``."""
    target = np.array([config.geometry.hole_x_max, config.geometry.hole_y_min], dtype=float)
    distances = np.linalg.norm(mesh.node_coordinates - target[None, :], axis=1)
    node_id = int(np.argmin(distances))
    w_a = float(solution[node_id])
    return node_id, w_a


def solve_plate_problem(config: ProblemConfig = ProblemConfig()) -> ProblemResult:
    """Full driver: mesh → model → BCs → hole load → K/F → u → sample deflection at point A."""
    mesh = generate_mesh(config)
    model = build_plate_model(config, mesh)
    apply_essential_boundary_conditions(config, model)
    apply_hole_top_line_loads(config, model)
    stiffness, force, solution = solve_displacement_system(model)
    point_a_node_id, point_a_deflection = extract_point_a_deflection(config, mesh, solution)
    return ProblemResult(
        model=model,
        stiffness_matrix=stiffness,
        force_vector=force,
        solution=solution,
        point_a_node_id=point_a_node_id,
        point_a_deflection=point_a_deflection,
    )


def solve_clamped_square_plate_line_load_case(
    config: SquarePlateCaseConfig = SquarePlateCaseConfig(),
) -> SquarePlateCaseResult:
    """
    Solve a square plate case with all outer edges clamped and a line load on the top boundary.
    """
    mesh = generate_rectangular_heterosis_mesh(
        width=config.side_length,
        height=config.side_length,
        nx=config.nx,
        ny=config.ny,
    )
    constitutive_material = PlateMaterial(
        young_modulus=config.young_modulus,
        poisson_ratio=config.poisson_ratio,
        thickness=config.thickness,
    )
    model = PlateModel(
        mesh=mesh,
        constitutive_material=constitutive_material,
        element_formulation=HeterosisPlateElement(),
    )

    # 1) Clamp all edges: w = theta_x = theta_y = 0
    for axis, value in (("x", 0.0), ("x", config.side_length), ("y", 0.0), ("y", config.side_length)):
        edge_nodes = mesh.find_w_nodes_on_line(axis=axis, value=value, tolerance=config.tolerance)
        for field_name in ("w", "theta_x", "theta_y"):
            model.add_essential_condition(
                EssentialBoundaryCondition(field_name=field_name, node_ids=edge_nodes.tolist(), value=0.0)
            )

    # 2) Apply uniform line load on top boundary edges (y = side_length).
    node_xy = mesh.node_coordinates
    w_lm = mesh.w_location_matrix
    y_top = config.side_length
    tol = config.tolerance
    for edge_id, local_edge_nodes in HeterosisPlateElement.local_edge_nodes.items():
        edge_node_ids = w_lm[local_edge_nodes, :]
        edge_y = node_xy[edge_node_ids, 1]
        is_top_edge = np.all(np.abs(edge_y - y_top) <= tol, axis=0)
        top_elements = np.flatnonzero(is_top_edge)
        for element_id in top_elements.tolist():
            model.add_line_load(
                ElementEdgeLineLoad(
                    element_id=int(element_id),
                    edge_id=edge_id,
                    magnitude=config.top_edge_line_load,
                )
            )

    # 3) Assemble and solve
    stiffness, force, solution = solve_displacement_system(model)

    # 4) Center deflection postprocess
    center = np.array([0.5 * config.side_length, 0.5 * config.side_length], dtype=float)
    distances = np.linalg.norm(mesh.node_coordinates - center[None, :], axis=1)
    center_node_id = int(np.argmin(distances))
    center_deflection = float(solution[center_node_id])

    return SquarePlateCaseResult(
        model=model,
        stiffness_matrix=stiffness,
        force_vector=force,
        solution=solution,
        center_node_id=center_node_id,
        center_deflection=center_deflection,
    )

