"""
Heterosis plate finite element package.

Solve pipeline (manual workflow)::

    mesh         = generate_rectangular_heterosis_mesh(width, height, nx, ny)
    material     = PlateMaterial(young_modulus=..., poisson_ratio=..., thickness=...)
    model        = PlateModel(mesh, material, HeterosisPlateElement())
    # attach boundary conditions and loads to model ...
    stiffness, force, displacement = solve_displacement_system(model)

Post-process and plot results::

    fields = sample_fields_at_quadrature_points(mesh, model, displacement)
    fig = plot_all_result_fields(mesh, model, displacement)   # 8-panel summary
    fig, ax = plot_w_field(mesh, displacement)                # displacement only
    fig, ax = plot_field_at_quadrature_points(               # any single field
        fields.x, fields.y, fields.gamma_xz, title=r"$\\gamma_{xz}$"
    )

One-call driver for the plate-with-hole benchmark::

    result = solve_plate_problem(ProblemConfig(...))

Module map:

    mesh.py                  HeterosisMesh — Q8/Q9 node layout and connectivity
    mesh_generation.py       Mesh generators for rectangles and plate-with-hole geometry
    materials.py             PlateMaterial — D_b and D_s constitutive matrices
    boundary_conditions.py   EssentialBoundaryCondition, ElementEdgeLineLoad, ElementSurfaceLoad
    model.py                 PlateModel — collects mesh + material + element + BCs + loads
    elements/                HeterosisPlateElement — local K, B-matrices, DOF scatter
    assembly.py              assemble_stiffness_matrix, assemble_force_vector
    solver.py                solve_linear_system, solve_displacement_system
    problem_orchestrator.py  High-level drivers: solve_plate_problem, ProblemConfig
    quadrature.py            Gauss-Legendre rules used inside element integration
    postprocessing.py        sample_fields_at_quadrature_points → SampledFields
    plotting.py              plot_w_field, plot_field_at_quadrature_points, plot_all_result_fields
    reference_solutions.py   Kirchhoff analytical solutions for SSSS/CCCC benchmarks
"""

from __future__ import annotations

# ── Mesh and geometry ──────────────────────────────────────────────────────────
from .mesh import HeterosisMesh
from .mesh_generation import (
    GmshBoundarySensitiveQ8Generator,
    PlateWithHoleGeometry,
    UniformBufferRingQ8Generator,
    generate_rectangular_heterosis_mesh,
)

# ── Material and element ───────────────────────────────────────────────────────
from .materials import PlateMaterial
from .elements import HeterosisPlateElement

# ── Model, boundary conditions, and loads ─────────────────────────────────────
from .model import PlateModel
from .boundary_conditions import (
    EssentialBoundaryCondition,
    ElementEdgeLineLoad,
    ElementSurfaceLoad,
    NodalPointLoad,
)

# ── Assembly and solve ─────────────────────────────────────────────────────────
from .assembly import assemble_force_vector, assemble_stiffness_matrix
from .solver import solve_displacement_system, solve_linear_system

# ── High-level drivers ─────────────────────────────────────────────────────────
from .problem_orchestrator import ProblemConfig, ProblemResult, solve_plate_problem

# ── Post-processing ────────────────────────────────────────────────────────────
from .postprocessing import SampledFields, sample_fields_at_quadrature_points

# ── Visualization ──────────────────────────────────────────────────────────────
from .plotting import (
    apply_report_style,
    plot_all_result_fields,
    plot_field_at_quadrature_points,
    plot_heterosis_mesh,
    plot_w_field,
)

__all__ = [
    # Mesh and geometry
    "HeterosisMesh",
    "PlateWithHoleGeometry",
    "generate_rectangular_heterosis_mesh",
    "GmshBoundarySensitiveQ8Generator",
    "UniformBufferRingQ8Generator",
    # Material and element
    "PlateMaterial",
    "HeterosisPlateElement",
    # Model, boundary conditions, and loads
    "PlateModel",
    "EssentialBoundaryCondition",
    "ElementEdgeLineLoad",
    "ElementSurfaceLoad",
    # Assembly and solve
    "assemble_stiffness_matrix",
    "assemble_force_vector",
    "solve_linear_system",
    "solve_displacement_system",
    # High-level drivers
    "ProblemConfig",
    "ProblemResult",
    "solve_plate_problem",
    # Post-processing
    "SampledFields",
    "sample_fields_at_quadrature_points",
    # Visualization
    "apply_report_style",
    "plot_heterosis_mesh",
    "plot_w_field",
    "plot_field_at_quadrature_points",
    "plot_all_result_fields",
]
