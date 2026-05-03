# FEMHeterosisPlate

Finite element implementation of a shear-deformable isotropic plate solver based on the heterosis quadrilateral element (Hughes & Cohen 1978).

- **Q8** interpolation for transverse displacement `w`
- **Q9** interpolation for rotations `θ_x`, `θ_y`
- Selective integration: **3×3** Gauss for bending, **2×2** Gauss for shear
- Sparse global assembly and direct linear solve

---

## 1) Problem target

The primary goal is to determine the transverse deflection at **Point A** (bottom-right corner of the cut-out) for a 500×300 mm plate with a centred 250×180 mm rectangular cut-out, using the heterosis element. Secondary goals include benchmark verification and convergence studies.

---

## 2) Install and run

From the repository root (conda environment `ce222fp` recommended):

```bash
pip install -e ".[dev]"
```

Run all tests:

```bash
python -m pytest tests/ -q
```

Run a specific category by mark:

```bash
python -m pytest -m patch -q               # element patch tests
python -m pytest -m benchmark -q           # classical plate benchmarks
python -m pytest -m final_project_specific -q  # plate-with-hole checks
```

---

## 3) Core formulation

Element local DOF ordering (26 DOFs per element):

```text
q_e = [w_1 … w_8,  θ_x1 θ_y1,  θ_x2 θ_y2,  …,  θ_x9 θ_y9]^T
```

Global DOF layout:

```text
u = [w (all w-nodes) | θ_x θ_y pairs (all θ-nodes)]^T
```

Index map:
- `w(node_id)           → node_id`
- `θ_x(theta_node_id)  → n_w + 2·theta_node_id`
- `θ_y(theta_node_id)  → n_w + 2·theta_node_id + 1`

---

## 4) Repository structure

### Source (`src/plate_fea/`)

```text
src/plate_fea/
├── __init__.py                # full public API — pipeline docstring + module map
├── assembly.py                # assemble_stiffness_matrix, assemble_force_vector
├── boundary_conditions.py     # EssentialBoundaryCondition, ElementEdgeLineLoad, ElementSurfaceLoad
├── materials.py               # PlateMaterial — precomputes D_b (3×3) and D_s (2×2)
├── mesh.py                    # HeterosisMesh — Q8/Q9 two-level node layout
├── mesh_generation.py         # structured and Gmsh-based Q8 generators; quarter-circle mesh
├── model.py                   # PlateModel — mesh + material + element + BCs + loads
├── postprocessing.py          # SampledFields, sample_fields_at_quadrature_points
├── plotting.py                # apply_report_style, plot_heterosis_mesh, plot_w_field,
│                              #   plot_field_at_quadrature_points, plot_all_result_fields
├── problem_orchestrator.py    # ProblemConfig, ProblemResult, solve_plate_problem
├── quadrature.py              # gauss_legendre_1d, tensor_product_rule (lru_cache)
├── reference_solutions.py     # Kirchhoff SSSS/CCCC analytical solutions
├── solver.py                  # solve_linear_system, solve_displacement_system
└── elements/
    ├── base.py                # PlateElementBase — abstract element interface
    └── heterosis_plate.py     # HeterosisPlateElement — shape functions, B-matrices, K/f
```

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `plot_assignment_results.py` | Solve the plate-with-hole problem and save all result figures to `output/assignment/` |
| `plot_results.py` | Solve a rectangular plate (SSSS or CCCC) and show result field figures |
| `convergence_study_point_a.py` | Convergence study: `w_A` vs mesh refinement for the assignment problem |
| `convergence_compare_uniform_vs_gmsh.py` | Compare uniform-buffer-ring vs Gmsh mesh strategies |
| `plot_mesh_strategies_comparison.py` | Visual comparison of mesh strategies |
| `plot_mesh.py` | Static mesh figure for the plate-with-hole |
| `plot_mesh_sliders.py` | Interactive slider control over mesh density |
| `plot_mesh_demo.py` | Single-element mesh demo |
| `run_problem.py` | Headless solve, prints `w_A` to stdout |
| `run_ssss_square_uniform_pressure.py` | SSSS square plate benchmark solve |
| `run_clamped_square_uniform_pressure.py` | CCCC square plate benchmark solve |
| `run_smoke_test.py` | Minimal end-to-end smoke test |

### Tests (`tests/`)

```text
tests/
├── unit/                       # Fast, isolated component tests (no full FEM solve)
│   ├── test_shape_functions.py
│   ├── test_material_constitutive_cache.py
│   ├── test_element_jacobian_and_stiffness.py
│   └── test_mesh_strategies.py
│
├── patch/                      # Element patch tests — strain recovery and eigenvalue spectrum
│   ├── _helpers.py             # shared mesh builders, kinematic fields, sampling helpers
│   ├── test_patch_linear_field.py
│   ├── test_simple_patch_cases.py   # constant shear (2×2 pts) and constant κ (3×3 pts)
│   ├── test_five_element_patch.py   # distorted 5-element enclosing patch
│   └── test_single_element_eigen.py
│
├── benchmarks/                 # Classical analytical comparisons
│   ├── test_ssss_uniform_pressure_vs_navier.py
│   ├── test_clamped_square_uniform_pressure.py
│   └── test_circular_plate_hughes.py   # Hughes Fig. 5.3.19 quarter-circle convergence
│
└── final_project_specific/     # CE 222 plate-with-hole assignment checks
    └── test_plate_with_hole_equilibrium.py
```

---

## 5) High-level workflows

### 5.1 Assignment problem — plate with cut-out

```bash
python scripts/plot_assignment_results.py
python scripts/plot_assignment_results.py --resolution 4 --hole-refine 3
```

Saves nine numbered figures to `output/assignment/`:

```
01_mesh.png          mesh with loaded edge and Point A
02_w.png             transverse displacement w
03.1_gamma_xz.png    shear strain γ_xz
03.2_gamma_yz.png    shear strain γ_yz
04.1_M_xx.png        bending moment M_xx
04.2_M_yy.png        bending moment M_yy
04.3_M_xy.png        bending moment M_xy
05.1_Q_x.png         shear force Q_x
05.2_Q_y.png         shear force Q_y
```

For a headless solve that just prints `w_A`:

```bash
python scripts/run_problem.py --resolution 2
```

### 5.2 Rectangular plate benchmarks

```bash
python scripts/plot_results.py                    # SSSS, 10×10 elements
python scripts/plot_results.py --bc clamped --nx 12 --ny 12
```

### 5.3 Convergence study

```bash
python scripts/convergence_study_point_a.py --out-dir output
```

Produces a convergence curve (`w_A` vs `N_el`) and a mesh gallery across eight refinement levels.

### 5.4 Post-processing and plotting

```python
from plate_fea import (
    sample_fields_at_quadrature_points,
    plot_all_result_fields,
    plot_w_field,
    plot_field_at_quadrature_points,
    apply_report_style,
)

apply_report_style()
fields = sample_fields_at_quadrature_points(mesh, model, displacement)
fig = plot_all_result_fields(mesh, model, displacement, hole_rect=(x0, x1, y0, y1))
```

Fields available on `SampledFields`: `x`, `y`, `kappa_xx`, `kappa_yy`, `kappa_xy`, `gamma_xz`, `gamma_yz`, `M_xx`, `M_yy`, `M_xy`, `Q_x`, `Q_y`.

---

## 6) Mesh generators

| Generator | Description |
|-----------|-------------|
| `generate_rectangular_heterosis_mesh` | Uniform structured Q8 grid for a rectangle |
| `generate_quarter_circle_heterosis_mesh` | Structured Q8 polar grid for a quarter-disc (Hughes circular plate benchmarks) |
| `UniformBufferRingQ8Generator` | Plate-with-hole, uniform buffer ring around cut-out |
| `GmshBoundarySensitiveQ8Generator` | Plate-with-hole, Gmsh distance-field sizing (requires `gmsh`) |

```bash
python scripts/plot_mesh.py --resolution 2 --hole-refine 2
python scripts/plot_mesh_sliders.py          # interactive
```

---

## 7) Implementation notes

- Constitutive matrices are precomputed once per `PlateMaterial` and stored read-only.
- Quadrature rules are `lru_cache`-memoised to avoid repeated allocation in element loops.
- Area Jacobian `det(∂(x,y)/∂(ξ,η))` is verified strictly positive at all integration points.
- The assembled stiffness matrix is never modified by the solver; the free/constrained partition is solved directly so `K @ u` correctly recovers both applied loads and reaction forces.
- Global sparse assembly uses SciPy LIL → CSR conversion.

---

## 8) Current status

Implemented and tested:
- Heterosis element: shape functions, geometry Jacobian, bending/shear B-matrices, local K and load vectors
- Sparse global assembly and constrained linear solve
- Plate-with-hole workflow with Gmsh and structured mesh strategies
- Post-processing: strain/stress resultant fields sampled at quadrature points (no nodal projection)
- Result plotting: individual figures per field, report-quality style matching convergence scripts
- Patch tests: constant shear (2×2 pts), constant κ (3×3 pts), distorted 5-element, eigenvalue spectrum
- Benchmarks: SSSS/CCCC rectangular plates vs Kirchhoff, circular plate convergence (Hughes Fig. 5.3.19)
- Equilibrium validation: applied force, reaction forces, and moment balance for the assignment problem
