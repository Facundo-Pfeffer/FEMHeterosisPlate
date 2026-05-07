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
| `plot_assignment_results.py` | Solve (Gmsh mesh strategy), save nine figures under `output/assignment/`; mesh/clamp styling and element-averaged γ/M/Q plots—see §5.1 |
| `convergence_study_point_a.py` | Gmsh convergence study: point-A `|w_A|` vs mesh refinement + per-level mesh exports |

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

The script fixes `ProblemConfig(mesh_strategy="gmsh_boundary_sensitive", …)` and calls `solve_plate_problem`; it then plots results (default save dir `output/assignment/`).

Saves nine numbered figures:

```
01_mesh.png          mesh + exterior hatched clamp support (left/top), legend (clamped, Q9 centres, loaded edge, Point A); no footer caption
02_w.png             transverse displacement w (nodal contours, hole masked)
03.1_gamma_xz.png    γ_xz — constant per element (mean over 3×3 Gauss samples)
03.2_gamma_yz.png    γ_yz — same element-average convention
04.1_M_xx.png …      M_xx, M_yy, M_xy — same element-average convention
05.1_Q_x.png …       Q_x, Q_y — same element-average convention
```

### 5.2 Convergence study (Gmsh)

Default figures go under `output/convergence/` (override with `--out-dir`).

Runs the assignment problem with `mesh_strategy="gmsh_boundary_sensitive"` over a resolution ladder (default `-1..8`), plots `|w_A|` vs `N_el`, and saves one mesh PNG per level.

Outputs:
- `convergence_point_a_gmsh.png`
- `meshes_by_level/gmsh/Lxx_res_<r>/mesh_nel_<N>.png`

Negative resolution indices must use an equals sign, e.g. `--resolutions=-1,0,...`.

```bash
python scripts/convergence_study_point_a.py
python scripts/convergence_study_point_a.py --resolutions=-1,0,1,2,3
```

### 5.3 Post-processing and plotting

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
| `UniformBufferRingQ8Generator` | Plate-with-hole, uniform buffer ring around cut-out |
| `GmshBoundarySensitiveQ8Generator` | Plate-with-hole, Gmsh distance-field sizing (requires `gmsh`) |

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
